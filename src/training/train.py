# =============================================================================
# Train — End-to-end training orchestrator
# =============================================================================
"""
Command-line entry point that orchestrates the full pipeline:
  1. Load CSV data.
  2. Extract features.
  3. Preprocess & scale.
  4. Generate time windows.
  5. Create sequences.
  6. Chronological train/val/test split.
  7. Train the world model.
  8. Evaluate on the test set.
  9. Save all artefacts.

Usage:
    python -m src.training.train --data data/sample/synthetic_traffic.csv
"""

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np

# Ensure project root is on sys.path so src.* imports work
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.dataset_loader import load_csv_dataset
from src.data.feature_extractor import FeatureExtractor
from src.data.preprocessor import Preprocessor
from src.data.time_windowing import TimeWindowGenerator
from src.models.world_model import NetworkWorldModel
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model
from src.utils.helpers import load_config, set_seed, get_device, save_json, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline(
    data_path: str,
    config_path: str = None,
    nrows: int = None,
) -> Dict[str, Any]:
    """
    Execute the complete training pipeline.

    Args:
        data_path: Path to the CSV training data.
        config_path: Optional path to config.yaml.
        nrows: Optional row limit for fast iteration.

    Returns:
        Dictionary with training history and evaluation results.
    """
    config = load_config(config_path)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()
    logger.info("Device: %s | Seed: %d", device, seed)

    # ── 1. Load data ─────────────────────────────────────────────────
    df, load_meta = load_csv_dataset(data_path, config, nrows=nrows)

    # ── 2. Feature extraction ────────────────────────────────────────
    extractor = FeatureExtractor(config)
    df, feat_meta = extractor.extract(df)

    # ── 3. Preprocessing ─────────────────────────────────────────────
    preprocessor = Preprocessor(config)
    df = preprocessor.parse_timestamps(df)
    df = preprocessor.map_labels(df, config)

    # Chronological split BEFORE fitting scaler (prevent leakage)
    pp_cfg = config.get("preprocessing", {})
    test_frac = pp_cfg.get("test_size", 0.15)
    val_frac = pp_cfg.get("val_size", 0.15)

    n = len(df)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    if n_train < 10:
        raise ValueError(
            f"Not enough data for training after split: {n_train} train rows. "
            f"Provide more data or reduce test/val fractions."
        )

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info("Split: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # Fit scaler on training data only
    df_train, scaled_train = preprocessor.fit_transform(df_train)
    df_val, scaled_val = preprocessor.transform(df_val)
    df_test, scaled_test = preprocessor.transform(df_test)

    feature_columns = preprocessor.feature_columns

    # ── 4. Time windowing ────────────────────────────────────────────
    windower = TimeWindowGenerator(config)

    states_train, labels_atk_train, labels_bin_train, _ = windower.create_windows(
        df_train, scaled_train, feature_columns
    )
    states_val, labels_atk_val, labels_bin_val, _ = windower.create_windows(
        df_val, scaled_val, feature_columns
    )
    states_test, labels_atk_test, labels_bin_test, ts_test = windower.create_windows(
        df_test, scaled_test, feature_columns
    )

    # ── 5. Create sequences ──────────────────────────────────────────
    X_train, y_st_train, y_atk_train, y_bin_train = windower.create_sequences(
        states_train, labels_atk_train, labels_bin_train
    )
    X_val, y_st_val, y_atk_val, y_bin_val = windower.create_sequences(
        states_val, labels_atk_val, labels_bin_val
    )
    X_test, y_st_test, y_atk_test, y_bin_test = windower.create_sequences(
        states_test, labels_atk_test, labels_bin_test
    )

    input_dim = X_train.shape[2]
    logger.info("Input dim: %d, Sequence length: %d", input_dim, X_train.shape[1])

    # ── 6. Build model ───────────────────────────────────────────────
    model = NetworkWorldModel.from_config(input_dim, config)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ── 7. Train ─────────────────────────────────────────────────────
    trainer = Trainer(model, config, device)
    history = trainer.train(
        X_train, y_st_train, y_atk_train, y_bin_train,
        X_val, y_st_val, y_atk_val, y_bin_val,
    )

    # ── 8. Evaluate ──────────────────────────────────────────────────
    # Load best model for evaluation
    best_path = os.path.join(
        config.get("training", {}).get("save_dir", "models/saved"),
        "best_world_model.pt",
    )
    if os.path.isfile(best_path):
        import torch
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    eval_results = evaluate_model(
        model, X_test, y_st_test, y_atk_test, y_bin_test, device
    )

    # ── 9. Save artefacts ────────────────────────────────────────────
    save_dir = config.get("training", {}).get("save_dir", "models/saved")
    os.makedirs(save_dir, exist_ok=True)

    preprocessor.save(save_dir)
    save_json(
        {"feature_columns": feature_columns, "input_dim": input_dim},
        os.path.join(save_dir, "feature_meta.json"),
    )
    save_json(history, os.path.join(save_dir, "training_history.json"))
    save_json(eval_results, os.path.join(save_dir, "evaluation_results.json"))
    save_json(config, os.path.join(save_dir, "config_snapshot.json"))

    logger.info("All artefacts saved to %s", save_dir)

    return {"history": history, "evaluation": eval_results}


# ── CLI entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the Network World Model")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the CSV training dataset.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Limit the number of rows loaded (for quick iteration).",
    )
    args = parser.parse_args()
    run_training_pipeline(args.data, args.config, args.nrows)


if __name__ == "__main__":
    main()
