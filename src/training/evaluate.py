# =============================================================================
# Evaluate — Post-training evaluation of the world model
# =============================================================================
"""
Loads the best saved model and evaluates it on the test set, reporting
regression metrics (state prediction), binary metrics (infiltration),
and multi-class metrics (attack stage).
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.world_model import NetworkWorldModel
from src.utils.metrics import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    regression_metrics,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


STAGE_NAMES = [
    "Normal", "Reconnaissance", "Initial Access",
    "Lateral Movement", "Command & Control", "Exfiltration",
]


def evaluate_model(
    model: NetworkWorldModel,
    X_test: np.ndarray,
    y_state_test: np.ndarray,
    y_attack_test: np.ndarray,
    y_binary_test: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Evaluate the trained world model on a held-out test set.

    Args:
        model: Trained NetworkWorldModel instance.
        X_test: (N, seq_len, D) test sequences.
        y_state_test: (N, D) ground-truth next states.
        y_attack_test: (N,) ground-truth attack stages.
        y_binary_test: (N,) ground-truth infiltration labels.
        device: Torch device.
        batch_size: Evaluation batch size.

    Returns:
        Dictionary with state_metrics, infiltration_metrics, and stage_metrics.
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    # Collect predictions
    all_pred_state, all_pred_infil, all_pred_stage = [], [], []

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            pred_state, pred_infil, pred_stage_logits, _ = model(batch_x)

            all_pred_state.append(pred_state.cpu().numpy())
            all_pred_infil.append(pred_infil.squeeze(-1).cpu().numpy())
            all_pred_stage.append(pred_stage_logits.cpu().numpy())

    pred_states = np.concatenate(all_pred_state, axis=0)
    pred_infil = np.concatenate(all_pred_infil, axis=0)
    pred_stage_logits = np.concatenate(all_pred_stage, axis=0)
    pred_stages = np.argmax(pred_stage_logits, axis=1)

    # ── Metrics ──────────────────────────────────────────────────────
    state_met = regression_metrics(y_state_test, pred_states)
    infil_met = binary_classification_metrics(y_binary_test, pred_infil)

    # Limit class names to classes present in ground truth
    present_classes = sorted(set(y_attack_test.astype(int)))
    class_names = [STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"Stage {i}" for i in present_classes]
    stage_met = multiclass_classification_metrics(
        y_attack_test, pred_stages, class_names=class_names
    )

    results = {
        "state_prediction": state_met,
        "infiltration_prediction": infil_met,
        "attack_stage_prediction": stage_met,
    }

    logger.info("=== Evaluation Results ===")
    logger.info("State prediction  — MSE: %.4f, MAE: %.4f", state_met["mse"], state_met["mae"])
    logger.info(
        "Infiltration      — F1: %.4f, AUC: %.4f, FPR: %.4f",
        infil_met["f1_score"], infil_met["auc_roc"], infil_met["false_positive_rate"],
    )
    logger.info(
        "Attack stage      — Acc: %.4f, Macro-F1: %.4f",
        stage_met["accuracy"], stage_met["macro_f1"],
    )

    return results


def load_and_evaluate(
    model_path: str,
    input_dim: int,
    config: Dict[str, Any],
    X_test: np.ndarray,
    y_state_test: np.ndarray,
    y_attack_test: np.ndarray,
    y_binary_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Load a saved model from disk and evaluate it.

    Args:
        model_path: Path to the .pt checkpoint.
        input_dim: Feature dimensionality.
        config: Config dictionary.
        X_test, y_*_test: Test data arrays.

    Returns:
        Evaluation results dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetworkWorldModel.from_config(input_dim, config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded from %s (epoch %d)", model_path, checkpoint.get("epoch", "?"))

    return evaluate_model(
        model, X_test, y_state_test, y_attack_test, y_binary_test, device
    )
