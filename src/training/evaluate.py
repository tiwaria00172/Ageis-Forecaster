# =============================================================================
# Evaluate — Post-training evaluation of the world model
# =============================================================================
"""
Loads the best saved model and evaluates it on the test set.

Reports:
- Regression metrics for next-state prediction
- Binary classification metrics for infiltration prediction
- Confusion matrix for infiltration prediction
- Test-set benign/attack distribution
- Multi-class metrics for attack-stage prediction
"""

import os
from typing import Any, Dict, Optional

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


# =============================================================================
# Attack stage names
# =============================================================================

STAGE_NAMES = [
    "Normal",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]


# =============================================================================
# Main evaluation function
# =============================================================================

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
        model:
            Trained NetworkWorldModel instance.

        X_test:
            Test sequences with shape:
            (N, seq_len, D)

        y_state_test:
            Ground-truth next states with shape:
            (N, D)

        y_attack_test:
            Ground-truth attack stages with shape:
            (N,)

        y_binary_test:
            Ground-truth binary infiltration labels with shape:
            (N,)

        device:
            Torch device.

        batch_size:
            Evaluation batch size.

    Returns:
        Dictionary containing:
            state_prediction
            infiltration_prediction
            attack_stage_prediction
    """

    # =========================================================================
    # Device
    # =========================================================================

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    logger.info("Starting evaluation...")
    logger.info("Test sequences: %d", len(X_test))

    # =========================================================================
    # Prepare test data
    # =========================================================================

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    dataset = torch.utils.data.TensorDataset(X_tensor)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # =========================================================================
    # Prediction containers
    # =========================================================================

    all_pred_state = []
    all_pred_infil = []
    all_pred_stage = []

    # =========================================================================
    # Model inference
    # =========================================================================

    with torch.no_grad():

        for (batch_x,) in loader:

            batch_x = batch_x.to(device)

            (
                pred_state,
                pred_infil,
                pred_stage_logits,
                _,
            ) = model(batch_x)

            # -------------------------------------------------------------
            # Next-state prediction
            # -------------------------------------------------------------

            all_pred_state.append(
                pred_state.cpu().numpy()
            )

            # -------------------------------------------------------------
            # Infiltration probability
            # -------------------------------------------------------------

            all_pred_infil.append(
                pred_infil
                .squeeze(-1)
                .cpu()
                .numpy()
            )

            # -------------------------------------------------------------
            # Attack-stage logits
            # -------------------------------------------------------------

            all_pred_stage.append(
                pred_stage_logits
                .cpu()
                .numpy()
            )

    # =========================================================================
    # Combine predictions
    # =========================================================================

    pred_states = np.concatenate(
        all_pred_state,
        axis=0,
    )

    pred_infil = np.concatenate(
        all_pred_infil,
        axis=0,
    )

    pred_stage_logits = np.concatenate(
        all_pred_stage,
        axis=0,
    )

    # Convert stage logits to predicted class IDs
    pred_stages = np.argmax(
        pred_stage_logits,
        axis=1,
    )

    # =========================================================================
    # Sanity checks
    # =========================================================================

    if len(pred_states) != len(y_state_test):
        raise ValueError(
            "State prediction count does not match test labels: "
            f"{len(pred_states)} != {len(y_state_test)}"
        )

    if len(pred_infil) != len(y_binary_test):
        raise ValueError(
            "Infiltration prediction count does not match test labels: "
            f"{len(pred_infil)} != {len(y_binary_test)}"
        )

    if len(pred_stages) != len(y_attack_test):
        raise ValueError(
            "Stage prediction count does not match test labels: "
            f"{len(pred_stages)} != {len(y_attack_test)}"
        )

    # =========================================================================
    # Regression metrics
    # =========================================================================

    state_met = regression_metrics(
        y_state_test,
        pred_states,
    )

    # =========================================================================
    # Binary infiltration metrics
    # =========================================================================

    infil_met = binary_classification_metrics(
        y_binary_test,
        pred_infil,
    )

    # =========================================================================
    # IMPORTANT DIAGNOSTIC INFORMATION
    # =========================================================================

    y_binary_test_array = np.asarray(
        y_binary_test
    ).astype(int).reshape(-1)

    pred_binary = (
        np.asarray(pred_infil) >= 0.5
    ).astype(int)

    # -------------------------------------------------------------------------
    # Ground-truth distribution
    # -------------------------------------------------------------------------

    benign_count = int(
        np.sum(y_binary_test_array == 0)
    )

    attack_count = int(
        np.sum(y_binary_test_array == 1)
    )

    total_count = len(
        y_binary_test_array
    )

    logger.info(
        "Test infiltration distribution — "
        "Benign: %d | Attack: %d | Total: %d",
        benign_count,
        attack_count,
        total_count,
    )

    if total_count > 0:

        benign_percentage = (
            benign_count / total_count
        ) * 100.0

        attack_percentage = (
            attack_count / total_count
        ) * 100.0

        logger.info(
            "Test infiltration percentage — "
            "Benign: %.2f%% | Attack: %.2f%%",
            benign_percentage,
            attack_percentage,
        )

    # -------------------------------------------------------------------------
    # Predicted distribution
    # -------------------------------------------------------------------------

    predicted_benign_count = int(
        np.sum(pred_binary == 0)
    )

    predicted_attack_count = int(
        np.sum(pred_binary == 1)
    )

    logger.info(
        "Predicted infiltration distribution — "
        "Benign: %d | Attack: %d",
        predicted_benign_count,
        predicted_attack_count,
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    tp = int(
        np.sum(
            (y_binary_test_array == 1)
            & (pred_binary == 1)
        )
    )

    tn = int(
        np.sum(
            (y_binary_test_array == 0)
            & (pred_binary == 0)
        )
    )

    fp = int(
        np.sum(
            (y_binary_test_array == 0)
            & (pred_binary == 1)
        )
    )

    fn = int(
        np.sum(
            (y_binary_test_array == 1)
            & (pred_binary == 0)
        )
    )

    logger.info(
        "Infiltration confusion matrix — "
        "TP: %d | TN: %d | FP: %d | FN: %d",
        tp,
        tn,
        fp,
        fn,
    )

    # =========================================================================
    # Attack-stage metrics
    # =========================================================================

    y_attack_test_array = np.asarray(
        y_attack_test
    ).astype(int).reshape(-1)

    pred_stages_array = np.asarray(
        pred_stages
    ).astype(int).reshape(-1)

    # -------------------------------------------------------------------------
    # Classes actually present in ground truth
    # -------------------------------------------------------------------------

    present_classes = sorted(
        set(y_attack_test_array.tolist())
    )

    # IMPORTANT:
    # We create names based on the actual class IDs rather than relying
    # on the position inside the list.
    #
    # This prevents:
    #
    # [0, 2, 5]
    #
    # from accidentally becoming:
    #
    # 0 -> Normal
    # 2 -> Initial Access
    # 5 -> Exfiltration
    #
    # with incorrect positional indexing inside metrics.py.
    # -------------------------------------------------------------------------

    class_name_map = {}

    for cls in present_classes:

        if 0 <= cls < len(STAGE_NAMES):
            class_name_map[cls] = STAGE_NAMES[cls]

        else:
            class_name_map[cls] = f"Stage {cls}"

    # The current multiclass metrics function accepts a list of names
    # corresponding to the sorted classes it discovers.
    #
    # Therefore, build the names using the complete class ordering that
    # metrics.py will encounter.
    all_stage_classes = sorted(
        set(y_attack_test_array.tolist())
        |
        set(pred_stages_array.tolist())
    )

    class_names = [
        class_name_map.get(
            cls,
            STAGE_NAMES[cls]
            if 0 <= cls < len(STAGE_NAMES)
            else f"Stage {cls}",
        )
        for cls in all_stage_classes
    ]

    stage_met = multiclass_classification_metrics(
        y_attack_test_array,
        pred_stages_array,
        class_names=class_names,
    )

    # =========================================================================
    # Stage distribution diagnostics
    # =========================================================================

    logger.info(
        "Ground-truth attack-stage distribution:"
    )

    for cls in all_stage_classes:

        count = int(
            np.sum(y_attack_test_array == cls)
        )

        stage_name = (
            STAGE_NAMES[cls]
            if 0 <= cls < len(STAGE_NAMES)
            else f"Stage {cls}"
        )

        logger.info(
            "  Stage %d (%s): %d",
            cls,
            stage_name,
            count,
        )

    logger.info(
        "Predicted attack-stage distribution:"
    )

    for cls in all_stage_classes:

        count = int(
            np.sum(pred_stages_array == cls)
        )

        stage_name = (
            STAGE_NAMES[cls]
            if 0 <= cls < len(STAGE_NAMES)
            else f"Stage {cls}"
        )

        logger.info(
            "  Stage %d (%s): %d",
            cls,
            stage_name,
            count,
        )

    # =========================================================================
    # Final results
    # =========================================================================

    results = {
        "state_prediction": state_met,

        "infiltration_prediction": infil_met,

        "attack_stage_prediction": stage_met,
    }

    # =========================================================================
    # Logging
    # =========================================================================

    logger.info(
        "=== Evaluation Results ==="
    )

    logger.info(
        "State prediction  — "
        "MSE: %.4f, MAE: %.4f, R2: %.4f",
        state_met["mse"],
        state_met["mae"],
        state_met["r2"],
    )

    logger.info(
        "Infiltration      — "
        "Accuracy: %.4f, Precision: %.4f, "
        "Recall: %.4f, F1: %.4f, "
        "AUC: %.4f, FPR: %.4f",
        infil_met["accuracy"],
        infil_met["precision"],
        infil_met["recall"],
        infil_met["f1_score"],
        infil_met["auc_roc"],
        infil_met["false_positive_rate"],
    )

    logger.info(
        "Attack stage      — "
        "Acc: %.4f, Macro-F1: %.4f",
        stage_met["accuracy"],
        stage_met["macro_f1"],
    )

    return results


# =============================================================================
# Load model and evaluate
# =============================================================================

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
        model_path:
            Path to saved .pt checkpoint.

        input_dim:
            Number of input features.

        config:
            Model configuration.

        X_test:
            Test sequences.

        y_state_test:
            Ground-truth next states.

        y_attack_test:
            Ground-truth attack stages.

        y_binary_test:
            Ground-truth binary infiltration labels.

    Returns:
        Evaluation results dictionary.
    """

    # =========================================================================
    # Device
    # =========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================================================================
    # Check checkpoint
    # =========================================================================

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    logger.info(
        "Loading model from %s",
        model_path,
    )

    # =========================================================================
    # Create model
    # =========================================================================

    model = NetworkWorldModel.from_config(
        input_dim,
        config,
    )

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    logger.info(
        "Model loaded from %s (epoch %s)",
        model_path,
        checkpoint.get("epoch", "?"),
    )

    # =========================================================================
    # Evaluate
    # =========================================================================

    return evaluate_model(
        model=model,
        X_test=X_test,
        y_state_test=y_state_test,
        y_attack_test=y_attack_test,
        y_binary_test=y_binary_test,
        device=device,
    )