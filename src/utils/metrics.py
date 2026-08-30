# =============================================================================
# Metrics — Evaluation metrics for the world model and baseline
# =============================================================================
"""
Provides classification and regression metrics used during training,
evaluation, and baseline comparison.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics for state prediction evaluation.

    Args:
        y_true: Ground truth state vectors (N, D).
        y_pred: Predicted state vectors (N, D).

    Returns:
        Dictionary with MSE, MAE, and per-feature RMSE.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return {"mse": float(mse), "mae": float(mae), "rmse": rmse}


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics for infiltration prediction.

    Args:
        y_true: Ground truth binary labels (N,).
        y_pred_proba: Predicted probabilities (N,).
        threshold: Classification threshold.

    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC, and FPR.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Guard against single-class splits
    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            auc = float("nan")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # False Positive Rate
    tn_fp = np.sum(y_true == 0)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fpr = float(fp / tn_fp) if tn_fp > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "false_positive_rate": float(fpr),
    }


def multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute multi-class metrics for attack stage prediction.

    Args:
        y_true: Ground truth stage labels (N,).
        y_pred: Predicted stage labels (N,).
        class_names: Optional list of stage names.

    Returns:
        Dictionary with accuracy, macro F1, per-class report, and
        confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm,
    }
