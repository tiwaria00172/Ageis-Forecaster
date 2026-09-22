# =============================================================================
# Metrics utilities
# =============================================================================

import numpy as np


# =============================================================================
# Regression metrics
# =============================================================================

def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics for next-state prediction.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


# =============================================================================
# Binary classification metrics
# =============================================================================

def binary_classification_metrics(y_true, y_score, threshold=0.5):
    """
    Calculate binary classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted attack probabilities.
        threshold: Probability threshold for converting scores to labels.

    Returns:
        Dictionary containing F1, precision, recall, accuracy,
        AUC-ROC and false-positive rate.
    """

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # AUC-ROC
    # -------------------------------------------------------------------------
    auc_roc = _binary_auc(y_true, y_score)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "false_positive_rate": false_positive_rate,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _binary_auc(y_true, y_score):
    """
    Calculate binary ROC-AUC using rank statistics.

    Returns 0.5 when AUC cannot be calculated because only one
    class is present.
    """

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    positives = y_true == 1
    negatives = y_true == 0

    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort scores and assign ranks.
    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.empty(len(y_score), dtype=float)

    # Handle tied scores by assigning the average rank.
    i = 0

    while i < len(sorted_scores):
        j = i + 1

        while (
            j < len(sorted_scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank

        i = j

    positive_rank_sum = np.sum(ranks[positives])

    auc = (
        positive_rank_sum
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)

    return float(auc)


# =============================================================================
# Multiclass classification metrics
# =============================================================================

def multiclass_classification_metrics(
    y_true,
    y_pred,
    class_names=None,
):
    """
    Calculate metrics for multi-class attack-stage prediction.
    """

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }

    classes = sorted(
        set(y_true.tolist()) |
        set(y_pred.tolist())
    )

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    precision_values = []
    recall_values = []
    f1_values = []

    per_class = {}

    for idx, cls in enumerate(classes):

        tp = int(
            np.sum(
                (y_true == cls) &
                (y_pred == cls)
            )
        )

        fp = int(
            np.sum(
                (y_true != cls) &
                (y_pred == cls)
            )
        )

        fn = int(
            np.sum(
                (y_true == cls) &
                (y_pred != cls)
            )
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        if class_names is not None and idx < len(class_names):
            class_name = class_names[idx]
        else:
            class_name = f"Stage {cls}"

        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(np.sum(y_true == cls)),
        }

    macro_precision = float(
        np.mean(precision_values)
    )

    macro_recall = float(
        np.mean(recall_values)
    )

    macro_f1 = float(
        np.mean(f1_values)
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }