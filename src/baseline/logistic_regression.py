# =============================================================================
# Logistic Regression Baseline — Static classifier for comparison
# =============================================================================
"""
Trains a Logistic Regression classifier on the SAME processed features
used by the GRU world model and reports standard classification metrics.

This baseline demonstrates the limitation of static classifiers: they
cannot model temporal state transitions or forecast future attack
progression.  The comparison highlights the value of the temporal
world model approach.

IMPORTANT: The baseline uses individual time-window feature vectors
(not sequences), which is equivalent to treating each window
independently — exactly the limitation we aim to surpass.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.utils.metrics import (
    binary_classification_metrics,
    multiclass_classification_metrics,
)
from src.utils.helpers import save_pickle, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


STAGE_NAMES = [
    "Normal", "Reconnaissance", "Initial Access",
    "Lateral Movement", "Command & Control", "Exfiltration",
]


class LogisticRegressionBaseline:
    """
    Baseline static classifier using Logistic Regression.

    Trained on the same windowed state vectors to provide a fair comparison
    with the temporal world model.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.infil_model = LogisticRegression(
            max_iter=max_iter, random_state=random_state, solver="lbfgs"
        )
        self.stage_model = LogisticRegression(
            max_iter=max_iter, random_state=random_state,
            solver="lbfgs", multi_class="multinomial",
        )
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_binary_train: np.ndarray,
        y_attack_train: np.ndarray,
    ) -> None:
        """
        Train the baseline on individual state vectors.

        If X_train is 3D (N, seq_len, D), only the *last* timestep of
        each sequence is used — this intentionally discards temporal
        context to demonstrate the baseline's limitation.
        """
        if X_train.ndim == 3:
            # Use only the last timestep (flatten temporal context)
            X_flat = X_train[:, -1, :]
            logger.info(
                "Baseline: using last timestep only. Shape: %s → %s",
                X_train.shape, X_flat.shape,
            )
        else:
            X_flat = X_train

        logger.info("Training Logistic Regression baseline...")

        # Infiltration (binary)
        self.infil_model.fit(X_flat, y_binary_train.astype(int))

        # Attack stage (multi-class)
        unique_stages = np.unique(y_attack_train)
        if len(unique_stages) > 1:
            self.stage_model.fit(X_flat, y_attack_train.astype(int))
        else:
            logger.warning(
                "Only one stage class in training data (%s). "
                "Stage model will predict constant class.", unique_stages
            )
            # Fit anyway to avoid downstream errors
            self.stage_model.fit(X_flat, y_attack_train.astype(int))

        self.is_trained = True
        logger.info("Baseline training complete.")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_binary_test: np.ndarray,
        y_attack_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate the baseline on the test set.

        Args:
            X_test: (N, seq_len, D) or (N, D) test features.
            y_binary_test: (N,) infiltration labels.
            y_attack_test: (N,) attack stage labels.

        Returns:
            Dictionary with infiltration and stage metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Baseline not trained. Call train() first.")

        if X_test.ndim == 3:
            X_flat = X_test[:, -1, :]
        else:
            X_flat = X_test

        # Infiltration
        infil_proba = self.infil_model.predict_proba(X_flat)
        # Handle case where model only learned one class
        if infil_proba.shape[1] == 2:
            infil_prob = infil_proba[:, 1]
        else:
            infil_prob = infil_proba[:, 0]

        infil_metrics = binary_classification_metrics(
            y_binary_test.astype(int), infil_prob
        )

        # Attack stage
        stage_pred = self.stage_model.predict(X_flat)
        present = sorted(set(y_attack_test.astype(int)))
        names = [STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"Stage {i}" for i in present]
        stage_metrics = multiclass_classification_metrics(
            y_attack_test.astype(int), stage_pred, class_names=names
        )

        logger.info("=== Baseline Evaluation ===")
        logger.info(
            "Infiltration — F1: %.4f, AUC: %.4f, FPR: %.4f",
            infil_metrics["f1_score"],
            infil_metrics["auc_roc"],
            infil_metrics["false_positive_rate"],
        )
        logger.info(
            "Attack stage — Acc: %.4f, Macro-F1: %.4f",
            stage_metrics["accuracy"],
            stage_metrics["macro_f1"],
        )

        return {
            "infiltration": infil_metrics,
            "attack_stage": stage_metrics,
        }

    def save(self, directory: str) -> None:
        """Save the baseline models to disk."""
        os.makedirs(directory, exist_ok=True)
        save_pickle(self.infil_model, os.path.join(directory, "baseline_infil.pkl"))
        save_pickle(self.stage_model, os.path.join(directory, "baseline_stage.pkl"))
        logger.info("Baseline saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load baseline models from disk."""
        self.infil_model = load_pickle(os.path.join(directory, "baseline_infil.pkl"))
        self.stage_model = load_pickle(os.path.join(directory, "baseline_stage.pkl"))
        self.is_trained = True
        logger.info("Baseline loaded from %s", directory)
