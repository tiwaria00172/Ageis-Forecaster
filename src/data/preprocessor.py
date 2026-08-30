# =============================================================================
# Preprocessor — Cleaning, scaling, and label encoding
# =============================================================================
"""
Handles missing/infinite values, selects numeric features, fits scalers
only on training data, and maps string labels to MITRE ATT&CK stage
integers via the configurable label mapping.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.utils.helpers import load_config, save_pickle, load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Stateful preprocessor.  Call ``fit_transform`` on training data,
    then ``transform`` on validation / test data to avoid leakage.
    """

    # Columns to never scale (identifiers, labels)
    NON_FEATURE_COLS = {
        "timestamp", "src_ip", "dst_ip", "label", "attack_stage",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()
        self.config = config
        pp = config.get("preprocessing", {})

        self.scaler_type: str = pp.get("scaler_type", "standard")
        self.missing_strategy: str = pp.get("missing_strategy", "median")
        self.handle_inf: bool = pp.get("handle_inf", True)

        self.scaler: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_numeric(self, df: pd.DataFrame) -> List[str]:
        """Select numeric columns excluding identifiers and labels."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in numeric_cols if c.lower() not in self.NON_FEATURE_COLS
        ]

    def _handle_inf_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinities with NaN and fill missing values."""
        if self.handle_inf:
            df = df.replace([np.inf, -np.inf], np.nan)

        if self.missing_strategy == "median":
            fill_values = df[self.feature_columns].median()
        elif self.missing_strategy == "mean":
            fill_values = df[self.feature_columns].mean()
        else:
            fill_values = 0

        df[self.feature_columns] = df[self.feature_columns].fillna(fill_values)
        return df

    def _create_scaler(self):
        """Instantiate the configured scaler."""
        if self.scaler_type == "robust":
            return RobustScaler()
        return StandardScaler()

    # ------------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------------

    def map_labels(
        self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Map the 'label' column to integer MITRE ATT&CK stage indices
        using the label_mapping in config.yaml.

        Creates two new columns:
          - attack_stage: integer stage (0–5)
          - is_attack: binary (0 = normal, 1 = any attack)

        Unknown labels default to stage 0 (Normal) with a warning.
        """
        if "label" not in df.columns:
            logger.warning("No 'label' column found — assuming all normal.")
            df["attack_stage"] = 0
            df["is_attack"] = 0
            return df

        cfg = config or self.config
        label_map = cfg.get("mitre", {}).get("label_mapping", {})

        # Normalise labels: strip whitespace
        df["label"] = df["label"].astype(str).str.strip()

        mapped = df["label"].map(label_map)
        unmapped = df["label"][mapped.isna()].unique()
        if len(unmapped) > 0:
            logger.warning(
                "Unmapped labels defaulting to Normal (stage 0): %s",
                unmapped.tolist(),
            )
        df["attack_stage"] = mapped.fillna(0).astype(int)
        df["is_attack"] = (df["attack_stage"] > 0).astype(int)

        logger.info(
            "Label distribution:\n%s",
            df["attack_stage"].value_counts().sort_index().to_string(),
        )
        return df

    # ------------------------------------------------------------------
    # Parse timestamps
    # ------------------------------------------------------------------

    @staticmethod
    def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to parse the 'timestamp' column into datetime.
        Falls back to row-index ordering if parsing fails.
        """
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column — using row index as order.")
            df["timestamp"] = pd.RangeIndex(len(df))
            return df

        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], infer_datetime_format=True, errors="coerce"
            )
            # If too many NaTs, fall back
            nat_ratio = df["timestamp"].isna().mean()
            if nat_ratio > 0.5:
                logger.warning(
                    "%.0f%% of timestamps unparseable — using row index.",
                    nat_ratio * 100,
                )
                df["timestamp"] = pd.RangeIndex(len(df))
            else:
                df = df.dropna(subset=["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                logger.info(
                    "Timestamps parsed.  Range: %s to %s",
                    df["timestamp"].min(),
                    df["timestamp"].max(),
                )
        except Exception as exc:
            logger.warning("Timestamp parsing failed (%s) — using row index.", exc)
            df["timestamp"] = pd.RangeIndex(len(df))

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the scaler on training data and return scaled feature matrix.

        Args:
            df: DataFrame with extracted features and labels mapped.

        Returns:
            (df, scaled_features) where scaled_features is (N, D).
        """
        self.feature_columns = self._select_numeric(df)
        logger.info("Selected %d numeric features for scaling.", len(self.feature_columns))

        df = self._handle_inf_nan(df)

        self.scaler = self._create_scaler()
        scaled = self.scaler.fit_transform(df[self.feature_columns].values)
        self.is_fitted = True

        logger.info("Scaler fitted (%s) on %d samples.", self.scaler_type, len(df))
        return df, scaled.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Scale features using the already-fitted scaler.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit_transform first.")

        # Ensure the same feature columns exist; fill missing with 0 and cast to numeric
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = self._handle_inf_nan(df)
        scaled = self.scaler.transform(df[self.feature_columns].values)
        return df, scaled.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save scaler and feature metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        save_pickle(self.scaler, os.path.join(directory, "scaler.pkl"))
        save_pickle(
            self.feature_columns,
            os.path.join(directory, "feature_columns.pkl"),
        )
        logger.info("Preprocessor saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load a previously saved preprocessor."""
        self.scaler = load_pickle(os.path.join(directory, "scaler.pkl"))
        self.feature_columns = load_pickle(
            os.path.join(directory, "feature_columns.pkl")
        )
        self.is_fitted = True
        logger.info("Preprocessor loaded from %s", directory)
