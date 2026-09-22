# =============================================================================
# Preprocessor — Cleaning, scaling, and label encoding
# =============================================================================
"""
Handles:
- Missing/infinite values
- Numeric feature selection
- Timestamp parsing and chronological sorting
- Label -> MITRE ATT&CK stage mapping
- Binary attack flag generation
- Feature scaling without target leakage
- Saving/loading preprocessing artifacts
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
    Stateful preprocessor.

    Training:
        fit_transform()

    Validation/test/inference:
        transform()

    The scaler is fitted ONLY on training data to avoid data leakage.
    """

    # ------------------------------------------------------------------
    # Columns that must NEVER become model input features.
    #
    # Important:
    # - attack_stage is the multiclass target
    # - is_attack is the binary target
    #
    # Keeping is_attack out of the features prevents target leakage.
    # ------------------------------------------------------------------
    NON_FEATURE_COLS = {
        "timestamp",
        "src_ip",
        "dst_ip",
        "label",
        "attack_stage",
        "is_attack",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()

        self.config = config

        pp = config.get("preprocessing", {})

        self.scaler_type: str = pp.get(
            "scaler_type",
            "standard"
        )

        self.missing_strategy: str = pp.get(
            "missing_strategy",
            "median"
        )

        self.handle_inf: bool = pp.get(
            "handle_inf",
            True
        )

        self.scaler: Optional[Any] = None

        self.feature_columns: List[str] = []

        self.is_fitted: bool = False

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================

    def _select_numeric(self, df: pd.DataFrame) -> List[str]:
        """
        Select numeric columns that are safe to use as model features.

        Excludes:
        - identifiers
        - timestamps
        - labels
        - target columns
        """

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        selected = [
            c
            for c in numeric_cols
            if c.lower() not in self.NON_FEATURE_COLS
        ]

        return selected

    # ------------------------------------------------------------------

    def _handle_inf_nan(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace infinities with NaN and fill missing feature values.
        """

        df = df.copy()

        if self.handle_inf:
            df = df.replace(
                [np.inf, -np.inf],
                np.nan
            )

        if not self.feature_columns:
            return df

        # --------------------------------------------------------------
        # Calculate replacement values
        # --------------------------------------------------------------
        if self.missing_strategy == "median":

            fill_values = df[
                self.feature_columns
            ].median()

        elif self.missing_strategy == "mean":

            fill_values = df[
                self.feature_columns
            ].mean()

        else:

            fill_values = pd.Series(
                0.0,
                index=self.feature_columns
            )

        # --------------------------------------------------------------
        # Fill missing values
        # --------------------------------------------------------------
        df[self.feature_columns] = (
            df[self.feature_columns]
            .fillna(fill_values)
        )

        # --------------------------------------------------------------
        # Final safety check
        # --------------------------------------------------------------
        df[self.feature_columns] = (
            df[self.feature_columns]
            .replace(
                [np.inf, -np.inf],
                0.0
            )
            .fillna(0.0)
        )

        return df

    # ------------------------------------------------------------------

    def _create_scaler(self):
        """
        Instantiate the configured scaler.
        """

        if self.scaler_type.lower() == "robust":
            return RobustScaler()

        return StandardScaler()

    # ==================================================================
    # LABEL MAPPING
    # ==================================================================

    def map_labels(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Map the dataset's string labels to MITRE ATT&CK stage integers.

        Creates:

            attack_stage
                0 = Normal
                1 = Reconnaissance
                2 = Initial Access
                3 = Lateral Movement
                4 = Command & Control
                5 = Exfiltration

            is_attack
                0 = Normal
                1 = Attack

        The exact mapping comes from config.yaml.
        """

        df = df.copy()

        # --------------------------------------------------------------
        # No label column
        # --------------------------------------------------------------
        if "label" not in df.columns:

            logger.warning(
                "No 'label' column found — assuming all normal."
            )

            df["attack_stage"] = 0
            df["is_attack"] = 0

            return df

        cfg = config or self.config

        label_map = (
            cfg
            .get("mitre", {})
            .get("label_mapping", {})
        )

        # --------------------------------------------------------------
        # Normalize labels
        # --------------------------------------------------------------
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------------
        # Map labels -> stage
        # --------------------------------------------------------------
        mapped = df["label"].map(label_map)

        # --------------------------------------------------------------
        # Detect labels not present in config
        # --------------------------------------------------------------
        unmapped = (
            df.loc[mapped.isna(), "label"]
            .unique()
        )

        if len(unmapped) > 0:

            logger.warning(
                "Unmapped labels defaulting to Normal "
                "(stage 0): %s",
                unmapped.tolist()
            )

        # --------------------------------------------------------------
        # Create targets
        # --------------------------------------------------------------
        df["attack_stage"] = (
            mapped
            .fillna(0)
            .astype(int)
        )

        df["is_attack"] = (
            df["attack_stage"] > 0
        ).astype(int)

        # --------------------------------------------------------------
        # Log distribution
        # --------------------------------------------------------------
        logger.info(
            "Label distribution:\n%s",
            (
                df["attack_stage"]
                .value_counts()
                .sort_index()
                .to_string()
            )
        )

        return df

    # ==================================================================
    # TIMESTAMP PARSING
    # ==================================================================

    @staticmethod
    def parse_timestamps(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Parse timestamps and sort records chronologically.

        Uses modern pandas-compatible to_datetime() syntax.

        If timestamps cannot be parsed reliably, falls back to
        row-index ordering.
        """

        df = df.copy()

        # --------------------------------------------------------------
        # No timestamp column
        # --------------------------------------------------------------
        if "timestamp" not in df.columns:

            logger.warning(
                "No timestamp column — using row index as order."
            )

            df["timestamp"] = pd.RangeIndex(
                len(df)
            )

            return df

        try:

            # ----------------------------------------------------------
            # Modern pandas-compatible timestamp parsing.
            #
            # DO NOT use:
            # infer_datetime_format=True
            #
            # Newer pandas versions no longer require it.
            # ----------------------------------------------------------
            parsed = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

            nat_ratio = parsed.isna().mean()

            # ----------------------------------------------------------
            # If more than half are invalid, timestamp information
            # isn't reliable enough.
            # ----------------------------------------------------------
            if nat_ratio > 0.5:

                logger.warning(
                    "%.0f%% of timestamps unparseable — "
                    "using row index.",
                    nat_ratio * 100
                )

                df["timestamp"] = pd.RangeIndex(
                    len(df)
                )

                return df

            # ----------------------------------------------------------
            # If only a small number are invalid, remove those rows.
            # ----------------------------------------------------------
            if nat_ratio > 0:

                logger.warning(
                    "%.0f%% of timestamps unparseable — "
                    "dropping those rows.",
                    nat_ratio * 100
                )

                df["timestamp"] = parsed

                df = (
                    df
                    .dropna(subset=["timestamp"])
                    .reset_index(drop=True)
                )

            else:

                df["timestamp"] = parsed

            # ----------------------------------------------------------
            # Sort chronologically
            # ----------------------------------------------------------
            df = (
                df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            logger.info(
                "Timestamps parsed. Range: %s to %s",
                df["timestamp"].min(),
                df["timestamp"].max()
            )

        except Exception as exc:

            logger.warning(
                "Timestamp parsing failed (%s) — "
                "using row index.",
                exc
            )

            df["timestamp"] = pd.RangeIndex(
                len(df)
            )

        return df

    # ==================================================================
    # FIT + TRANSFORM
    # ==================================================================

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the scaler on training data.

        Returns:
            df
            scaled feature matrix with shape (N, D)
        """

        df = df.copy()

        # --------------------------------------------------------------
        # Determine model features
        # --------------------------------------------------------------
        self.feature_columns = (
            self._select_numeric(df)
        )

        logger.info(
            "Selected %d numeric features for scaling.",
            len(self.feature_columns)
        )

        if not self.feature_columns:

            raise ValueError(
                "No numeric features were found."
            )

        # --------------------------------------------------------------
        # Clean NaN / Inf
        # --------------------------------------------------------------
        df = self._handle_inf_nan(df)

        # --------------------------------------------------------------
        # Create scaler
        # --------------------------------------------------------------
        self.scaler = self._create_scaler()

        # --------------------------------------------------------------
        # Fit ONLY on training data
        # --------------------------------------------------------------
        scaled = self.scaler.fit_transform(
            df[self.feature_columns].values
        )

        self.is_fitted = True

        logger.info(
            "Scaler fitted (%s) on %d samples.",
            self.scaler_type,
            len(df)
        )

        return (
            df,
            scaled.astype(np.float32)
        )

    # ==================================================================
    # TRANSFORM
    # ==================================================================

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform validation/test/inference data using the
        already-fitted scaler.
        """

        if (
            not self.is_fitted
            or self.scaler is None
        ):

            raise RuntimeError(
                "Preprocessor is not fitted. "
                "Call fit_transform first."
            )

        df = df.copy()

        # --------------------------------------------------------------
        # Ensure every training feature exists
        # --------------------------------------------------------------
        for col in self.feature_columns:

            if col not in df.columns:

                df[col] = 0.0

            else:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # --------------------------------------------------------------
        # Clean values
        # --------------------------------------------------------------
        df = self._handle_inf_nan(df)

        # --------------------------------------------------------------
        # Transform using training scaler
        # --------------------------------------------------------------
        scaled = self.scaler.transform(
            df[self.feature_columns].values
        )

        return (
            df,
            scaled.astype(np.float32)
        )

    # ==================================================================
    # PERSISTENCE
    # ==================================================================

    def save(
        self,
        directory: str
    ) -> None:
        """
        Save scaler and feature metadata.
        """

        os.makedirs(
            directory,
            exist_ok=True
        )

        save_pickle(
            self.scaler,
            os.path.join(
                directory,
                "scaler.pkl"
            )
        )

        save_pickle(
            self.feature_columns,
            os.path.join(
                directory,
                "feature_columns.pkl"
            )
        )

        logger.info(
            "Preprocessor saved to %s",
            directory
        )

    # ------------------------------------------------------------------

    def load(
        self,
        directory: str
    ) -> None:
        """
        Load previously saved preprocessing artifacts.
        """

        self.scaler = load_pickle(
            os.path.join(
                directory,
                "scaler.pkl"
            )
        )

        self.feature_columns = load_pickle(
            os.path.join(
                directory,
                "feature_columns.pkl"
            )
        )

        self.is_fitted = True

        logger.info(
            "Preprocessor loaded from %s",
            directory
        )