# =============================================================================
# Time Windowing — Convert chronological flows into network state vectors
# =============================================================================
"""
Groups chronologically sorted network flows into fixed-duration time windows,
aggregates features within each window to create network state vectors S_t,
and then constructs overlapping sequences [S_{t-n}, ..., S_t] → S_{t+1}
for training the temporal world model.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeWindowGenerator:
    """
    Converts a chronological DataFrame + scaled feature matrix into
    windowed state vectors and training sequences.
    """

    def __init__(self, config: Dict[str, Any]):
        wc = config.get("windowing", {})
        self.window_size_seconds: float = wc.get("window_size_seconds", 10)
        self.sequence_length: int = wc.get("sequence_length", 10)
        self.forecast_steps: int = wc.get("forecast_steps", 5)
        self.aggregation: str = wc.get("aggregation", "mean")
        self.min_flows: int = wc.get("min_flows_per_window", 1)

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------

    def create_windows(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Aggregate flows into fixed-duration time windows.

        Args:
            df: DataFrame with a 'timestamp' column (datetime or int index),
                'attack_stage', and 'is_attack' columns.
            scaled_features: (N, D) array of scaled numeric features.
            feature_columns: Names corresponding to columns in scaled_features.

        Returns:
            state_vectors: (W, D) — one state per window.
            window_labels_attack: (W,) — majority attack stage per window.
            window_labels_binary: (W,) — 1 if any attack flow present.
            window_timestamps: list of window-start timestamps / indices.
        """
        has_datetime = pd.api.types.is_datetime64_any_dtype(df["timestamp"])

        if has_datetime:
            return self._window_by_time(df, scaled_features, feature_columns)
        else:
            return self._window_by_index(df, scaled_features, feature_columns)

    def _aggregate(self, block: np.ndarray) -> np.ndarray:
        """Aggregate a (K, D) block to a (D,) state vector."""
        if self.aggregation == "mean":
            return np.nanmean(block, axis=0)
        elif self.aggregation == "median":
            return np.nanmedian(block, axis=0)
        elif self.aggregation == "sum":
            return np.nansum(block, axis=0)
        elif self.aggregation == "max":
            return np.nanmax(block, axis=0)
        return np.nanmean(block, axis=0)

    def _window_by_time(
        self, df, scaled_features, feature_columns
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Group by fixed time intervals."""
        timestamps = df["timestamp"].values
        t_start = timestamps[0]
        t_end = timestamps[-1]

        window_delta = np.timedelta64(int(self.window_size_seconds), "s")
        states, attack_labels, binary_labels, win_ts = [], [], [], []

        current = t_start
        while current < t_end:
            next_boundary = current + window_delta
            mask = (timestamps >= current) & (timestamps < next_boundary)
            indices = np.where(mask)[0]

            if len(indices) >= self.min_flows:
                block = scaled_features[indices]
                state = self._aggregate(block)
                states.append(state)

                # Majority attack stage
                stages = df.iloc[indices]["attack_stage"].values
                attack_labels.append(int(np.bincount(stages.astype(int)).argmax()))

                # Binary: any attack present
                binary_labels.append(int(df.iloc[indices]["is_attack"].any()))
                win_ts.append(pd.Timestamp(current))

            current = next_boundary

        logger.info("Created %d time windows from datetime timestamps.", len(states))
        return (
            np.array(states, dtype=np.float32),
            np.array(attack_labels, dtype=np.int64),
            np.array(binary_labels, dtype=np.float32),
            win_ts,
        )

    def _window_by_index(
        self, df, scaled_features, feature_columns
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Fall back to index-based windowing (fixed number of rows)."""
        window_size = max(1, int(self.window_size_seconds))  # rows per window
        n = len(df)
        states, attack_labels, binary_labels, win_ts = [], [], [], []

        for start in range(0, n, window_size):
            end = min(start + window_size, n)
            if (end - start) < self.min_flows:
                continue

            block = scaled_features[start:end]
            state = self._aggregate(block)
            states.append(state)

            stages = df.iloc[start:end]["attack_stage"].values
            attack_labels.append(int(np.bincount(stages.astype(int)).argmax()))
            binary_labels.append(int(df.iloc[start:end]["is_attack"].any()))
            win_ts.append(start)

        logger.info("Created %d index-based windows.", len(states))
        return (
            np.array(states, dtype=np.float32),
            np.array(attack_labels, dtype=np.int64),
            np.array(binary_labels, dtype=np.float32),
            win_ts,
        )

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        state_vectors: np.ndarray,
        attack_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create overlapping sequences for the temporal model.

        Input state_vectors : (W, D)
        Output:
            X_seq : (N, seq_len, D)
            y_state : (N, D)           — next state S_{t+1}
            y_attack : (N,)            — attack stage of next window
            y_binary : (N,)            — infiltration label of next window

        The target for each sequence is the *next* time window after the
        final element in the sequence.
        """
        W, D = state_vectors.shape
        seq_len = self.sequence_length

        if W < seq_len + 1:
            raise ValueError(
                f"Not enough windows ({W}) for sequence_length={seq_len}. "
                f"Need at least {seq_len + 1} windows."
            )

        N = W - seq_len  # number of valid sequences
        X_seq = np.zeros((N, seq_len, D), dtype=np.float32)
        y_state = np.zeros((N, D), dtype=np.float32)
        y_attack = np.zeros(N, dtype=np.int64)
        y_binary = np.zeros(N, dtype=np.float32)

        for i in range(N):
            X_seq[i] = state_vectors[i : i + seq_len]
            y_state[i] = state_vectors[i + seq_len]
            y_attack[i] = attack_labels[i + seq_len]
            y_binary[i] = binary_labels[i + seq_len]

        logger.info(
            "Created %d sequences (seq_len=%d, features=%d).", N, seq_len, D
        )
        return X_seq, y_state, y_attack, y_binary
