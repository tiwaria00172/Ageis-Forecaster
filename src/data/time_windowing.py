# =============================================================================
# Time Windowing — Convert chronological flows into network state vectors
# =============================================================================
"""
Groups chronologically sorted network flows into fixed-duration time windows,
aggregates features within each window to create network state vectors S_t,
and then constructs overlapping sequences:

    [S_{t-n}, ..., S_t] -> S_{t+1}

Labels:
    - Binary attack label:
        1 if any attack flow is present in the window
        0 otherwise

    - Attack stage:
        0 if the window contains no attack flows
        otherwise the majority stage among ATTACK FLOWS ONLY

This prevents a window from having:

    is_attack = 1
    attack_stage = 0

merely because benign flows outnumber attack flows.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class TimeWindowGenerator:
    """
    Converts a chronological DataFrame + scaled feature matrix into
    windowed network-state vectors and training sequences.
    """

    def __init__(self, config: Dict[str, Any]):

        wc = config.get("windowing", {})

        self.window_size_seconds: float = wc.get(
            "window_size_seconds",
            10,
        )

        self.sequence_length: int = wc.get(
            "sequence_length",
            10,
        )

        self.forecast_steps: int = wc.get(
            "forecast_steps",
            5,
        )

        self.aggregation: str = wc.get(
            "aggregation",
            "mean",
        )

        self.min_flows: int = wc.get(
            "min_flows_per_window",
            1,
        )

    # =========================================================================
    # Window creation
    # =========================================================================

    def create_windows(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        feature_columns: List[str],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[pd.Timestamp],
    ]:
        """
        Aggregate flows into fixed-duration time windows.

        Args:
            df:
                DataFrame containing:
                    timestamp
                    attack_stage
                    is_attack

            scaled_features:
                Scaled feature matrix of shape (N, D).

            feature_columns:
                Names corresponding to scaled_features.

        Returns:
            state_vectors:
                (W, D) one state per time window.

            window_labels_attack:
                (W,) attack stage per window.

            window_labels_binary:
                (W,) binary attack label per window.

            window_timestamps:
                Window start timestamps.
        """

        if len(df) == 0:
            raise ValueError(
                "Cannot create windows from an empty DataFrame."
            )

        if len(df) != len(scaled_features):
            raise ValueError(
                "DataFrame and scaled feature matrix have different "
                f"lengths: {len(df)} != {len(scaled_features)}"
            )

        required_columns = [
            "timestamp",
            "attack_stage",
            "is_attack",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns for time windowing: "
                + ", ".join(missing)
            )

        has_datetime = pd.api.types.is_datetime64_any_dtype(
            df["timestamp"]
        )

        if has_datetime:
            return self._window_by_time(
                df,
                scaled_features,
                feature_columns,
            )

        return self._window_by_index(
            df,
            scaled_features,
            feature_columns,
        )

    # =========================================================================
    # Aggregation
    # =========================================================================

    def _aggregate(
        self,
        block: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate a (K, D) block into a (D,) network-state vector.
        """

        if self.aggregation == "mean":

            result = np.nanmean(
                block,
                axis=0,
            )

        elif self.aggregation == "median":

            result = np.nanmedian(
                block,
                axis=0,
            )

        elif self.aggregation == "sum":

            result = np.nansum(
                block,
                axis=0,
            )

        elif self.aggregation == "max":

            result = np.nanmax(
                block,
                axis=0,
            )

        else:

            result = np.nanmean(
                block,
                axis=0,
            )

        # Protect against NaN/Inf values.
        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return result

    # =========================================================================
    # Label generation
    # =========================================================================

    def _get_window_labels(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Determine binary attack and attack-stage labels for one window.

        IMPORTANT:

        Binary label:
            1 if ANY attack flow exists.

        Stage label:
            - 0 (Normal) if there are no attack flows.
            - Otherwise, majority stage among ATTACK FLOWS ONLY.

        This keeps the binary and multiclass targets logically aligned.
        """

        attack_values = (
            df.iloc[indices]["is_attack"]
            .to_numpy()
            .astype(int)
        )

        stage_values = (
            df.iloc[indices]["attack_stage"]
            .to_numpy()
            .astype(int)
        )

        # ---------------------------------------------------------------------
        # Binary attack label
        # ---------------------------------------------------------------------

        is_attack = int(
            np.any(attack_values == 1)
        )

        # ---------------------------------------------------------------------
        # No attack -> Normal stage
        # ---------------------------------------------------------------------

        if is_attack == 0:

            attack_stage = 0

            return attack_stage, is_attack

        # ---------------------------------------------------------------------
        # Attack exists.
        #
        # IMPORTANT:
        # Ignore benign stage=0 flows when determining the attack stage.
        # ---------------------------------------------------------------------

        attack_mask = (
            attack_values == 1
        )

        attack_stages = stage_values[
            attack_mask
        ]

        # Safety fallback.
        if len(attack_stages) == 0:

            return 0, is_attack

        # Majority attack stage.
        unique_stages, counts = np.unique(
            attack_stages,
            return_counts=True,
        )

        attack_stage = int(
            unique_stages[
                np.argmax(counts)
            ]
        )

        return attack_stage, is_attack

    # =========================================================================
    # Datetime-based windowing
    # =========================================================================

    def _window_by_time(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        feature_columns: List[str],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
    ]:
        """
        Group flows into fixed-duration time intervals.
        """

        timestamps = df["timestamp"].values

        if len(timestamps) == 0:

            raise ValueError(
                "No timestamps available for datetime windowing."
            )

        t_start = timestamps[0]
        t_end = timestamps[-1]

        window_delta = np.timedelta64(
            int(self.window_size_seconds),
            "s",
        )

        states = []
        attack_labels = []
        binary_labels = []
        win_ts = []

        current = t_start

        while current < t_end:

            next_boundary = (
                current + window_delta
            )

            mask = (
                (timestamps >= current)
                &
                (timestamps < next_boundary)
            )

            indices = np.where(mask)[0]

            if len(indices) >= self.min_flows:

                # -------------------------------------------------------------
                # Network state
                # -------------------------------------------------------------

                block = scaled_features[
                    indices
                ]

                state = self._aggregate(
                    block
                )

                states.append(
                    state
                )

                # -------------------------------------------------------------
                # Labels
                # -------------------------------------------------------------

                attack_stage, is_attack = (
                    self._get_window_labels(
                        df,
                        indices,
                    )
                )

                attack_labels.append(
                    attack_stage
                )

                binary_labels.append(
                    is_attack
                )

                win_ts.append(
                    pd.Timestamp(current)
                )

            current = next_boundary

        states_array = np.asarray(
            states,
            dtype=np.float32,
        )

        attack_array = np.asarray(
            attack_labels,
            dtype=np.int64,
        )

        binary_array = np.asarray(
            binary_labels,
            dtype=np.float32,
        )

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        logger.info(
            "Created %d time windows from datetime timestamps.",
            len(states_array),
        )

        self._log_window_distribution(
            attack_array,
            binary_array,
        )

        return (
            states_array,
            attack_array,
            binary_array,
            win_ts,
        )

    # =========================================================================
    # Index-based windowing
    # =========================================================================

    def _window_by_index(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        feature_columns: List[str],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
    ]:
        """
        Fall back to index-based windowing.

        Here window_size_seconds is interpreted as the number of rows.
        """

        window_size = max(
            1,
            int(self.window_size_seconds),
        )

        n = len(df)

        states = []
        attack_labels = []
        binary_labels = []
        win_ts = []

        for start in range(
            0,
            n,
            window_size,
        ):

            end = min(
                start + window_size,
                n,
            )

            if (
                end - start
            ) < self.min_flows:

                continue

            indices = np.arange(
                start,
                end,
            )

            # -----------------------------------------------------------------
            # Network state
            # -----------------------------------------------------------------

            block = scaled_features[
                indices
            ]

            state = self._aggregate(
                block
            )

            states.append(
                state
            )

            # -----------------------------------------------------------------
            # Labels
            # -----------------------------------------------------------------

            attack_stage, is_attack = (
                self._get_window_labels(
                    df,
                    indices,
                )
            )

            attack_labels.append(
                attack_stage
            )

            binary_labels.append(
                is_attack
            )

            win_ts.append(
                start
            )

        states_array = np.asarray(
            states,
            dtype=np.float32,
        )

        attack_array = np.asarray(
            attack_labels,
            dtype=np.int64,
        )

        binary_array = np.asarray(
            binary_labels,
            dtype=np.float32,
        )

        logger.info(
            "Created %d index-based windows.",
            len(states_array),
        )

        self._log_window_distribution(
            attack_array,
            binary_array,
        )

        return (
            states_array,
            attack_array,
            binary_array,
            win_ts,
        )

    # =========================================================================
    # Window distribution diagnostics
    # =========================================================================

    def _log_window_distribution(
        self,
        attack_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> None:
        """
        Log binary and multiclass window distributions.
        """

        if len(binary_labels) == 0:

            logger.warning(
                "No windows were created."
            )

            return

        benign = int(
            np.sum(binary_labels == 0)
        )

        attack = int(
            np.sum(binary_labels == 1)
        )

        logger.info(
            "Window binary distribution — "
            "Normal: %d | Attack: %d",
            benign,
            attack,
        )

        unique_stages, counts = np.unique(
            attack_labels,
            return_counts=True,
        )

        distribution = ", ".join(
            f"{int(stage)}={int(count)}"
            for stage, count
            in zip(unique_stages, counts)
        )

        logger.info(
            "Window stage distribution — %s",
            distribution,
        )

    # =========================================================================
    # Sequence construction
    # =========================================================================

    def create_sequences(
        self,
        state_vectors: np.ndarray,
        attack_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Create overlapping sequences for the temporal world model.

        Input:
            state_vectors : (W, D)

        Output:
            X_seq:
                (N, seq_len, D)

            y_state:
                (N, D)
                Next network state S_{t+1}

            y_attack:
                (N,)
                Attack stage of S_{t+1}

            y_binary:
                (N,)
                Binary attack label of S_{t+1}

        For each sequence:

            X = [S_t-9, ..., S_t]

            Target = S_t+1

        The attack-stage and binary targets correspond to exactly
        the same future window as y_state.
        """

        if state_vectors.ndim != 2:

            raise ValueError(
                "state_vectors must have shape (W, D), "
                f"got {state_vectors.shape}"
            )

        W, D = state_vectors.shape

        seq_len = self.sequence_length

        if len(attack_labels) != W:

            raise ValueError(
                "attack_labels length does not match "
                "state_vectors."
            )

        if len(binary_labels) != W:

            raise ValueError(
                "binary_labels length does not match "
                "state_vectors."
            )

        if W < seq_len + 1:

            raise ValueError(
                f"Not enough windows ({W}) for "
                f"sequence_length={seq_len}. "
                f"Need at least {seq_len + 1} windows."
            )

        # Number of valid sequences.
        N = W - seq_len

        X_seq = np.zeros(
            (
                N,
                seq_len,
                D,
            ),
            dtype=np.float32,
        )

        y_state = np.zeros(
            (
                N,
                D,
            ),
            dtype=np.float32,
        )

        y_attack = np.zeros(
            N,
            dtype=np.int64,
        )

        y_binary = np.zeros(
            N,
            dtype=np.float32,
        )

        for i in range(N):

            # -------------------------------------------------------------
            # Input context
            # -------------------------------------------------------------

            X_seq[i] = state_vectors[
                i : i + seq_len
            ]

            # -------------------------------------------------------------
            # Future state
            # -------------------------------------------------------------

            target_index = (
                i + seq_len
            )

            y_state[i] = state_vectors[
                target_index
            ]

            # -------------------------------------------------------------
            # Future attack stage
            # -------------------------------------------------------------

            y_attack[i] = attack_labels[
                target_index
            ]

            # -------------------------------------------------------------
            # Future binary attack label
            # -------------------------------------------------------------

            y_binary[i] = binary_labels[
                target_index
            ]

        logger.info(
            "Created %d sequences "
            "(seq_len=%d, features=%d).",
            N,
            seq_len,
            D,
        )

        # ---------------------------------------------------------------------
        # Sequence-target diagnostics
        # ---------------------------------------------------------------------

        self._log_sequence_distribution(
            y_attack,
            y_binary,
        )

        return (
            X_seq,
            y_state,
            y_attack,
            y_binary,
        )

    # =========================================================================
    # Sequence distribution diagnostics
    # =========================================================================

    def _log_sequence_distribution(
        self,
        attack_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> None:
        """
        Log distributions of actual sequence targets.
        """

        if len(binary_labels) == 0:

            logger.warning(
                "No sequences were created."
            )

            return

        benign = int(
            np.sum(binary_labels == 0)
        )

        attack = int(
            np.sum(binary_labels == 1)
        )

        logger.info(
            "Sequence target distribution — "
            "Normal: %d | Attack: %d",
            benign,
            attack,
        )

        unique_stages, counts = np.unique(
            attack_labels,
            return_counts=True,
        )

        distribution = ", ".join(
            f"{int(stage)}={int(count)}"
            for stage, count
            in zip(unique_stages, counts)
        )

        logger.info(
            "Sequence target stage distribution — %s",
            distribution,
        )