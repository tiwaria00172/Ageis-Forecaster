# =============================================================================
# Feature Extractor — Derives flow, TCP, temporal, and attack features
# =============================================================================
"""
Extracts and derives network features from a canonical DataFrame.

The extractor:
- Preserves existing dataset features
- Derives flow-level features
- Resolves common CIC-IDS TCP flag aliases
- Derives TCP flag ratios
- Derives temporal/IAT features
- Derives attack-behaviour heuristics
- Handles missing source columns gracefully

Important:
The attack-related features in this module are heuristic indicators.
They are NOT ground-truth attack labels.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Stateless feature extractor.

    Call:

        enriched_df, metadata = extractor.extract(df)

    to derive all computable features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.available_features: List[str] = []
        self.missing_features: List[str] = []

    # ==================================================================
    # COLUMN RESOLUTION HELPERS
    # ==================================================================

    @staticmethod
    def _normalise_column_name(name: str) -> str:
        """
        Normalize a column name so aliases can be compared reliably.
        """

        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

    # ------------------------------------------------------------------

    def _build_column_lookup(
        self,
        df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Build:

            normalized_name -> original_dataframe_column
        """

        lookup = {}

        for col in df.columns:
            lookup[
                self._normalise_column_name(col)
            ] = col

        return lookup

    # ------------------------------------------------------------------

    def _resolve_column(
        self,
        df: pd.DataFrame,
        aliases: List[str]
    ) -> Optional[str]:
        """
        Find the first matching dataframe column from a list of aliases.
        """

        lookup = self._build_column_lookup(df)

        for alias in aliases:

            normalized = self._normalise_column_name(
                alias
            )

            if normalized in lookup:
                return lookup[normalized]

        return None

    # ------------------------------------------------------------------

    @staticmethod
    def _numeric(
        series: pd.Series,
        default: float = 0.0
    ) -> pd.Series:
        """
        Safely convert a Series to numeric.
        """

        result = pd.to_numeric(
            series,
            errors="coerce"
        )

        return result.fillna(default)

    # ==================================================================
    # FLOW-LEVEL FEATURES
    # ==================================================================

    def _extract_flow_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        # --------------------------------------------------------------
        # Total bytes / packets
        # --------------------------------------------------------------

        bytes_col = self._resolve_column(
            df,
            [
                "total_bytes",
                "Total Bytes",
                "Total Length of Fwd Packets",
            ]
        )

        packets_col = self._resolve_column(
            df,
            [
                "total_packets",
                "Total Packets",
            ]
        )

        duration_col = self._resolve_column(
            df,
            [
                "flow_duration",
                "Flow Duration",
                "duration",
            ]
        )

        # --------------------------------------------------------------
        # bytes_per_packet
        # --------------------------------------------------------------

        if bytes_col and packets_col:

            total_bytes = self._numeric(
                df[bytes_col]
            )

            total_packets = (
                self._numeric(df[packets_col])
                .replace(0, np.nan)
            )

            df["bytes_per_packet"] = (
                total_bytes / total_packets
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "bytes_per_packet"
            )

        # --------------------------------------------------------------
        # Duration normalization helper
        # --------------------------------------------------------------

        if duration_col:

            duration = self._numeric(
                df[duration_col]
            )

            # CIC-IDS Flow Duration is normally microseconds.
            #
            # We use the distribution rather than blindly assuming
            # every dataset uses microseconds.
            if duration.median() > 1e5:

                duration_sec = (
                    duration / 1e6
                )

            else:

                duration_sec = duration

            duration_sec = (
                duration_sec
                .replace(0, np.nan)
            )

        else:

            duration_sec = None

        # --------------------------------------------------------------
        # packets_per_second
        # --------------------------------------------------------------

        if packets_col and duration_sec is not None:

            total_packets = self._numeric(
                df[packets_col]
            )

            df["packets_per_second"] = (
                total_packets / duration_sec
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "packets_per_second"
            )

        # --------------------------------------------------------------
        # bytes_per_second
        # --------------------------------------------------------------

        if bytes_col and duration_sec is not None:

            total_bytes = self._numeric(
                df[bytes_col]
            )

            df["bytes_per_second"] = (
                total_bytes / duration_sec
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "bytes_per_second"
            )

        # --------------------------------------------------------------
        # Forward / backward packet ratio
        # --------------------------------------------------------------

        fwd_packets_col = self._resolve_column(
            df,
            [
                "total_fwd_packets",
                "Total Fwd Packets",
                "fwd_packet_count",
                "fwd_packets",
            ]
        )

        bwd_packets_col = self._resolve_column(
            df,
            [
                "total_backward_packets",
                "Total Backward Packets",
                "bwd_packet_count",
                "bwd_packets",
            ]
        )

        if fwd_packets_col and bwd_packets_col:

            fwd = self._numeric(
                df[fwd_packets_col]
            )

            bwd = (
                self._numeric(
                    df[bwd_packets_col]
                )
                .replace(0, np.nan)
            )

            df["fwd_bwd_packet_ratio"] = (
                fwd / bwd
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "fwd_bwd_packet_ratio"
            )

        return df

    # ==================================================================
    # TCP FLAG FEATURES
    # ==================================================================

    def _extract_tcp_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        # --------------------------------------------------------------
        # Common CIC-IDS aliases
        # --------------------------------------------------------------

        flag_aliases = {

            "syn_count": [
                "syn_count",
                "syn_flag_cnt",
                "syn flag count",
                "syn flags",
                "SYN Flag Count",
            ],

            "ack_count": [
                "ack_count",
                "ack_flag_cnt",
                "ack flag count",
                "ack flags",
                "ACK Flag Count",
            ],

            "fin_count": [
                "fin_count",
                "fin_flag_cnt",
                "fin flag count",
                "fin flags",
                "FIN Flag Count",
            ],

            "rst_count": [
                "rst_count",
                "rst_flag_cnt",
                "rst flag count",
                "rst flags",
                "RST Flag Count",
            ],

            "psh_count": [
                "psh_count",
                "psh_flag_cnt",
                "psh flag count",
                "psh flags",
                "PSH Flag Count",
            ],

            "urg_count": [
                "urg_count",
                "urg_flag_cnt",
                "urg flag count",
                "urg flags",
                "URG Flag Count",
            ],
        }

        resolved: Dict[str, str] = {}

        # --------------------------------------------------------------
        # Resolve each TCP flag
        # --------------------------------------------------------------

        for canonical, aliases in flag_aliases.items():

            original = self._resolve_column(
                df,
                aliases
            )

            if original:

                resolved[canonical] = original

                # Create canonical feature if needed
                if canonical not in df.columns:

                    df[canonical] = self._numeric(
                        df[original]
                    )

            else:

                self.missing_features.append(
                    canonical
                )

        # --------------------------------------------------------------
        # Total TCP flags
        # --------------------------------------------------------------

        total_flags = pd.Series(
            0.0,
            index=df.index
        )

        for canonical in resolved:

            if canonical in df.columns:

                total_flags += self._numeric(
                    df[canonical]
                )

        total_flags = (
            total_flags
            .replace(0, np.nan)
        )

        # --------------------------------------------------------------
        # SYN ratio
        # --------------------------------------------------------------

        if "syn_count" in df.columns:

            df["syn_ratio"] = (
                self._numeric(
                    df["syn_count"]
                ) / total_flags
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "syn_ratio"
            )

        # --------------------------------------------------------------
        # ACK ratio
        # --------------------------------------------------------------

        if "ack_count" in df.columns:

            df["ack_ratio"] = (
                self._numeric(
                    df["ack_count"]
                ) / total_flags
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "ack_ratio"
            )

        # --------------------------------------------------------------
        # Reset ratio
        # --------------------------------------------------------------

        if "rst_count" in df.columns:

            df["rst_ratio"] = (
                self._numeric(
                    df["rst_count"]
                ) / total_flags
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        return df

    # ==================================================================
    # TEMPORAL FEATURES
    # ==================================================================

    def _extract_temporal_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        # --------------------------------------------------------------
        # Locate IAT features
        # --------------------------------------------------------------

        iat_cols = [
            c
            for c in df.columns
            if "iat" in str(c).lower()
        ]

        if iat_cols:

            iat_values = (
                df[iat_cols]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            df["iat_mean"] = (
                iat_values
                .mean(axis=1)
                .fillna(0)
            )

            df["iat_variance"] = (
                iat_values
                .var(axis=1)
                .fillna(0)
            )

            df["iat_max"] = (
                iat_values
                .max(axis=1)
                .fillna(0)
            )

            std_iat = (
                df["iat_variance"]
                .clip(lower=0)
                .apply(np.sqrt)
            )

            mean_iat = (
                df["iat_mean"]
                .replace(0, np.nan)
            )

            df["burstiness"] = (
                (std_iat - mean_iat)
                /
                (std_iat + mean_iat)
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            for feature in [
                "iat_mean",
                "iat_variance",
                "iat_max",
                "burstiness",
            ]:

                self.missing_features.append(
                    feature
                )

        # --------------------------------------------------------------
        # Packet rate
        # --------------------------------------------------------------

        if "packets_per_second" in df.columns:

            df["packet_rate"] = (
                df["packets_per_second"]
            )

        else:

            packets_col = self._resolve_column(
                df,
                [
                    "total_packets",
                    "Total Packets",
                ]
            )

            duration_col = self._resolve_column(
                df,
                [
                    "flow_duration",
                    "Flow Duration",
                    "duration",
                ]
            )

            if packets_col and duration_col:

                packets = self._numeric(
                    df[packets_col]
                )

                duration = self._numeric(
                    df[duration_col]
                )

                if duration.median() > 1e5:

                    duration = duration / 1e6

                duration = (
                    duration
                    .replace(0, np.nan)
                )

                df["packet_rate"] = (
                    packets / duration
                ).replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

            else:

                self.missing_features.append(
                    "packet_rate"
                )

        return df

    # ==================================================================
    # ATTACK BEHAVIOUR FEATURES
    # ==================================================================

    def _extract_attack_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Derive heuristic attack-behaviour indicators.

        These are features only. They are not ground-truth labels.
        """

        # --------------------------------------------------------------
        # Destination port features
        # --------------------------------------------------------------

        dst_port_col = self._resolve_column(
            df,
            [
                "dst_port",
                "destination_port",
                "Destination Port",
                "dest_port",
            ]
        )

        src_ip_col = self._resolve_column(
            df,
            [
                "src_ip",
                "source_ip",
                "Source IP",
            ]
        )

        if dst_port_col:

            dst_port_numeric = self._numeric(
                df[dst_port_col]
            )

            # ----------------------------------------------------------
            # Destination-port diversity
            # ----------------------------------------------------------

            if src_ip_col:

                port_diversity = (
                    df.assign(
                        __dst_port_numeric=dst_port_numeric
                    )
                    .groupby(src_ip_col)[
                        "__dst_port_numeric"
                    ]
                    .transform("nunique")
                )

                df["dst_port_diversity"] = (
                    port_diversity
                    .astype(float)
                    .fillna(1)
                )

            else:

                df["dst_port_diversity"] = 1.0

                self.missing_features.append(
                    "dst_port_diversity (src_ip absent)"
                )

            # ----------------------------------------------------------
            # Port scan score
            # ----------------------------------------------------------

            if src_ip_col:

                port_std = (
                    df.assign(
                        __dst_port_numeric=dst_port_numeric
                    )
                    .groupby(src_ip_col)[
                        "__dst_port_numeric"
                    ]
                    .transform("std")
                    .fillna(0)
                )

                port_mean = (
                    df.assign(
                        __dst_port_numeric=dst_port_numeric
                    )
                    .groupby(src_ip_col)[
                        "__dst_port_numeric"
                    ]
                    .transform("mean")
                    .replace(0, np.nan)
                    .fillna(1)
                )

                df["port_scan_score"] = (
                    port_std / port_mean
                ).replace(
                    [np.inf, -np.inf],
                    np.nan
                ).fillna(0)

            else:

                df["port_scan_score"] = 0.0

        else:

            self.missing_features.extend(
                [
                    "dst_port_diversity",
                    "port_scan_score",
                ]
            )

        # --------------------------------------------------------------
        # Resolve total packet count
        # --------------------------------------------------------------

        total_packets_col = self._resolve_column(
            df,
            [
                "total_packets",
                "Total Packets",
            ]
        )

        # --------------------------------------------------------------
        # SYN burst score
        # --------------------------------------------------------------

        if (
            "syn_count" in df.columns
            and total_packets_col
        ):

            total_packets = (
                self._numeric(
                    df[total_packets_col]
                )
                .replace(0, np.nan)
            )

            df["syn_burst_score"] = (
                self._numeric(
                    df["syn_count"]
                )
                / total_packets
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "syn_burst_score"
            )

        # --------------------------------------------------------------
        # Connection attempt rate
        # --------------------------------------------------------------

        if src_ip_col:

            connection_rate = (
                df.groupby(src_ip_col)[
                    src_ip_col
                ]
                .transform("count")
            )

            df["connection_attempt_rate"] = (
                connection_rate
                .astype(float)
                .fillna(0)
            )

        else:

            self.missing_features.append(
                "connection_attempt_rate"
            )

        # --------------------------------------------------------------
        # Failed connection ratio
        #
        # RST packets / total packets
        # --------------------------------------------------------------

        if (
            "rst_count" in df.columns
            and total_packets_col
        ):

            total_packets = (
                self._numeric(
                    df[total_packets_col]
                )
                .replace(0, np.nan)
            )

            df["failed_connection_ratio"] = (
                self._numeric(
                    df["rst_count"]
                )
                / total_packets
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        else:

            self.missing_features.append(
                "failed_connection_ratio"
            )

        # --------------------------------------------------------------
        # SYN/ACK relationship
        #
        # Useful as an additional behavioural indicator.
        # --------------------------------------------------------------

        if (
            "syn_count" in df.columns
            and "ack_count" in df.columns
        ):

            ack = (
                self._numeric(
                    df["ack_count"]
                )
                .replace(0, np.nan)
            )

            df["syn_ack_ratio"] = (
                self._numeric(
                    df["syn_count"]
                )
                / ack
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

        return df

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def extract(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete feature-extraction pipeline.

        Returns:

            enriched_df
            extraction_metadata
        """

        # --------------------------------------------------------------
        # Reset state
        # --------------------------------------------------------------

        self.available_features = []
        self.missing_features = []

        df = df.copy()

        original_cols = set(
            df.columns
        )

        # --------------------------------------------------------------
        # Extraction pipeline
        # --------------------------------------------------------------

        df = self._extract_flow_features(df)

        df = self._extract_tcp_features(df)

        df = self._extract_temporal_features(df)

        df = self._extract_attack_features(df)

        # --------------------------------------------------------------
        # Determine derived features
        # --------------------------------------------------------------

        new_cols = (
            set(df.columns)
            - original_cols
        )

        self.available_features = sorted(
            new_cols
        )

        # --------------------------------------------------------------
        # Remove duplicate missing entries
        # --------------------------------------------------------------

        self.missing_features = sorted(
            set(self.missing_features)
        )

        # --------------------------------------------------------------
        # Log extraction status
        # --------------------------------------------------------------

        if self.missing_features:

            logger.warning(
                "Features that could not be derived: %s",
                ", ".join(
                    self.missing_features
                )
            )

        logger.info(
            "Feature extraction complete: "
            "%d derived, %d missing",
            len(self.available_features),
            len(self.missing_features)
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        metadata = {

            "derived_features":
                self.available_features,

            "missing_features":
                self.missing_features,

            "total_features":
                len(df.columns),
        }

        return (
            df,
            metadata
        )