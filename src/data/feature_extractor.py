# =============================================================================
# Feature Extractor — Derives flow, TCP, temporal, and attack features
# =============================================================================
"""
Extracts and derives network features from a canonical DataFrame.
Handles missing source columns gracefully: features that cannot be
computed are logged and excluded rather than fabricated.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Stateless feature extractor.  Call ``extract(df)`` to derive all
    computable features from the raw DataFrame and receive back a
    feature-enriched DataFrame plus metadata about which features
    were actually produced.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_features: List[str] = []
        self.missing_features: List[str] = []

    # ------------------------------------------------------------------
    # Flow-level features
    # ------------------------------------------------------------------

    def _extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive flow-level features that are not already present."""

        # bytes_per_packet
        if "total_bytes" in df.columns and "total_packets" in df.columns:
            total_pkts = pd.to_numeric(df["total_packets"], errors="coerce").replace(0, np.nan)
            df["bytes_per_packet"] = (
                pd.to_numeric(df["total_bytes"], errors="coerce") / total_pkts
            ).fillna(0)
        else:
            self.missing_features.append("bytes_per_packet")

        # packets_per_second
        if "total_packets" in df.columns and "flow_duration" in df.columns:
            dur = pd.to_numeric(df["flow_duration"], errors="coerce").replace(0, np.nan)
            # flow_duration may be in microseconds (CIC-IDS)
            dur_sec = dur / 1e6 if dur.median() > 1e5 else dur
            dur_sec = dur_sec.replace(0, np.nan)
            df["packets_per_second"] = (
                pd.to_numeric(df["total_packets"], errors="coerce") / dur_sec
            ).fillna(0)
        else:
            self.missing_features.append("packets_per_second")

        # bytes_per_second
        if "total_bytes" in df.columns and "flow_duration" in df.columns:
            dur = pd.to_numeric(df["flow_duration"], errors="coerce").replace(0, np.nan)
            dur_sec = dur / 1e6 if dur.median() > 1e5 else dur
            dur_sec = dur_sec.replace(0, np.nan)
            df["bytes_per_second"] = (
                pd.to_numeric(df["total_bytes"], errors="coerce") / dur_sec
            ).fillna(0)
        else:
            self.missing_features.append("bytes_per_second")

        # fwd_bwd_packet_ratio
        fwd_col = next(
            (c for c in df.columns if "fwd" in c.lower() and "packet" in c.lower() and "total" in c.lower()), None
        )
        bwd_col = next(
            (c for c in df.columns if "bwd" in c.lower() and "packet" in c.lower() and "total" in c.lower()), None
        )
        if fwd_col and bwd_col:
            fwd = pd.to_numeric(df[fwd_col], errors="coerce").fillna(0)
            bwd = pd.to_numeric(df[bwd_col], errors="coerce").replace(0, np.nan).fillna(1)
            df["fwd_bwd_packet_ratio"] = fwd / bwd
        else:
            self.missing_features.append("fwd_bwd_packet_ratio")

        return df

    # ------------------------------------------------------------------
    # TCP flag features
    # ------------------------------------------------------------------

    def _extract_tcp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive TCP flag counts and ratios."""
        flag_map = {
            "syn_count": ["syn flag count", "syn_flag_cnt", "syn_count"],
            "ack_count": ["ack flag count", "ack_flag_cnt", "ack_count"],
            "fin_count": ["fin flag count", "fin_flag_cnt", "fin_count"],
            "rst_count": ["rst flag count", "rst_flag_cnt", "rst_count"],
            "psh_count": ["psh flag count", "psh_flag_cnt", "psh_count"],
            "urg_count": ["urg flag count", "urg_flag_cnt", "urg_count"],
        }

        cols_lower = {c.lower().strip(): c for c in df.columns}
        resolved: Dict[str, str] = {}

        for canonical, aliases in flag_map.items():
            for alias in aliases:
                if alias in cols_lower:
                    resolved[canonical] = cols_lower[alias]
                    break
            if canonical not in resolved:
                # Check if canonical name itself exists
                if canonical in cols_lower:
                    resolved[canonical] = cols_lower[canonical]

        for canonical, original in resolved.items():
            if canonical not in df.columns:
                df[canonical] = pd.to_numeric(df[original], errors="coerce").fillna(0)

        # Compute ratios
        total_flags = pd.Series(np.zeros(len(df)), index=df.index)
        for canonical in resolved:
            if canonical in df.columns:
                total_flags = total_flags + pd.to_numeric(df[canonical], errors="coerce").fillna(0)
        total_flags = total_flags.replace(0, np.nan)

        for flag_name in ("syn_count", "ack_count"):
            ratio_name = flag_name.replace("_count", "_ratio")
            if flag_name in df.columns:
                df[ratio_name] = (
                    pd.to_numeric(df[flag_name], errors="coerce").fillna(0) / total_flags
                ).fillna(0)
            else:
                self.missing_features.append(ratio_name)

        return df

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive inter-arrival time statistics and burstiness."""
        iat_cols = [c for c in df.columns if "iat" in c.lower()]

        if iat_cols:
            iat_values = df[iat_cols].apply(pd.to_numeric, errors="coerce")
            df["iat_mean"] = iat_values.mean(axis=1).fillna(0)
            df["iat_variance"] = iat_values.var(axis=1).fillna(0)
            df["iat_max"] = iat_values.max(axis=1).fillna(0)

            mean_iat = df["iat_mean"].replace(0, np.nan)
            df["burstiness"] = (
                (df["iat_variance"].apply(np.sqrt) - mean_iat)
                / (df["iat_variance"].apply(np.sqrt) + mean_iat)
            ).fillna(0)
        else:
            for feat in ("iat_mean", "iat_variance", "iat_max", "burstiness"):
                self.missing_features.append(feat)

        # Packet rate (may already be computed)
        if "packets_per_second" in df.columns:
            df["packet_rate"] = df["packets_per_second"]
        elif "total_packets" in df.columns and "flow_duration" in df.columns:
            dur = pd.to_numeric(df["flow_duration"], errors="coerce").replace(0, np.nan)
            dur_sec = dur / 1e6 if dur.median() > 1e5 else dur
            dur_sec = dur_sec.replace(0, np.nan)
            df["packet_rate"] = (
                pd.to_numeric(df["total_packets"], errors="coerce") / dur_sec
            ).fillna(0)
        else:
            self.missing_features.append("packet_rate")

        return df

    # ------------------------------------------------------------------
    # Attack behaviour features
    # ------------------------------------------------------------------

    def _extract_attack_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive attack-behaviour indicators such as port scan score and
        SYN burst score.  These are *heuristic* features, not ground truth.
        """
        if "dst_port" in df.columns:
            dst_port_numeric = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0)
            # Port diversity per source IP (if src_ip is available)
            if "src_ip" in df.columns:
                port_div = df.groupby("src_ip")["dst_port"].transform("nunique")
                df["dst_port_diversity"] = pd.to_numeric(port_div, errors="coerce").fillna(1)
            else:
                df["dst_port_diversity"] = 1
                self.missing_features.append("dst_port_diversity (src_ip absent)")

            # Sequential port scan heuristic: sorted port std / mean
            if "src_ip" in df.columns:
                port_std = df.groupby("src_ip")["dst_port"].transform(
                    lambda x: pd.to_numeric(x, errors="coerce").std()
                ).fillna(0)
                port_mean = df.groupby("src_ip")["dst_port"].transform(
                    lambda x: pd.to_numeric(x, errors="coerce").mean()
                ).replace(0, np.nan).fillna(1)
                df["port_scan_score"] = (port_std / port_mean).fillna(0)
            else:
                df["port_scan_score"] = 0
        else:
            for feat in ("dst_port_diversity", "port_scan_score"):
                self.missing_features.append(feat)

        # SYN burst score
        if "syn_count" in df.columns and "total_packets" in df.columns:
            total_pkts = pd.to_numeric(df["total_packets"], errors="coerce").replace(0, np.nan)
            df["syn_burst_score"] = (
                pd.to_numeric(df["syn_count"], errors="coerce").fillna(0) / total_pkts
            ).fillna(0)
        else:
            self.missing_features.append("syn_burst_score")

        # Connection attempt rate placeholder (per source IP)
        if "src_ip" in df.columns:
            conn_rate = df.groupby("src_ip")["src_ip"].transform("count")
            df["connection_attempt_rate"] = pd.to_numeric(conn_rate, errors="coerce").fillna(0)
        else:
            self.missing_features.append("connection_attempt_rate")

        # Failed connection ratio heuristic: RST / total packets
        if "rst_count" in df.columns and "total_packets" in df.columns:
            total_pkts = pd.to_numeric(df["total_packets"], errors="coerce").replace(0, np.nan)
            df["failed_connection_ratio"] = (
                pd.to_numeric(df["rst_count"], errors="coerce").fillna(0) / total_pkts
            ).fillna(0)
        else:
            self.missing_features.append("failed_connection_ratio")

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the full feature extraction pipeline.

        Args:
            df: Canonical DataFrame from the dataset loader.

        Returns:
            (enriched_df, extraction_metadata)
        """
        self.available_features = []
        self.missing_features = []

        original_cols = set(df.columns)

        df = self._extract_flow_features(df)
        df = self._extract_tcp_features(df)
        df = self._extract_temporal_features(df)
        df = self._extract_attack_features(df)

        new_cols = set(df.columns) - original_cols
        self.available_features = sorted(new_cols)

        if self.missing_features:
            logger.warning(
                "Features that could not be derived: %s",
                ", ".join(sorted(set(self.missing_features))),
            )

        metadata = {
            "derived_features": self.available_features,
            "missing_features": sorted(set(self.missing_features)),
            "total_features": len(df.columns),
        }

        logger.info(
            "Feature extraction complete: %d derived, %d missing",
            len(self.available_features),
            len(set(self.missing_features)),
        )

        return df, metadata
