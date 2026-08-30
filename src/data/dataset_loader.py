# =============================================================================
# Dataset Loader — Flexible CSV / PCAP ingestion with schema auto-detection
# =============================================================================
"""
Loads network traffic datasets from CSV files and auto-detects the column
schema (CIC-IDS, CTU-13, or generic).  Provides a unified DataFrame with
canonical column names regardless of the source format.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


# ── Canonical column names that the rest of the pipeline expects ─────────

CANONICAL_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "total_bytes", "total_packets", "flow_duration", "label",
]


# ── Schema detection ────────────────────────────────────────────────────

def _score_schema(df_columns: List[str], mapping: Dict[str, str]) -> int:
    """Count how many mapping values exist in the DataFrame columns."""
    df_cols_lower = {c.strip().lower() for c in df_columns}
    return sum(
        1 for v in mapping.values() if v.strip().lower() in df_cols_lower
    )


def detect_schema(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[str, Dict[str, str]]:
    """
    Auto-detect the dataset schema by scoring each known column mapping.

    Returns:
        (schema_name, best_mapping) — e.g. ("cic_ids", {...})
    """
    mappings = config.get("data", {}).get("column_mappings", {})
    best_name, best_score, best_map = "generic", 0, {}

    for name, mapping in mappings.items():
        score = _score_schema(list(df.columns), mapping)
        if score > best_score:
            best_name, best_score, best_map = name, score, mapping

    logger.info(
        "Schema detected: '%s' (matched %d columns)", best_name, best_score
    )
    return best_name, best_map


# ── Column normalisation ────────────────────────────────────────────────

def _find_column(df_columns: List[str], target: str) -> Optional[str]:
    """Case-insensitive column lookup."""
    target_lower = target.strip().lower()
    for c in df_columns:
        if c.strip().lower() == target_lower:
            return c
    return None


def normalise_columns(
    df: pd.DataFrame, mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Rename columns in *df* from dataset-specific names to canonical names
    using the provided mapping.  Columns not present are skipped.
    """
    rename_map: Dict[str, str] = {}
    cols = list(df.columns)

    for canonical, raw in mapping.items():
        found = _find_column(cols, raw)
        if found is not None and found != canonical:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)
    return df


# ── Public API ──────────────────────────────────────────────────────────

def load_csv_dataset(
    csv_path: str,
    config: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a CSV network traffic dataset, detect its schema, and normalise
    column names to canonical form.

    Args:
        csv_path: Path to the CSV file.
        config: Configuration dict (loaded from YAML if None).
        nrows: Optional row limit for quick prototyping.

    Returns:
        (normalised_df, metadata) where metadata includes schema info,
        original columns, and record count.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CSV is empty or unreadable.
    """
    if config is None:
        config = load_config()

    csv_path = str(Path(csv_path).resolve())
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading CSV: %s", csv_path)

    # Try common encodings
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                nrows=nrows,
                low_memory=False,
            )
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to decode CSV file: {csv_path}")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    original_columns = list(df.columns)
    schema_name, mapping = detect_schema(df, config)
    df = normalise_columns(df, mapping)

    # ── Derive total_bytes / total_packets if absent ────────────────
    if "total_bytes" not in df.columns:
        fwd_len = next(
            (c for c in df.columns if "fwd" in c.lower() and "length" in c.lower()),
            None,
        )
        bwd_len = next(
            (c for c in df.columns if "bwd" in c.lower() and "length" in c.lower()),
            None,
        )
        if fwd_len and bwd_len:
            df["total_bytes"] = (
                pd.to_numeric(df[fwd_len], errors="coerce").fillna(0)
                + pd.to_numeric(df[bwd_len], errors="coerce").fillna(0)
            )

    if "total_packets" not in df.columns:
        fwd_pkt = next(
            (c for c in df.columns if "fwd" in c.lower() and "packet" in c.lower()),
            None,
        )
        bwd_pkt = next(
            (c for c in df.columns if "bwd" in c.lower() and "packet" in c.lower()),
            None,
        )
        if fwd_pkt and bwd_pkt:
            df["total_packets"] = (
                pd.to_numeric(df[fwd_pkt], errors="coerce").fillna(0)
                + pd.to_numeric(df[bwd_pkt], errors="coerce").fillna(0)
            )

    metadata = {
        "file": csv_path,
        "schema": schema_name,
        "original_columns": original_columns,
        "canonical_columns": list(df.columns),
        "num_records": len(df),
        "num_features": len(df.columns),
    }

    logger.info(
        "Loaded %d records with %d columns (schema: %s)",
        metadata["num_records"],
        metadata["num_features"],
        schema_name,
    )

    return df, metadata
