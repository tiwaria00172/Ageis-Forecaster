# =============================================================================
# PCAP Processor — Extract network features from packet captures
# =============================================================================
"""
Reads PCAP/PCAPNG files via Scapy and aggregates packets into the same
canonical DataFrame format used by the CSV pipeline.  Extracted packet-level
features are aggregated per source-destination flow and then per time window.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Guard Scapy import — it is a heavy optional dependency
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning(
        "Scapy is not installed.  PCAP processing will not be available. "
        "Install with: pip install scapy"
    )


def _extract_packet_record(pkt) -> Optional[Dict[str, Any]]:
    """Extract a flat dict of features from a single Scapy packet."""
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    record: Dict[str, Any] = {
        "timestamp": float(pkt.time),
        "src_ip": ip.src,
        "dst_ip": ip.dst,
        "protocol": ip.proto,
        "packet_length": len(pkt),
        "ttl": ip.ttl,
        "fragment_flags": int(ip.flags),
    }

    # TCP layer
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        record["src_port"] = tcp.sport
        record["dst_port"] = tcp.dport
        record["tcp_flags"] = str(tcp.flags)
        record["tcp_window"] = tcp.window
        record["syn"] = int("S" in str(tcp.flags))
        record["ack"] = int("A" in str(tcp.flags))
        record["fin"] = int("F" in str(tcp.flags))
        record["rst"] = int("R" in str(tcp.flags))
        record["psh"] = int("P" in str(tcp.flags))
        record["urg"] = int("U" in str(tcp.flags))
        # Payload size
        payload = bytes(tcp.payload)
        record["payload_length"] = len(payload)
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        record["src_port"] = udp.sport
        record["dst_port"] = udp.dport
        record["tcp_flags"] = ""
        record["tcp_window"] = 0
        for flag in ("syn", "ack", "fin", "rst", "psh", "urg"):
            record[flag] = 0
        payload = bytes(udp.payload)
        record["payload_length"] = len(payload)
    else:
        record["src_port"] = 0
        record["dst_port"] = 0
        record["tcp_flags"] = ""
        record["tcp_window"] = 0
        for flag in ("syn", "ack", "fin", "rst", "psh", "urg"):
            record[flag] = 0
        record["payload_length"] = 0

    return record


def load_pcap(
    pcap_path: str,
    max_packets: int = 100_000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a PCAP/PCAPNG file and convert to a canonical DataFrame.

    Args:
        pcap_path: Path to the capture file.
        max_packets: Safety limit to avoid OOM on huge captures.

    Returns:
        (df, metadata) where df has canonical columns plus packet-level
        features (ttl, tcp_window, etc.).

    Raises:
        ImportError: If Scapy is not installed.
        FileNotFoundError: If pcap_path does not exist.
        ValueError: If no IP packets can be extracted.
    """
    if not SCAPY_AVAILABLE:
        raise ImportError(
            "Scapy is required for PCAP processing. "
            "Install with: pip install scapy"
        )

    import os
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    logger.info("Reading PCAP: %s (max %d packets)", pcap_path, max_packets)

    try:
        packets = rdpcap(pcap_path, count=max_packets)
    except Exception as exc:
        raise ValueError(f"Failed to read PCAP file: {exc}") from exc

    records: List[Dict[str, Any]] = []
    for pkt in packets:
        rec = _extract_packet_record(pkt)
        if rec is not None:
            records.append(rec)

    if not records:
        raise ValueError("No IP packets found in PCAP file.")

    df = pd.DataFrame(records)

    # Convert epoch timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # ── Aggregate to flow-level stats ────────────────────────────────
    # Group by (src_ip, dst_ip, src_port, dst_port, protocol)
    group_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    flows = (
        df.groupby(group_cols, dropna=False)
        .agg(
            timestamp=("timestamp", "first"),
            total_packets=("packet_length", "count"),
            total_bytes=("packet_length", "sum"),
            flow_duration=("timestamp", lambda x: (x.max() - x.min()).total_seconds()),
            ttl_mean=("ttl", "mean"),
            ttl_var=("ttl", "var"),
            tcp_window_mean=("tcp_window", "mean"),
            packet_size_mean=("packet_length", "mean"),
            packet_size_var=("packet_length", "var"),
            payload_mean=("payload_length", "mean"),
            syn_count=("syn", "sum"),
            ack_count=("ack", "sum"),
            fin_count=("fin", "sum"),
            rst_count=("rst", "sum"),
            psh_count=("psh", "sum"),
            urg_count=("urg", "sum"),
        )
        .reset_index()
    )

    # Fill NaN variance with 0
    for col in ("ttl_var", "packet_size_var"):
        flows[col] = flows[col].fillna(0)

    # PCAP data is unlabelled — assume Normal
    flows["label"] = "Normal"

    metadata = {
        "file": pcap_path,
        "total_raw_packets": len(packets),
        "ip_packets": len(records),
        "flows": len(flows),
        "timestamp_range": (
            str(flows["timestamp"].min()),
            str(flows["timestamp"].max()),
        ),
    }

    logger.info(
        "PCAP processed: %d raw packets → %d IP packets → %d flows",
        len(packets),
        len(records),
        len(flows),
    )

    return flows, metadata
