"""
Synthetic Network Traffic Generator
------------------------------------

Generates chronological synthetic network traffic for the
Network Attack Forecasting World Model.

IMPORTANT:
- This is synthetic data for development/demo/testing.
- It is NOT real network telemetry.
- Multiple attack campaigns are distributed across the timeline.
- Temporal ordering is preserved.
- Each campaign progresses through:
    Normal
    -> Reconnaissance
    -> Initial Access
    -> Lateral Movement
    -> Command & Control
    -> Exfiltration

Output contains:
- CIC-IDS-style flow features
- attack_stage
- is_attack
- Label

MITRE-style stage mapping:
    0 = Normal
    1 = Reconnaissance
    2 = Initial Access
    3 = Lateral Movement
    4 = Command & Control
    5 = Exfiltration
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------

STAGE_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration",
}


# ---------------------------------------------------------------------
# Attack labels used by the project
# ---------------------------------------------------------------------

STAGE_LABELS = {
    0: ["BENIGN"],
    1: [
        "PortScan",
        "SSH-Patator",
    ],
    2: [
        "Web Attack – Brute Force",
        "Infiltration",
        "DoS Hulk",
    ],
    3: [
        "Infiltration",
        "Web Attack – XSS",
    ],
    4: [
        "Bot",
    ],
    5: [
        "Bot",
        "Infiltration",
    ],
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    """Generate a clipped normally distributed value."""
    return float(np.clip(rng.normal(mean, std), low, high))


def random_ip(rng: np.random.Generator, internal: bool = False) -> str:
    """Generate a synthetic IPv4 address."""
    if internal:
        return (
            f"10."
            f"{rng.integers(0, 256)}."
            f"{rng.integers(0, 256)}."
            f"{rng.integers(1, 255)}"
        )

    return (
        f"{rng.integers(1, 223)}."
        f"{rng.integers(0, 256)}."
        f"{rng.integers(0, 256)}."
        f"{rng.integers(1, 255)}"
    )


def choose_protocol(
    rng: np.random.Generator,
    stage: int,
) -> str:
    """Choose a protocol based on attack stage."""

    if stage == 1:
        return str(rng.choice(["TCP", "UDP"]))

    if stage == 2:
        return str(rng.choice(["TCP", "TCP", "HTTP", "HTTPS"]))

    if stage == 3:
        return str(rng.choice(["TCP", "TCP", "SMB", "SSH"]))

    if stage == 4:
        return str(rng.choice(["TCP", "TCP", "UDP"]))

    if stage == 5:
        return str(rng.choice(["TCP", "TCP", "UDP"]))

    return str(rng.choice(["TCP", "TCP", "UDP", "ICMP"]))


def choose_port(
    rng: np.random.Generator,
    stage: int,
) -> int:
    """Choose destination port based on attack stage."""

    if stage == 1:
        return int(
            rng.choice(
                [
                    21,
                    22,
                    23,
                    25,
                    53,
                    80,
                    110,
                    135,
                    139,
                    143,
                    443,
                    445,
                    3389,
                    rng.integers(1, 65535),
                ]
            )
        )

    if stage == 2:
        return int(rng.choice([80, 443, 8080, 8000]))

    if stage == 3:
        return int(rng.choice([22, 135, 139, 445, 3389]))

    if stage == 4:
        return int(rng.choice([53, 80, 443, 8080]))

    if stage == 5:
        return int(rng.choice([80, 443, 21, 22]))

    return int(
        rng.choice(
            [
                53,
                80,
                443,
                22,
                25,
                110,
                123,
                3306,
                5432,
            ]
        )
    )


# ---------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------

def generate_features(
    rng: np.random.Generator,
    stage: int,
    label: str,
) -> Dict[str, float]:
    """
    Generate CIC-IDS-style flow and packet-level features.

    Values are intentionally synthetic but stage-dependent so that
    temporal dynamics can be learned by the demo World Model.
    """

    # -------------------------------------------------------------
    # Normal traffic baseline
    # -------------------------------------------------------------

    if stage == 0:

        duration = clipped_normal(
            rng,
            mean=2.5,
            std=1.2,
            low=0.05,
            high=8.0,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=18,
                std=8,
                low=2,
                high=60,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=9000,
                std=4000,
                low=500,
                high=30000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.45, 0.65)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(rng.binomial(3, 0.25))
        ack = int(rng.binomial(8, 0.75))
        fin = int(rng.binomial(3, 0.35))
        rst = int(rng.binomial(2, 0.08))

        packet_size_mean = clipped_normal(
            rng,
            mean=500,
            std=150,
            low=50,
            high=1500,
        )

        iat_mean = clipped_normal(
            rng,
            mean=0.15,
            std=0.08,
            low=0.005,
            high=0.5,
        )

        ttl = clipped_normal(
            rng,
            mean=64,
            std=8,
            low=32,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=64240,
            std=10000,
            low=8192,
            high=65535,
        )

    # -------------------------------------------------------------
    # Reconnaissance
    # -------------------------------------------------------------

    elif stage == 1:

        duration = clipped_normal(
            rng,
            mean=0.20,
            std=0.10,
            low=0.01,
            high=0.8,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=5,
                std=3,
                low=2,
                high=20,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=500,
                std=250,
                low=100,
                high=2000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.75, 0.95)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(
            clipped_normal(
                rng,
                mean=4,
                std=2,
                low=1,
                high=10,
            )
        )

        ack = int(
            clipped_normal(
                rng,
                mean=1,
                std=1,
                low=0,
                high=4,
            )
        )

        fin = int(rng.binomial(2, 0.1))
        rst = int(
            clipped_normal(
                rng,
                mean=1,
                std=1,
                low=0,
                high=4,
            )
        )

        packet_size_mean = clipped_normal(
            rng,
            mean=100,
            std=40,
            low=40,
            high=300,
        )

        iat_mean = clipped_normal(
            rng,
            mean=0.02,
            std=0.01,
            low=0.001,
            high=0.1,
        )

        ttl = clipped_normal(
            rng,
            mean=50,
            std=15,
            low=20,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=32000,
            std=9000,
            low=4096,
            high=65535,
        )

    # -------------------------------------------------------------
    # Initial Access
    # -------------------------------------------------------------

    elif stage == 2:

        duration = clipped_normal(
            rng,
            mean=3.0,
            std=1.5,
            low=0.1,
            high=10,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=45,
                std=18,
                low=5,
                high=120,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=25000,
                std=12000,
                low=1000,
                high=90000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.55, 0.8)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(
            clipped_normal(
                rng,
                mean=8,
                std=3,
                low=1,
                high=20,
            )
        )

        ack = int(
            clipped_normal(
                rng,
                mean=12,
                std=5,
                low=1,
                high=30,
            )
        )

        fin = int(rng.binomial(5, 0.25))
        rst = int(rng.binomial(4, 0.15))

        packet_size_mean = clipped_normal(
            rng,
            mean=650,
            std=200,
            low=100,
            high=1500,
        )

        iat_mean = clipped_normal(
            rng,
            mean=0.08,
            std=0.04,
            low=0.003,
            high=0.3,
        )

        ttl = clipped_normal(
            rng,
            mean=58,
            std=10,
            low=20,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=48000,
            std=9000,
            low=8192,
            high=65535,
        )

    # -------------------------------------------------------------
    # Lateral Movement
    # -------------------------------------------------------------

    elif stage == 3:

        duration = clipped_normal(
            rng,
            mean=4.5,
            std=2,
            low=0.2,
            high=15,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=65,
                std=25,
                low=10,
                high=160,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=45000,
                std=20000,
                low=2000,
                high=150000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.50, 0.75)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(
            clipped_normal(
                rng,
                mean=7,
                std=3,
                low=1,
                high=20,
            )
        )

        ack = int(
            clipped_normal(
                rng,
                mean=25,
                std=8,
                low=5,
                high=50,
            )
        )

        fin = int(rng.binomial(5, 0.30))
        rst = int(rng.binomial(5, 0.15))

        packet_size_mean = clipped_normal(
            rng,
            mean=720,
            std=220,
            low=100,
            high=1500,
        )

        iat_mean = clipped_normal(
            rng,
            mean=0.12,
            std=0.05,
            low=0.005,
            high=0.4,
        )

        ttl = clipped_normal(
            rng,
            mean=61,
            std=9,
            low=20,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=52000,
            std=8000,
            low=8192,
            high=65535,
        )

    # -------------------------------------------------------------
    # Command & Control
    # -------------------------------------------------------------

    elif stage == 4:

        duration = clipped_normal(
            rng,
            mean=8,
            std=3,
            low=0.5,
            high=25,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=35,
                std=12,
                low=5,
                high=100,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=18000,
                std=7000,
                low=1000,
                high=60000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.40, 0.60)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(rng.binomial(3, 0.35))

        ack = int(
            clipped_normal(
                rng,
                mean=20,
                std=7,
                low=3,
                high=50,
            )
        )

        fin = int(rng.binomial(4, 0.20))
        rst = int(rng.binomial(3, 0.10))

        packet_size_mean = clipped_normal(
            rng,
            mean=450,
            std=150,
            low=50,
            high=1200,
        )

        # Periodic communication
        iat_mean = clipped_normal(
            rng,
            mean=0.8,
            std=0.25,
            low=0.1,
            high=2.0,
        )

        ttl = clipped_normal(
            rng,
            mean=58,
            std=10,
            low=20,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=50000,
            std=9000,
            low=8192,
            high=65535,
        )

    # -------------------------------------------------------------
    # Exfiltration
    # -------------------------------------------------------------

    else:

        duration = clipped_normal(
            rng,
            mean=12,
            std=5,
            low=1,
            high=40,
        )

        total_packets = int(
            clipped_normal(
                rng,
                mean=120,
                std=45,
                low=20,
                high=300,
            )
        )

        total_bytes = int(
            clipped_normal(
                rng,
                mean=160000,
                std=60000,
                low=10000,
                high=500000,
            )
        )

        fwd_packets = max(
            1,
            int(total_packets * rng.uniform(0.65, 0.85)),
        )

        bwd_packets = max(
            1,
            total_packets - fwd_packets,
        )

        syn = int(rng.binomial(3, 0.25))

        ack = int(
            clipped_normal(
                rng,
                mean=65,
                std=20,
                low=10,
                high=120,
            )
        )

        fin = int(rng.binomial(8, 0.30))
        rst = int(rng.binomial(4, 0.08))

        packet_size_mean = clipped_normal(
            rng,
            mean=1200,
            std=180,
            low=300,
            high=1500,
        )

        iat_mean = clipped_normal(
            rng,
            mean=0.04,
            std=0.015,
            low=0.002,
            high=0.15,
        )

        ttl = clipped_normal(
            rng,
            mean=60,
            std=9,
            low=20,
            high=128,
        )

        tcp_window = clipped_normal(
            rng,
            mean=60000,
            std=5000,
            low=8192,
            high=65535,
        )

    # -----------------------------------------------------------------
    # Derived values
    # -----------------------------------------------------------------

    bytes_per_packet = total_bytes / max(total_packets, 1)
    packets_per_second = total_packets / max(duration, 0.001)
    bytes_per_second = total_bytes / max(duration, 0.001)

    fwd_bwd_packet_ratio = (
        fwd_packets / max(bwd_packets, 1)
    )

    syn_ratio = syn / max(total_packets, 1)
    ack_ratio = ack / max(total_packets, 1)
    rst_ratio = rst / max(total_packets, 1)

    iat_variance = max(
        0.000001,
        iat_mean * iat_mean * rng.uniform(0.4, 1.8),
    )

    iat_max = max(
        iat_mean,
        iat_mean * rng.uniform(2.0, 8.0),
    )

    # Packet-level characteristics
    payload_mean = max(
        0,
        packet_size_mean * rng.uniform(0.35, 0.85),
    )

    payload_std = max(
        1,
        packet_size_mean * rng.uniform(0.05, 0.30),
    )

    payload_max = min(
        1500,
        max(
            payload_mean,
            packet_size_mean * rng.uniform(1.1, 1.8),
        ),
    )

    fragment_flags = int(
        rng.binomial(
            2,
            0.15 if stage in [1, 3] else 0.03,
        )
    )

    retransmissions = int(
        rng.poisson(
            2.0 if stage in [1, 2, 3] else 0.7
        )
    )

    # Attack-specific heuristic signals
    dst_port_diversity = (
        rng.uniform(8, 40)
        if stage == 1
        else rng.uniform(1, 5)
    )

    port_scan_score = (
        rng.uniform(0.65, 1.0)
        if stage == 1
        else rng.uniform(0.0, 0.25)
    )

    syn_burst_score = (
        rng.uniform(0.60, 1.0)
        if stage == 1
        else rng.uniform(0.0, 0.30)
    )

    connection_attempt_rate = (
        packets_per_second
        * rng.uniform(0.5, 1.5)
    )

    failed_connection_ratio = (
        rng.uniform(0.35, 0.90)
        if stage == 1
        else rng.uniform(0.02, 0.25)
    )

    # -----------------------------------------------------------------
    # Return CIC-IDS-style feature dictionary
    # -----------------------------------------------------------------

    return {
        "Flow Duration": duration * 1_000_000,
        "Total Fwd Packets": fwd_packets,
        "Total Backward Packets": bwd_packets,
        "Total Length of Fwd Packets": total_bytes * 0.60,
        "Total Length of Bwd Packets": total_bytes * 0.40,
        "Fwd Packet Length Max": packet_size_mean * 1.5,
        "Fwd Packet Length Mean": packet_size_mean,
        "Bwd Packet Length Max": packet_size_mean * 1.2,
        "Bwd Packet Length Mean": packet_size_mean * 0.8,
        "Flow Bytes/s": bytes_per_second,
        "Flow Packets/s": packets_per_second,
        "Flow IAT Mean": iat_mean,
        "Flow IAT Std": np.sqrt(iat_variance),
        "Flow IAT Max": iat_max,
        "Fwd IAT Mean": iat_mean * rng.uniform(0.8, 1.2),
        "Bwd IAT Mean": iat_mean * rng.uniform(0.8, 1.3),
        "Fwd PSH Flags": int(rng.binomial(4, 0.25)),
        "Bwd PSH Flags": int(rng.binomial(4, 0.15)),
        "Fwd URG Flags": int(rng.binomial(2, 0.05)),
        "Bwd URG Flags": int(rng.binomial(2, 0.03)),
        "Fwd Header Length": int(fwd_packets * rng.uniform(20, 40)),
        "Bwd Header Length": int(bwd_packets * rng.uniform(20, 40)),
        "Fwd Packets/s": fwd_packets / max(duration, 0.001),
        "Bwd Packets/s": bwd_packets / max(duration, 0.001),
        "Min Packet Length": max(20, packet_size_mean * 0.2),
        "Max Packet Length": payload_max,
        "Packet Length Mean": packet_size_mean,
        "Packet Length Std": payload_std,
        "Packet Length Variance": payload_std ** 2,
        "Average Packet Size": bytes_per_packet,
        "Avg Fwd Segment Size": packet_size_mean,
        "Avg Bwd Segment Size": packet_size_mean * 0.8,
        "Subflow Fwd Packets": fwd_packets,
        "Subflow Fwd Bytes": total_bytes * 0.60,
        "Subflow Bwd Packets": bwd_packets,
        "Subflow Bwd Bytes": total_bytes * 0.40,
        "Init_Win_bytes_forward": tcp_window,
        "Init_Win_bytes_backward": tcp_window * rng.uniform(0.6, 1.0),
        "act_data_pkt_fwd": max(1, int(fwd_packets * rng.uniform(0.3, 0.9))),
        "min_seg_size_forward": rng.uniform(20, 40),

        # Explicit TCP flags
        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "FIN Flag Count": fin,
        "RST Flag Count": rst,
        "PSH Flag Count": int(rng.binomial(6, 0.2)),
        "URG Flag Count": int(rng.binomial(3, 0.05)),

        # Packet-level / derived telemetry
        "TTL Mean": ttl,
        "TCP Window Mean": tcp_window,
        "Fragment Flag Count": fragment_flags,
        "Payload Size Mean": payload_mean,
        "Payload Size Std": payload_std,
        "Payload Size Max": payload_max,
        "Retransmission Count": retransmissions,

        # Derived attack indicators
        "bytes_per_packet": bytes_per_packet,
        "packets_per_second": packets_per_second,
        "bytes_per_second": bytes_per_second,
        "fwd_bwd_packet_ratio": fwd_bwd_packet_ratio,
        "syn_ratio": syn_ratio,
        "ack_ratio": ack_ratio,
        "rst_ratio": rst_ratio,
        "syn_ack_ratio": syn / max(ack, 1),

        "iat_mean": iat_mean,
        "iat_variance": iat_variance,
        "iat_max": iat_max,

        "dst_port_diversity": dst_port_diversity,
        "port_scan_score": port_scan_score,
        "syn_burst_score": syn_burst_score,
        "connection_attempt_rate": connection_attempt_rate,
        "failed_connection_ratio": failed_connection_ratio,
    }


# ---------------------------------------------------------------------
# Campaign generation
# ---------------------------------------------------------------------

def generate_campaign(
    rng: np.random.Generator,
    campaign_id: int,
    records_per_stage: List[int],
) -> List[Dict]:
    """
    Generate one chronological attack campaign.

    Every campaign progresses:

        Normal
          ↓
        Reconnaissance
          ↓
        Initial Access
          ↓
        Lateral Movement
          ↓
        Command & Control
          ↓
        Exfiltration
    """

    records: List[Dict] = []

    # Small amount of benign traffic before the attack starts
    warmup = max(5, records_per_stage[0])

    for _ in range(warmup):

        stage = 0
        label = "BENIGN"

        records.append(
            {
                "campaign_id": campaign_id,
                "stage": stage,
                "label": label,
                "features": generate_features(
                    rng,
                    stage,
                    label,
                ),
            }
        )

    # Attack stages
    for stage in range(1, 6):

        count = max(
            5,
            records_per_stage[stage],
        )

        for _ in range(count):

            label = str(
                rng.choice(
                    STAGE_LABELS[stage]
                )
            )

            records.append(
                {
                    "campaign_id": campaign_id,
                    "stage": stage,
                    "label": label,
                    "features": generate_features(
                        rng,
                        stage,
                        label,
                    ),
                }
            )

    # Recovery / cleanup period
    cleanup_count = max(
        5,
        records_per_stage[0] // 2,
    )

    for _ in range(cleanup_count):

        stage = 0
        label = "BENIGN"

        records.append(
            {
                "campaign_id": campaign_id,
                "stage": stage,
                "label": label,
                "features": generate_features(
                    rng,
                    stage,
                    label,
                ),
            }
        )

    return records


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_synthetic_traffic(
    n_records: int = 10000,
    attack_ratio: float = 0.35,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate multiple chronological attack campaigns.

    Parameters
    ----------
    n_records:
        Number of traffic records.

    attack_ratio:
        Approximate fraction of records belonging to attack stages.

    output_path:
        Optional CSV output path.

    seed:
        Random seed.

    Returns
    -------
    pandas.DataFrame
    """

    if n_records < 1000:
        raise ValueError(
            "n_records should be at least 1000."
        )

    if not 0.05 <= attack_ratio <= 0.80:
        raise ValueError(
            "attack_ratio should be between 0.05 and 0.80."
        )

    rng = np.random.default_rng(seed)

    random.seed(seed)

    logger.info(
        "Generating %d synthetic records "
        "(attack_ratio=%.2f, seed=%d)",
        n_records,
        attack_ratio,
        seed,
    )

    # -----------------------------------------------------------------
    # Determine number of campaigns
    # -----------------------------------------------------------------

    # Multiple campaigns are essential for chronological evaluation.
    n_campaigns = max(
        5,
        min(10, n_records // 900),
    )

    logger.info(
        "Generating %d chronological attack campaigns.",
        n_campaigns,
    )

    # -----------------------------------------------------------------
    # Build campaign sizes
    # -----------------------------------------------------------------

    # Allocate approximately equal record counts to campaigns.
    base_size = n_records // n_campaigns

    campaign_sizes = [
        base_size
        for _ in range(n_campaigns)
    ]

    campaign_sizes[-1] += (
        n_records - sum(campaign_sizes)
    )

    all_records: List[Dict] = []

    # -----------------------------------------------------------------
    # Generate campaigns
    # -----------------------------------------------------------------

    for campaign_id, campaign_size in enumerate(
        campaign_sizes,
        start=1,
    ):

        # Approximate stage proportions within each campaign.
        #
        # The first value is benign warmup.
        # Remaining stages form the attack progression.

        benign_before = int(
            campaign_size * 0.12
        )

        recon = int(
            campaign_size * attack_ratio * 0.18
        )

        initial_access = int(
            campaign_size * attack_ratio * 0.20
        )

        lateral = int(
            campaign_size * attack_ratio * 0.20
        )

        c2 = int(
            campaign_size * attack_ratio * 0.18
        )

        exfil = int(
            campaign_size * attack_ratio * 0.24
        )

        allocated = (
            benign_before
            + recon
            + initial_access
            + lateral
            + c2
            + exfil
        )

        # Remaining records become benign traffic.
        remaining = max(
            0,
            campaign_size - allocated,
        )

        # Divide remaining benign traffic between
        # pre-attack and post-attack periods.
        benign_before += remaining // 2

        benign_after = (
            remaining
            - remaining // 2
        )

        # The campaign generator uses the first value as
        # warmup and automatically creates a cleanup section.
        #
        # Therefore compensate slightly for cleanup.
        stage_counts = [
            max(
                5,
                benign_before,
            ),
            max(5, recon),
            max(5, initial_access),
            max(5, lateral),
            max(5, c2),
            max(5, exfil),
        ]

        campaign_records = generate_campaign(
            rng=rng,
            campaign_id=campaign_id,
            records_per_stage=stage_counts,
        )

        # Add additional benign traffic after the campaign.
        for _ in range(benign_after):

            campaign_records.append(
                {
                    "campaign_id": campaign_id,
                    "stage": 0,
                    "label": "BENIGN",
                    "features": generate_features(
                        rng,
                        0,
                        "BENIGN",
                    ),
                }
            )

        all_records.extend(
            campaign_records
        )

    # -----------------------------------------------------------------
    # Trim / expand exactly to requested size
    # -----------------------------------------------------------------

    if len(all_records) > n_records:

        all_records = all_records[:n_records]

    elif len(all_records) < n_records:

        missing = n_records - len(all_records)

        last_campaign = n_campaigns

        for _ in range(missing):

            all_records.append(
                {
                    "campaign_id": last_campaign,
                    "stage": 0,
                    "label": "BENIGN",
                    "features": generate_features(
                        rng,
                        0,
                        "BENIGN",
                    ),
                }
            )

    # -----------------------------------------------------------------
    # Convert to rows
    # -----------------------------------------------------------------

    start_time = pd.Timestamp(
        "2025-06-15 08:00:00"
    )

    rows: List[Dict] = []

    for i, record in enumerate(
        all_records
    ):

        stage = int(
            record["stage"]
        )

        label = record["label"]

        is_attack = int(
            stage > 0
        )

        # Small random interval between flows.
        # Keeps timestamps strictly chronological.
        timestamp = (
            start_time
            + pd.Timedelta(
                milliseconds=(
                    i * 500
                    + rng.integers(
                        0,
                        250,
                    )
                )
            )
        )

        # Synthetic endpoints.
        if stage == 0:

            src_ip = random_ip(
                rng,
                internal=True,
            )

            dst_ip = random_ip(
                rng,
                internal=False,
            )

        else:

            src_ip = random_ip(
                rng,
                internal=False,
            )

            dst_ip = random_ip(
                rng,
                internal=True,
            )

        dst_port = choose_port(
            rng,
            stage,
        )

        protocol = choose_protocol(
            rng,
            stage,
        )

        row = {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "dst_port": dst_port,
            "protocol": protocol,
        }

        row.update(
            record["features"]
        )

        # Labels required by the project.
        row["Label"] = label
        row["attack_stage"] = stage
        row["is_attack"] = is_attack

        # Campaign information is useful for debugging and analysis.
        row["campaign_id"] = int(
            record["campaign_id"]
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Final sorting
    # -----------------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    assert len(df) == n_records

    assert df["timestamp"].is_monotonic_increasing

    assert set(
        df["attack_stage"].unique()
    ).issubset(
        {0, 1, 2, 3, 4, 5}
    )

    # Verify stage/binary consistency.
    expected_attack = (
        df["attack_stage"] > 0
    ).astype(int)

    if not np.array_equal(
        expected_attack.values,
        df["is_attack"].values,
    ):
        raise RuntimeError(
            "attack_stage and is_attack are inconsistent."
        )

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    logger.info(
        "Synthetic dataset generated successfully."
    )

    logger.info(
        "Records: %d",
        len(df),
    )

    logger.info(
        "Campaigns: %d",
        df["campaign_id"].nunique(),
    )

    logger.info(
        "Time range: %s → %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )

    logger.info(
        "Attack percentage: %.2f%%",
        df["is_attack"].mean() * 100,
    )

    logger.info(
        "Label distribution:"
    )

    for label, count in (
        df["Label"]
        .value_counts()
        .items()
    ):
        logger.info(
            "  %-30s %d",
            label,
            count,
        )

    logger.info(
        "Stage distribution:"
    )

    for stage in range(6):

        count = int(
            (
                df["attack_stage"]
                == stage
            ).sum()
        )

        percentage = (
            count
            / len(df)
            * 100
        )

        logger.info(
            "  Stage %d (%s): %d (%.2f%%)",
            stage,
            STAGE_NAMES[stage],
            count,
            percentage,
        )

    logger.info(
        "Campaign distribution:"
    )

    for campaign_id, count in (
        df["campaign_id"]
        .value_counts()
        .sort_index()
        .items()
    ):
        logger.info(
            "  Campaign %d: %d records",
            campaign_id,
            count,
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Saved synthetic dataset to: %s",
            output_path.resolve(),
        )

    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate chronological synthetic "
            "network traffic for the World Model."
        )
    )

    parser.add_argument(
        "--records",
        type=int,
        default=10000,
        help=(
            "Number of records to generate "
            "(default: 10000)"
        ),
    )

    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.35,
        help=(
            "Approximate attack ratio "
            "(default: 0.35)"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/sample/"
            "synthetic_traffic.csv"
        ),
        help=(
            "Output CSV path."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed "
            "(default: 42)"
        ),
    )

    args = parser.parse_args()

    generate_synthetic_traffic(
        n_records=args.records,
        attack_ratio=args.attack_ratio,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()