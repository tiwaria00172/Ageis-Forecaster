# =============================================================================
# Synthetic Data Generator — Demo data for testing the pipeline
# =============================================================================
"""
Generates synthetic network traffic data that simulates a multi-stage
attack progressing over time.  This data is for DEMONSTRATION AND
TESTING ONLY — it should never be presented as real benchmark results.

The generator creates a realistic time-ordered CSV with interleaved
benign and malicious traffic, where attack stages progress
chronologically to test the temporal world model's ability to learn
state transitions.

LABEL: Synthetic Demo Data
"""

import os
import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def generate_synthetic_traffic(
    n_records: int = 10000,
    attack_ratio: float = 0.35,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic network traffic with progressive attack stages.

    The traffic is generated in chronological order:
      - Phase 1 (0–20%):   Mostly benign
      - Phase 2 (20–40%):  Reconnaissance activity begins
      - Phase 3 (40–60%):  Initial Access attempts
      - Phase 4 (60–75%):  Lateral Movement
      - Phase 5 (75–85%):  Command & Control
      - Phase 6 (85–100%): Exfiltration + continued benign

    Args:
        n_records: Total number of flow records.
        attack_ratio: Approximate fraction of malicious flows.
        output_path: If provided, save CSV to this path.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with synthetic traffic.
    """
    rng = np.random.RandomState(seed)
    start_time = datetime(2025, 6, 15, 8, 0, 0)

    records = []
    internal_ips = [f"192.168.1.{i}" for i in range(1, 51)]
    external_ips = [f"10.0.{rng.randint(1, 255)}.{rng.randint(1, 255)}" for _ in range(30)]
    attacker_ip = "203.0.113.42"
    c2_ip = "198.51.100.77"

    for i in range(n_records):
        progress = i / n_records
        timestamp = start_time + timedelta(seconds=i * 0.5 + rng.uniform(0, 0.3))

        # Determine if this flow is an attack based on phase
        is_attack = False
        label = "BENIGN"
        stage = 0

        if progress < 0.20:
            # Phase 1: mostly benign
            is_attack = rng.random() < 0.02
            if is_attack:
                label = "PortScan"
                stage = 1
        elif progress < 0.40:
            # Phase 2: Reconnaissance
            is_attack = rng.random() < 0.25
            if is_attack:
                label = rng.choice(["PortScan", "SSH-Patator"])
                stage = 1
        elif progress < 0.60:
            # Phase 3: Initial Access
            is_attack = rng.random() < 0.35
            if is_attack:
                label = rng.choice(["Web Attack – Brute Force", "Infiltration", "DoS Hulk"])
                stage = 2
        elif progress < 0.75:
            # Phase 4: Lateral Movement
            is_attack = rng.random() < 0.40
            if is_attack:
                label = rng.choice(["Infiltration", "Web Attack – XSS"])
                stage = 3
        elif progress < 0.85:
            # Phase 5: C2
            is_attack = rng.random() < 0.30
            if is_attack:
                label = "Bot"
                stage = 4
        else:
            # Phase 6: Exfiltration
            is_attack = rng.random() < 0.20
            if is_attack:
                label = "Bot"
                stage = 5

        # Generate flow features based on attack type
        if is_attack:
            src_ip = attacker_ip if stage <= 2 else rng.choice(internal_ips)
            dst_ip = rng.choice(internal_ips) if stage >= 3 else rng.choice(internal_ips)

            if stage == 1:  # Recon
                dst_port = rng.randint(1, 65535)
                src_port = rng.randint(40000, 65535)
                protocol = 6  # TCP
                total_fwd_packets = rng.randint(1, 5)
                total_bwd_packets = rng.randint(0, 2)
                fwd_length = rng.randint(40, 200)
                bwd_length = rng.randint(0, 100)
                flow_duration = rng.randint(100, 5000)
                syn_count = rng.randint(1, 10)
                ack_count = rng.randint(0, 3)
                rst_count = rng.randint(0, 5)
                fin_count = 0
                psh_count = 0
                urg_count = 0
            elif stage == 2:  # Initial Access
                dst_port = rng.choice([22, 80, 443, 8080, 3389])
                src_port = rng.randint(40000, 65535)
                protocol = 6
                total_fwd_packets = rng.randint(10, 100)
                total_bwd_packets = rng.randint(5, 50)
                fwd_length = rng.randint(500, 5000)
                bwd_length = rng.randint(200, 3000)
                flow_duration = rng.randint(10000, 100000)
                syn_count = rng.randint(5, 20)
                ack_count = rng.randint(10, 50)
                rst_count = rng.randint(2, 15)
                fin_count = rng.randint(0, 3)
                psh_count = rng.randint(5, 30)
                urg_count = 0
            elif stage == 3:  # Lateral
                dst_ip = rng.choice(internal_ips)
                dst_port = rng.choice([135, 139, 445, 3389, 5985])
                src_port = rng.randint(40000, 65535)
                protocol = 6
                total_fwd_packets = rng.randint(20, 200)
                total_bwd_packets = rng.randint(15, 150)
                fwd_length = rng.randint(1000, 10000)
                bwd_length = rng.randint(500, 8000)
                flow_duration = rng.randint(50000, 500000)
                syn_count = rng.randint(3, 10)
                ack_count = rng.randint(20, 80)
                rst_count = rng.randint(0, 5)
                fin_count = rng.randint(1, 5)
                psh_count = rng.randint(10, 50)
                urg_count = 0
            elif stage == 4:  # C2
                dst_ip = c2_ip
                dst_port = rng.choice([443, 8443, 4444, 53])
                src_port = rng.randint(40000, 65535)
                protocol = rng.choice([6, 17])  # TCP or UDP
                total_fwd_packets = rng.randint(5, 30)
                total_bwd_packets = rng.randint(5, 30)
                fwd_length = rng.randint(200, 2000)
                bwd_length = rng.randint(200, 2000)
                flow_duration = rng.randint(100000, 1000000)
                syn_count = rng.randint(1, 3)
                ack_count = rng.randint(10, 30)
                rst_count = 0
                fin_count = rng.randint(0, 2)
                psh_count = rng.randint(5, 20)
                urg_count = 0
            else:  # Exfiltration
                dst_ip = c2_ip
                dst_port = rng.choice([443, 8443, 21, 22])
                src_port = rng.randint(40000, 65535)
                protocol = 6
                total_fwd_packets = rng.randint(50, 500)
                total_bwd_packets = rng.randint(10, 50)
                fwd_length = rng.randint(10000, 500000)
                bwd_length = rng.randint(500, 5000)
                flow_duration = rng.randint(200000, 2000000)
                syn_count = rng.randint(1, 3)
                ack_count = rng.randint(30, 100)
                rst_count = 0
                fin_count = rng.randint(1, 5)
                psh_count = rng.randint(20, 80)
                urg_count = 0
        else:
            # Benign traffic
            src_ip = rng.choice(internal_ips)
            dst_ip = rng.choice(external_ips + internal_ips)
            dst_port = rng.choice([80, 443, 53, 8080, 993, 587, 25, 110])
            src_port = rng.randint(1024, 65535)
            protocol = rng.choice([6, 17])
            total_fwd_packets = rng.randint(1, 50)
            total_bwd_packets = rng.randint(1, 50)
            fwd_length = rng.randint(100, 10000)
            bwd_length = rng.randint(100, 50000)
            flow_duration = rng.randint(1000, 500000)
            syn_count = rng.randint(0, 2)
            ack_count = rng.randint(1, 20)
            rst_count = rng.randint(0, 1)
            fin_count = rng.randint(0, 2)
            psh_count = rng.randint(0, 15)
            urg_count = 0

        # IAT features
        iat_mean = rng.uniform(10, 100000) if not is_attack else rng.uniform(1, 10000)
        iat_std = iat_mean * rng.uniform(0.1, 2.0)

        total_pkts = total_fwd_packets + total_bwd_packets
        total_bytes = fwd_length + bwd_length

        records.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "Src IP": src_ip,
            "Dst IP": dst_ip,
            "Src Port": src_port,
            "Dst Port": dst_port,
            "Protocol": protocol,
            "Total Fwd Packet": total_fwd_packets,
            "Total Bwd packets": total_bwd_packets,
            "Total Length of Fwd Packet": fwd_length,
            "Total Length of Bwd Packet": bwd_length,
            "Flow Duration": flow_duration,
            "Flow Bytes/s": total_bytes / max(flow_duration / 1e6, 1e-6),
            "Flow Packets/s": total_pkts / max(flow_duration / 1e6, 1e-6),
            "Fwd IAT Mean": iat_mean,
            "Bwd IAT Mean": iat_mean * rng.uniform(0.8, 1.2),
            "Fwd IAT Std": iat_std,
            "Bwd IAT Std": iat_std * rng.uniform(0.5, 1.5),
            "SYN Flag Count": syn_count,
            "ACK Flag Count": ack_count,
            "FIN Flag Count": fin_count,
            "RST Flag Count": rst_count,
            "PSH Flag Count": psh_count,
            "URG Flag Count": urg_count,
            "Label": label,
        })

    df = pd.DataFrame(records)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[OK] Synthetic data saved: {output_path}")
        print(f"    Records: {len(df)}")
        print(f"    Label distribution:")
        print(df["Label"].value_counts().to_string(header=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic network traffic")
    parser.add_argument("--records", type=int, default=10000, help="Number of records")
    parser.add_argument(
        "--output", type=str, default="data/sample/synthetic_traffic.csv",
        help="Output CSV path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_synthetic_traffic(
        n_records=args.records,
        output_path=args.output,
        seed=args.seed,
    )
