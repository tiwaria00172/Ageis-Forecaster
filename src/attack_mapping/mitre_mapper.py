# =============================================================================
# MITRE ATT&CK Mapper — Map predicted behaviour to ATT&CK stages
# =============================================================================
"""
Maps predicted network behaviour to MITRE ATT&CK stages and provides
human-readable descriptions.  Clearly distinguishes *observed* stages
(from labelled data) from *predicted/forecasted* stages (from the
world model).

NOTE: The mapping from raw dataset labels to ATT&CK stages is a
HEURISTIC simplification.  Real ATT&CK mapping requires detailed
threat intelligence.  This module documents this limitation explicitly.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── ATT&CK stage definitions ────────────────────────────────────────────

ATTACK_STAGES = {
    0: {
        "name": "Normal",
        "mitre_id": "N/A",
        "description": "Normal network activity — no indicators of compromise.",
        "color": "#2ecc71",
        "icon": "🟢",
    },
    1: {
        "name": "Reconnaissance",
        "mitre_id": "TA0043",
        "description": (
            "Adversary is gathering information about the target network. "
            "Indicators include port scanning, DNS enumeration, and service "
            "probing."
        ),
        "color": "#f39c12",
        "icon": "🟡",
        "indicators": [
            "High destination port diversity",
            "Sequential port access patterns",
            "Elevated SYN-to-ACK ratio",
            "Low payload sizes (probing)",
            "High connection attempt rate",
        ],
    },
    2: {
        "name": "Initial Access",
        "mitre_id": "TA0001",
        "description": (
            "Adversary is attempting to gain initial foothold in the network. "
            "Indicators include brute force attempts, exploit delivery, and "
            "credential stuffing."
        ),
        "color": "#e67e22",
        "icon": "🟠",
        "indicators": [
            "Repeated failed authentication attempts",
            "Unusual protocol usage",
            "Large payload transfer to single host",
            "Elevated RST/FIN ratios (failed connections)",
        ],
    },
    3: {
        "name": "Lateral Movement",
        "mitre_id": "TA0008",
        "description": (
            "Adversary is moving through the network to reach valuable targets. "
            "Indicators include internal scanning, credential reuse, and "
            "SMB/RDP traffic spikes."
        ),
        "color": "#e74c3c",
        "icon": "🔴",
        "indicators": [
            "Internal-to-internal scanning",
            "Multiple destination IPs from single source",
            "Unusual internal port access",
            "SMB/RDP traffic anomalies",
        ],
    },
    4: {
        "name": "Command and Control",
        "mitre_id": "TA0011",
        "description": (
            "Adversary has established communication with compromised hosts. "
            "Indicators include beaconing patterns, DNS tunnelling, and "
            "encrypted traffic to unknown destinations."
        ),
        "color": "#9b59b6",
        "icon": "🟣",
        "indicators": [
            "Regular beaconing intervals",
            "DNS query anomalies",
            "Encrypted traffic to rare destinations",
            "Persistent outbound connections",
        ],
    },
    5: {
        "name": "Exfiltration",
        "mitre_id": "TA0010",
        "description": (
            "Adversary is extracting data from the network. "
            "Indicators include large outbound data transfers, data "
            "staging, and unusual upload patterns."
        ),
        "color": "#1a1a2e",
        "icon": "⚫",
        "indicators": [
            "Large outbound data volumes",
            "Unusual upload-to-download ratio",
            "Data transfer to external IPs",
            "Off-hours activity",
        ],
    },
}


class MitreMapper:
    """
    Maps integer stage predictions to MITRE ATT&CK stage information
    and generates risk narratives.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.stages = ATTACK_STAGES
        self.config = config or {}

        # Custom label mapping from config
        mitre_cfg = self.config.get("mitre", {})
        self.label_mapping = mitre_cfg.get("label_mapping", {})

    def get_stage_info(self, stage_id: int) -> Dict[str, Any]:
        """Return the full stage metadata for a given stage ID."""
        return self.stages.get(stage_id, self.stages[0])

    def get_stage_name(self, stage_id: int) -> str:
        """Return just the stage name."""
        return self.stages.get(stage_id, self.stages[0])["name"]

    def get_stage_color(self, stage_id: int) -> str:
        """Return the stage colour for visualisation."""
        return self.stages.get(stage_id, self.stages[0])["color"]

    def map_label_to_stage(self, label: str) -> int:
        """Map a dataset label string to an ATT&CK stage integer."""
        return self.label_mapping.get(str(label).strip(), 0)

    def generate_risk_narrative(
        self,
        current_stage: int,
        predicted_stage: int,
        infiltration_prob: float,
        top_features: Optional[List[Tuple[str, float]]] = None,
    ) -> str:
        """
        Generate a template-based risk narrative without an external LLM.

        Args:
            current_stage: Currently observed ATT&CK stage.
            predicted_stage: Forecasted next stage.
            infiltration_prob: Predicted infiltration probability.
            top_features: List of (feature_name, importance) tuples.

        Returns:
            Human-readable risk narrative string.
        """
        curr_info = self.get_stage_info(current_stage)
        pred_info = self.get_stage_info(predicted_stage)

        risk_level = self._risk_label(infiltration_prob)

        narrative = (
            f"CURRENT STATE: {curr_info['icon']} {curr_info['name']} "
            f"({curr_info['mitre_id']})\n"
            f"PREDICTED NEXT STATE: {pred_info['icon']} {pred_info['name']} "
            f"({pred_info['mitre_id']})\n\n"
            f"RISK LEVEL: {risk_level} "
            f"(infiltration probability: {infiltration_prob:.1%})\n\n"
        )

        if predicted_stage > current_stage and predicted_stage > 0:
            narrative += (
                f"⚠ ESCALATION DETECTED: The model forecasts progression from "
                f"'{curr_info['name']}' to '{pred_info['name']}'. "
                f"{pred_info['description']}\n\n"
            )

        if top_features:
            narrative += "KEY RISK DRIVERS:\n"
            for feat_name, importance in top_features[:5]:
                direction = "↑ Increased risk" if importance > 0 else "↓ Reduced risk"
                narrative += f"  • {feat_name}: {importance:+.3f} ({direction})\n"

        # Add indicators for the predicted stage
        indicators = pred_info.get("indicators", [])
        if indicators:
            narrative += f"\nINDICATORS TO WATCH ({pred_info['name']}):\n"
            for ind in indicators:
                narrative += f"  • {ind}\n"

        return narrative

    def _risk_label(self, prob: float) -> str:
        """Map probability to a risk label string."""
        if prob >= 0.8:
            return "🔴 CRITICAL"
        elif prob >= 0.6:
            return "🟠 HIGH"
        elif prob >= 0.3:
            return "🟡 MEDIUM"
        return "🟢 LOW"

    def get_progression_path(
        self, forecast_stages: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Build a progression path from a list of forecasted stage IDs.

        Returns:
            List of dicts with stage info + step index.
        """
        path = []
        for i, stage_id in enumerate(forecast_stages):
            info = self.get_stage_info(stage_id)
            info_copy = dict(info)
            info_copy["step"] = i + 1
            info_copy["stage_id"] = stage_id
            path.append(info_copy)
        return path
