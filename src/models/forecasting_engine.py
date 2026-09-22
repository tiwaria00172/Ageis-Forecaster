# =============================================================================
# Forecasting Engine — K-step recursive forward simulation
# =============================================================================
"""
Uses a trained NetworkWorldModel to recursively predict future network
states.  Given a historical sequence [S_{t-n}, ..., S_t], the engine:

  1. Predicts S_{t+1} and its infiltration / stage estimates.
  2. Appends S_{t+1} to the sequence (dropping S_{t-n}).
  3. Repeats for K steps to produce S_{t+2}, ..., S_{t+K}.

This is the core innovation: the world model treats an attack as an
*evolving process* and forecasts its progression, rather than merely
classifying individual flows.

IMPORTANT SIMPLIFICATION:
When recursively feeding predicted states, the model only has access to
its own predictions (not ground truth) for future steps.  This means
prediction errors compound over time, so confidence naturally degrades.
A configurable `confidence_decay` factor is applied to reflect this.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.world_model import NetworkWorldModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForecastResult:
    """Container for a single forecast step's predictions."""

    __slots__ = (
        "step", "predicted_state", "infiltration_prob",
        "attack_stage", "stage_probs", "confidence",
    )

    def __init__(
        self,
        step: int,
        predicted_state: np.ndarray,
        infiltration_prob: float,
        attack_stage: int,
        stage_probs: np.ndarray,
        confidence: float,
    ):
        self.step = step
        self.predicted_state = predicted_state
        self.infiltration_prob = infiltration_prob
        self.attack_stage = attack_stage
        self.stage_probs = stage_probs
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "step": self.step,
            "infiltration_probability": round(self.infiltration_prob, 4),
            "predicted_stage": int(self.attack_stage),
            "stage_probabilities": [round(float(p), 4) for p in self.stage_probs],
            "confidence": round(self.confidence, 4),
        }


class ForecastingEngine:
    """
    Performs K-step recursive forward simulation using a trained world model.
    """

    STAGE_NAMES = {
        0: "Normal",
        1: "Reconnaissance",
        2: "Initial Access",
        3: "Infiltration",
        4: "Lateral Movement",
        5: "Exfiltration",
    }

    def __init__(
        self,
        model: NetworkWorldModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        fc = config.get("forecasting", {})
        self.default_k: int = fc.get("default_k_steps", 5)
        self.confidence_decay: float = fc.get("confidence_decay", 0.95)

    # ------------------------------------------------------------------
    # Core recursive simulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(
        self,
        historical_states: np.ndarray,
        k_steps: Optional[int] = None,
    ) -> List[ForecastResult]:
        """
        Recursively predict k future network states.

        Args:
            historical_states: (seq_len, num_features) — the most recent
                sequence of observed state vectors.
            k_steps: Number of future steps to simulate (default from config).

        Returns:
            List of ForecastResult objects, one per future step.

        The method works as follows:
          - Start with the observed sequence as context.
          - At each step, run the model to get S_{t+1}, infiltration
            probability, and attack stage.
          - Slide the context window forward: drop the oldest state and
            append the predicted S_{t+1}.
          - Repeat K times.
        """
        if k_steps is None:
            k_steps = self.default_k

        seq_len = historical_states.shape[0]
        if seq_len < 2:
            raise ValueError(
                f"Need at least 2 historical states for forecasting, got {seq_len}."
            )

        # Convert to tensor: (1, seq_len, D)
        context = torch.tensor(
            historical_states, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        results: List[ForecastResult] = []
        confidence = 1.0

        for step in range(1, k_steps + 1):
            next_state, infiltration, stage_logits, _ = self.model(context)

            # Extract predictions
            pred_state = next_state.squeeze(0).cpu().numpy()          # (D,)
            infil_prob = float(infiltration.squeeze().cpu().item())
            stage_probs = torch.softmax(stage_logits, dim=-1).squeeze(0).cpu().numpy()
            # Use:
            # predicted_stage = class_with_highest_probability
            # confidence = highest_class_probability
            # To ensure logical consistency: if infiltration probability is high (> 0.5) OR the argmax class is an attack,
            # we choose the attack class (1 to 5) with the highest probability.
            if infil_prob > 0.5 or np.argmax(stage_probs) > 0:
                pred_stage = int(np.argmax(stage_probs[1:]) + 1)
            else:
                pred_stage = 0

            confidence = float(stage_probs[pred_stage])

            results.append(ForecastResult(
                step=step,
                predicted_state=pred_state,
                infiltration_prob=infil_prob,
                attack_stage=pred_stage,
                stage_probs=stage_probs,
                confidence=confidence,
            ))

            # ── Slide context window ────────────────────────────────
            # Drop oldest state, append predicted state
            pred_tensor = torch.tensor(
                pred_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

            context = torch.cat([context[:, 1:, :], pred_tensor], dim=1)

        logger.info("Forecast complete: %d steps simulated.", k_steps)
        return results

    # ------------------------------------------------------------------
    # Single-step prediction (used during evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_single(
        self, sequence: np.ndarray
    ) -> Tuple[np.ndarray, float, int, np.ndarray]:
        """
        Predict the next state from a single sequence.

        Args:
            sequence: (seq_len, D) observed states.

        Returns:
            (predicted_state, infiltration_prob, predicted_stage, stage_probs)
        """
        x = torch.tensor(
            sequence, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        next_state, infiltration, stage_logits, _ = self.model(x)

        pred_state = next_state.squeeze(0).cpu().numpy()
        infil_prob = float(infiltration.squeeze().cpu().item())
        stage_probs = torch.softmax(stage_logits, dim=-1).squeeze(0).cpu().numpy()
        
        if infil_prob > 0.5 or np.argmax(stage_probs) > 0:
            pred_stage = int(np.argmax(stage_probs[1:]) + 1)
        else:
            pred_stage = 0

        return pred_state, infil_prob, pred_stage, stage_probs

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def format_forecast(self, results: List[ForecastResult]) -> str:
        """Return a human-readable forecast summary."""
        lines = ["=" * 55, "  K-STEP FORWARD SIMULATION RESULTS", "=" * 55]
        for r in results:
            stage_name = self.STAGE_NAMES.get(r.attack_stage, "Unknown")
            lines.append(
                f"  T+{r.step:>2}  │  "
                f"Infiltration: {r.infiltration_prob:5.1%}  │  "
                f"Stage: {stage_name:<22}  │  "
                f"Conf: {r.confidence:.0%}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)

    def get_risk_level(self, infiltration_prob: float) -> str:
        """Map infiltration probability to a risk label based on SIH thresholds."""
        p_pct = infiltration_prob * 100.0
        if p_pct <= 20.0:
            return "LOW"
        elif p_pct <= 50.0:
            return "MODERATE"
        elif p_pct <= 75.0:
            return "HIGH"
        else:
            return "CRITICAL"
