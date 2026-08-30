# =============================================================================
# Explainer — Feature importance for world model predictions
# =============================================================================
"""
Provides prediction explanations via feature perturbation importance.
For each prediction, the explainer measures how much each input feature
contributes to the infiltration probability by perturbing features one
at a time and measuring the change in model output.

Methods available:
  - perturbation: Fast, model-agnostic, robust for sequence models.
  - integrated_gradients: Gradient-based, more precise but slower.

SHAP (TreeExplainer / KernelExplainer) is NOT well-suited for GRU
sequence models.  Feature perturbation is the recommended approach
for this architecture.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.world_model import NetworkWorldModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Explainer:
    """
    Explains world model predictions via feature importance analysis.
    """

    def __init__(
        self,
        model: NetworkWorldModel,
        feature_names: List[str],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        ec = config.get("explainability", {})
        self.method: str = ec.get("method", "perturbation")
        self.n_samples: int = ec.get("num_perturbation_samples", 100)
        self.top_k: int = ec.get("top_k_features", 10)

    # ------------------------------------------------------------------
    # Feature perturbation importance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def perturbation_importance(
        self, sequence: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Compute feature importance by zeroing each feature across the
        sequence and measuring the change in infiltration probability.

        Args:
            sequence: (seq_len, D) observed state sequence.

        Returns:
            Sorted list of (feature_name, importance_score) tuples.
            Positive scores = feature increases infiltration risk.
            Negative scores = feature decreases infiltration risk.
        """
        x = torch.tensor(
            sequence, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, seq_len, D)

        # Baseline prediction
        _, base_infil, _, _ = self.model(x)
        base_prob = float(base_infil.squeeze().cpu().item())

        importances: List[Tuple[str, float]] = []
        num_features = sequence.shape[1]

        for feat_idx in range(num_features):
            # Create perturbed input: zero out this feature across all timesteps
            x_perturbed = x.clone()
            x_perturbed[:, :, feat_idx] = 0.0

            _, perturbed_infil, _, _ = self.model(x_perturbed)
            perturbed_prob = float(perturbed_infil.squeeze().cpu().item())

            # Importance = how much the probability drops when the feature
            # is removed.  Positive = feature was increasing risk.
            importance = base_prob - perturbed_prob

            feat_name = (
                self.feature_names[feat_idx]
                if feat_idx < len(self.feature_names)
                else f"feature_{feat_idx}"
            )
            importances.append((feat_name, importance))

        # Sort by absolute importance (descending)
        importances.sort(key=lambda t: abs(t[1]), reverse=True)
        return importances

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        sequence: np.ndarray,
        n_steps: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Compute feature importance using Integrated Gradients.

        Approximates the integral of gradients along a straight-line
        path from a zero baseline to the actual input.

        Args:
            sequence: (seq_len, D) observed state sequence.
            n_steps: Number of interpolation steps.

        Returns:
            Sorted list of (feature_name, importance_score) tuples.
        """
        x = torch.tensor(
            sequence, dtype=torch.float32, device=self.device
        ).unsqueeze(0).requires_grad_(True)

        baseline = torch.zeros_like(x)

        # Accumulate gradients along the path
        total_grads = torch.zeros_like(x)

        for step in range(n_steps + 1):
            alpha = step / n_steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            _, infil, _, _ = self.model(interpolated)
            infil_scalar = infil.sum()
            infil_scalar.backward()

            if interpolated.grad is not None:
                total_grads += interpolated.grad.detach()

        # Average gradients × (input - baseline)
        avg_grads = total_grads / (n_steps + 1)
        ig = (x.detach() - baseline) * avg_grads  # (1, seq_len, D)

        # Aggregate over the sequence dimension (mean)
        feature_ig = ig.squeeze(0).mean(dim=0).cpu().numpy()  # (D,)

        importances: List[Tuple[str, float]] = []
        for feat_idx in range(len(feature_ig)):
            feat_name = (
                self.feature_names[feat_idx]
                if feat_idx < len(self.feature_names)
                else f"feature_{feat_idx}"
            )
            importances.append((feat_name, float(feature_ig[feat_idx])))

        importances.sort(key=lambda t: abs(t[1]), reverse=True)
        return importances

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self, sequence: np.ndarray
    ) -> Dict[str, Any]:
        """
        Explain a prediction using the configured method.

        Args:
            sequence: (seq_len, D) observed state sequence.

        Returns:
            Dictionary with:
              - method: name of the explanation method
              - feature_importances: list of (name, score) tuples
              - top_features: top-K features
              - narrative: template-based text explanation
        """
        try:
            if self.method == "integrated_gradients":
                importances = self.integrated_gradients(sequence)
            else:
                importances = self.perturbation_importance(sequence)
        except Exception as e:
            logger.error("Explainability failed: %s", e)
            importances = [
                (name, 0.0) for name in self.feature_names[:self.top_k]
            ]

        top = importances[: self.top_k]

        # Generate template-based narrative
        narrative = self._generate_narrative(top)

        return {
            "method": self.method,
            "feature_importances": importances,
            "top_features": top,
            "narrative": narrative,
        }

    def _generate_narrative(
        self, top_features: List[Tuple[str, float]]
    ) -> str:
        """
        Generate a human-readable explanation from feature importances.
        No external LLM required.
        """
        if not top_features:
            return "Insufficient data for explanation."

        risk_drivers = [f for f in top_features if f[1] > 0]
        risk_reducers = [f for f in top_features if f[1] < 0]

        parts = []
        if risk_drivers:
            driver_names = [
                self._humanise_feature(f[0]) for f in risk_drivers[:3]
            ]
            parts.append(
                "Elevated infiltration risk is primarily associated with "
                + ", ".join(driver_names[:-1])
                + (f", and {driver_names[-1]}" if len(driver_names) > 1 else driver_names[0])
                + "."
            )

        if risk_reducers:
            reducer_names = [
                self._humanise_feature(f[0]) for f in risk_reducers[:2]
            ]
            parts.append(
                "Risk is partially mitigated by "
                + " and ".join(reducer_names)
                + "."
            )

        return " ".join(parts) if parts else "No significant risk drivers identified."

    @staticmethod
    def _humanise_feature(name: str) -> str:
        """Convert feature names like 'syn_ratio' to 'high SYN ratio'."""
        name = name.replace("_", " ").strip()
        # Add descriptors for common features
        if "syn" in name.lower() and "ratio" in name.lower():
            return "unusually high SYN activity"
        if "port" in name.lower() and "diversity" in name.lower():
            return "increased destination port diversity"
        if "iat" in name.lower() and "variance" in name.lower():
            return "abnormal packet timing"
        if "burst" in name.lower():
            return "burst traffic patterns"
        if "connection" in name.lower() and "rate" in name.lower():
            return "elevated connection attempt rate"
        if "rst" in name.lower() or "failed" in name.lower():
            return "high failed connection ratio"
        if "bytes_per" in name.lower():
            return "unusual byte transfer patterns"
        return f"anomalous {name}"
