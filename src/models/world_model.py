# =============================================================================
# World Model — GRU-based temporal network state transition model
# =============================================================================
"""
Core world model that learns the dynamics of network state transitions.

Architecture:
    Network State Sequence  →  StateEncoder  →  GRU  →  Latent h_T
                                                         │
                                          ┌──────────────┼──────────────┐
                                          ▼              ▼              ▼
                                   Next-State      Infiltration    Attack-Stage
                                   Prediction      Probability     Prediction
                                   (regression)    (sigmoid)       (softmax)

The model is trained with a multi-task loss:
    L = α · L_state  +  β · L_infiltration  +  γ · L_stage

This is NOT a standard classifier — it learns P(S_{t+1} | S_t) and
can recursively simulate future network states for K-step forecasting.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.state_encoder import StateEncoder


class NetworkWorldModel(nn.Module):
    """
    Multi-task GRU world model for network attack forecasting.

    Three prediction heads share a common GRU temporal backbone:
      1. next_state_head  — predicts the feature vector of the next time
         window (regression via MSE / SmoothL1).
      2. infiltration_head — predicts probability that the next window
         contains malicious activity (binary, sigmoid).
      3. stage_head — predicts MITRE ATT&CK stage of the next window
         (multi-class, softmax / cross-entropy).
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list = None,
        encoder_dropout: float = 0.2,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.2,
        gru_bidirectional: bool = False,
        num_attack_stages: int = 6,
    ):
        """
        Args:
            input_dim: Dimensionality of raw state vectors S_t.
            encoder_hidden_dims: Hidden sizes for the StateEncoder.
            encoder_dropout: Dropout in the encoder.
            gru_hidden_size: GRU hidden state dimensionality.
            gru_num_layers: Number of stacked GRU layers.
            gru_dropout: Dropout between GRU layers (only if num_layers > 1).
            gru_bidirectional: Use bidirectional GRU (False for forecasting).
            num_attack_stages: Number of MITRE ATT&CK stage classes.
        """
        super().__init__()

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.num_directions = 2 if gru_bidirectional else 1

        # ── Feature Encoder ──────────────────────────────────────────
        self.encoder = StateEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=encoder_dropout,
        )

        # ── GRU Temporal Model ───────────────────────────────────────
        self.gru = nn.GRU(
            input_size=self.encoder.output_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )

        latent_dim = gru_hidden_size * self.num_directions

        # ── HEAD 1: Next State Prediction (regression) ───────────────
        self.next_state_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, input_dim),
        )

        # ── HEAD 2: Infiltration Prediction (binary) ─────────────────
        self.infiltration_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),
        )

        # ── HEAD 3: Attack Stage Prediction (multi-class) ────────────
        self.stage_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, num_attack_stages),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the world model.

        Args:
            x: (batch, seq_len, input_dim) — sequence of state vectors.
            hidden: Optional initial GRU hidden state.

        Returns:
            next_state:  (batch, input_dim)   — predicted S_{t+1}
            infiltration: (batch, 1)          — infiltration probability
            stage_logits: (batch, num_stages) — attack stage logits
            h_n: (num_layers*dirs, batch, hidden) — final GRU hidden state
        """
        batch_size = x.size(0)

        # Encode each state in the sequence
        encoded = self.encoder(x)  # (batch, seq_len, encoder_output_dim)

        # Temporal modelling
        if hidden is None:
            h_0 = torch.zeros(
                self.gru_num_layers * self.num_directions,
                batch_size,
                self.gru_hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h_0 = hidden

        gru_out, h_n = self.gru(encoded, h_0)
        # gru_out: (batch, seq_len, hidden * dirs)

        # Use the final time-step output as the temporal latent state
        latent = gru_out[:, -1, :]  # (batch, hidden * dirs)

        # Prediction heads
        next_state = self.next_state_head(latent)       # (batch, input_dim)
        infiltration = self.infiltration_head(latent)    # (batch, 1)
        stage_logits = self.stage_head(latent)           # (batch, num_stages)

        return next_state, infiltration, stage_logits, h_n

    # ------------------------------------------------------------------
    # Factory from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, input_dim: int, config: Dict[str, Any]) -> "NetworkWorldModel":
        """Instantiate a model from the YAML config dictionary."""
        mc = config.get("model", {})
        return cls(
            input_dim=input_dim,
            encoder_hidden_dims=mc.get("encoder_hidden_dims", [128, 64]),
            encoder_dropout=mc.get("encoder_dropout", 0.2),
            gru_hidden_size=mc.get("gru_hidden_size", 128),
            gru_num_layers=mc.get("gru_num_layers", 2),
            gru_dropout=mc.get("gru_dropout", 0.2),
            gru_bidirectional=mc.get("gru_bidirectional", False),
            num_attack_stages=mc.get("num_attack_stages", 6),
        )
