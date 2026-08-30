# =============================================================================
# State Encoder — Dense feature encoder for network state vectors
# =============================================================================
"""
A feed-forward encoder that projects raw state vectors into a compact
latent representation before temporal modelling.  This decouples the
feature dimensionality from the GRU input size and allows the model to
learn non-linear feature interactions.
"""

from typing import List

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Multi-layer dense encoder: input_dim → h1 → h2 → ... → output_dim.

    Each hidden layer uses ReLU activation followed by dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Dimensionality of the raw state vector.
            hidden_dims: List of hidden layer sizes, e.g. [128, 64].
                         The last element becomes the encoder output dim.
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim).

        Returns:
            Encoded tensor with same leading dims and output_dim last.
        """
        return self.encoder(x)
