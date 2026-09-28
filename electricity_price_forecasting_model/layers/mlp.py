"""PyTorch MLP layer used by the corresponding forecasting wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int],
        dropout: float | Sequence[float],
    ) -> None:
        super().__init__()

        hidden_sizes = tuple(hidden_sizes)
        if isinstance(dropout, (int, float)):
            dropout_rates = (float(dropout),) * len(hidden_sizes)
        else:
            dropout_rates = tuple(float(rate) for rate in dropout)
        if len(dropout_rates) != len(hidden_sizes):
            raise ValueError("dropout must provide one rate per hidden layer")

        layers: list[nn.Module] = []
        input_size = n_features
        for position, (hidden_size, dropout_rate) in enumerate(
            zip(hidden_sizes, dropout_rates)
        ):
            layers.append(nn.Linear(input_size, hidden_size))
            # The research model normalises its first two hidden layers but
            # leaves the final 32-unit representation unnormalised.
            if position < len(hidden_sizes) - 1:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
