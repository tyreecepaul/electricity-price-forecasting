"""Spatio-temporal graph encoder that produces one embedding per region."""

from __future__ import annotations

import torch
from torch import nn

from electricity_price_forecasting_model.layers.gnn_layers.graph_attention import (
    GraphAttentionLayer,
)


class SpatioTemporalGraphEncoder(nn.Module):
    """Graph attention at each timestep, then a GRU over the lookback per node.

    Input: (batch, lookback, num_nodes, input_size).
    Returns ``(embedding, prediction)`` with shapes
    (batch, num_nodes, embedding_dim) and (batch, num_nodes). The prediction
    head exists only to train the encoder; callers keep the embedding.
    """

    def __init__(
        self,
        input_size: int,
        adjacency: torch.Tensor,
        hidden_dim: int = 32,
        embedding_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        num_nodes = adjacency.shape[0]
        self.input_proj = nn.Linear(input_size, hidden_dim)
        # Weights are shared across regions; this lets a node still be told apart.
        self.node_embedding = nn.Parameter(torch.zeros(num_nodes, hidden_dim))
        self.graph_layers = nn.ModuleList(
            GraphAttentionLayer(hidden_dim, adjacency, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(hidden_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, lookback, num_nodes, _ = x.shape
        hidden = self.input_proj(x) + self.node_embedding
        hidden = hidden.reshape(batch_size * lookback, num_nodes, -1)
        for layer in self.graph_layers:
            hidden = layer(hidden)

        # (batch, lookback, nodes, hidden) -> one sequence per (batch, node).
        hidden = hidden.view(batch_size, lookback, num_nodes, -1).transpose(1, 2)
        hidden = hidden.reshape(batch_size * num_nodes, lookback, -1)
        _, last = self.gru(hidden)

        embedding = self.bottleneck(self.dropout(last[-1]))
        prediction = self.head(embedding)
        return (
            embedding.view(batch_size, num_nodes, -1),
            prediction.view(batch_size, num_nodes),
        )
