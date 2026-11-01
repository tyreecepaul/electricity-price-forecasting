"""A minimal, hand-rolled multi-head Graph Attention layer over a fixed
5-node NEM-region graph. No external graph library dependency.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """Multi-head additive-mask graph attention, applied per-timestep.

    Input:  [batch, num_nodes, in_features]
    Output: [batch, num_nodes, out_features]

    ``adjacency`` is a [num_nodes, num_nodes] 0/1 matrix (self-loops included)
    used as an additive mask: non-edges get -inf before the softmax so a node
    only attends to itself and its directly connected regions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        adjacency: torch.Tensor,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if out_features % num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features

        self.register_buffer(
            "attention_mask",
            torch.where(
                adjacency.bool(),
                torch.zeros_like(adjacency),
                torch.full_like(adjacency, float("-inf")),
            ),
        )

        self.query_proj = nn.Linear(in_features, out_features)
        self.key_proj = nn.Linear(in_features, out_features)
        self.value_proj = nn.Linear(in_features, out_features)
        self.output_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query_proj(x))  # [batch, heads, nodes, head_dim]
        k = split_heads(self.key_proj(x))
        v = split_heads(self.value_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + self.attention_mask
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)  # [batch, heads, nodes, head_dim]
        context = context.transpose(1, 2).reshape(batch_size, num_nodes, self.out_features)
        return self.output_proj(context)
