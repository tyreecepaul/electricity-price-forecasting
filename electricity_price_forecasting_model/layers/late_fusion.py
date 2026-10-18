"""TCN encoder, cross-attention decoder and a late-fused graph-embedding branch"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from electricity_price_forecasting_model.layers.tcn import TemporalBlock


class LateFusionNetwork(nn.Module):
    """Forecast from temporal features and a region's graph embedding, fused late

    Columns are split by position lists supplied at construction, so the
    network does not depend on where the embedding columns sit in the frame.
    """

    def __init__(
        self,
        temporal_indices: Sequence[int],
        spatial_indices: Sequence[int],
        sequence_length: int,
        channels: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 3,
        d_model: int = 32,
        nhead: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        n_queries: int = 1,
        spatial_hidden: int = 16,
        head_hidden: int = 32,
        dropout: float = 0.1
    ) -> None:
        """Build both branches and the fusion head"""
        super().__init__()
        if not len(temporal_indices):
            raise ValueError("At least one temporal feature is required")
        if set(temporal_indices) & set(spatial_indices):
            raise ValueError("Temporal and spatial features must not overlap")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        # Buffers follow .to(device) and are saved with the weights.
        self.register_buffer(
            "temporal_idx", torch.as_tensor(list(temporal_indices), dtype=torch.long)
        )
        self.register_buffer(
            "spatial_idx", torch.as_tensor(list(spatial_indices), dtype=torch.long)
        )
        self.input_size: int = len(temporal_indices) + len(spatial_indices)

        # Branch A: causal TCN encoder whose every step becomes a memory token.
        self.encoder: nn.Sequential = nn.Sequential(*[
            TemporalBlock(
                len(temporal_indices) if index == 0 else channels,
                channels,
                kernel_size,
                dilation=2**index,
                dropout=dropout
            ) for index in range(num_blocks)
        ])
        self.memory_proj: nn.Linear = nn.Linear(channels, d_model)
        self.position: nn.Parameter = nn.Parameter(torch.zeros(1, sequence_length, d_model))

        # Regression has no target sequence to decode, so the query is a shared
        # learned token shifted by the latest encoder state: "given now, which
        # past steps matter". Memory never extends past the forecast origin, so
        # no attention masks are needed.
        self.query: nn.Parameter = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.latest_proj: nn.Linear = nn.Linear(d_model, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder: nn.TransformerDecoder = nn.TransformerDecoder(
            layer, num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        # Branch B: the embedding already summarises history, so only the
        # forecast origin's vector is used.
        self.spatial: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(len(spatial_indices), spatial_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(spatial_hidden, spatial_hidden)
            )
            if len(spatial_indices) else None
        )

        fused: int = d_model + (spatial_hidden if self.spatial is not None else 0)
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(fused, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def temporal_summary(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the decoder's pooled output h_t shaped (batch, d_model)"""
        temporal: torch.Tensor = inputs.index_select(-1, self.temporal_idx)
        encoded: torch.Tensor = self.encoder(temporal.transpose(1, 2)).transpose(1, 2)
        # Align positions to the final step so shorter windows keep their meaning.
        memory: torch.Tensor = self.memory_proj(encoded) + self.position[:, -inputs.shape[1]:]
        tgt: torch.Tensor = self.query + self.latest_proj(memory[:, -1:])
        return self.decoder(tgt, memory).mean(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one scalar forecast per (batch, time, features) window"""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("Expected (batch, time, input_size)")
        if inputs.shape[1] > self.position.shape[1]:
            raise ValueError("Window is longer than sequence_length")
        summary: torch.Tensor = self.temporal_summary(inputs)
        if self.spatial is None:
            return self.head(summary)
        spatial: torch.Tensor = inputs[:, -1].index_select(-1, self.spatial_idx)
        return self.head(torch.cat([summary, self.spatial(spatial)], dim=-1))
