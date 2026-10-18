"""Late-fusion TCN-Transformer forecaster using the shared PyTorch training pipeline"""
from __future__ import annotations

from torch import nn

from electricity_price_forecasting_model.layers.late_fusion import LateFusionNetwork
from pipeline.neural import TorchForecastModel

SPATIAL_PREFIX: str = "gnn_emb_"


class LateFusionModel(TorchForecastModel):
    """Fuse a TCN-Transformer over temporal features with graph embeddings

    Without ``gnn_emb_*`` columns the spatial branch is omitted, leaving a
    plain TCN-Transformer for comparison.
    """

    model_name: str = "LateFusion"
    requires_sequence: bool = True

    def __init__(
        self,
        sequence_length: int = 24,
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
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        prediction_batch_size: int = 1024,
        max_epochs: int = 20,
        patience: int = 3,
        min_delta: float = 0.001,
        loss: str = "huber",
        huber_delta: float = 1.0,
        random_seed: int = 42,
        device: str | None = None,
        selection_metric: str = "mae",
        gradient_clip_norm: float | None = 1.0,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float = 0.5,
        mixed_precision: bool = False,
        interval_minutes: int = 5,
        show_progress: bool = False
    ) -> None:
        """Validate architecture settings and configure training options"""
        architecture_sizes: tuple[tuple[str, int], ...] = (
            ("sequence_length", sequence_length),
            ("channels", channels),
            ("kernel_size", kernel_size),
            ("num_blocks", num_blocks),
            ("d_model", d_model),
            ("nhead", nhead),
            ("num_decoder_layers", num_decoder_layers),
            ("dim_feedforward", dim_feedforward),
            ("n_queries", n_queries),
            ("spatial_hidden", spatial_hidden),
            ("head_hidden", head_hidden)
        )
        for name, value in architecture_sizes:
            if value <= 0 or int(value) != value:
                raise ValueError(f"{name} must be a positive integer")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.sequence_length: int = int(sequence_length)
        self.channels: int = int(channels)
        self.kernel_size: int = int(kernel_size)
        self.num_blocks: int = int(num_blocks)
        self.d_model: int = int(d_model)
        self.nhead: int = int(nhead)
        self.num_decoder_layers: int = int(num_decoder_layers)
        self.dim_feedforward: int = int(dim_feedforward)
        self.n_queries: int = int(n_queries)
        self.spatial_hidden: int = int(spatial_hidden)
        self.head_hidden: int = int(head_hidden)
        self.dropout: float = float(dropout)
        self.show_progress: bool = show_progress

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            prediction_batch_size=prediction_batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            loss=loss,
            huber_delta=huber_delta,
            random_seed=random_seed,
            device=device,
            selection_metric=selection_metric,
            gradient_clip_norm=gradient_clip_norm,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            mixed_precision=mixed_precision,
            interval_minutes=interval_minutes
        )

    def _build_network(self, input_size: int) -> nn.Module:
        """Split the fitted feature columns by name and create an untrained network"""
        # _train() records feature_columns before building; fall back to all
        # temporal when called directly without column names.
        columns: list[str] = (
            self.feature_columns if len(self.feature_columns) == input_size else []
        )
        spatial: list[int] = [
            index for index, column in enumerate(columns)
            if column.startswith(SPATIAL_PREFIX)
        ]
        spatial_set: set[int] = set(spatial)
        temporal: list[int] = [index for index in range(input_size) if index not in spatial_set]
        return LateFusionNetwork(
            temporal,
            spatial,
            self.sequence_length,
            channels=self.channels,
            kernel_size=self.kernel_size,
            num_blocks=self.num_blocks,
            d_model=self.d_model,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            n_queries=self.n_queries,
            spatial_hidden=self.spatial_hidden,
            head_hidden=self.head_hidden,
            dropout=self.dropout
        )
