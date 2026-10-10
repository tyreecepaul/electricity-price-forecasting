"""Fit the graph encoder on an early training window and embed every interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from electricity_price_forecasting_model.layers.gnn_layers.encoder import (
    SpatioTemporalGraphEncoder,
)
from pipeline.neural import set_random_seed
from pipeline.preprocessing.adapters import LearnedFeatures, add_model_target
from pipeline.preprocessing.scaling import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)
from pipeline.sequences import ContextSequenceDataset, SequenceDataset
from pipeline.training import TorchEarlyStoppingTrainer

from .config import GraphEmbeddingConfig
from .graph import adjacency
from .panel import GraphPanel, build_graph_panel, resolve_node_features


class GraphEmbedder:
    """Supervised encoder whose per-region hidden state becomes model features.

    The encoder learns to forecast every region's target from a window of all
    regions' features. Only labels that resolve before ``fit_end`` are used,
    and the weights are frozen afterwards, so embeddings for later intervals
    are computed by a model that never saw their outcomes.
    """

    def __init__(self, config: GraphEmbeddingConfig, *, interval_minutes: int = 5):
        self.config = config
        self.interval_minutes = interval_minutes
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = None
        self.regions: tuple[str, ...] = ()
        self.node_features: tuple[str, ...] = ()
        self.feature_medians: list[pd.Series] = []
        self.preprocessors: list[NumericPreprocessor] = []
        self.target_transformers: list[TargetTransformer] = []
        self.fit_end = None
        self.diagnostics: dict = {}

    def _scale(self, x: np.ndarray) -> torch.Tensor:
        """Impute and scale each node with its own fitted statistics."""
        scaled = np.empty_like(x, dtype=np.float32)
        for node, (medians, preprocessor) in enumerate(
            zip(self.feature_medians, self.preprocessors)
        ):
            frame = pd.DataFrame(x[:, node, :], columns=list(self.node_features))
            scaled[:, node, :] = preprocessor.transform(frame.fillna(medians)).to_numpy(
                dtype=np.float32
            )
        return torch.from_numpy(scaled)

    def _forward(self, history: torch.Tensor, inputs: torch.Tensor):
        """Return (embedding, transformed prediction) for every row of ``inputs``."""
        loader = DataLoader(
            ContextSequenceDataset(history, inputs, self.config.sequence_length),
            batch_size=self.config.prediction_batch_size,
            shuffle=False,
        )
        self.network.eval()
        embeddings, predictions = [], []
        with torch.inference_mode():
            for (xb,) in loader:
                embedding, prediction = self.network(xb.to(self.device))
                embeddings.append(embedding.cpu().numpy())
                predictions.append(prediction.cpu().numpy())
        return np.concatenate(embeddings), np.concatenate(predictions)

    def fit(self, panel: GraphPanel, horizon_minutes: int, train_end=None):
        """Fit on the first ``fit_fraction`` of the intervals before ``train_end``.

        The final ``inner_validation_fraction`` of that window selects the
        checkpoint. Labels within one horizon of either boundary are masked.
        """
        config = self.config
        length = config.sequence_length
        times = panel.times
        horizon = pd.Timedelta(minutes=horizon_minutes)
        available = len(times) if train_end is None else int((times < train_end).sum())
        fit_index = int(available * config.fit_fraction)
        if fit_index <= length:
            raise ValueError("Too few training intervals to fit the graph encoder")
        fit_end = times[fit_index]

        resolved = (times + horizon < fit_end) & np.isfinite(panel.y).any(axis=1)
        eligible = np.flatnonzero(resolved)
        if len(eligible) < 2:
            raise ValueError("Graph encoder fit window contains too few targets")
        inner_start = times[
            eligible[int(len(eligible) * (1 - config.inner_validation_fraction))]
        ]
        train_rows = resolved & (times + horizon < inner_start)
        validation_rows = np.flatnonzero(resolved & (times >= inner_start))
        if not train_rows.any() or len(validation_rows) == 0:
            raise ValueError("Graph encoder split produced an empty subset")

        self.regions, self.node_features = panel.regions, panel.node_features
        groups = suggested_scaling_groups(list(panel.node_features))
        self.feature_medians, self.preprocessors, self.target_transformers = [], [], []
        targets = np.full(panel.y.shape, np.nan, dtype=np.float32)
        for node, region in enumerate(panel.regions):
            fitting = pd.DataFrame(
                panel.x[train_rows, node, :], columns=list(panel.node_features)
            )
            medians = fitting.median().fillna(0.0)
            self.feature_medians.append(medians)
            self.preprocessors.append(
                NumericPreprocessor(
                    asinh_columns=tuple(groups["asinh_then_standardise"]),
                    standardise_columns=tuple(groups["standardise_only"]),
                ).fit(fitting.fillna(medians))
            )
            labels = panel.y[train_rows, node]
            if not np.isfinite(labels).any():
                raise ValueError(f"Region {region} has no targets in the encoder fit window")
            transformer = TargetTransformer().fit(labels)
            self.target_transformers.append(transformer)
            targets[:, node] = transformer.transform(panel.y[:, node])

        inputs = self._scale(panel.x)
        target_tensor = torch.from_numpy(targets)
        dataset = SequenceDataset(inputs, target_tensor, length, target_mask=train_rows)
        validation_start, validation_stop = validation_rows[0], validation_rows[-1] + 1
        validation_history = inputs[:validation_start]
        validation_inputs = inputs[validation_start:validation_stop]
        validation_targets = targets[validation_start:validation_stop]
        # Rows between the two boundaries are embargoed, not scored.
        scored = np.zeros(len(validation_inputs), dtype=bool)
        scored[validation_rows - validation_start] = True
        scored = scored[:, None] & np.isfinite(validation_targets)

        set_random_seed(config.random_seed)
        self.network = SpatioTemporalGraphEncoder(
            len(panel.node_features),
            torch.tensor(adjacency(panel.regions, config.adjacency_mode)),
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.random_seed),
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.HuberLoss(delta=config.huber_delta)

        def train_epoch() -> float:
            self.network.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                mask = torch.isfinite(yb)
                optimizer.zero_grad(set_to_none=True)
                _, prediction = self.network(xb)
                loss = criterion(prediction[mask], yb[mask])
                loss.backward()
                optimizer.step()
                total += loss.item() * int(mask.sum())
                count += int(mask.sum())
            return total / count

        def validation_score() -> float:
            _, prediction = self._forward(validation_history, validation_inputs)
            return float(np.mean(np.abs(prediction[scored] - validation_targets[scored])))

        result = TorchEarlyStoppingTrainer(
            max_epochs=config.max_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
        ).fit(
            network=self.network,
            train_epoch=train_epoch,
            validation_score=validation_score,
        )

        _, prediction = self._forward(validation_history, validation_inputs)
        actual = panel.y[validation_start:validation_stop]
        validation_mae, zero_forecast_mae = {}, {}
        for node, region in enumerate(panel.regions):
            rows = scored[:, node]
            forecast = self.target_transformers[node].inverse_transform(prediction[rows, node])
            validation_mae[region] = float(np.mean(np.abs(actual[rows, node] - forecast)))
            zero_forecast_mae[region] = float(np.mean(np.abs(actual[rows, node])))

        self.fit_end = fit_end
        self.diagnostics = {
            "adjacency_mode": config.adjacency_mode,
            "node_features": list(panel.node_features),
            "fit_end": str(fit_end),
            "inner_validation_start": str(inner_start),
            "training_windows": len(dataset),
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "training_loss_history": result.training_loss_history,
            "validation_score_history": result.validation_score_history,
            # Encoder head on its inner validation slice, in target units. A
            # zero forecast equals persistence when the target is ``change``.
            "validation_mae": validation_mae,
            "zero_forecast_mae": zero_forecast_mae,
        }
        return self

    def transform(self, panel: GraphPanel) -> pd.DataFrame:
        """Return ``gnn_emb_*`` columns per region and interval, from past rows only.

        The first ``sequence_length - 1`` intervals lack context and are NaN.
        """
        if self.network is None:
            raise RuntimeError("fit() must be called before transform()")
        if panel.regions != self.regions or panel.node_features != self.node_features:
            raise ValueError("Panel regions and features must match the fitted encoder")
        columns = self.config.embedding_columns
        count = self.config.sequence_length - 1
        embedding = np.full(
            (len(panel.times), len(panel.regions), len(columns)), np.nan, dtype=np.float32
        )
        if len(panel.times) > count:
            inputs = self._scale(panel.x)
            embedding[count:], _ = self._forward(inputs[:count], inputs[count:])
        frames = []
        for node, region in enumerate(panel.regions):
            frame = pd.DataFrame(embedding[:, node, :], columns=columns)
            frame.insert(0, "settlement_date", panel.times)
            frame.insert(0, "region", region)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def compute_graph_embeddings(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    config: GraphEmbeddingConfig,
    horizon: int,
    target: str,
    interval_minutes: int = 5,
) -> LearnedFeatures:
    """Fit the encoder on early training data and embed both splits.

    Both splits share one time axis so the first test windows continue from
    the training tail. Fitting still sees training-period labels only.
    """
    train = add_model_target(train_features, target, horizon, split="training")
    test = add_model_target(test_features, target, horizon, split="test")
    node_features = resolve_node_features(train, config, horizon)
    absent = [name for name in node_features if name not in test.columns]
    if absent:
        raise ValueError(f"Graph node features are missing from test data: {absent}")
    keep = ["region", "settlement_date", *node_features, "target"]
    combined = pd.concat([train[keep], test[keep]], ignore_index=True)
    panel = build_graph_panel(
        combined, config.regions, node_features, interval_minutes=interval_minutes
    )
    train_end = pd.to_datetime(test["settlement_date"]).min()

    embedder = GraphEmbedder(config, interval_minutes=interval_minutes)
    embedder.fit(panel, horizon, train_end=train_end)
    embedded = embedder.transform(panel)
    is_test = embedded["settlement_date"] >= train_end
    return LearnedFeatures(
        train=embedded.loc[~is_test].reset_index(drop=True),
        test=embedded.loc[is_test].reset_index(drop=True),
        columns=config.embedding_columns,
        fit_end=embedder.fit_end,
        diagnostics=embedder.diagnostics,
    )
