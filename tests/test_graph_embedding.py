"""Graph-embedding stage: topology, alignment, leakage safety and integration."""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.layers.gnn_layers.encoder import (
    SpatioTemporalGraphEncoder,
)
from electricity_price_forecasting_model.layers.gnn_layers.graph_attention import (
    GraphAttentionLayer,
)
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    ExperimentData,
    check_embedding_precedes_validation,
    chronological_validation_split,
)
from pipeline.graph_embedding import (
    DEFAULT_REGIONS,
    NEM_INTERCONNECTORS,
    GraphEmbeddingConfig,
    adjacency,
    edge_pairs,
)
from pipeline.graph_embedding.embedder import GraphEmbedder, compute_graph_embeddings
from pipeline.graph_embedding.panel import build_graph_panel
from pipeline.model_data import final_training_data, predict_test, selection_data
from pipeline.preprocessing import FeatureConfig, prepare_model_data
from pipeline.preprocessing.adapters import add_model_target, build_split_features

HORIZON = 10
FEATURES = ["rrp", "totaldemand"]
FEATURE_CONFIG = FeatureConfig(price_lags=(1,), demand_lags=(1,), rolling_windows=(2,))
EMBEDDING = GraphEmbeddingConfig(
    node_features=("rrp", "totaldemand"),
    sequence_length=3,
    embedding_dim=4,
    hidden_dim=8,
    num_heads=2,
    num_layers=1,
    dropout=0.0,
    batch_size=16,
    max_epochs=2,
    device="cpu",
)


def raw_frame(intervals: int = 200, interconnectors: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=intervals, freq="5min")
    links = {}
    if interconnectors:
        link_rng = np.random.default_rng(1)
        for prefix in NEM_INTERCONNECTORS:
            links[f"{prefix}_mwflow"] = link_rng.normal(0, 200, intervals)
            links[f"{prefix}_importlimit"] = -500.0
            links[f"{prefix}_exportlimit"] = 600.0
    frames = []
    for offset, region in enumerate(DEFAULT_REGIONS):
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "settlement_date": times,
                    "rrp": 80 + 10 * offset + rng.normal(0, 5, intervals).cumsum(),
                    "totaldemand": 1000 + 100 * offset + rng.normal(0, 20, intervals),
                    "availablegeneration": 2000.0,
                    **links,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def split(raw: pd.DataFrame, train_intervals: int = 160):
    cutoff = raw["settlement_date"].sort_values().unique()[train_intervals]
    return (
        SimpleNamespace(df=raw[raw["settlement_date"] < cutoff]),
        SimpleNamespace(df=raw[raw["settlement_date"] >= cutoff]),
    )


def split_features(raw: pd.DataFrame | None = None):
    train_data, test_data = split(raw_frame() if raw is None else raw)
    train, test, _ = build_split_features(
        horizon=HORIZON,
        target="change",
        config=FEATURE_CONFIG,
        train_data=train_data,
        test_data=test_data,
    )
    return train, test


def embed(train, test, config=EMBEDDING):
    return compute_graph_embeddings(
        train, test, config=config, horizon=HORIZON, target="change"
    )


def prepared_data(stage=None, region="VIC1"):
    train_data, test_data = split(raw_frame())
    return prepare_model_data(
        region=region,
        horizon=HORIZON,
        features=FEATURES,
        config=FEATURE_CONFIG,
        train_data=train_data,
        test_data=test_data,
        learned_features=stage,
    )


def test_adjacency_matches_nem_topology():
    matrix = adjacency()
    index = {region: position for position, region in enumerate(DEFAULT_REGIONS)}
    edges = {("NSW1", "QLD1"), ("NSW1", "VIC1"), ("VIC1", "SA1"), ("VIC1", "TAS1")}
    assert (matrix == matrix.T).all() and (np.diag(matrix) == 1).all()
    for first in DEFAULT_REGIONS:
        for second in DEFAULT_REGIONS:
            linked = (first, second) in edges or (second, first) in edges
            assert matrix[index[first], index[second]] == (first == second or linked)

    reordered = adjacency(("TAS1", "VIC1", "QLD1"))
    np.testing.assert_array_equal(reordered, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(adjacency(mode="identity"), np.eye(5))


def test_panel_keeps_missing_intervals_and_matches_single_region_targets():
    train, _ = split_features()
    train = add_model_target(train, "change", HORIZON)
    missing_time = train["settlement_date"].sort_values().unique()[20]
    dropped = train[
        ~((train["region"] == "QLD1") & (train["settlement_date"] == missing_time))
    ]
    panel = build_graph_panel(dropped, DEFAULT_REGIONS, FEATURES)

    assert panel.x.shape == (160, 5, 2)
    assert panel.y.shape == (160, 5)
    assert np.isnan(panel.x[20, 1]).all()
    assert np.isfinite(panel.x[20, 0]).all() and np.isfinite(panel.x[21, 1]).all()

    single = prepared_data().train_timeline
    np.testing.assert_allclose(
        panel.y[:, DEFAULT_REGIONS.index("VIC1")],
        single["target"].to_numpy(),
        rtol=1e-5,
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="missing"):
        build_graph_panel(train[train["region"] != "SA1"], DEFAULT_REGIONS, FEATURES)


def test_attention_reaches_neighbours_only():
    torch.manual_seed(0)
    layer = GraphAttentionLayer(8, torch.tensor(adjacency()), num_heads=2).eval()
    x = torch.randn(2, 5, 8)
    sa, qld, vic = (DEFAULT_REGIONS.index(name) for name in ("SA1", "QLD1", "VIC1"))
    base = layer(x)

    far = x.clone()
    far[:, qld] += 5.0
    torch.testing.assert_close(layer(far)[:, sa], base[:, sa])

    near = x.clone()
    near[:, vic] += 5.0
    assert not torch.allclose(layer(near)[:, sa], base[:, sa])


def test_encoder_returns_embedding_and_prediction_per_node():
    encoder = SpatioTemporalGraphEncoder(
        3, torch.tensor(adjacency()), hidden_dim=8, embedding_dim=4, num_heads=2
    )
    embedding, prediction, spread = encoder(torch.zeros(6, 7, 5, 3))
    assert embedding.shape == (6, 5, 4)
    assert prediction.shape == (6, 5)
    assert spread.shape == (6, 0)


def test_embeddings_cover_every_interval_and_are_deterministic():
    train, test = split_features()
    learned = embed(train, test)

    assert learned.columns == [f"gnn_emb_{i:02d}" for i in range(4)]
    assert len(learned.train) == 160 * 5 and len(learned.test) == 40 * 5
    assert not learned.train.duplicated(["region", "settlement_date"]).any()
    first = learned.train.groupby("region").head(2)
    assert first[learned.columns].isna().all().all()
    assert learned.train.groupby("region").tail(158)[learned.columns].notna().all().all()
    # The first test intervals continue from the training tail.
    assert learned.test[learned.columns].notna().all().all()
    assert learned.fit_end < test["settlement_date"].min()
    assert set(learned.diagnostics["validation_mae"]) == set(DEFAULT_REGIONS)

    again = embed(train, test)
    pd.testing.assert_frame_equal(learned.train, again.train)
    pd.testing.assert_frame_equal(learned.test, again.test)


def test_identity_graph_changes_the_embedding():
    train, test = split_features()
    graph = embed(train, test)
    control = embed(train, test, replace(EMBEDDING, adjacency_mode="identity"))
    assert not np.allclose(
        graph.test[graph.columns].to_numpy(), control.test[control.columns].to_numpy()
    )


def test_embedding_ignores_later_features():
    train, _ = split_features()
    train = add_model_target(train, "change", HORIZON)
    panel = build_graph_panel(train, DEFAULT_REGIONS, FEATURES)
    embedder = GraphEmbedder(EMBEDDING).fit(panel, HORIZON)
    original = embedder.transform(panel)

    changed = replace(panel, x=panel.x.copy())
    changed.x[120:] += 1000.0
    altered = embedder.transform(changed)

    early = original["settlement_date"] < panel.times[120]
    pd.testing.assert_frame_equal(original[early], altered[early])
    assert not np.allclose(
        original.loc[~early, embedder.config.embedding_columns],
        altered.loc[~early, embedder.config.embedding_columns],
    )


def test_encoder_never_learns_from_labels_after_its_fit_window():
    train, test = split_features()
    learned = embed(train, test)

    raw = raw_frame()
    # Prices at or after fit_end are the only labels the encoder must not use.
    # Changing them also changes later features, so compare the fitted weights
    # through embeddings of intervals that precede the change.
    late = raw["settlement_date"] >= learned.fit_end
    raw.loc[late, "rrp"] = raw.loc[late, "rrp"] * 3 + 500
    changed_train, changed_test = split_features(raw)
    changed = embed(changed_train, changed_test)

    assert changed.fit_end == learned.fit_end
    before = learned.train["settlement_date"] < learned.fit_end
    pd.testing.assert_frame_equal(learned.train[before], changed.train[before])
    assert not np.allclose(
        learned.test[learned.columns].to_numpy(),
        changed.test[changed.columns].to_numpy(),
    )


def test_stage_adds_features_without_changing_scored_rows():
    plain = prepared_data()
    staged = prepared_data(lambda train, test: embed(train, test))

    assert staged.features == [*FEATURES, *staged.embedding_columns]
    assert staged.embedding_columns == EMBEDDING.embedding_columns
    for name in ("train_df", "test_df", "train_timeline", "test_timeline"):
        before, after = getattr(plain, name), getattr(staged, name)
        pd.testing.assert_frame_equal(before, after[before.columns])
    assert list(staged.X_test.columns) == staged.features
    assert staged.X_test[staged.embedding_columns].notna().all().all()

    config = ExperimentConfig(
        region="VIC1",
        horizon_minutes=HORIZON,
        validation_fraction=0.2,
        model_features=FEATURES,
        feature_config=FEATURE_CONFIG,
        graph_embedding=EMBEDDING,
    )
    fit, validation = chronological_validation_split(staged.train_df, config)
    experiment = ExperimentData(config, staged, fit, validation, staged.test_df)
    check_embedding_precedes_validation(experiment)

    model = GRUModel(
        sequence_length=3, hidden_size=4, max_epochs=1, batch_size=8, device="cpu"
    )
    data = selection_data(model, experiment)
    assert list(data.x_train.columns) == staged.features
    model.train(*final_training_data(model, staged))
    predictions = predict_test(model, staged)
    assert len(predictions) == len(staged.test_df) and np.isfinite(predictions).all()


def test_validation_overlapping_the_encoder_fit_is_rejected():
    staged = prepared_data(lambda train, test: embed(train, test))
    config = ExperimentConfig(
        region="VIC1",
        horizon_minutes=HORIZON,
        validation_fraction=0.2,
        model_features=FEATURES,
        feature_config=FEATURE_CONFIG,
        graph_embedding=EMBEDDING,
        walk_forward_folds=3,
    )
    fit, validation = chronological_validation_split(staged.train_df, config)
    experiment = ExperimentData(config, staged, fit, validation, staged.test_df)
    with pytest.raises(ValueError, match="fit_fraction"):
        check_embedding_precedes_validation(experiment)

    two_folds = replace(experiment, config=replace(config, walk_forward_folds=2))
    # This cutoff still includes outcomes after the first validation origin.
    with pytest.raises(ValueError, match="fit_fraction"):
        check_embedding_precedes_validation(two_folds)


def test_cache_identity_changes_only_when_the_stage_is_enabled(monkeypatch):
    import pipeline.experiment_pipeline as module

    monkeypatch.setattr(module, "source_fingerprint", lambda: "source")
    plain = ExperimentConfig(region="VIC1")
    staged = replace(plain, graph_embedding=EMBEDDING)
    control = replace(
        plain, graph_embedding=replace(EMBEDDING, adjacency_mode="identity")
    )

    assert "gnn" not in plain.cache_file.name
    assert "graph_embedding" not in plain.cache_metadata()
    assert len({plain.cache_file, staged.cache_file, control.cache_file}) == 3
    assert staged.graph_embedding_cache_file != control.graph_embedding_cache_file
    # One encoder fit serves every region's experiment.
    other_region = replace(staged, region="NSW1")
    assert staged.cache_file != other_region.cache_file
    assert staged.graph_embedding_cache_file == other_region.graph_embedding_cache_file
    assert (
        staged.graph_embedding_cache_metadata()
        == other_region.graph_embedding_cache_metadata()
    )


def linked_panel():
    train, _ = split_features(raw_frame(interconnectors=True))
    train = add_model_target(train, "change", HORIZON)
    return train, build_graph_panel(train, DEFAULT_REGIONS, FEATURES)


def test_edge_tensor_follows_the_interconnectors():
    train, panel = linked_panel()
    index = {region: position for position, region in enumerate(DEFAULT_REGIONS)}
    edges = panel.edges
    flow, utilisation, headroom = (edges[..., i] for i in range(3))

    assert edges.shape == (160, 5, 5, 3)
    np.testing.assert_allclose(flow, -flow.transpose(0, 2, 1))
    np.testing.assert_allclose(utilisation, utilisation.transpose(0, 2, 1))
    np.testing.assert_allclose(headroom, headroom.transpose(0, 2, 1))
    assert (edges[:, adjacency() == 0] == 0).all()
    assert (edges[:, np.eye(5, dtype=bool)] == 0).all()

    # QNI and Terranora are one NSW1-QLD1 edge.
    rows = train[train["region"] == "NSW1"].sort_values("settlement_date")
    parallel = ["ic_nsw1_qld1", "ic_n_q_mnsp1"]
    np.testing.assert_allclose(
        flow[:, index["NSW1"], index["QLD1"]],
        rows[[f"{p}_mwflow" for p in parallel]].sum(axis=1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        utilisation[:, index["QLD1"], index["NSW1"]],
        rows[[f"{p}_utilisation" for p in parallel]].max(axis=1),
        rtol=1e-5,
    )
    assert len(edge_pairs()) == 4

    _, unlinked = split_features()
    assert build_graph_panel(
        add_model_target(unlinked, "change", HORIZON), DEFAULT_REGIONS, FEATURES
    ).edges is None


def test_edge_attributes_shift_attention_between_linked_regions_only():
    torch.manual_seed(0)
    layer = GraphAttentionLayer(
        8, torch.tensor(adjacency()), num_heads=2, edge_size=3
    ).eval()
    x, edge_attr = torch.randn(2, 5, 8), torch.zeros(2, 5, 5, 3)
    sa, qld, vic = (DEFAULT_REGIONS.index(name) for name in ("SA1", "QLD1", "VIC1"))
    base = layer(x, edge_attr)

    unlinked = edge_attr.clone()
    unlinked[:, sa, qld] = 5.0
    torch.testing.assert_close(layer(x, unlinked), base)

    linked = edge_attr.clone()
    linked[:, sa, vic] = 5.0
    changed = layer(x, linked)
    assert not torch.allclose(changed[:, sa], base[:, sa])
    torch.testing.assert_close(changed[:, qld], base[:, qld])


def test_encoder_unpacks_edges_and_forecasts_spreads():
    pairs = [(a, b) for a, b, _ in edge_pairs()]
    encoder = SpatioTemporalGraphEncoder(
        3,
        torch.tensor(adjacency()),
        hidden_dim=8,
        embedding_dim=4,
        num_heads=2,
        edge_size=3,
        spread_pairs=pairs,
    ).eval()
    x = torch.randn(6, 7, 5, 3 + 5 * 3)
    embedding, prediction, spread = encoder(x)
    assert embedding.shape == (6, 5, 4)
    assert prediction.shape == (6, 5)
    assert spread.shape == (6, 4)

    # A uniform shift would cancel in the softmax, so vary it by pair.
    changed = x.clone()
    changed[..., 3:] += torch.randn(5, 15)
    assert not torch.allclose(encoder(changed)[0], embedding)


def test_spread_targets_are_future_price_differences():
    train, panel = linked_panel()
    pairs = [(a, b) for a, b, _ in edge_pairs()]
    spreads = GraphEmbedder._spreads(panel, pairs)

    wide = train.pivot(index="settlement_date", columns="region", values="target_rrp")
    for column, (a, b) in enumerate(pairs):
        expected = wide[DEFAULT_REGIONS[a]] - wide[DEFAULT_REGIONS[b]]
        np.testing.assert_allclose(
            spreads[:, column], expected.to_numpy(), rtol=1e-4, equal_nan=True
        )


def test_edges_and_spread_objective_are_recorded_and_change_the_embedding():
    train, test = split_features(raw_frame(interconnectors=True))
    full = embed(train, test)
    plain = embed(
        train, test, replace(EMBEDDING, edge_features=False, spread_loss_weight=0.0)
    )
    control = embed(train, test, replace(EMBEDDING, adjacency_mode="identity"))

    assert full.diagnostics["edge_features"] == ["flow", "utilisation", "headroom"]
    assert set(full.diagnostics["spread_validation_mae"]) == {
        "NSW1-QLD1",
        "VIC1-NSW1",
        "VIC1-SA1",
        "TAS1-VIC1",
    }
    assert plain.diagnostics["edge_features"] == []
    assert plain.diagnostics["spread_validation_mae"] == {}
    # The control keeps the spread objective but has no edges to describe.
    assert control.diagnostics["edge_features"] == []
    assert control.diagnostics["spread_loss_weight"] == 0.5
    assert not np.allclose(
        full.test[full.columns].to_numpy(), plain.test[plain.columns].to_numpy()
    )

    without_links = embed(*split_features())
    assert without_links.diagnostics["edge_features"] == []


def test_flows_without_limits_use_flow_alone_without_warnings():
    raw = raw_frame(interconnectors=True)
    raw = raw.drop(columns=[c for c in raw.columns if c.endswith("limit")])
    train, test = split_features(raw)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        learned = embed(train, test)

    assert learned.diagnostics["edge_features"] == ["flow"]
    assert learned.test[learned.columns].notna().all().all()


def test_embedding_ignores_later_edge_attributes():
    _, panel = linked_panel()
    embedder = GraphEmbedder(EMBEDDING).fit(panel, HORIZON)
    original = embedder.transform(panel)

    changed = replace(panel, edges=panel.edges.copy())
    changed.edges[120:] *= 50.0
    altered = embedder.transform(changed)

    early = original["settlement_date"] < panel.times[120]
    columns = embedder.config.embedding_columns
    pd.testing.assert_frame_equal(original[early], altered[early])
    assert not np.allclose(original.loc[~early, columns], altered.loc[~early, columns])
