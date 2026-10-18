from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from electricity_price_forecasting_model.final_model.late_fusion import (  # noqa: E402
    LateFusionModel,
)
from electricity_price_forecasting_model.layers.late_fusion import (  # noqa: E402
    LateFusionNetwork,
)
from pipeline.neural import TorchForecastModel  # noqa: E402


def small_network(temporal=range(83), spatial=range(83, 91), **settings):
    torch.manual_seed(0)
    options = dict(
        sequence_length=12, channels=8, num_blocks=2, d_model=8, nhead=2,
        dim_feedforward=16, spatial_hidden=4, head_hidden=8, dropout=0.0,
    )
    options.update(settings)
    return LateFusionNetwork(list(temporal), list(spatial), **options).eval()


def test_network_returns_one_forecast_per_window():
    network = small_network()

    assert network(torch.randn(4, 12, 91)).shape == (4, 1)
    assert network(torch.randn(2, 5, 91)).shape == (2, 1)
    with pytest.raises(ValueError):
        network(torch.randn(2, 12, 90))
    with pytest.raises(ValueError):
        network(torch.randn(2, 13, 91))


def test_spatial_branch_reads_only_the_forecast_origin_embedding():
    network = small_network()
    inputs = torch.randn(3, 12, 91)
    earlier = inputs.clone()
    earlier[:, :-1, 83:] += 5.0
    latest = inputs.clone()
    latest[:, -1, 83:] += 5.0

    with torch.no_grad():
        torch.testing.assert_close(network(inputs), network(earlier))
        assert not torch.allclose(network(inputs), network(latest))
        # Embeddings never enter the temporal branch.
        torch.testing.assert_close(
            network.temporal_summary(inputs), network.temporal_summary(latest)
        )


def test_temporal_encoder_is_causal():
    network = small_network()
    inputs = torch.randn(2, 12, 91)
    changed = inputs.clone()
    changed[:, -1, :83] += 5.0

    with torch.no_grad():
        original = network.encoder(inputs[..., :83].transpose(1, 2))
        perturbed = network.encoder(changed[..., :83].transpose(1, 2))
    torch.testing.assert_close(original[:, :, :-1], perturbed[:, :, :-1])


def test_network_without_embeddings_uses_the_temporal_branch_alone():
    network = small_network(temporal=range(5), spatial=[])

    assert network.spatial is None
    assert network(torch.randn(3, 12, 5)).shape == (3, 1)


def test_model_exposes_the_shared_sequence_contract():
    model = LateFusionModel(sequence_length=12, d_model=16, nhead=4, device="cpu")

    assert isinstance(model, TorchForecastModel)
    assert model.get_model_name() == "LateFusion"
    assert model.requires_sequence is True
    assert isinstance(model._build_network(input_size=3), LateFusionNetwork)


def test_model_splits_columns_by_name_not_position():
    model = LateFusionModel(device="cpu")
    model.feature_columns = ["price_lag", "gnn_emb_00", "demand", "gnn_emb_01"]

    network = model._build_network(input_size=4)

    assert network.temporal_idx.tolist() == [0, 2]
    assert network.spatial_idx.tolist() == [1, 3]


@pytest.mark.parametrize(
    "settings",
    [
        {"sequence_length": 0},
        {"kernel_size": 1},
        {"d_model": 30, "nhead": 4},
        {"n_queries": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_model_rejects_invalid_architecture_settings(settings):
    with pytest.raises(ValueError):
        LateFusionModel(**settings)


def test_model_trains_and_predicts_with_explicit_history():
    rng = np.random.default_rng(0)
    x_train = pd.DataFrame(
        {
            "price_lag": rng.normal(50, 10, 16),
            "demand": rng.normal(1000, 50, 16),
            "gnn_emb_00": rng.normal(size=16),
            "gnn_emb_01": rng.normal(size=16),
        }
    )
    x_train.iloc[2, 0] = np.nan
    y_train = pd.Series(rng.normal(50, 10, 16))
    current = x_train.iloc[:3].reset_index(drop=True) + 1.0
    model = LateFusionModel(
        sequence_length=3, channels=4, num_blocks=1, d_model=8, nhead=2,
        dim_feedforward=8, spatial_hidden=4, head_hidden=4,
        batch_size=4, max_epochs=1, random_seed=7, device="cpu",
    )

    model.train(x_train, y_train)
    all_at_once = model.predict(current, history=x_train)
    first = model.predict(current.iloc[:1], history=x_train)
    rest = model.predict(
        current.iloc[1:], history=pd.concat([x_train, current.iloc[:1]])
    )

    assert model.network.spatial_idx.tolist() == [2, 3]
    assert all_at_once.shape == (3,)
    assert np.isfinite(all_at_once).all()
    np.testing.assert_allclose(all_at_once, np.concatenate([first, rest]), rtol=1e-5)
