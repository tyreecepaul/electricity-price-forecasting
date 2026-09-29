from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pipeline.neural import TorchForecastModel  # noqa: E402


def load_network_type():
    name = "electricity_price_forecasting_model.layers.tcn"
    assert importlib.util.find_spec(name) is not None, "TCN network module is missing"
    return importlib.import_module(name).TCNNetwork


def load_model_type():
    name = "electricity_price_forecasting_model.final_model.tcn"
    assert importlib.util.find_spec(name) is not None, "TCN model module is missing"
    return importlib.import_module(name).TCNModel


def test_tcn_network_preserves_time_and_returns_one_forecast():
    network_type = load_network_type()
    network = network_type(
        input_size=3, channels=8, kernel_size=3, num_blocks=2, dropout=0.0
    )
    inputs = torch.randn(4, 12, 3)

    assert network.temporal_features(inputs).shape == (4, 8, 12)
    assert network(inputs).shape == (4, 1)
    assert network(torch.randn(2, 5, 3)).shape == (2, 1)
    assert network.receptive_field == 13
    dilations = [
        layer.dilation[0]
        for layer in network.modules()
        if isinstance(layer, torch.nn.Conv1d) and layer.kernel_size == (3,)
    ]
    assert dilations == [1, 1, 2, 2]


def test_tcn_temporal_features_are_causal():
    network_type = load_network_type()
    torch.manual_seed(7)
    network = network_type(
        input_size=2, channels=4, kernel_size=3, num_blocks=2, dropout=0.0
    )
    network.eval()
    inputs = torch.randn(2, 12, 2)
    changed = inputs.clone()
    changed[:, 7:, :] += 1000

    with torch.inference_mode():
        original_prefix = network.temporal_features(inputs)[:, :, :7]
        changed_prefix = network.temporal_features(changed)[:, :, :7]
        truncated_prefix = network.temporal_features(inputs[:, :7, :])

    torch.testing.assert_close(original_prefix, changed_prefix)
    torch.testing.assert_close(original_prefix, truncated_prefix)


def test_tcn_model_exposes_the_shared_sequence_contract():
    model_type = load_model_type()
    network_type = load_network_type()
    model = model_type(
        sequence_length=12,
        channels=16,
        kernel_size=3,
        num_blocks=2,
        dropout=0.1,
        learning_rate=3e-4,
        batch_size=8,
        max_epochs=2,
        device="cpu",
    )

    assert isinstance(model, TorchForecastModel)
    assert model.get_model_name() == "TCN"
    assert model.requires_sequence is True
    assert model.sequence_length == 12
    assert model.channels == 16
    assert model.kernel_size == 3
    assert model.num_blocks == 2
    assert model.dropout == pytest.approx(0.1)
    assert model.learning_rate == pytest.approx(3e-4)
    assert isinstance(model._build_network(input_size=2), network_type)


@pytest.mark.parametrize(
    "settings",
    [
        {"sequence_length": 0},
        {"channels": 0},
        {"kernel_size": 1},
        {"num_blocks": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_tcn_model_rejects_invalid_architecture_settings(settings):
    model_type = load_model_type()
    with pytest.raises(ValueError):
        model_type(**settings)


def test_tcn_trains_and_predicts_with_explicit_history():
    model_type = load_model_type()
    network_type = load_network_type()
    x_train = pd.DataFrame(
        {
            "price_lag": [10.0, 12.0, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0],
            "demand": [100.0, 105.0, 110.0, np.nan, 120.0, 125.0, 130.0, 135.0],
        }
    )
    y_train = pd.Series([11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0])
    current = pd.DataFrame(
        {"price_lag": [26.0, np.nan, 30.0], "demand": [140.0, 145.0, 150.0]}
    )
    model = model_type(
        sequence_length=3,
        channels=4,
        kernel_size=3,
        num_blocks=1,
        batch_size=2,
        max_epochs=1,
        random_seed=7,
        device="cpu",
    )

    model.train(x_train, y_train)
    all_at_once = model.predict(current, history=x_train)
    first = model.predict(current.iloc[:1], history=x_train)
    later_history = pd.concat([x_train, current.iloc[:1]])
    rest = model.predict(current.iloc[1:], history=later_history)

    assert isinstance(model.network, network_type)
    assert all_at_once.shape == (3,)
    assert np.isfinite(all_at_once).all()
    assert len(model.training_context) == 2
    assert len(model.training_loss_history) == 1
    np.testing.assert_allclose(all_at_once, np.concatenate([first, rest]))
