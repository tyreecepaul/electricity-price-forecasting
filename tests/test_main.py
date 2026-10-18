from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_main_does_not_start_an_experiment(monkeypatch):
    from pipeline import experiment_pipeline

    def unexpected_run(*args, **kwargs):
        raise AssertionError("importing main.py started the experiment")

    monkeypatch.setattr(
        experiment_pipeline,
        "prepare_experiment",
        unexpected_run,
    )

    load_main_module()


def test_cli_selects_requested_models_in_registry_order():
    module = load_main_module()

    selected = module.select_model_registry(["mlp"])

    assert [factory.model_name for factory, _ in selected] == ["MLP"]


def test_cli_selects_gru():
    module = load_main_module()

    selected = module.select_model_registry(["gru"])

    assert [factory.model_name for factory, _ in selected] == ["GRU"]


def test_cli_selects_lstm():
    module = load_main_module()

    selected = module.select_model_registry(["lstm"])

    assert [factory.model_name for factory, _ in selected] == ["LSTM"]


def test_cli_selects_arima_with_the_experiment_horizon_and_target():
    module = load_main_module()

    selected = module.select_model_registry(["arima"])

    assert [factory.model_name for factory, _ in selected] == ["ARIMA"]
    grid = selected[0][1]
    assert grid["horizon_minutes"] == [module.EXPERIMENT.horizon_minutes]
    assert grid["target_type"] == [module.EXPERIMENT.target_type]


def test_cli_selects_tcn_with_the_initial_comparison_window():
    module = load_main_module()

    assert "tcn" in [module.model_cli_name(factory) for factory, _ in module.MODEL_REGISTRY]
    selected = module.select_model_registry(["tcn"])

    assert [factory.model_name for factory, _ in selected] == ["TCN"]
    grid = selected[0][1]
    assert grid["sequence_length"] == [12]
    assert grid["channels"] == [16]
    assert grid["kernel_size"] == [3]
    assert grid["num_blocks"] == [2]
    assert all(len(values) == 1 for values in grid.values())
    assert module.parse_args(["--models", "tcn"]).models == ["tcn"]


def test_cli_defaults_to_all_registered_models():
    module = load_main_module()

    args = module.parse_args([])

    assert args.models is None
    assert module.select_model_registry(args.models) == module.MODEL_REGISTRY


def test_cli_accepts_uppercase_tcn_and_uses_automatic_device_selection():
    module = load_main_module()

    args = module.parse_args(["--models", "TCN"])
    assert args.models == ["tcn"]
    selected = module.select_model_registry(["TCN"])
    assert [factory.model_name for factory, _ in selected] == ["TCN"]
    assert selected[0][1]["device"] == [None]


def test_cli_parses_multiple_models():
    module = load_main_module()

    args = module.parse_args(["--models", "mlp", "gru", "xgboost"])

    assert args.models == ["mlp", "gru", "xgboost"]


def test_cli_graph_embedding_choices():
    module = load_main_module()

    assert module.parse_args([]).graph_embedding == "per-model"
    assert module.parse_args(["--graph-embedding"]).graph_embedding == "nem"
    assert module.parse_args(["--graph-embedding", "none"]).graph_embedding == "none"
    assert module.parse_args(["--graph-embedding", "--models", "lstm"]).models == ["lstm"]
    assert module.EXPERIMENT.graph_embedding is None
    assert module.GRAPH_EMBEDDING_CHOICES["none"] is None
    assert module.GRAPH_EMBEDDING_CHOICES["nem"].adjacency_mode == "nem"
    assert module.GRAPH_EMBEDDING_CHOICES["identity"].adjacency_mode == "identity"


def test_each_model_is_assigned_its_graph_embedding():
    module = load_main_module()
    assigned = {
        factory.model_name: embedding
        for factory, embedding in module.MODEL_GRAPH_EMBEDDING.items()
    }

    assert assigned["XGBoost"].adjacency_mode == "identity"
    assert assigned["LSTM"].adjacency_mode == "nem"
    assert assigned["LateFusion"].adjacency_mode == "nem"
    assert "GRU" not in assigned


def run_fake_models(module, monkeypatch, *, test_rows=None, **arguments):
    """Run ``main`` with stub models and report what each one was given."""

    def fake_model(name):
        class Fake:
            model_name = name

            def __init__(self, depth):
                self.depth = depth

            def get_model_name(self):
                return name

        return Fake

    first, second, third = (fake_model(name) for name in ("First", "Second", "Third"))
    nem = module.GRAPH_EMBEDDING_CHOICES["nem"]
    prepared_configs, searches = [], []

    def fake_prepare(config):
        prepared_configs.append(config.graph_embedding)
        rows = (test_rows or {}).get(config.graph_embedding, 2)
        test_df = pd.DataFrame(
            {
                "settlement_date": pd.date_range("2025-01-01", periods=rows, freq="5min"),
                "target_rrp": 1.0,
            }
        )
        return SimpleNamespace(prepared="prepared", test_df=test_df, config=config)

    def fake_search(factory, grid, experiment):
        searches.append((factory.model_name, experiment.config.graph_embedding, grid))
        return SimpleNamespace(best_config={"depth": 1})

    registry = [(first, {"depth": [1]}), (second, {"depth": [1]}), (third, {"depth": [1]})]
    monkeypatch.setattr(module, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(module, "MODEL_GRAPH_EMBEDDING", {first: nem, third: nem})
    monkeypatch.setattr(module, "SWEEP_GRIDS", {second: {"depth": [1, 2]}})
    monkeypatch.setattr(module, "prepare_experiment", fake_prepare)
    monkeypatch.setattr(module, "print_experiment_summary", lambda value: None)
    monkeypatch.setattr(module, "search_hyperparameters", fake_search)
    monkeypatch.setattr(module, "final_training_data", lambda model, prepared: (None, None))
    monkeypatch.setattr(first, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(second, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(third, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(module, "predict_test", lambda model, prepared: np.zeros(2))
    monkeypatch.setattr(module, "reconstruct_price_forecast", lambda frame, y, config: y)
    monkeypatch.setattr(
        module, "evaluate", lambda model, actual, predicted: model.get_model_name()
    )
    monkeypatch.setattr(module, "print_evaluation_summary", lambda *args: None)

    module.main(save_results=False, **arguments)
    return nem, prepared_configs, searches


def test_models_sharing_an_embedding_share_one_prepared_experiment(monkeypatch):
    module = load_main_module()

    nem, prepared, searches = run_fake_models(module, monkeypatch)

    assert prepared == [nem, None]
    assert [(name, embedding) for name, embedding, _ in searches] == [
        ("First", nem),
        ("Third", nem),
        ("Second", None),
    ]


def test_explicit_embedding_overrides_the_per_model_map(monkeypatch):
    module = load_main_module()

    _, prepared, searches = run_fake_models(module, monkeypatch, graph_embeddings={})

    assert prepared == [None]
    assert [name for name, _, _ in searches] == ["First", "Second", "Third"]


def test_sweep_replaces_only_grids_that_have_one(monkeypatch):
    module = load_main_module()

    _, _, searches = run_fake_models(module, monkeypatch, sweep=True)
    grids = {name: grid for name, _, grid in searches}

    assert grids == {
        "First": {"depth": [1]},
        "Second": {"depth": [1, 2]},
        "Third": {"depth": [1]},
    }
    lstm = next(f for f, _ in load_main_module().MODEL_REGISTRY if f.model_name == "LSTM")
    registry_grid = next(g for f, g in load_main_module().MODEL_REGISTRY if f is lstm)
    sweep_grid = next(
        g for f, g in load_main_module().SWEEP_GRIDS.items() if f.model_name == "LSTM"
    )
    assert set(sweep_grid) >= set(registry_grid)
    assert len(sweep_grid["sequence_length"]) > 1 and len(sweep_grid["hidden_size"]) > 1


def test_embedding_variants_must_score_the_same_test_rows(monkeypatch):
    import pytest

    module = load_main_module()
    nem = module.GRAPH_EMBEDDING_CHOICES["nem"]

    with pytest.raises(RuntimeError, match="different test rows"):
        run_fake_models(module, monkeypatch, test_rows={nem: 2, None: 3})


def test_main_uses_prepared_features_target_and_reconstructed_prices(monkeypatch):
    module = load_main_module()

    train_x = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    train_y = pd.Series([0.5, -0.5], name="target")
    test_x = pd.DataFrame({"feature_a": [5.0, 6.0], "feature_b": [7.0, 8.0]})
    test_df = pd.DataFrame(
        {
            "rrp": [100.0, 200.0],
            "target_rrp": [110.0, 190.0],
            "target": [10.0, -10.0],
        }
    )
    prepared = SimpleNamespace(
        X_train=train_x,
        y_train=train_y,
        X_test=test_x,
        test_df=test_df,
        features=["feature_a", "feature_b"],
    )
    experiment = SimpleNamespace(
        prepared=prepared,
        test_df=test_df,
        features=prepared.features,
        config=module.EXPERIMENT,
    )

    trained = {}

    class FakeModel:
        model_name = "Fake model"

        def __init__(self, depth):
            trained["depth"] = depth

        def train(self, x_train, y_train):
            trained["x"] = x_train.copy()
            trained["y"] = y_train.copy()

        def predict(self, x_test):
            pd.testing.assert_frame_equal(x_test, test_x)
            return np.array([5.0, -5.0])

        def get_model_name(self):
            return self.model_name

    def fake_search(model_factory, grid, supplied_experiment):
        assert model_factory is FakeModel
        assert grid == {"depth": [1]}
        assert supplied_experiment is experiment
        return SimpleNamespace(best_config={"depth": 1})

    def fake_reconstruct(frame, predicted_target, config):
        assert frame is test_df
        assert config is module.EXPERIMENT
        np.testing.assert_array_equal(predicted_target, [5.0, -5.0])
        return frame["rrp"].to_numpy() + predicted_target

    evaluated = {}

    def fake_evaluate(model, actual, predicted):
        evaluated["actual"] = actual.copy()
        evaluated["predicted"] = predicted.copy()
        return SimpleNamespace(model_name=model.get_model_name())

    monkeypatch.setattr(module, "MODEL_REGISTRY", [(FakeModel, {"depth": [1]})])
    monkeypatch.setattr(module, "prepare_experiment", lambda config: experiment)
    monkeypatch.setattr(module, "print_experiment_summary", lambda value: None)
    monkeypatch.setattr(module, "search_hyperparameters", fake_search)
    monkeypatch.setattr(module, "reconstruct_price_forecast", fake_reconstruct)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "print_evaluation_summary", lambda *args: None)

    module.main(save_results=False)

    pd.testing.assert_frame_equal(trained["x"], train_x)
    pd.testing.assert_series_equal(trained["y"], train_y)
    assert trained["depth"] == 1
    np.testing.assert_array_equal(evaluated["actual"], [110.0, 190.0])
    np.testing.assert_array_equal(evaluated["predicted"], [105.0, 195.0])
