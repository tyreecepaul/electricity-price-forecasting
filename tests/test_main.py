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
    import pipeline.experiment_pipeline as experiment_pipeline

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


def test_cli_defaults_to_all_registered_models():
    module = load_main_module()

    args = module.parse_args([])

    assert args.models is None
    assert module.select_model_registry(args.models) == module.MODEL_REGISTRY


def test_cli_parses_multiple_models():
    module = load_main_module()

    args = module.parse_args(["--models", "mlp", "xgboost"])

    assert args.models == ["mlp", "xgboost"]


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

    module.main()

    pd.testing.assert_frame_equal(trained["x"], train_x)
    pd.testing.assert_series_equal(trained["y"], train_y)
    assert trained["depth"] == 1
    np.testing.assert_array_equal(evaluated["actual"], [110.0, 190.0])
    np.testing.assert_array_equal(evaluated["predicted"], [105.0, 195.0])
