from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.tuning import search_hyperparameters


def experiment_fixture():
    fit_df = pd.DataFrame(
        {
            "settlement_date": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "region": ["NSW1"] * 3,
            "rrp": [100.0, 105.0, 103.0],
            "target_rrp": [105.0, 103.0, 110.0],
            "target": [5.0, -2.0, 7.0],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        }
    )
    validation_df = pd.DataFrame(
        {
            "settlement_date": pd.date_range("2026-02-01", periods=2, freq="5min"),
            "region": ["NSW1"] * 2,
            "rrp": [90.0, 95.0],
            "target_rrp": [92.0, 91.0],
            "target": [2.0, -4.0],
            "feature_a": [7.0, 8.0],
            "feature_b": [9.0, 10.0],
        }
    )
    return SimpleNamespace(
        fit_df=fit_df,
        validation_df=validation_df,
        features=["feature_a", "feature_b"],
    )


def test_search_uses_only_selected_features_and_configured_target():
    experiment = experiment_fixture()
    observed = {}

    class RecordingModel:
        model_name = "Recording model"

        def __init__(self, depth):
            observed["depth"] = depth

        def train(self, x_train, y_train):
            observed["x_train"] = x_train.copy()
            observed["y_train"] = y_train.copy()

        def predict(self, x_validation):
            observed["x_validation"] = x_validation.copy()
            return experiment.validation_df["target"].to_numpy()

        def get_model_name(self):
            return self.model_name

    result = search_hyperparameters(
        RecordingModel,
        {"depth": [2]},
        experiment,
        verbose=False,
    )

    pd.testing.assert_frame_equal(
        observed["x_train"],
        experiment.fit_df[experiment.features],
    )
    pd.testing.assert_series_equal(
        observed["y_train"],
        experiment.fit_df["target"],
    )
    pd.testing.assert_frame_equal(
        observed["x_validation"],
        experiment.validation_df[experiment.features],
    )
    assert result.best_config == {"depth": 2}
    assert result.best_score == 0.0


def test_search_reports_failed_trials_before_raising():
    experiment = experiment_fixture()

    class FailingModel:
        model_name = "Failing model"

        def __init__(self, depth):
            pass

        def train(self, x_train, y_train):
            raise ValueError("deliberate training failure")

        def get_model_name(self):
            return self.model_name

    with pytest.raises(RuntimeError, match="No configuration produced a usable score"):
        search_hyperparameters(
            FailingModel,
            {"depth": [1, 2]},
            experiment,
            verbose=True,
        )
