"""Tests for the evaluate() function (evaluation/evaluation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluation import evaluate
from evaluation.Model import Model
from preprocessing import PreparedModelData


def _make_prepared_data(target_rrp: list[float]) -> PreparedModelData:
    n = len(target_rrp)
    test_df = pd.DataFrame(
        {
            "settlement_date": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "region": ["NSW1"] * n,
            "rrp": [0.0] * n,
            "target_rrp": target_rrp,
            "target": [0.0] * n,
        }
    )
    return PreparedModelData(
        X_train=pd.DataFrame(),
        y_train=pd.Series(dtype=float),
        X_test=test_df[[]],
        y_test=test_df["target"],
        train_df=pd.DataFrame(),
        test_df=test_df,
        features=[],
        region="NSW1",
        horizon=5,
        target_type="price",
    )


class _FixedPredictionModel(Model):
    """Returns a pre-set prediction array regardless of input."""

    model_name = "fixed"

    def __init__(self, prediction: list[float]) -> None:
        self._prediction = np.asarray(prediction, dtype=float)

    def train(self, training_data: PreparedModelData) -> None:
        pass

    def predict(self, test_data: PreparedModelData) -> np.ndarray:
        return self._prediction


def test_evaluate_prints_correct_sse_against_target_rrp(capsys):
    prepared = _make_prepared_data(target_rrp=[50.0, 60.0, 70.0])
    model = _FixedPredictionModel(prediction=[52.0, 58.0, 75.0])

    # errors: 2, -2, 5 -> squared: 4, 4, 25 -> SSE: 33.0
    evaluate(model, prepared)

    captured = capsys.readouterr()
    assert "SSE for fixed: 33.0" in captured.out


def test_evaluate_reports_zero_sse_for_perfect_predictions(capsys):
    prepared = _make_prepared_data(target_rrp=[10.0, 20.0])
    model = _FixedPredictionModel(prediction=[10.0, 20.0])

    evaluate(model, prepared)

    captured = capsys.readouterr()
    assert "SSE for fixed: 0.0" in captured.out


def test_evaluate_calls_predict_with_the_prepared_data_object():
    prepared = _make_prepared_data(target_rrp=[1.0, 2.0])
    received: list[PreparedModelData] = []

    class _RecordingModel(Model):
        model_name = "recorder"

        def __init__(self) -> None:
            pass

        def train(self, training_data: PreparedModelData) -> None:
            pass

        def predict(self, test_data: PreparedModelData) -> np.ndarray:
            received.append(test_data)
            return np.array([1.0, 2.0])

    evaluate(_RecordingModel(), prepared)

    assert received == [prepared]
