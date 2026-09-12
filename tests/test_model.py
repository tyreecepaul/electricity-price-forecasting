"""Tests for the shared Model contract (evaluation/Model.py)."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from evaluation.Model import Model
from preprocessing import PreparedModelData


def _make_prepared_data() -> PreparedModelData:
    test_df = pd.DataFrame(
        {
            "settlement_date": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "region": ["NSW1"] * 3,
            "rrp": [50.0, 55.0, 60.0],
            "target_rrp": [52.0, 58.0, 65.0],
            "target": [2.0, 3.0, 5.0],
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
        target_type="change",
    )


class _ConstantModel(Model):
    """Minimal concrete Model used only to exercise the abstract contract."""

    model_name = "constant"

    def __init__(self, prediction: np.ndarray) -> None:
        self._prediction = prediction

    def train(self, training_data: PreparedModelData) -> None:
        pass

    def predict(self, test_data: PreparedModelData) -> np.ndarray:
        return self._prediction


def test_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Model()


def test_train_and_predict_are_typed_against_prepared_model_data():
    # Pins the DoD decision: the shared Model contract takes the
    # feature-engineered PreparedModelData, not the raw ingestion Data.
    train_hints = typing.get_type_hints(Model.train)
    predict_hints = typing.get_type_hints(Model.predict)

    assert train_hints["training_data"] is PreparedModelData
    assert predict_hints["test_data"] is PreparedModelData


def test_get_model_name_returns_subclass_name():
    model = _ConstantModel(prediction=np.zeros(3))
    assert model.get_model_name() == "constant"


def test_predict_returns_one_prediction_per_test_row():
    prepared = _make_prepared_data()
    model = _ConstantModel(prediction=np.array([52.0, 58.0, 65.0]))

    model.train(prepared)
    predictions = model.predict(prepared)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(prepared.test_df),)
