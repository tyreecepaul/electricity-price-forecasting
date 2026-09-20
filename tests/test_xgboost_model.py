from __future__ import annotations

import pandas as pd

from electricity_price_forecasting_model.xgboost import XGBoostModel


def test_xgboost_receives_named_hyperparameters():
    model = XGBoostModel(
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=4,
        max_estimators=7,
        early_stopping_rounds=None,
    )

    params = model.model.get_params()

    assert params["objective"] == "reg:absoluteerror"
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.1
    assert params["min_child_weight"] == 4
    assert params["n_estimators"] == 7
    assert params["early_stopping_rounds"] is None


def test_xgboost_can_train_without_a_validation_set():
    model = XGBoostModel(max_estimators=3, early_stopping_rounds=None)
    x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])

    model.train(x_train, y_train)

    predictions = model.predict(x_train)
    assert predictions.shape == (4,)
