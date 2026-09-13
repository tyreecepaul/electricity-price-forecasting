import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pipeline.model import Model

FIXED_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_bin": 256,
    "n_jobs": -1,
    "random_state": 42
}

class XGBoostModel(Model):
    model_name = "XGBoost"
    model: XGBRegressor

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        min_child_weight: int = 8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 8.0,
        max_estimators: int = 2000,
        early_stopping_rounds: int = 100,
    ):
        self.model = XGBRegressor(
            max_depth,
            learning_rate,
            subsample,
            colsample_bytree,
            min_child_weight,
            reg_alpha,
            reg_lambda,
            max_estimators,
            early_stopping_rounds,
            **FIXED_PARAMS,
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(x_train, y_train, verbose=False)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        if (self.model is None):
            raise RuntimeError("train() must be called before predict()")

        return self.model.predict(x_test)