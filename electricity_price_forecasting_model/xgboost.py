"""XGBoost regressor"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pipeline.model import Model


class XGBoostModel(Model):
    model_name = "XGBoost"

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        min_child_weight: float = 8.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 8.0,
        max_estimators: int = 1000,
        max_bin: int = 256,
        random_seed: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int | None = None,
    ) -> None:
        if max_depth <= 0 or learning_rate <= 0 or min_child_weight < 0:
            raise ValueError("Tree depth and learning rate must be positive")
        if not 0 < subsample <= 1 or not 0 < colsample_bytree <= 1:
            raise ValueError("subsample values must be in (0, 1]")
        if reg_alpha < 0 or reg_lambda < 0:
            raise ValueError("regularization values must be non-negative")
        if max_estimators <= 0 or max_bin <= 1:
            raise ValueError("max_estimators must be positive and max_bin greater than 1")
        if early_stopping_rounds is not None:
            raise ValueError(
                "The shared pipeline selects candidates on its validation split; "
                "tune max_estimators instead of using internal early stopping"
            )

        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.max_estimators = int(max_estimators)
        self.max_bin = int(max_bin)
        self.random_seed = int(random_seed)
        self.n_jobs = int(n_jobs)

        self.feature_columns: list[str] = []
        self.is_fitted = False
        self.model = self._build_model()

    def _build_model(self) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="rmse",
            tree_method="hist",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_estimators=self.max_estimators,
            max_bin=self.max_bin,
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        if x_train.empty:
            raise ValueError("x_train must contain at least one row")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same number of rows")

        target = pd.to_numeric(y_train, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid_target = target.notna().to_numpy()
        if not valid_target.any():
            raise ValueError("y_train contains no finite targets")

        self.feature_columns = list(x_train.columns)
        features = self._numeric_frame(x_train.loc[valid_target])
        self.model = self._build_model()
        self.model.fit(features, target.loc[valid_target], verbose=False)
        self.is_fitted = True

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("train() must be called before predict()")
        if x_test.empty:
            return np.empty(0, dtype=float)

        missing_columns = set(self.feature_columns).difference(x_test.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing model features: {names}")
        features = self._numeric_frame(x_test[self.feature_columns])
        return np.asarray(self.model.predict(features), dtype=float)
