"""PyTorch multilayer perceptron"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.model import Model
from pipeline.training import TrainingResult, TorchEarlyStoppingTrainer
from pipeline.preprocessing import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int],
        dropout: float | Sequence[float],
    ) -> None:
        super().__init__()

        hidden_sizes = tuple(hidden_sizes)
        if isinstance(dropout, (int, float)):
            dropout_rates = (float(dropout),) * len(hidden_sizes)
        else:
            dropout_rates = tuple(float(rate) for rate in dropout)
        if len(dropout_rates) != len(hidden_sizes):
            raise ValueError("dropout must provide one rate per hidden layer")

        layers: list[nn.Module] = []
        input_size = n_features
        for position, (hidden_size, dropout_rate) in enumerate(
            zip(hidden_sizes, dropout_rates)
        ):
            layers.append(nn.Linear(input_size, hidden_size))
            # The research model normalises its first two hidden layers but
            # leaves the final 32-unit representation unnormalised.
            if position < len(hidden_sizes) - 1:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MLPModel(Model):
    model_name = "MLP"

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (128, 64, 32),
        dropout: float | Sequence[float] = (0.2, 0.15, 0.0),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        prediction_batch_size: int = 1024,
        max_epochs: int = 30,
        patience: int = 3,
        min_delta: float = 0.001,
        loss: str = "huber",
        huber_delta: float = 1.0,
        random_seed: int = 42,
        device: str | None = None,
    ) -> None:
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if isinstance(dropout, (int, float)):
            dropout_rates = (float(dropout),) * len(hidden_sizes)
        else:
            dropout_rates = tuple(float(rate) for rate in dropout)
        if len(dropout_rates) != len(hidden_sizes):
            raise ValueError("dropout must provide one rate per hidden layer")
        if any(not 0 <= rate < 1 for rate in dropout_rates):
            raise ValueError("dropout rates must be in [0, 1)")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if batch_size <= 0 or prediction_batch_size <= 0 or max_epochs <= 0:
            raise ValueError("batch sizes and max_epochs must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        loss = loss.lower()
        if loss not in {"huber", "mse", "mae"}:
            raise ValueError("loss must be 'huber', 'mse', or 'mae'")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

        self.hidden_sizes = hidden_sizes
        self.dropout = dropout_rates
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.prediction_batch_size = int(prediction_batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.loss = loss
        self.huber_delta = float(huber_delta)
        self.random_seed = int(random_seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.network: FeedForwardNetwork | None = None
        self.preprocessor: NumericPreprocessor | None = None
        self.target_transformer: TargetTransformer | None = None
        self.feature_columns: list[str] = []
        self.feature_medians: pd.Series | None = None
        self.training_loss_history: list[float] = []
        self.validation_mae_history: list[float] = []
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self.training_result: TrainingResult | None = None

    def _set_random_seed(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_criterion(self) -> nn.Module:
        if self.loss == "mse":
            return nn.MSELoss()
        if self.loss == "mae":
            return nn.L1Loss()
        return nn.HuberLoss(delta=self.huber_delta)

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    def _impute_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_medians is None:
            raise RuntimeError("train() must be called before predict()")
        missing_columns = set(self.feature_columns).difference(frame.columns)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing model features: {names}")
        numeric = self._numeric_frame(frame[self.feature_columns])
        return numeric.fillna(self.feature_medians)

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._train(x_train, y_train)

    def train_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> TrainingResult:
        result = self._train(
            x_train,
            y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )
        return result

    def _train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        x_validation: pd.DataFrame | None = None,
        y_validation: pd.Series | None = None,
    ) -> TrainingResult:
        if x_train.empty:
            raise ValueError("x_train must contain at least one row")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same number of rows")
        if (x_validation is None) != (y_validation is None):
            raise ValueError(
                "x_validation and y_validation must either both be provided or omitted"
            )

        target = pd.to_numeric(y_train, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid_target = target.notna().to_numpy()
        if not valid_target.any():
            raise ValueError("y_train contains no finite targets")

        self.feature_columns = list(x_train.columns)
        numeric_features = self._numeric_frame(x_train.loc[valid_target])
        self.feature_medians = numeric_features.median().fillna(0.0)
        imputed_features = numeric_features.fillna(self.feature_medians)
        valid_target_values = target.loc[valid_target]

        scaling_groups = suggested_scaling_groups(self.feature_columns)
        self.preprocessor = NumericPreprocessor(
            asinh_columns=tuple(scaling_groups["asinh_then_standardise"]),
            standardise_columns=tuple(scaling_groups["standardise_only"]),
        )
        scaled_features = self.preprocessor.fit_transform(imputed_features)

        self.target_transformer = TargetTransformer()
        scaled_target = self.target_transformer.fit_transform(valid_target_values)

        inputs = torch.tensor(
            scaled_features.to_numpy(dtype=np.float32), dtype=torch.float32
        )
        targets = torch.tensor(scaled_target, dtype=torch.float32).reshape(-1, 1)

        validation_inputs: torch.Tensor | None = None
        validation_actual: np.ndarray | None = None
        if x_validation is not None and y_validation is not None:
            if x_validation.empty:
                raise ValueError("x_validation must contain at least one row")
            if len(x_validation) != len(y_validation):
                raise ValueError(
                    "x_validation and y_validation must have the same number of rows"
                )
            validation_target = pd.to_numeric(
                y_validation, errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            valid_validation_target = validation_target.notna().to_numpy()
            if not valid_validation_target.any():
                raise ValueError("y_validation contains no finite targets")
            validation_features = self._impute_features(
                x_validation.loc[valid_validation_target]
            )
            scaled_validation_features = self.preprocessor.transform(
                validation_features
            )
            validation_inputs = torch.tensor(
                scaled_validation_features.to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            validation_actual = validation_target.loc[
                valid_validation_target
            ].to_numpy(dtype=float)

        self._set_random_seed()
        self.network = FeedForwardNetwork(
            n_features=len(self.feature_columns),
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        generator = torch.Generator().manual_seed(self.random_seed)
        loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        criterion = self._build_criterion()
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def train_epoch() -> float:
            self.network.train()
            total_loss = 0.0
            for input_batch, target_batch in loader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                optimizer.zero_grad()
                predictions = self.network(input_batch)
                loss = criterion(predictions, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(input_batch)
            return total_loss / len(inputs)

        validation_callback = None
        if validation_inputs is not None and validation_actual is not None:
            validation_callback = lambda: self._validation_mae(
                validation_inputs,
                validation_actual,
            )

        trainer = TorchEarlyStoppingTrainer(
            max_epochs=self.max_epochs,
            patience=self.patience,
            min_delta=self.min_delta,
        )
        result = trainer.fit(
            network=self.network,
            train_epoch=train_epoch,
            validation_score=validation_callback,
        )
        self.training_result = result
        self.training_loss_history = result.training_loss_history
        self.validation_mae_history = result.validation_score_history
        self.best_epoch = result.best_epoch
        self.stopped_epoch = result.stopped_epoch
        return result

    def _validation_mae(
        self,
        inputs: torch.Tensor,
        actual: np.ndarray,
    ) -> float:
        if self.network is None or self.target_transformer is None:
            raise RuntimeError("The model must be initialised before validation")

        loader = DataLoader(
            TensorDataset(inputs),
            batch_size=self.prediction_batch_size,
            shuffle=False,
        )
        self.network.eval()
        prediction_batches: list[np.ndarray] = []
        with torch.no_grad():
            for (input_batch,) in loader:
                predictions = self.network(input_batch.to(self.device))
                prediction_batches.append(predictions.cpu().numpy())

        transformed = np.concatenate(prediction_batches).reshape(-1)
        predicted = self.target_transformer.inverse_transform(transformed)
        return float(np.mean(np.abs(actual - predicted)))

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        if self.network is None or self.preprocessor is None:
            raise RuntimeError("train() must be called before predict()")
        if self.target_transformer is None:
            raise RuntimeError("Target transformer is not fitted")
        if x_test.empty:
            return np.empty(0, dtype=float)

        imputed_features = self._impute_features(x_test)
        scaled_features = self.preprocessor.transform(imputed_features)
        inputs = torch.tensor(
            scaled_features.to_numpy(dtype=np.float32), dtype=torch.float32
        )
        loader = DataLoader(
            TensorDataset(inputs),
            batch_size=self.prediction_batch_size,
            shuffle=False,
        )

        self.network.eval()
        prediction_batches: list[np.ndarray] = []
        with torch.no_grad():
            for (input_batch,) in loader:
                predictions = self.network(input_batch.to(self.device))
                prediction_batches.append(predictions.cpu().numpy())

        transformed = np.concatenate(prediction_batches).reshape(-1)
        return np.asarray(
            self.target_transformer.inverse_transform(transformed), dtype=float
        )
