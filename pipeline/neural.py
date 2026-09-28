"""Shared preprocessing, training and inference for single-horizon neural models."""

from __future__ import annotations

import random
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.contracts import (
    METRIC_NAMES,
    forecast_vector,
    score_forecast,
    temporal_metadata,
    validate_context,
    validate_timeline,
)
from pipeline.model import Model
from pipeline.preprocessing import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)
from pipeline.sequences import ContextSequenceDataset, SequenceDataset
from pipeline.training import TorchEarlyStoppingTrainer


class TorchForecastModel(Model):
    """Subclasses supply a network; this wrapper owns the forecasting lifecycle.

    Sequence inputs use a DatetimeIndex or a (region, settlement_date) index.
    Unindexed frames remain supported for positional experiments, but cannot
    receive timestamp checks. Production adapters always supply metadata.

    Subclasses define architecture parameters and implement ``_build_network``.
    Row networks receive (batch, features); sequence networks receive
    (batch, lookback, features). Both return (batch, 1) transformed targets.
    Public ``predict`` converts these back to the experiment's target units.
    """

    sequence_length = 1

    def __init__(
        self,
        *,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        prediction_batch_size=1024,
        max_epochs=30,
        patience=3,
        min_delta=0.001,
        loss="huber",
        huber_delta=1.0,
        random_seed=42,
        device=None,
        selection_metric="mae",
        gradient_clip_norm=None,
        scheduler_step_size=None,
        scheduler_gamma=0.5,
        mixed_precision=False,
        interval_minutes=5,
    ):
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if min(batch_size, prediction_batch_size, max_epochs, patience) <= 0:
            raise ValueError("batch sizes, max_epochs and patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        loss = loss.lower()
        if loss not in {"huber", "mse", "mae"}:
            raise ValueError("loss must be 'huber', 'mse', or 'mae'")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if selection_metric not in METRIC_NAMES:
            raise ValueError(f"selection_metric must be one of {METRIC_NAMES}")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if scheduler_step_size is not None and scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive")
        if not 0 < scheduler_gamma <= 1 or interval_minutes <= 0:
            raise ValueError("Invalid scheduler_gamma or interval_minutes")
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
        self.selection_metric = selection_metric
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.mixed_precision = mixed_precision
        self.interval_minutes = interval_minutes
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = None
        self.preprocessor = None
        self.target_transformer = None
        self.feature_columns = []
        self.feature_medians = None
        self.training_context = None
        self.context_frame = None
        self.training_loss_history = []
        self.validation_score_history = []
        self.validation_mae_history = []
        self.best_epoch = None
        self.stopped_epoch = None
        self.training_result = None

    @abstractmethod
    def _build_network(self, input_size: int) -> nn.Module:
        """Return an untrained network producing one scalar per sample."""

    def _set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_criterion(self):
        """Choose the optimization loss, evaluated in transformed target space."""
        if self.loss == "mse":
            return nn.MSELoss()
        if self.loss == "mae":
            return nn.L1Loss()
        return nn.HuberLoss(delta=self.huber_delta)

    @staticmethod
    def _numeric_frame(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    def _impute_features(self, frame):
        """Restore training column order and fill missing values with fitted medians."""
        if self.feature_medians is None:
            raise RuntimeError("train() must be called before predict()")
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")
        return self._numeric_frame(frame[self.feature_columns]).fillna(
            self.feature_medians
        )

    def _inputs(self, frame):
        """Apply existing transforms and create float32 CPU tensors; never refit here."""
        scaled = self.preprocessor.transform(self._impute_features(frame))
        return torch.tensor(scaled.to_numpy(dtype=np.float32))

    def _check_timeline(self, frame):
        metadata = temporal_metadata(frame)
        if metadata is not None:
            validate_timeline(metadata, self.interval_minutes)

    def train(self, x_train, y_train):
        """Fit for max_epochs without validation, as used in the final refit."""
        self._train(x_train, y_train)

    def train_with_validation(self, x_train, y_train, x_validation, y_validation):
        """Fit and restore the checkpoint selected by chronological validation."""
        return self._train(
            x_train, y_train, x_validation=x_validation, y_validation=y_validation
        )

    def _train(self, x_train, y_train, *, x_validation=None, y_validation=None):
        """Fit transforms, build samples, then delegate epoch selection to the trainer.

        NaN targets mark rows excluded from the loss. Their features remain in
        sequence windows, so label embargoes do not create artificial time gaps.
        """
        if x_train.empty or len(x_train) != len(y_train):
            raise ValueError(
                "Training inputs and targets must be nonempty and have equal lengths"
            )
        if not x_train.index.equals(y_train.index):
            raise ValueError("Training input and target indices must match")
        if (x_validation is None) != (y_validation is None):
            raise ValueError("Validation inputs and targets must both be provided")
        if self.requires_sequence:
            self._check_timeline(x_train)
        target = pd.to_numeric(y_train, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = target.notna().to_numpy(copy=True)
        if not valid.any():
            raise ValueError("y_train contains no finite targets")

        self.feature_columns = list(x_train.columns)
        # Fit transformations only on eligible fitting rows. Unlabeled/embargo
        # rows are transformed using these fitted statistics for context.
        fitting = self._numeric_frame(x_train.loc[valid])
        self.feature_medians = fitting.median().fillna(0.0)
        groups = suggested_scaling_groups(self.feature_columns)
        self.preprocessor = NumericPreprocessor(
            asinh_columns=tuple(groups["asinh_then_standardise"]),
            standardise_columns=tuple(groups["standardise_only"]),
        ).fit(fitting.fillna(self.feature_medians))
        self.target_transformer = TargetTransformer().fit(target.loc[valid])
        inputs = self._inputs(x_train)
        targets = torch.tensor(
            self.target_transformer.transform(target), dtype=torch.float32
        ).reshape(-1, 1)
        if self.requires_sequence:
            dataset = SequenceDataset(
                inputs, targets, self.sequence_length, target_mask=valid
            )
            count = self.sequence_length - 1
            # Retain both forms: tensors for inference and indexed features for
            # checking whether a later prediction call follows this history.
            self.training_context = (
                inputs[-count:].clone() if count else inputs[:0].clone()
            )
            self.context_frame = x_train.tail(count).copy()
        else:
            dataset = TensorDataset(inputs[valid], targets[valid])

        validation_inputs = validation_actual = None
        if x_validation is not None:
            if x_validation.empty or len(x_validation) != len(y_validation):
                raise ValueError(
                    "Validation inputs and targets must be nonempty and have equal lengths"
                )
            if not x_validation.index.equals(y_validation.index):
                raise ValueError("Validation input and target indices must match")
            if self.requires_sequence:
                self._check_timeline(x_validation)
                if self.sequence_length > 1:
                    validate_context(
                        self.context_frame, x_validation, self.interval_minutes
                    )
            validation_inputs = self._inputs(x_validation)
            validation_actual = pd.to_numeric(y_validation, errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(validation_actual).any():
                raise ValueError("y_validation contains no finite targets")

        self._set_random_seed()
        self.network = self._build_network(len(self.feature_columns)).to(self.device)
        # Shuffle complete samples, never timesteps within a window. Each sample
        # is independent; these networks do not carry hidden state between batches.
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = None
        if self.scheduler_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        criterion = self._build_criterion()
        amp_enabled = bool(self.mixed_precision and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        def train_epoch():
            """Run one pass over fitting samples and return their mean loss."""
            self.network.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    prediction = self.network(xb)
                    if prediction.shape != yb.shape:
                        raise ValueError(
                            f"Network output {prediction.shape} must match targets {yb.shape}"
                        )
                    loss = criterion(prediction, yb)
                scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    # AMP scales gradients to protect small values. Undo that
                    # scaling before clipping so the threshold has its usual units.
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(xb)
            # Weight by batch size above so a short final batch is not overcounted.
            if scheduler is not None:
                scheduler.step()
            return total / len(dataset)

        callback = None
        if validation_inputs is not None:
            callback = lambda: (
                self._validation_mae(validation_inputs, validation_actual)
                if self.selection_metric == "mae"
                else self._validation_score(
                    validation_inputs, validation_actual, self.selection_metric
                )
            )
        result = TorchEarlyStoppingTrainer(
            max_epochs=self.max_epochs, patience=self.patience, min_delta=self.min_delta
        ).fit(network=self.network, train_epoch=train_epoch, validation_score=callback)
        self.training_result = result
        self.training_loss_history = result.training_loss_history
        self.validation_score_history = result.validation_score_history
        self.validation_mae_history = (
            result.validation_score_history if self.selection_metric == "mae" else []
        )
        self.best_epoch, self.stopped_epoch = result.best_epoch, result.stopped_epoch
        return result

    def _predict_transformed(self, inputs, history=None):
        """Run batched inference in transformed target space, preserving row order.

        Inputs are already scaled tensors. ``history`` overrides the saved tail
        for this call only; it does not advance or modify the model's context.
        """
        if self.network is None:
            raise RuntimeError("train() must be called before predict()")
        if not len(inputs):
            return np.empty(0, dtype=float)
        dataset = (
            ContextSequenceDataset(
                self.training_context if history is None else history,
                inputs,
                self.sequence_length,
            )
            if self.requires_sequence
            else TensorDataset(inputs)
        )
        loader = DataLoader(
            dataset, batch_size=self.prediction_batch_size, shuffle=False
        )
        self.network.eval()
        batches = []
        with torch.inference_mode():
            for (xb,) in loader:
                batches.append(
                    forecast_vector(
                        self.network(xb.to(self.device)).cpu().numpy(), len(xb)
                    )
                )
        return np.concatenate(batches)

    def _validation_score(self, inputs, actual, metric):
        """Evaluate in original target units so scores are comparable across models."""
        prediction = self.target_transformer.inverse_transform(
            self._predict_transformed(inputs)
        )
        valid = np.isfinite(actual)
        if not np.isfinite(prediction[valid]).all():
            raise RuntimeError("Validation produced non-finite predictions")
        return score_forecast(actual[valid], prediction[valid], metric)

    def _validation_mae(self, inputs, actual):
        return self._validation_score(inputs, actual, "mae")

    def predict(self, x_test, *, history=None):
        """Return one target-scale forecast per current row, in the same order.

        For sequence models, history contains observed feature rows immediately
        preceding x_test. Pass it explicitly for successive prediction chunks;
        otherwise the stored training tail is used. No targets are read here,
        and prediction does not update model weights, scalers, or saved history.
        """
        if self.network is None or self.preprocessor is None:
            raise RuntimeError("train() must be called before predict()")
        if x_test.empty:
            return np.empty(0, dtype=float)
        history_inputs = None
        if self.requires_sequence:
            self._check_timeline(x_test)
            count = self.sequence_length - 1
            context = self.context_frame if history is None else history
            if count:
                # Validate all supplied history, including future/duplicate rows.
                self._check_timeline(context)
                validate_context(context.tail(count), x_test, self.interval_minutes)
            history_inputs = (
                self.training_context
                if history is None
                else self._inputs(context.tail(count))
            )
        transformed = self._predict_transformed(self._inputs(x_test), history_inputs)
        return forecast_vector(
            self.target_transformer.inverse_transform(transformed), len(x_test)
        )
