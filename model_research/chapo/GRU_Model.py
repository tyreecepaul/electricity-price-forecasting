"""Train and evaluate a GRU on the shared dispatch-price experiment data."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_research.common.experiment_pipeline import (
    ExperimentConfig,
    build_results_frame,
    complete_case_rows,
    evaluate_aemo_rows,
    print_experiment_summary,
    print_improvement,
    print_metrics,
    print_price_regime_results,
    print_target_diagnostics,
    prepare_experiment,
    reconstruct_price_forecast,
)
from preprocessing import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)

from typing import Sequence

features: Sequence[str] = [
    "rrp",
    "totaldemand",
    "availablegeneration",
    "demandforecast",
    "dispatchablegeneration",
    "dispatchableload",
    "netinterchange",
    "reserve_margin_mw",
    "reserve_ratio",
    "demand_forecast_error_mw",

    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",

    # Keep longer-range information that may sit
    # outside the GRU sequence window
    "rrp_lag_288",
    "demand_lag_288",

    # Short-term changes
    "rrp_delta_1",
    "demand_delta_1",
    "rrp_delta_3",
    "demand_delta_3",
    "rrp_delta_6",
    "demand_delta_6",
    "rrp_delta_12",
    "demand_delta_12",

    # Rolling statistics
    "rrp_roll_mean_6",
    "rrp_roll_std_6",
    "demand_roll_mean_6",

    "rrp_roll_mean_12",
    "rrp_roll_std_12",
    "demand_roll_mean_12",

    "rrp_roll_mean_36",
    "rrp_roll_std_36",
    "demand_roll_mean_36",

    "rrp_roll_mean_288",
    "rrp_roll_std_288",
    "demand_roll_mean_288",

    # Interconnectors
    "ic_n_q_mnsp1_mwflow",
    "ic_nsw1_qld1_mwflow",
    "ic_t_v_mnsp1_mwflow",
    "ic_v_s_mnsp1_mwflow",
    "ic_v_sa_mwflow",
    "ic_vic1_nsw1_mwflow",

    # P5 forecasts
    "p5_h05_rrp",
    "p5_h05_totaldemand",
    "p5_h05_availablegeneration",

    "p5_h10_rrp",
    "p5_h10_totaldemand",
    "p5_h10_availablegeneration",

    "p5_h15_rrp",
    "p5_h15_totaldemand",
    "p5_h15_availablegeneration",

    "p5_h30_rrp",
    "p5_h30_totaldemand",
    "p5_h30_availablegeneration",

    "p5_h60_rrp",
    "p5_h60_totaldemand",
    "p5_h60_availablegeneration",

    # Forecast relative to current conditions
    "p5_h05_rrp_vs_now",
    "p5_h05_demand_vs_now",
    "p5_h05_reserve_margin_mw",

    "p5_h10_rrp_vs_now",
    "p5_h10_demand_vs_now",
    "p5_h10_reserve_margin_mw",

    "p5_h15_rrp_vs_now",
    "p5_h15_demand_vs_now",
    "p5_h15_reserve_margin_mw",

    "p5_h30_rrp_vs_now",
    "p5_h30_demand_vs_now",
    "p5_h30_reserve_margin_mw",

    "p5_h60_rrp_vs_now",
    "p5_h60_demand_vs_now",
    "p5_h60_reserve_margin_mw",
]

EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=30,
    target_type="change",
    model_features=features,
    validation_fraction=0.20,
)

experiment = prepare_experiment(EXPERIMENT)
data = experiment.prepared
features = experiment.features
fit_df = experiment.fit_df
validation_df = experiment.validation_df
test_df = data.test_df

print_experiment_summary(experiment)


from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------
# Fit leakage-safe preprocessing and build model-selection tensors
# ---------------------------------------------------------

fit_df = complete_case_rows(fit_df, features)
validation_df = complete_case_rows(validation_df, features)

scaling_groups = suggested_scaling_groups(features)
scaler = NumericPreprocessor(
    asinh_columns=tuple(scaling_groups["asinh_then_standardise"]),
    standardise_columns=tuple(scaling_groups["standardise_only"]),
)

X_fit_scaled = scaler.fit_transform(fit_df[features])
X_validation_scaled = scaler.transform(validation_df[features])

target_transformer = TargetTransformer()
y_fit_scaled = target_transformer.fit_transform(fit_df["target"])
y_validation_scaled = target_transformer.transform(validation_df["target"])

X_train = torch.tensor(
    X_fit_scaled.to_numpy(),
    dtype=torch.float32
)

y_train = torch.tensor(
    y_fit_scaled,
    dtype=torch.float32
).reshape(-1, 1)

X_test = torch.tensor(
    X_validation_scaled.to_numpy(),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_validation_scaled,
    dtype=torch.float32
).reshape(-1, 1)


# ---------------------------------------------------------
# Lazy sequence datasets and model-selection data loaders
# ---------------------------------------------------------
class SequenceDataset(Dataset):
    """Create training windows on demand instead of storing every window."""

    def __init__(self, X, y, sequence_length):
        if len(X) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} rows, got {len(X)}."
            )
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, index):
        target_index = index + self.sequence_length - 1
        return self.X[index:target_index + 1], self.y[target_index]


class ContextSequenceDataset(Dataset):
    """Create evaluation windows using preceding training rows as context."""

    def __init__(self, history_X, X, sequence_length, y=None):
        context_length = sequence_length - 1
        if len(history_X) < context_length:
            raise ValueError(
                f"Need at least {context_length} history rows, got {len(history_X)}."
            )
        self.context = (
            history_X[-context_length:] if context_length else history_X[:0]
        )
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        context_length = self.sequence_length - 1
        if index < context_length:
            window = torch.cat(
                (self.context[index:], self.X[:index + 1]),
                dim=0,
            )
        else:
            window = self.X[index - context_length:index + 1]

        if self.y is None:
            return (window,)
        return window, self.y[index]


SEQUENCE_LENGTH = 36

train_dataset = SequenceDataset(X_train, y_train, SEQUENCE_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

validation_dataset = ContextSequenceDataset(
    X_train,
    X_test,
    SEQUENCE_LENGTH,
    y=y_test,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1024,
    shuffle=False,
)


# ---------------------------------------------------------

class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=1,
        output_size=1,
        dropout=0.0
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        output, _ = self.gru(x)

        # Use the final timestep
        x = output[:, -1, :]

        return self.fc(x)


model = GRU(
    input_size=len(data.features)
).to(device)

print(model)


# ---------------------------------------------------------
# Robust objective, regularized optimizer, and plateau scheduler
# ---------------------------------------------------------

criterion = nn.HuberLoss(delta=1.0)
loss_name = type(criterion).__name__.removesuffix("Loss")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
)


def evaluate_validation():
    """Evaluate in transformed space and in the original $/MWh target scale."""
    model.eval()
    total_validation_loss = 0.0
    prediction_batches = []

    with torch.no_grad():
        for X_batch, y_batch in validation_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_validation_loss += loss.item() * X_batch.size(0)
            prediction_batches.append(predictions.cpu().numpy())

    transformed_predictions = np.concatenate(prediction_batches).reshape(-1)
    predicted_target = target_transformer.inverse_transform(
        transformed_predictions
    )
    actual_target = validation_df["target"].to_numpy(dtype=float)
    errors = predicted_target - actual_target

    return {
        "loss": total_validation_loss / len(validation_dataset),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "predicted_target": predicted_target,
    }


# ---------------------------------------------------------
# Select the epoch count using chronological validation
# ---------------------------------------------------------

num_epochs = 20
early_stopping_patience = 4
best_validation_mse = float("inf")
best_epoch = 0
best_model_state = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0
learning_rate_history = []

for epoch in range(num_epochs):

    model.train()
    learning_rate_history.append(optimizer.param_groups[0]["lr"])

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Move batch to GPU / MPS / CPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)

        # Regression loss
        loss = criterion(predictions, y_batch)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    train_loss = total_loss / len(train_dataset)

    validation_metrics = evaluate_validation()

    if validation_metrics["mse"] < best_validation_mse:
        best_validation_mse = validation_metrics["mse"]
        best_epoch = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    scheduler.step(validation_metrics["mse"])
    learning_rate = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:03d}/{num_epochs} "
        f"| train transformed {loss_name}: {train_loss:.4f} "
        f"| validation transformed {loss_name}: "
        f"{validation_metrics['loss']:.4f} "
        f"| raw validation MSE: {validation_metrics['mse']:.2f} "
        f"| MAE: {validation_metrics['mae']:.2f} "
        f"| RMSE: {validation_metrics['rmse']:.2f} "
        f"| lr: {learning_rate:.1e}"
    )

    if epochs_without_improvement >= early_stopping_patience:
        print(
            f"Early stopping after {epoch + 1} epochs; "
            f"best epoch was {best_epoch}."
        )
        break

model.load_state_dict(best_model_state)
model.eval()
best_metrics = evaluate_validation()
print(
    f"Restored epoch {best_epoch} model "
    f"with validation MSE: {best_metrics['mse']:.2f}"
)


# ---------------------------------------------------------
# Report model-selection validation results
# ---------------------------------------------------------

validation_predictions = reconstruct_price_forecast(
    validation_df,
    best_metrics["predicted_target"],
    EXPERIMENT,
)
actual_validation = validation_df["target_rrp"].to_numpy(dtype=float)
persistence_predictions = validation_df["rrp"].to_numpy(dtype=float)

comparison_metrics = print_metrics(
    "BEST VALIDATION RESULTS",
    actual_validation,
    {
        "GRU": validation_predictions,
        "Persistence": persistence_predictions,
    },
)

print_improvement(
    "GRU",
    comparison_metrics["GRU"]["mae"],
    "Persistence",
    comparison_metrics["Persistence"]["mae"],
)

evaluate_aemo_rows(
    "VALIDATION: PAIRED GRU VS AEMO",
    validation_df,
    validation_predictions,
    EXPERIMENT,
    model_name="GRU",
    include_persistence=True,
)


# ---------------------------------------------------------
# Refit a fresh model on all pre-test observations
# ---------------------------------------------------------

final_train_df = complete_case_rows(data.train_df, features)
final_test_df = complete_case_rows(data.test_df, features)

print("\nRefitting final GRU on all complete pre-test rows...")
print("Final training rows:", len(final_train_df))
print("Comparable test rows:", len(final_test_df))

final_scaler = NumericPreprocessor(
    asinh_columns=tuple(scaling_groups["asinh_then_standardise"]),
    standardise_columns=tuple(scaling_groups["standardise_only"]),
)
X_final_train_scaled = final_scaler.fit_transform(final_train_df[features])
X_final_test_scaled = final_scaler.transform(final_test_df[features])

final_target_transformer = TargetTransformer()
y_final_train_scaled = final_target_transformer.fit_transform(
    final_train_df["target"]
)

X_final_train = torch.tensor(
    X_final_train_scaled.to_numpy(),
    dtype=torch.float32,
)
y_final_train = torch.tensor(
    y_final_train_scaled,
    dtype=torch.float32,
).reshape(-1, 1)
X_final_test = torch.tensor(
    X_final_test_scaled.to_numpy(),
    dtype=torch.float32,
)

final_train_dataset = SequenceDataset(
    X_final_train,
    y_final_train,
    SEQUENCE_LENGTH,
)
final_train_loader = DataLoader(
    final_train_dataset,
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)
final_test_dataset = ContextSequenceDataset(
    X_final_train,
    X_final_test,
    SEQUENCE_LENGTH,
)
final_test_loader = DataLoader(
    final_test_dataset,
    batch_size=1024,
    shuffle=False,
)

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

final_model = GRU(input_size=len(features)).to(device)
final_optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

for epoch in range(best_epoch):
    final_model.train()
    total_loss = 0.0
    for parameter_group in final_optimizer.param_groups:
        parameter_group["lr"] = learning_rate_history[epoch]

    for X_batch, y_batch in final_train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        final_optimizer.zero_grad()
        predictions = final_model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        final_optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    print(
        f"Final fit epoch {epoch + 1:03d}/{best_epoch} "
        f"| train transformed {loss_name}: "
        f"{total_loss / len(final_train_dataset):.4f}"
    )


# ---------------------------------------------------------
# Evaluate the final model once on the untouched test period
# ---------------------------------------------------------

final_model.eval()
test_prediction_batches = []

with torch.no_grad():
    for (X_batch,) in final_test_loader:
        predictions = final_model(X_batch.to(device))
        test_prediction_batches.append(predictions.cpu().numpy())

transformed_predicted_target = np.concatenate(
    test_prediction_batches
).reshape(-1)
predicted_target = final_target_transformer.inverse_transform(
    transformed_predicted_target
)
predictions = reconstruct_price_forecast(
    final_test_df,
    predicted_target,
    EXPERIMENT,
)
actual_test = final_test_df["target_rrp"].to_numpy(dtype=float)

test_metrics = print_metrics(
    "FINAL TEST RESULTS",
    actual_test,
    {
        "GRU": predictions,
        "Persistence": final_test_df["rrp"].to_numpy(dtype=float),
    },
)
print_improvement(
    "GRU",
    test_metrics["GRU"]["mae"],
    "Persistence",
    test_metrics["Persistence"]["mae"],
)

evaluate_aemo_rows(
    "FINAL TEST: PAIRED GRU VS AEMO",
    final_test_df,
    predictions,
    EXPERIMENT,
    model_name="GRU",
    include_persistence=True,
)

results = build_results_frame(
    final_test_df,
    predictions,
    predicted_target,
    EXPERIMENT,
)
print_price_regime_results(results, model_name="GRU")
print_target_diagnostics(
    final_test_df["target"].to_numpy(dtype=float),
    predicted_target,
    EXPERIMENT,
)
