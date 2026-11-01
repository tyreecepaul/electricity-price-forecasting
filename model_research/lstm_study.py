"""Compare LSTM settings with the original LSTM and the MLP.

Every run uses main.py's experiment (30 minutes ahead, VIC1 unless --region is
given) and follows its steps: the number of epochs is chosen on validation data,
the model is refit on all pre-test data, and the test year is forecast once.
Results are saved under results/lstm_study/, and finished runs are skipped.

    python model_research/lstm_study.py --runs lstm_2x64_dropout --seeds 42 1 2
    python model_research/lstm_study.py --summary
    python model_research/lstm_study.py --summary --compact --runs mlp lstm_original lstm_2x64_dropout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repository root, so main.py and the pipeline are importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_model.lstm import LSTMModel  # noqa: E402
from src.final_model.mlp import MLPModel  # noqa: E402
from src.layers.lstm import LSTMNetwork  # noqa: E402
from main import EXPERIMENT, MODEL_REGISTRY  # noqa: E402
from pipeline.evaluation import evaluate  # noqa: E402
from pipeline.experiment_pipeline import (  # noqa: E402
    aemo_forecast_values,
    prepare_experiment,
    reconstruct_price_forecast,
)
from pipeline.model_data import final_training_data, predict_test  # noqa: E402
from pipeline.neural import TorchForecastModel  # noqa: E402
from pipeline.tuning import search_hyperparameters  # noqa: E402

RESULTS = HERE / "results" / "lstm_study"
MLP = next(grid for model, grid in MODEL_REGISTRY if model is MLPModel)
# The LSTM settings main.py had before this study, kept so the comparison can be rerun.
ORIGINAL_LSTM = {
    "sequence_length": [12],
    "hidden_size": [16],
    "num_layers": [1],
    "dropout": [0.0],
    "learning_rate": [0.001],
    "weight_decay": [0.0001],
    "batch_size": [512],
    "prediction_batch_size": [2048],
    "max_epochs": [10],
    "patience": [2],
    "min_delta": [0.001],
    "loss": ["huber"],
    "huber_delta": [1.0],
}
# The settings now in main.py: two layers of 64 units with dropout between
# them, clipped gradients, and room to train for longer.
LARGER_LSTM = {
    **ORIGINAL_LSTM,
    "hidden_size": [64],
    "num_layers": [2],
    "dropout": [0.25],
    "gradient_clip_norm": [1.0],
    "max_epochs": [20],
    "patience": [3],
}
LOW_DROPOUT = {**LARGER_LSTM, "dropout": [0.1]}
# Shrink the learning rate after every epoch so the weights settle before
# training stops, and allow more epochs for that to happen.
DECAY = {
    "scheduler_step_size": [1],
    "scheduler_gamma": [0.6],
    "max_epochs": [12],
    "patience": [4],
}


class SpacedLSTMNetwork(LSTMNetwork):
    """An LSTM that reads every ``spacing``-th row of its window, then the latest ``recent`` rows in full.

    With ``recent=0`` the spaced rows always end on the latest row.
    """

    def __init__(self, *arguments, spacing: int, recent: int = 0, **settings) -> None:
        super().__init__(*arguments, **settings)
        self.spacing, self.recent = spacing, recent

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        older = inputs[:, :inputs.shape[1] - self.recent, :]
        first = (older.shape[1] - 1) % self.spacing
        rows = torch.cat((older[:, first::self.spacing, :], inputs[:, inputs.shape[1] - self.recent:, :]), dim=1)
        return super().forward(rows)


def spaced_lstm(spacing: int, recent: int = 0) -> type[LSTMModel]:
    """The LSTM model reading one row in every ``spacing`` of a longer window."""

    class SpacedLSTMModel(LSTMModel):
        model_name = f"LSTM, every {spacing} rows" + (f" then the last {recent}" if recent else "")

        def _build_network(self, input_size: int) -> torch.nn.Module:
            return SpacedLSTMNetwork(input_size, self.hidden_size, self.num_layers, dropout=self.dropout,
                                     spacing=spacing, recent=recent)

    return SpacedLSTMModel


# name: (model, settings, use the forecast reliability features)
RUNS = {
    "mlp": (MLPModel, MLP, False),
    "lstm_original": (LSTMModel, ORIGINAL_LSTM, False),
    "lstm_2x64_dropout": (LSTMModel, LARGER_LSTM, False),
    # Other amounts of dropout between the two layers.
    "lstm_2x64": (LSTMModel, LOW_DROPOUT, False),
    "lstm_2x64_dropout40": (LSTMModel, {**LARGER_LSTM, "dropout": [0.4]}, False),
    "lstm_2x64_dropout50": (LSTMModel, {**LARGER_LSTM, "dropout": [0.5]}, False),
    # Wider and deeper.
    "lstm_2x128": (LSTMModel, {**LOW_DROPOUT, "hidden_size": [128]}, False),
    "lstm_3x64": (LSTMModel, {**LOW_DROPOUT, "num_layers": [3]}, False),
    # A learning rate that shrinks every epoch.
    "lstm_2x64_decay": (LSTMModel, {**LOW_DROPOUT, **DECAY}, False),
    "lstm_2x64_dropout_decay": (LSTMModel, {**LARGER_LSTM, **DECAY}, False),
    "lstm_original_decay": (LSTMModel, {**ORIGINAL_LSTM, **DECAY}, False),
    "mlp_decay": (MLPModel, {**MLP, **DECAY}, False),
    # With the 18 forecast reliability features as extra inputs.
    "lstm_2x64_reliability": (LSTMModel, LOW_DROPOUT, True),
    "lstm_2x64_dropout_reliability": (LSTMModel, LARGER_LSTM, True),
    # A longer look-back: 12 rows spread over 3 or 6 hours, and the latest hour
    # in full plus the five hours before it every 30 minutes (22 rows).
    "lstm_2x64_dropout_3h_spaced": (spaced_lstm(3), {**LARGER_LSTM, "sequence_length": [34]}, False),
    "lstm_2x64_dropout_6h_spaced": (spaced_lstm(6), {**LARGER_LSTM, "sequence_length": [67]}, False),
    "lstm_2x64_dropout_6h_hybrid": (spaced_lstm(6, recent=12), {**LARGER_LSTM, "sequence_length": [72]}, False),
}


def run(name: str, seed: int, experiments: dict, folder: Path, region: str) -> dict:
    """Search, refit and test one named run, or load it if it is already saved."""
    path = folder / f"{name}_seed{seed}"
    if (path / "run.json").exists():
        return json.loads((path / "run.json").read_text())
    model_class, settings, reliability = RUNS[name]
    if reliability not in experiments:
        extra_inputs = {"forecast_reliability": True} if reliability else {}
        experiments[reliability] = prepare_experiment(replace(EXPERIMENT, region=region, **extra_inputs))
    experiment = experiments[reliability]

    started = time.time()
    search = search_hyperparameters(model_class, {**settings, "random_seed": [seed]}, experiment)
    searched = time.time()
    model = model_class(**search.best_config)
    model.train(*final_training_data(model, experiment.prepared))
    change = predict_test(model, experiment.prepared)
    test = experiment.test_df
    forecast = reconstruct_price_forecast(test, change, experiment.config)
    result = evaluate(model, test["target_rrp"].to_numpy(), forecast)

    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "settlement_date": test["settlement_date"].to_numpy(),
        "rrp": test["rrp"].to_numpy(),
        "target_rrp": test["target_rrp"].to_numpy(),
        "aemo": aemo_forecast_values(test, experiment.config),
        "forecast": forecast,
    }).to_csv(path / "test.csv", index=False, float_format="%.5f")
    summary = {
        "name": name,
        "model": model_class.model_name,
        "seed": seed,
        "reliability_features": reliability,
        "inputs": len(experiment.features),
        "parameters": sum(weights.numel() for weights in model.network.parameters()),
        "best_config": search.best_config,
        "validation_mae": search.best_score,
        "validation_rmse": float(search.trials.iloc[0]["rmse"]),
        "test_mae": result.metrics["mae"],
        "test_rmse": result.metrics["rmse"],
        "search_minutes": (searched - started) / 60,
        "refit_minutes": (time.time() - searched) / 60,
    }
    (path / "run.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarise(folder: Path, names: list[str] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    """One row per run: validation and test error averaged over the saved seeds, or over ``seeds``."""
    by_name: dict[str, list[dict]] = {}
    benchmarks = {}
    for path in sorted(folder.glob("*_seed*/run.json")):
        summary = json.loads(path.read_text())
        if (names and summary["name"] not in names) or (seeds and summary["seed"] not in seeds):
            continue
        test = pd.read_csv(path.parent / "test.csv")
        error = (test["forecast"] - test["target_rrp"]).abs()
        summary["high_mae"] = error[test["target_rrp"] >= 300].mean()
        by_name.setdefault(summary["name"], []).append(summary)
        for name, column in (("persistence", "rrp"), ("AEMO P5MIN", "aemo")):
            miss = (test[column] - test["target_rrp"]).abs()
            benchmarks[name] = {"test_mae": miss.mean(), "test_rmse": (miss**2).mean() ** 0.5,
                                "test_mae_300_plus": miss[test["target_rrp"] >= 300].mean()}
    rows = {}
    for name, saved in by_name.items():
        saved = sorted(saved, key=lambda summary: (summary["seed"] != EXPERIMENT.random_seed, summary["seed"]))
        rows[name] = {
            "seeds": len(saved),
            "parameters": saved[0]["parameters"],
            "validation_mae": sum(s["validation_mae"] for s in saved) / len(saved),
            "test_mae": sum(s["test_mae"] for s in saved) / len(saved),
            "test_mae_by_seed": " / ".join(f"{s['test_mae']:.2f}" for s in saved),
            "test_rmse": sum(s["test_rmse"] for s in saved) / len(saved),
            "test_mae_300_plus": sum(s["high_mae"] for s in saved) / len(saved),
        }
    table = pd.DataFrame.from_dict(rows, orient="index").sort_values("validation_mae")
    return pd.concat([table, pd.DataFrame.from_dict(benchmarks, orient="index")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", default=[], choices=list(RUNS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        help=f"seeds to run (default {EXPERIMENT.random_seed}) or to include in the summary (default all)")
    parser.add_argument("--region", default=EXPERIMENT.region)
    parser.add_argument("--summary", action="store_true", help="print the saved results as one table")
    parser.add_argument("--compact", action="store_true", help="a narrow summary that fits in a chat message")
    parser.add_argument("--progress", action="store_true", help="show a progress bar for every training epoch")
    parser.add_argument("--threads", type=int, help="CPU threads to use, to share the machine between runs")
    arguments = parser.parse_args()
    if arguments.threads:
        torch.set_num_threads(arguments.threads)
    if arguments.progress:
        TorchForecastModel.show_progress = True

    folder = RESULTS / f"{arguments.region}_{EXPERIMENT.horizon_minutes}min"
    experiments: dict = {}
    for name in arguments.runs:
        for seed in arguments.seeds or [EXPERIMENT.random_seed]:
            summary = run(name, seed, experiments, folder, arguments.region)
            if not arguments.summary:
                print(f"{name} seed {seed}: {summary['parameters']:,} parameters, "
                      f"{summary['best_config']['max_epochs']} epochs, validation MAE {summary['validation_mae']:.3f}, "
                      f"test MAE {summary['test_mae']:.3f} "
                      f"({summary['search_minutes'] + summary['refit_minutes']:.0f} min)", flush=True)
    if arguments.summary and arguments.compact:
        table = summarise(folder, arguments.runs, arguments.seeds)
        width = max(map(len, table.index)) + 2
        print(f"{arguments.region}, {EXPERIMENT.horizon_minutes} min ahead, MAE in $/MWh")
        print(f"{'model':<{width}}{'valid.':>7}{'test':>7}   test by seed")
        for name, row in table.iterrows():
            validation = "" if pd.isna(row["validation_mae"]) else f"{row['validation_mae']:.2f}"
            by_seed = "" if pd.isna(row["test_mae_by_seed"]) else row["test_mae_by_seed"]
            print(f"{name:<{width}}{validation:>7}{row['test_mae']:>7.2f}   {by_seed}")
    elif arguments.summary:
        print(f"{arguments.region}, {EXPERIMENT.horizon_minutes} minutes ahead, test year, sorted by validation MAE")
        whole = {name: lambda value: "" if pd.isna(value) else f"{int(value):,}" for name in ("seeds", "parameters")}
        table = summarise(folder, arguments.runs, arguments.seeds)
        print(table.to_string(float_format=lambda value: f"{value:,.3f}", na_rep="", formatters=whole))
