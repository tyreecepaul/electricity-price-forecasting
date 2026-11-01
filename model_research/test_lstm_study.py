"""Unit tests for the LSTM study.

    pytest model_research/connor/test_lstm_study.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lstm_study  # noqa: E402
from lstm_study import (  # noqa: E402
    LARGER_LSTM,
    ORIGINAL_LSTM,
    RUNS,
    SpacedLSTMNetwork,
    summarise,
)
from main import MODEL_REGISTRY  # noqa: E402


def rows_read(spacing: int, recent: int, window: int) -> list[int]:
    """Which rows of a window the spaced LSTM passes to its LSTM layers."""
    network = SpacedLSTMNetwork(1, hidden_size=4, spacing=spacing, recent=recent)
    seen = []
    network.lstm.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0][0, :, 0].int().tolist()))
    network(torch.arange(window, dtype=torch.float32).reshape(1, window, 1))
    return seen[0]


def test_larger_lstm_is_the_one_registered_in_main():
    registered = next(grid for model, grid in MODEL_REGISTRY if model is lstm_study.LSTMModel)

    assert registered == LARGER_LSTM
    assert RUNS["lstm_2x64_dropout"] == (lstm_study.LSTMModel, LARGER_LSTM, False)


def test_larger_lstm_changes_only_size_dropout_and_training_length():
    changed = {key for key in LARGER_LSTM if LARGER_LSTM[key] != ORIGINAL_LSTM.get(key)}

    assert changed == {"hidden_size", "num_layers", "dropout", "gradient_clip_norm", "max_epochs", "patience"}
    assert LARGER_LSTM["sequence_length"] == ORIGINAL_LSTM["sequence_length"] == [12]


def test_every_run_uses_settings_its_model_accepts():
    for name, (model, settings, _) in RUNS.items():
        model(**{key: values[0] for key, values in settings.items()}, device="cpu")


@pytest.mark.parametrize(
    ("spacing", "recent", "window", "expected"),
    [
        (3, 0, 34, list(range(0, 34, 3))),
        (6, 0, 67, list(range(0, 67, 6))),
        (6, 12, 72, [*range(5, 60, 6), *range(60, 72)]),
    ],
)
def test_spaced_lstm_reads_the_intended_rows(spacing, recent, window, expected):
    rows = rows_read(spacing, recent, window)

    assert rows == expected
    assert rows[-1] == window - 1  # always ends on the latest row


def test_summary_averages_seeds_and_adds_the_benchmarks(tmp_path):
    for seed, forecast in ((42, [11.0, 21.0]), (1, [13.0, 23.0])):
        folder = tmp_path / f"lstm_2x64_dropout_seed{seed}"
        folder.mkdir()
        pd.DataFrame({"settlement_date": ["2026-01-01 00:05", "2026-01-01 00:10"], "rrp": [9.0, 19.0],
                      "target_rrp": [10.0, 400.0], "aemo": [10.0, 390.0], "forecast": forecast}).to_csv(
            folder / "test.csv", index=False)
        summary = {"name": "lstm_2x64_dropout", "seed": seed, "parameters": 71489, "validation_mae": 20.0 + seed % 2,
                   "test_mae": (abs(forecast[0] - 10) + abs(forecast[1] - 400)) / 2, "test_rmse": 1.0}
        (folder / "run.json").write_text(json.dumps(summary))

    table = summarise(tmp_path)

    row = table.loc["lstm_2x64_dropout"]
    assert row["seeds"] == 2
    assert row["validation_mae"] == pytest.approx(20.5)
    assert row["test_mae"] == pytest.approx((190.0 + 190.0) / 2)
    assert row["test_mae_by_seed"] == "190.00 / 190.00"
    assert row["test_mae_300_plus"] == pytest.approx((379.0 + 377.0) / 2)
    assert table.loc["persistence", "test_mae"] == pytest.approx((1.0 + 381.0) / 2)
    assert table.loc["AEMO P5MIN", "test_mae"] == pytest.approx(5.0)
    assert list(summarise(tmp_path, seeds=[1])["seeds"].dropna()) == [1]
