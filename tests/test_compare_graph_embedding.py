"""Paired with/without-embedding comparison built from saved run artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import compare_graph_embedding as compare

TIMES = pd.date_range("2025-01-01", periods=288 * 6, freq="5min")


def write_run(root, arm, seed, error, *, model="mlp", times=TIMES, invalid=()):
    """Save a run whose forecast misses the actual price by ``error``."""
    actual = np.where(np.arange(len(times)) % 50 == 0, 500.0, 60.0)
    cell = compare.cell_directory(root, "VIC1", 30, arm, seed) / "run"
    directory = cell / model
    directory.mkdir(parents=True)
    valid = np.ones(len(times), dtype=bool)
    valid[list(invalid)] = False
    pd.DataFrame(
        {
            "region": "VIC1",
            "settlement_date": times,
            "rrp": actual + 10.0,
            "target_rrp": actual,
            "p5_h30_rrp": actual + 4.0,
            "prediction_rrp": actual + error,
            "valid": valid,
        }
    ).to_csv(directory / "predictions.csv", index=False)
    (directory / "model.json").write_text(json.dumps({"validation_score": abs(error)}))
    manifest = {}
    if arm != "baseline":
        manifest["graph_embedding_diagnostics"] = {
            "validation_mae": {"VIC1": 3.0},
            "zero_forecast_mae": {"VIC1": 5.0},
            "best_epoch": 2,
        }
    (cell / "experiment.json").write_text(json.dumps(manifest))
    (cell / "summary.csv").write_text("model\n")


def test_summary_pairs_arms_on_shared_rows(tmp_path):
    for seed, error in ((42, 6.0), (1, 8.0)):
        write_run(tmp_path, "baseline", seed, error)
    write_run(tmp_path, "gnn", 42, 2.0, invalid=[0, 1])
    write_run(tmp_path, "gnn", 1, 4.0)
    write_run(tmp_path, "no_graph", 42, 5.0)

    tables = compare.summarise(30, tmp_path)

    overall = tables["overall"].set_index("arm")
    assert overall.loc["baseline", "mae"] == pytest.approx(7.0)
    assert overall.loc["baseline", ["mae_min", "mae_max"]].tolist() == [6.0, 8.0]
    assert overall.loc["gnn", "mae"] == pytest.approx(3.0)
    assert overall.loc["gnn", "mae_vs_baseline_pct"] == pytest.approx(-400 / 7)
    assert overall.loc["gnn", "mae_vs_no_graph_pct"] == pytest.approx(-40.0)
    assert overall.loc["persistence", "mae"] == pytest.approx(10.0)
    assert overall.loc["aemo", "mae"] == pytest.approx(4.0)
    # Rows invalid in any arm are dropped from every arm.
    assert (overall["n_rows"] == len(TIMES) - 2).all()

    paired = tables["paired"].set_index("comparison")
    gain = paired.loc["gnn_vs_baseline"]
    assert gain["mae_difference"] == pytest.approx(-4.0)
    assert gain["interval_excludes_zero"] and gain["exceeds_seed_spread"]
    assert gain["reference_seed_spread"] == pytest.approx(2.0)
    assert gain["days"] == 6
    assert np.isnan(paired.loc["gnn_vs_no_graph", "reference_seed_spread"])

    regimes = tables["regimes"]
    assert set(regimes["regime"]) == {"Normal ($0-$300/MWh)", "High ($300-$1,000/MWh)"}
    assert regimes.groupby("regime")["n_rows"].first().sum() == len(TIMES) - 2

    assert set(tables["encoder"]["arm"]) == {"gnn", "no_graph"}
    assert (tmp_path / "overall_30min.csv").exists()


def test_arms_with_different_test_rows_are_rejected(tmp_path):
    write_run(tmp_path, "baseline", 42, 6.0)
    write_run(tmp_path, "gnn", 42, 2.0, times=TIMES[1:])

    with pytest.raises(ValueError, match="different test rows"):
        compare.summarise(30, tmp_path)


def test_day_block_bootstrap_is_seeded_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    difference = pd.Series(rng.normal(-1.0, 3.0, len(TIMES)), index=TIMES)

    estimate, low, high = compare.day_block_bootstrap(difference)

    assert estimate == pytest.approx(difference.mean())
    assert low < estimate < high
    assert compare.day_block_bootstrap(difference) == (estimate, low, high)


def test_run_skips_finished_cells_and_repeats_seeds_for_networks_only(
    tmp_path, monkeypatch
):
    import main

    calls = []

    def fake_main(models, *, config, output_directory, graph_embeddings):
        assert graph_embeddings == {}
        calls.append((list(models), config.random_seed, config.graph_embedding))
        for model in models:
            directory = output_directory / "run" / model
            directory.mkdir(parents=True)
            (directory / "predictions.csv").write_text("")
        (output_directory / "run" / "summary.csv").write_text("")

    monkeypatch.setattr(main, "main", fake_main)
    compare.run(["VIC1"], ["xgboost", "mlp"], [42, 1], 30, tmp_path)

    arms = len(compare.ARMS)
    assert [models for models, seed, _ in calls if seed == 42] == [["xgboost", "mlp"]] * arms
    assert [models for models, seed, _ in calls if seed == 1] == [["mlp"]] * arms
    assert [embedding for _, seed, embedding in calls if seed == 42] == list(
        compare.ARMS.values()
    )

    compare.run(["VIC1"], ["xgboost", "mlp"], [42, 1], 30, tmp_path)
    assert len(calls) == 2 * arms

    # GRU is left out of graph comparisons unless it is named.
    calls.clear()
    compare.run(["NSW1"], None, [42], 30, tmp_path)
    assert {tuple(models) for models, _, _ in calls} == {("xgboost", "lstm")}


def test_plain_arms_switch_off_both_encoder_additions():
    for name in ("gnn", "no_graph"):
        full, plain = compare.ARMS[name], compare.ARMS[f"{name}_plain"]
        assert full.edge_features and full.spread_loss_weight > 0
        assert not plain.edge_features and plain.spread_loss_weight == 0
        assert full.adjacency_mode == plain.adjacency_mode
    assert ("gnn", "gnn_plain") in compare.COMPARISONS
