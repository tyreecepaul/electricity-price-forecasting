"""Unit tests for the forecast reliability features.

    pytest model_research/test_reliability_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reliability_features import reliability_features  # noqa: E402

INTERVALS = 288  # one day of 5-minute intervals


def make_timeline(price: np.ndarray, bias_5: float = 0.0, bias_30: float = 0.0) -> pd.DataFrame:
    """AEMO forecasts that equal the realised price plus a known bias.

    The forecast issued at t for t + 5 min is p5_h05_rrp[t], so it is the price
    one row later; the 30-minute forecast is the price six rows later.
    """
    times = pd.date_range("2025-01-01 00:05", periods=len(price), freq="5min")
    series = pd.Series(price, index=times)
    return pd.DataFrame({
        "settlement_date": times,
        "rrp": price,
        "p5_h05_rrp": series.shift(-1).to_numpy() + bias_5,
        "p5_h30_rrp": series.shift(-6).to_numpy() + bias_30,
    })


def test_builds_18_named_features_for_every_interval():
    timeline = make_timeline(np.full(INTERVALS, 100.0))

    features = reliability_features(timeline)

    expected = [f"reliability_rrp_err{m:02d}_{kind}_{w}" for m in (5, 30) for w in (12, 72, 288)
                for kind in ("mean", "abs")]
    assert len(features.columns) == 18
    assert set(expected) <= set(features.columns)
    assert {"reliability_false_alarms05_288", "reliability_missed30_288"} <= set(features.columns)
    assert features.index.equals(pd.DatetimeIndex(timeline["settlement_date"]))


def test_errors_match_a_known_forecast_bias():
    # AEMO always forecasts $10 too high at 5 minutes and $5 too low at 30 minutes.
    timeline = make_timeline(np.linspace(50, 150, INTERVALS), bias_5=10.0, bias_30=-5.0)

    row = reliability_features(timeline).iloc[100]

    assert row["reliability_rrp_err05_mean_12"] == pytest.approx(-10.0)
    assert row["reliability_rrp_err05_abs_72"] == pytest.approx(10.0)
    assert row["reliability_rrp_err30_mean_12"] == pytest.approx(5.0)
    assert row["reliability_rrp_err30_abs_12"] == pytest.approx(5.0)


def test_counts_warnings_false_alarms_and_missed_spikes():
    price = np.full(INTERVALS, 100.0)
    price[150] = 400.0  # a spike
    timeline = make_timeline(price)
    timeline.loc[99, "p5_h05_rrp"] = 500.0  # forecast for row 100 says $500, but the price stays $100
    timeline.loc[149, "p5_h05_rrp"] = 100.0  # forecast for row 150 says $100, but the price is $400

    last = reliability_features(timeline).iloc[-1]

    assert last["reliability_warnings05_288"] == 1
    assert last["reliability_false_alarms05_288"] == 1
    assert last["reliability_missed05_288"] == 1


def test_values_never_depend_on_later_data():
    rng = np.random.default_rng(0)
    timeline = make_timeline(100 + rng.normal(0, 30, 3 * INTERVALS), bias_5=3.0)
    cut = 2 * INTERVALS

    full = reliability_features(timeline)
    past_only = reliability_features(timeline.iloc[:cut])

    pd.testing.assert_frame_equal(past_only, full.iloc[:cut])


def test_missing_forecast_column_is_reported():
    timeline = make_timeline(np.full(INTERVALS, 100.0)).drop(columns="p5_h30_rrp")

    with pytest.raises(KeyError, match="p5_h30_rrp"):
        reliability_features(timeline)
