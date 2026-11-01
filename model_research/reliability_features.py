"""Forecast reliability features: how accurate AEMO's P5MIN price forecast has been recently.

For each of AEMO's 5- and 30-minute-ahead price forecasts, the forecast issued for
the current interval is compared with the price that was realised. Nine features
are computed per forecast horizon, 18 in total:

* mean error and mean absolute error over the last 1 h, 6 h and 24 h;
* counts over the last 24 h of $300+ forecasts, false alarms ($300+ forecast,
  price below) and missed spikes ($300+ price, no $300+ forecast).

The value at interval t only uses forecasts issued before t and prices up to t,
so every feature is known at forecast time. The features are appended to the
prepared experiment; the shared pipeline itself is not changed.

    from reliability_features import reliability_experiment
    experiment = reliability_experiment("VIC1", horizon=5)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repository root, so the pipeline is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.experiment_pipeline import (  # noqa: E402
    ExperimentConfig,
    ExperimentData,
    chronological_validation_split,
    prepare_experiment,
)

FORECAST_HORIZONS = (5, 30)  # AEMO forecast horizons that are scored, in minutes
WINDOWS = (12, 72, 288)  # 1 h, 6 h and 24 h of 5-minute intervals
SPIKE_LEVEL = 300.0  # $/MWh level for the warning, false-alarm and missed-spike counts


def reliability_features(timeline: pd.DataFrame, level: float = SPIKE_LEVEL) -> pd.DataFrame:
    """The 18 features for every interval of `timeline`, indexed by settlement_date.

    `timeline` needs the columns settlement_date, rrp, p5_h05_rrp and p5_h30_rrp,
    on a 5-minute grid.
    """
    timeline = timeline.drop_duplicates("settlement_date").set_index("settlement_date").sort_index()
    times, price = timeline.index, timeline["rrp"]
    features = pd.DataFrame(index=times)
    for minutes in FORECAST_HORIZONS:
        column = f"p5_h{minutes:02d}_rrp"
        if column not in timeline:
            raise KeyError(f"timeline is missing {column}")
        # AEMO's forecast for this interval, issued `minutes` earlier.
        issued = pd.Series(timeline[column].reindex(times - pd.to_timedelta(minutes, unit="min")).to_numpy(),
                           index=times)
        error = price - issued
        for window in WINDOWS:
            features[f"reliability_rrp_err{minutes:02d}_mean_{window}"] = error.rolling(window, min_periods=1).mean()
            features[f"reliability_rrp_err{minutes:02d}_abs_{window}"] = (
                error.abs().rolling(window, min_periods=1).mean())
        warned = issued >= level
        features[f"reliability_warnings{minutes:02d}_288"] = warned.astype(float).rolling(288, min_periods=1).sum()
        features[f"reliability_false_alarms{minutes:02d}_288"] = (
            (warned & (price < level)).astype(float).rolling(288, min_periods=1).sum())
        features[f"reliability_missed{minutes:02d}_288"] = (
            (~warned & issued.notna() & (price >= level)).astype(float).rolling(288, min_periods=1).sum())
    return features


def baseline_experiment(region: str = "VIC1", horizon: int = 5) -> ExperimentData:
    """The standard experiment (main.py's settings) for one region and horizon."""
    config = ExperimentConfig(
        region=region,
        horizon_minutes=horizon,
        target_type="change",
        model_features="all",
        validation_fraction=0.20,
    )
    return prepare_experiment(config)


def add_reliability_features(experiment: ExperimentData) -> ExperimentData:
    """The same experiment with the 18 features appended to every frame and to the input list."""
    prepared = experiment.prepared
    extra = reliability_features(pd.concat([prepared.train_timeline, prepared.test_timeline]))

    def with_features(frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(extra, left_on="settlement_date", right_index=True, how="left")
        merged.index = frame.index
        return merged

    features = [*prepared.features, *extra.columns]
    train_df, test_df = with_features(prepared.train_df), with_features(prepared.test_df)
    prepared = replace(
        prepared,
        X_train=train_df[features].copy(),
        X_test=test_df[features].copy(),
        train_df=train_df,
        test_df=test_df,
        features=features,
        train_timeline=with_features(prepared.train_timeline),
        test_timeline=with_features(prepared.test_timeline),
    )
    fit_df, validation_df = chronological_validation_split(prepared.train_df, experiment.config)
    return ExperimentData(experiment.config, prepared, fit_df, validation_df, prepared.test_df)


def reliability_experiment(region: str = "VIC1", horizon: int = 5) -> ExperimentData:
    """The standard experiment plus the forecast reliability features."""
    return add_reliability_features(baseline_experiment(region, horizon))
