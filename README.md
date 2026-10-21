# Forecast-Informed Electricity Price Forecasting

This STAT3007/7007 project forecasts five-minute Regional Reference Prices
(RRP) in Australia's National Electricity Market. It combines historical AEMO
dispatch data, interconnector conditions, and aligned P5MIN forecasts in a
shared experiment pipeline for tree-based and deep-learning models.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The first run may download and process several years of AEMO data. Raw data and
prepared experiment data are cached for later runs.

Run the test suite with:

```bash
python -m pytest -q
```

## Running an Experiment

Configure the experiment and model grid in `main.py`:

```bash
# All runs forecast the complete 5-to-60-minute curve; GNN features default on.
python main.py --models mlp gru lstm tcn transformer --device xpu
python main.py --models xgboost arima
```

Deep models use a 12-value output head, XGBoost fits one
horizon-conditioned model, and ARIMA projects one fitted price model across the
12 leads. Targets are matched by timestamp and the 60-minute embargo applies
to the complete vector. All models score the same complete forecast origins.

```python
EXPERIMENT = ExperimentConfig(
    region="NSW1",
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
)

MODEL_REGISTRY = [
    (
        XGBoostModel,
        {
            "max_depth": [3, 6],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [4, 8],
            "max_estimators": [500, 1000],
            "random_seed": [42],
        },
    ),
    (
        MLPModel,
        {
            "hidden_sizes": [(128, 64, 32)],
            "learning_rate": [1e-3],
            "weight_decay": [1e-4],
            "dropout": [(0.2, 0.15, 0.0)],
            "batch_size": [256],
            "max_epochs": [5],
            "loss": ["huber", "mse", "mae"],
            "huber_delta": [1.0],
            "random_seed": [42],
        },
    ),
    (
        GRUModel,
        {
            "sequence_length": [12],
            "hidden_size": [16],
            "num_layers": [1],
            "dropout": [0.0],
            "learning_rate": [1e-3],
            "weight_decay": [1e-4],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [10],
            "patience": [2],
            "loss": ["huber"],
            "huber_delta": [1.0],
            "random_seed": [42],
        },
    ),
]
```

Sequence construction is shared in `pipeline/sequences.py`. Its lazy training
and context datasets can be reused by GRU, LSTM, TCN, and Transformer wrappers
without materialising every overlapping window. The array-based
`make_sequences` preprocessing helper also accepts `history=` when an eager
sequence batch is useful for analysis or another deep-learning framework.

The common neural lifecycle lives in `pipeline/neural.py`: fitted imputation
and scaling, target transforms, deterministic seeding, batched training and
inference, checkpoint selection, optional gradient clipping, StepLR scheduling,
and CUDA mixed precision. MLP, GRU, and LSTM use it, and future sequence
architectures such as Transformers can reuse the same lifecycle.

### Temporal data and model extensions

`PreparedModelData.train_timeline` and `test_timeline` retain all observed
feature rows, including rows without targets. The existing `train_df` and
`test_df` contain eligible supervised/scoring rows. During validation, the
forecast-horizon embargo masks training labels but retains preceding features
as sequence context. Neural preprocessing statistics are fitted on eligible
training rows only.

`pipeline/model_data.py` selects the appropriate representation for each model.
Sequence inputs carry `(region, settlement_date)` in their index, not in their
numeric feature columns. Production sequence models reject mixed regions,
duplicate/reversed timestamps and missing intervals. They do not interpolate
missing observations. Unindexed frames still support positional experiments,
but timestamp checks require the metadata supplied by the pipeline adapters.

For separate prediction calls, explicitly pass recent observations:

```python
predictions = model.predict(current_features, history=recent_features)
```

History must end immediately before the current frame and contain at least
`sequence_length - 1` observations. Without explicit history, the stored
training tail is used and checked against current timestamps. History is not
mutated by prediction calls. Both frames must use the same temporal index.

To add another neural architecture, subclass `TorchForecastModel`, implement
`_build_network(input_size)`, and set `requires_sequence = True` plus a
`sequence_length` for temporal models. Forward passes return
`(batch, self.output_size)`. The shared wrapper returns `(N,)` for one target
or `(N, 12)` for the hourly curve, on the configured target scale.
New row-based models can continue implementing `Model.train()` and `predict()`.
Register a model factory and its hyperparameter grid in `main.py`.

The interface forecasts one region at a time. Prepared feature columns,
including learned embeddings when available, are separate from target columns;
future actuals must never be used as known-future inputs.

### Reproducible model comparisons

```bash
python main.py --models gru --folds 3 --metric rmse --seed 42
```

Omitting `--models` runs all models in `MODEL_REGISTRY`.
`--folds 1` uses the usual holdout; larger counts use expanding training windows
and disjoint validation blocks, each of size approximately
`validation_fraction * training_rows`. Leave enough initial training data for
the lookback and embargo. Scores pool the validation errors across folds. The
selection metric also controls neural checkpoint selection. Final neural refits
use the median selected epoch across folds, even when `max_epochs` was omitted
from the grid. Seeds come from `ExperimentConfig`/`--seed` unless a candidate
explicitly sets its own `random_seed`.

Runs print one compact leaderboard with MAE, RMSE, baseline improvements and
fit time (validation search plus final refit, excluding SHAP). Sample counts
are printed once. Add `--verbose` for full SHAP, timestep, price-regime and
per-horizon tables.

```bash
python main.py --models mlp gru xgboost --plot
python main.py --plot --plot-details
python main.py --verbose
```

`--plot` displays MAE/RMSE curves for all models, six randomly sampled test
price reconstructions, and a separate SHAP view for every explained deep model.
Each reconstruction compares all models, actual prices, persistence and available
AEMO forecasts at the same origin. Sampling uses `--seed` for reproducibility.
The MLP uses a ranked horizontal bar chart with a forecast-horizon slider;
its top features are selected separately for each target. Sequence models also
have an overview of the top 20 features across all forecast horizons.
Set `--shap-top-features 30` to show more, or `--shap-top-features 0`
to show every feature. Matching plot types share colour scales across models.
All GradientSHAP plots show **percentages of total absolute attribution per
forecast horizon**, using every feature and sequence step in the denominator
before selecting the top features. These describe attribution shares, not
accuracy or explained variance. An all-zero attribution horizon stays zero.
Price curves and MAE/RMSE retain their $/MWh units.

Sequence models also show sequence-step importance across forecast horizons,
and a feature-by-sequence heatmap with a forecast-horizon slider. This separates
minutes **ahead** (the forecast lead) from minutes **before** the forecast origin
(observed context). Every sequence step is drawn; long windows thin tick labels
only. Hovering a cell shows its exact value in Matplotlib's toolbar. The top
features are selected and reordered by their total importance for the chosen
horizon; the colour scale remains fixed when switching horizons.

```bash
python main.py --models gru lstm --plot --shap-top-features 30
python main.py --models tcn --plot --shap-3d
```

`--shap-3d` optionally adds rotatable surfaces of forecast horizon versus sequence
step, with height showing percentage attribution summed over features.
The 2D views retain the individual feature details. No additional dependency is
needed. The saved feature-by-sequence image shows the longest explained horizon;
CSV summaries include every explained horizon and sequence step.
The console also compares P5 and top-five feature shares, the latest input row's
share, the most recent 30 minutes of sequence rows, and the sequence lookback covering 90%
of attribution (shortest/longest forecast targets; all targets with `--verbose`).
For five-minute data, the recent-30-minute share covers lags 0, 5, ..., 25.
These sequence diagnostics measure input-row positions. A feature such as
`rrp_lag_288` already represents older information even in the latest row;
engineered feature lookbacks are additional to the sequence lookback.
Saved CSVs retain raw magnitudes alongside percentage and concentration summaries.
`--plot-details` adds price-regime improvements. It
requires `--plot` or `--save`. Plots use the same evaluation masks as the tables;
missing AEMO horizons remain gaps. With `--skip-shap`, the SHAP heatmap is omitted.

Runs do not save artifacts by default. Add `--save` to save
configuration, dependency versions, data provenance, trial/fold
scores, per-origin predictions, training history and fitted CPU model bundles
(including scalers and history) under `reports/runs/<unique-run-id>/`.
Use `--save --output-dir` to choose a save location. Load
your own model bundles with `joblib.load`; do not load untrusted joblib files.
Saved runs also include `leaderboard.csv`, `errors_by_horizon.png`, and model
SHAP plots (`gradient_shap_top_features.png` for MLP, heatmaps for sequence models).
`--save` exports
plots without opening windows unless `--plot` is also supplied.

Neural runs print compact Captum GradientShap feature summaries for up to 128 test
observations, using 64 sampled training references and 128 random draws per
observation. Multioutput mode now explains all twelve horizons by default. This
requires four times as many attribution calls as explaining only 5, 30 and 60
minutes. For a faster run select `--shap-horizons 5 30 60`. Adjust computation with
`--shap-samples` / `--shap-draws`, or omit it with `--skip-shap`.
Sequence references are complete historical windows. Contributions
explain the learned target in $/MWh relative to the reference predictions; the
printed residual reflects Monte Carlo approximation. `--save` writes
`gradient_shap.csv`, `feature_importance.csv` and sequence lag contributions.
Multioutput filenames include the horizon. No result/model/attribution folders
are created without `--save`; reusable raw and prepared-data caches remain enabled.

Prepared caches live under this repository's `model_cache`. Compatibility now
includes the feature configuration, preparation code and local raw-cache file
identities (path, size and modification time). Changes in upstream data that
have not been downloaded cannot be detected automatically. `--refresh` rebuilds
prepared data through ingestion; existing ingestion caches may still be reused.

Rerunning a region with matching data and feature settings loads the complete
prepared dataset, including all twelve target columns and any GNN embeddings.
It skips ingestion, feature construction, target construction, and encoder fitting.
Compatible older feature-only caches are upgraded once in place. Model search
and training still run; changing only a forecaster's hyperparameters does not
rebuild the data. Switching GNN settings uses a separate prepared cache.

`python main.py` then:

1. loads or builds leakage-safe model data;
2. creates a chronological fit/validation split with a forecast-horizon embargo;
3. searches each model's hyperparameter grid on the validation period;
4. retrains the selected configuration on all pre-test training data;
5. reconstructs target-scale predictions into RRP forecasts; and
6. evaluates the forecasts once on the held-out test period.

Run every registered model with:

```bash
python main.py
```

Or select one or more models from the command line:

```bash
python main.py --models mlp
python main.py --models gru
python main.py --models lstm
python main.py --models xgboost
python main.py --models mlp gru lstm xgboost
```

### Graph embedding features

The models above forecast one region at a time. A shared preprocessing stage
adds what the other regions are doing: a graph encoder is trained on all five
NEM regions (nodes) linked by the interconnectors (edges), and its per-region
embedding is appended to the prepared data as `gnn_emb_00`, `gnn_emb_01`, ...
Every registered model then trains on those columns like any other feature.

By default, every model uses the full NEM graph embedding alongside the other
engineered features. A run prepares one shared dataset for all models. All runs
use twelve outputs. To disable GNN features, set `graph_embedding=None` in the
`ExperimentConfig` in `main.py` (or when calling `prepare_experiment`):

```python
EXPERIMENT = ExperimentConfig(region="VIC1", graph_embedding=None)
```

Omit that parameter to enable the GNN, or pass `GraphEmbeddingConfig(...)` to
tune the encoder. This setting has no CLI switch and applies to every model.

```bash
python main.py --models xgboost lstm
python main.py --models gru tcn
```

The encoder attends between linked regions using interconnector flow,
utilisation and headroom (`edge_features`), and is trained to forecast both
each region's target and the future price difference across each
interconnector (`spread_loss_weight`). The encoder lives in `pipeline/graph_embedding/`, with its layers in
`electricity_price_forecasting_model/layers/gnn_layers/`. It is fitted once per
target and cached under `model_cache`, so experiments on different
regions share it.

The encoder learns the 60-minute auxiliary target, and its
features feed the downstream model's twelve forecast outputs. `--device` applies
to both the encoder and deep forecasters; automatic selection prefers CUDA,
then Intel XPU, then CPU. GradientSHAP values for `gnn_emb_*` describe learned
embedding features, rather than individual inputs from neighbouring regions.
Saved graph-enabled forecasters expect these embedding columns as inputs;
their `model.joblib` does not bundle the separate graph encoder.

Because the encoder is trained on prices, it is fitted only on the first
`fit_fraction` (default 0.6) of the training period and then frozen.
Embeddings for later intervals, including every validation and test interval,
come from an encoder that never saw their outcomes. `prepare_experiment` raises
if a validation block would overlap the encoder's fit window; lower
`fit_fraction` or use fewer `--folds` in that case.

### Hyperparameter sweeps

`SWEEP_GRIDS` in `main.py` holds wider grids than `MODEL_REGISTRY` (currently
18 LSTM candidates over `sequence_length`, `hidden_size` and `num_layers`).
`--sweep` uses them for the models that have one:

```bash
python main.py --models lstm --sweep
```

Candidates are ranked on the validation period only; the test period is scored
once, for the selected configuration.

### Target Types

| Value | Model target | Price reconstruction |
| --- | --- | --- |
| `change` | future RRP minus current RRP | current RRP plus predicted change |
| `price` | future RRP | prediction is already an RRP forecast |
| `aemo_error` | future RRP minus aligned AEMO P5MIN RRP | AEMO forecast plus predicted error |

`model_features="all"` uses the standard engineered feature set. Pass a list
of column names to run a feature-specific experiment.

## Data Pipeline

`pipeline/data_cleaning/data.py` loads the following source tables:

- `DISPATCHPRICE`;
- `DISPATCHREGIONSUM`;
- `DISPATCHINTERCONNECTORRES`; and
- P5MIN `REGIONSOLUTION` forecasts.

The ingestion defaults cover all five NEM regions from 1 January 2022 to
31 July 2026. The final 20% of the requested period is reserved as the
chronological test split. `main.py` currently selects `NSW1` from that data.

Use the joint loader when working directly with raw data:

```python
from datetime import datetime

from pipeline.data_cleaning.data import getTrainTestData

train, test = getTrainTestData(
    regions=["NSW1"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
```

The preprocessing package joins the source tables and constructs price and
demand lags, rolling statistics, reserve measures, calendar features,
interconnector utilisation, aligned P5MIN features, and future targets. It
matches future targets by timestamp rather than row offset.

## Public Data Shapes

### Raw ingestion

```python
@dataclass
class Data:
    prices: pd.DataFrame
    region_summary: pd.DataFrame
    interconnectors: pd.DataFrame | None
    p5min: pd.DataFrame | None
    regions: list[str]
    start: datetime
    end: datetime

    @property
    def df(self) -> pd.DataFrame:
        return self.prices
```

`Data.df` is retained for compatibility and aliases the raw price table. It is
not the joined or feature-engineered model frame.

### Prepared model data

```python
@dataclass
class PreparedModelData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    features: list[str]
    region: str
    horizon: int
    target_type: str
```

`X_train` and `X_test` contain exactly the selected model features. The full
frames also contain metadata, current `rrp`, future `target_rrp`, and the
configured `target`.

### Experiment data

```python
@dataclass(frozen=True)
class ExperimentConfig:
    region: str = "NSW1"
    target_type: Literal["change", "price", "aemo_error"] = "change"
    model_features: Literal["all"] | Sequence[str] = "all"
    validation_fraction: float = 0.15
    cache_directory: Path = ...
    graph_embedding: GraphEmbeddingConfig | None = ...


@dataclass(frozen=True)
class ExperimentData:
    config: ExperimentConfig
    prepared: PreparedModelData
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame

    @property
    def features(self) -> list[str]: ...
```

`prepare_experiment(config, force_refresh=False)` returns `ExperimentData`.
Set `force_refresh=True` when cached prepared data must be rebuilt.

## Model, Tuning, and Evaluation Contracts

Every production model implements:

```python
class Model(ABC):
    model_name: str

    def __init__(self, **model_params) -> None: ...
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None: ...
    def predict(self, x_test: pd.DataFrame) -> np.ndarray: ...
    def get_model_name(self) -> str: ...
```

### Adding a Model Step by Step

#### 1. Create the model module

Add a Python file under `electricity_price_forecasting_model/`. The following
complete example wraps scikit-learn's histogram gradient boosting regressor,
which can consume the missing feature values retained by the shared pipeline.

```python
# electricity_price_forecasting_model/hist_gradient_boosting.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline.model import Model


class HistGradientBoostingModel(Model):
    model_name = "HistGradientBoosting"

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 0.0,
    ) -> None:
        self.model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=42,
        )

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x_test)
```

The wrapper must obey four rules:

- define a stable `model_name`;
- accept every tunable value as a named constructor argument;
- fit preprocessing only from `x_train` and `y_train`; and
- return one target-scale prediction per input row from `predict()`.

The prediction must remain on the configured target scale. For example, a
`change` experiment returns predicted price changes. `main.py` performs the
final conversion back to RRP.

Models that cannot consume missing values must fit and store an imputer inside
`train()`, then reuse it in `predict()`. Neural models may similarly fit a
`NumericPreprocessor` and `TargetTransformer` on training data only; they must
inverse-transform their predictions before returning them.

#### 2. Import the model in `main.py`

```python
from electricity_price_forecasting_model.hist_gradient_boosting import (
    HistGradientBoostingModel,
)
```

#### 3. Add the class and its grid to `MODEL_REGISTRY`

Register the class itself, not an already-created instance. Grid keys must
exactly match constructor parameter names, and every grid value must be a
sequence.

```python
MODEL_REGISTRY = [
    (
        XGBoostModel,
        {
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [4, 8],
        },
    ),
    (
        HistGradientBoostingModel,
        {
            "learning_rate": [0.03, 0.1],
            "max_iter": [100, 300],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
        },
    ),
]
```

This example evaluates 2 x 2 x 2 x 2 = 16 configurations for
`HistGradientBoostingModel`. Keep the first grid small when checking a new
model.

#### 4. Select the shared experiment

All registered models receive the same region, twelve horizons, target, features, and
chronological splits from `EXPERIMENT`:

```python
EXPERIMENT = ExperimentConfig(
    region="NSW1",
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
)
```

Do not load or split data inside the model wrapper. This keeps comparisons
between models consistent and prevents test-set leakage.

#### 5. Test and run

```bash
python -m pytest -q
python main.py
```

For each registry entry, the pipeline tunes on `fit_df` and `validation_df`,
recreates the model with `SearchResult.best_config`, trains it on the complete
pre-test training data, predicts the held-out test rows, reconstructs RRP, and
stores a `ModelResult`.

Hyperparameter search returns:

```python
@dataclass
class SearchResult:
    best_config: dict
    best_score: float
    metric: str
    trials: pd.DataFrame
```

The tuner uses `ExperimentData.fit_df` and `validation_df`; it never uses the
held-out test period for model selection.

Evaluation has the following interface:

```python
evaluate(
    model: Model,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> ModelResult


@dataclass
class ModelResult:
    model_name: str
    predictions: np.ndarray
    valid_mask: np.ndarray
    metrics: dict[str, float]
```

Reported metrics are MAE, RMSE, SSE, and median absolute error. Final model
evaluation should use reconstructed RRP forecasts against `target_rrp`.

## Repository Layout

```text
.
├── assets/                              course resources and reference papers
├── docs/                                deliverables, timeline, and workflow docs
├── electricity_price_forecasting_model/ production model implementations
├── model_research/                      individual model experiments
├── notebooks/                           exploratory notebooks
├── pipeline/
│   ├── data_cleaning/                   AEMO ingestion and raw caches
│   ├── preprocessing/                   shared feature and sequence preparation
│   ├── graph_embedding/                 cross-region graph encoder features
│   ├── sequences.py                     lazy datasets for sequence models
│   ├── evaluation.py                    common result metrics
│   ├── experiment_pipeline.py           experiment preparation and diagnostics
│   ├── model.py                         abstract model interface
│   └── tuning.py                        hyperparameter grid search
├── tests/                               pipeline regression tests
├── main.py                              experiment entry point
└── requirements.txt
```

See [Git and GitHub conventions](docs/git-and-github-conventions.md) for the
team workflow.

## Course Deliverables

- Proposal: 5%, due 5 pm 14 September
- Seminar: 10%, due in week 13
- Report: 20%, due 5 pm 9 November
- Individual reflective essay: 5%, due 5 pm 10 November

Further assessment details are in [`docs/README.md`](docs/README.md).

## References

- [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS)
- [Multi-Task Graph Adaptive Learning for Multivariate Electricity Price Short-Term Forecasting in Australia's National Electricity Market](https://ieeexplore.ieee.org/document/10486198)
