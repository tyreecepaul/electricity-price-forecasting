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

```python
EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=30,
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
`sequence_length` for temporal models. Forward passes return `(batch, 1)`.
The shared wrapper returns `(N,)` forecasts on the configured target scale.
New row-based models can continue implementing `Model.train()` and `predict()`.
Register a model factory and its hyperparameter grid in `main.py`.

The current interface deliberately covers one region and one forecast horizon.
Multi-horizon targets and separate known-future covariate tensors would require
an explicit extended output/input contract; future actuals must never be used
as known-future inputs.

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

Runs print results without saving artifacts by default. Add `--save` to save
configuration, dependency versions, data provenance, trial/fold
scores, per-origin predictions, training history and fitted CPU model bundles
(including scalers and history) under `reports/runs/<unique-run-id>/`.
Use `--save --output-dir` to choose a save location. Load
your own model bundles with `joblib.load`; do not load untrusted joblib files.

Neural runs also print Captum GradientShap feature rankings for up to 1,024 test
observations, using 256 sampled training references and 2,048 random draws per
observation. Sequence references are complete historical windows. Contributions
explain the learned target in $/MWh relative to the reference predictions; the
printed residual reflects Monte Carlo approximation. `--save` writes
`gradient_shap.csv`, `feature_importance.csv` and sequence lag contributions.

Prepared caches live under this repository's `model_cache`. Compatibility now
includes the feature configuration, preparation code and local raw-cache file
identities (path, size and modification time). Changes in upstream data that
have not been downloaded cannot be detected automatically. `--refresh` rebuilds
prepared data through ingestion; existing ingestion caches may still be reused.

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

The models above see one region at a time. An optional preprocessing stage
adds what the other regions are doing: a graph encoder is trained on all five
NEM regions (nodes) linked by the interconnectors (edges), and its per-region
embedding is appended to the prepared data as `gnn_emb_00`, `gnn_emb_01`, ...
Every registered model then trains on those columns like any other feature.

Which embedding a model trains on is set per model in `main.py`:

```python
MODEL_GRAPH_EMBEDDING = {
    XGBoostModel: GraphEmbeddingConfig(adjacency_mode="identity"),  # no cross-region edges
    LSTMModel: GraphEmbeddingConfig(),                              # full interconnector graph
}
```

Models not listed there (GRU, MLP, TCN, Transformer) get no embedding. A run
prepares one dataset per distinct embedding and still scores every model on
the same test rows.

```bash
python main.py --models xgboost lstm               # the per-model map above
python main.py --models lstm --graph-embedding none  # plain baseline for every model
python main.py --models gru --graph-embedding        # full graph for every model
python main.py --models gru --graph-embedding identity
```

The encoder attends between linked regions using interconnector flow,
utilisation and headroom (`edge_features`), and is trained to forecast both
each region's target and the future price difference across each
interconnector (`spread_loss_weight`). The encoder lives in `pipeline/graph_embedding/`, with its layers in
`electricity_price_forecasting_model/layers/gnn_layers/`. It is fitted once per
horizon and target and cached under `model_cache`, so experiments on different
regions share it.

Because the encoder is trained on prices, it is fitted only on the first
`fit_fraction` (default 0.6) of the training period and then frozen.
Embeddings for later intervals, including every validation and test interval,
come from an encoder that never saw their outcomes. `prepare_experiment` raises
if a validation block would overlap the encoder's fit window; lower
`fit_fraction` or use fewer `--folds` in that case.

To measure whether the stage helps, compare arms that differ only in the
embedding: `baseline` (none), `gnn` (interconnector graph), `no_graph` (the
same encoder with cross-region edges removed), and `gnn_plain` /
`no_graph_plain` (those two without edge attributes or the spread objective):

```bash
python compare_graph_embedding.py run --regions VIC1 NSW1
python compare_graph_embedding.py summarise
```

`run` trains XGBoost and LSTM unless `--models` names others, and gives every
model in a cell that cell's arm regardless of `MODEL_GRAPH_EMBEDDING`.

`run` is resumable and writes ordinary run artifacts under `reports/ablation/`.
`summarise` scores all arms on identical test rows and writes overall metrics,
a day-block bootstrap interval for each MAE difference, results by price
regime, the encoder's own validation error and XGBoost's reliance on the
embedding columns.

### Hyperparameter sweeps

`SWEEP_GRIDS` in `main.py` holds wider grids than `MODEL_REGISTRY` (currently
18 LSTM candidates over `sequence_length`, `hidden_size` and `num_layers`).
`--sweep` uses them for the models that have one:

```bash
python main.py --models lstm --sweep
python main.py --models lstm --sweep --graph-embedding none
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
    horizon_minutes: int = 60
    target_type: Literal["change", "price", "aemo_error"] = "change"
    model_features: Literal["all"] | Sequence[str] = "all"
    validation_fraction: float = 0.15
    cache_directory: Path = ...


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

All registered models receive the same region, horizon, target, features, and
chronological splits from `EXPERIMENT`:

```python
EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=30,
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
├── compare_graph_embedding.py           with/without graph-embedding comparison
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
