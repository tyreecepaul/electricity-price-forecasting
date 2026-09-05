# Shared model benchmarking workflow

`model_research/common/experiment_pipeline.py` contains the data splitting,
evaluation, and reporting components that must remain identical across model
comparisons:

- prepared-data loading and configuration-aware caching;
- chronological validation with a forecast-horizon embargo;
- a shared complete-case row policy for directly comparable models;
- reconstruction of prices from either price-change or AEMO-error targets;
- persistence and AEMO baseline comparisons;
- common MAE, MSE, RMSE, median-error, price-regime, and target diagnostics;
- separation of model selection, final refitting, and untouched test evaluation.

The model scripts keep only model-specific behavior:

- `XGBoost_Model.py` selects the number of trees with chronological validation,
  refits on all pre-test rows, and reports final test metrics and feature
  importance.
- `MLP_Model.py` applies train-only feature and target transformations, selects
  an epoch count with validation and early stopping, then refits a fresh network
  on all pre-test rows before reporting the same final test metrics.

## Fair model comparisons

Before comparing outputs, use identical `ExperimentConfig` values in both model
scripts, especially `region`, `horizon_minutes`, `target_type`,
`model_features`, and `validation_fraction`. Both scripts must also evaluate the
same complete test rows. The `FINAL TEST RESULTS` and `RESULTS BY PRICE REGIME`
sections are comparable only when these settings and row counts match.

## Starting another model

The minimum shared setup is:

```python
from model_research.common.experiment_pipeline import (
    ExperimentConfig,
    prepare_experiment,
)

EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=30,
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
)

experiment = prepare_experiment(EXPERIMENT)
data = experiment.prepared

X_fit = experiment.fit_df[experiment.features]
y_fit = experiment.fit_df["target"]
X_validation = experiment.validation_df[experiment.features]
y_validation = experiment.validation_df["target"]
```

A model using the same experiment configuration reuses the same prepared-data
cache. Model-specific fitted artefacts, scalers, and checkpoints should be stored
separately. Transformations for an MLP must be fitted on `X_fit` during model
selection, then fitted from scratch on `data.X_train` for the final model. The
validation-selected epoch count is reused for this final refit; the test set is
evaluated once and never controls training.

## Cache identity

Cache filenames include the region, horizon, target, feature-set identifier, and
cache schema version. Change `CACHE_SCHEMA_VERSION` in `experiment_pipeline.py`
when shared preprocessing or cache structure changes in a way that requires all
prepared data to be rebuilt.
