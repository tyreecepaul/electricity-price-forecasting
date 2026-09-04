# Model research workflow

`model_research/common/experiment_pipeline.py` contains the components that should remain identical
across model comparisons:

- prepared-data loading and configuration-aware caching;
- chronological validation with a forecast-horizon embargo;
- reconstruction of prices from either price-change or AEMO-error targets;
- persistence and AEMO baseline comparisons;
- interval-level results, price-regime metrics, and target diagnostics.

`XGBoost_Model.py` contains the XGBoost specification, fitting calls, and
feature-importance output. It can be used as the structural reference for a new
model without duplicating the data pipeline.

## Starting another model

The minimum shared setup is:

```python
from model_research.common.experiment_pipeline import (
    ExperimentConfig,
    prepare_experiment,
)

EXPERIMENT = ExperimentConfig(
    region="NSW1",
    horizon_minutes=60,
    target_type="change",
    model_features="all",
    validation_fraction=0.15,
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
selection, then refitted on `data.X_train` for the final model.

## Cache identity

Cache filenames include the region, horizon, target, feature-set identifier, and
cache schema version. Change `CACHE_SCHEMA_VERSION` in `experiment_pipeline.py`
when shared preprocessing or cache structure changes in a way that requires all
prepared data to be rebuilt.
