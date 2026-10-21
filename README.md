# Forecast-Informed Electricity Price Forecasting

A STAT3007/7007 project forecasting five-minute Regional Reference Prices
(RRP) in Australia's National Electricity Market. At each forecast origin,
models predict the next hour: **12 prices, from 5 to 60 minutes ahead**.

We compare MLP, GRU, LSTM, TCN, Transformer, XGBoost and ARIMA against
persistence (the current price) and available AEMO P5MIN forecasts. Models
share data preparation, chronological validation and held-out test evaluation.

## Setup

Run these commands from the repository root. Python and Git must be installed;
Git is needed for the pinned NEMSEER dependency in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU runs need a PyTorch installation compatible with your CUDA or Intel XPU
device. CPU runs are supported.

## Run an experiment

Start with one model and skip attribution to reduce computation:

```bash
python main.py --models mlp --device cpu --epochs 3 --skip-shap
```

**The first run can take substantial time and disk space.** Default ingestion
covers all five NEM regions from January 2022 to July 2026, and GNN feature
training is enabled. `--epochs` limits the deep forecaster, not the GNN encoder.
Later runs reuse raw data, prepared features and cached embeddings.

Common commands:

```bash
# Compare selected models using automatic device selection
python main.py --models mlp gru xgboost --skip-shap

# Display forecast errors, example price curves and SHAP views
python main.py --models mlp --plot --shap-horizons 5 30 60

# Save results, plots and fitted models
python main.py --models lstm --save --skip-shap

# Show all command-line options
python main.py --help
```

Omitting `--models` runs all seven registered models. Deep models produce
12 outputs, XGBoost shares one horizon-conditioned ensemble across the leads,
and ARIMA projects one fitted price model over the hour.

## Change the configuration

Edit [`main.py`](main.py):

| Setting | Where to change it |
| --- | --- |
| Region, target and input features | `EXPERIMENT` |
| GNN features | `EXPERIMENT.graph_embedding`; set to `None` to disable |
| Model hyperparameters | `MODEL_REGISTRY` |
| Wider search grids | `SWEEP_GRIDS`, enabled with `--sweep` |

The current experiment uses **NSW1**, price-change targets, all standard
features plus GNN embeddings, and a 20% validation fraction. All experiments
use the complete 12-horizon curve; there is no horizon-selection CLI flag.

Use `--device cpu|cuda|xpu|auto` to select the device for the GNN encoder and
deep models. Automatic selection prefers CUDA, then XPU, then CPU.

For configuration examples, target definitions, folds and cache behaviour,
see the [experiment guide](docs/experiments.md).

## Results and interpretation

The console reports MAE, RMSE, improvements over baselines and model fit time.
Errors are measured in **$/MWh** after reconstructing price forecasts.
Add `--verbose` for per-horizon, price-regime and detailed attribution tables.

Hyperparameters and neural checkpoints are selected on chronological validation
data. Training labels are embargoed by the longest horizon (60 minutes), and
the selected model is refitted before evaluation on the held-out test period.
Model comparisons use shared eligible forecast values.

- `--plot` opens forecast and attribution plots.
- `--save` writes to `reports/runs/<run-id>/`; use `--output-dir PATH` to
  choose another parent directory.
- Runs create no result artifacts unless `--save` is supplied. Data caches
  remain enabled.

Saved runs include comparison tables, per-horizon predictions, forecast curves,
plots and CPU model bundles. Graph-enabled bundles require `gnn_emb_*` input
features; the separate graph encoder is not included in `model.joblib`.

GradientSHAP runs for deep models by default. Its plots show each feature's
share of total absolute attribution, **not accuracy or explained variance**.
Use `--shap-horizons 5 30 60` to explain fewer outputs, or `--skip-shap` to
skip attribution. See the [experiment guide](docs/experiments.md) for details.

## Tests and development

```bash
python -m pytest -q
```

See the [development guide](docs/development.md) for data shapes, model contracts
and adding an architecture, and [Git conventions](docs/git-and-github-conventions.md)
for the team workflow.

| Directory | Contents |
| --- | --- |
| `pipeline/` | Ingestion, features, GNN encoding, training and evaluation |
| `electricity_price_forecasting_model/` | Model wrappers and network layers |
| `model_research/`, `notebooks/` | Research scripts and exploration |
| `tests/` | Regression tests |
| `docs/` | Guides, course requirements and deliverables |

[Documentation and coursework](docs/README.md) · [Report template](docs/report/report.tex) ·
[Reference papers](assets/reference_papers/)
