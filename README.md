### Placeholder

Hello from Dev

## Directory structure
```
.
├── assets # contains data sets, course resources, and intresting research papers
├── data_cleaning # data cleaning scripts
├── electricity_price_forecasting_model # main training scripts 
├── model_research # researching directorties for experimentation
├── notebooks # juptyer notebooks
├── README.md 
├── reports # submission latex papers orgonized by milestones
└── tests # model evaluation and testing
```

## Important Dates
- Proposal 5% due 5pm 14 Sep
- Seminar 10% due week 13 (Slide submission as well)
- Report 20%  due 5pm 9 Nov
- Reflective essay (I think individual) 5% due 5pm 10th Nov

## Data Stage 1: Ingestion & Cleaning Contract

`data_cleaning/data.py` is the shared contract for the rest of the pipeline (EDA, all modelling tracks, evaluation). It exposes:

```python
from data_cleaning.data import getTrainingData, getTestData

train = getTrainingData(regions=["NSW1"], start=datetime(2024, 1, 1), end=datetime(2024, 12, 31))
test = getTestData(regions=["NSW1"], start=datetime(2024, 1, 1), end=datetime(2024, 12, 31))
```

Do not read `data_cleaning/data.py` to understand what you're working with — the decisions below are all you need.

### Region
The NEM covers NSW, QLD, SA, TAS and VIC. This pipeline defaults to **NSW1 only**, but `regions` is a function argument — pass a list of any subset (`NSW1`, `QLD1`, `SA1`, `TAS1`, `VIC1`) to include more.

### Target
**RRP ($/MWh)** — the Regional Reference Price — is the forecast target. `total_demand` (MW) is also present in AEMO data but is **not** used by this pipeline.

### Frequency
**5-minute dispatch intervals**, sourced from AEMO's `DISPATCHPRICE` table via [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) (not the 30-minute `TRADINGPRICE` table).

### Date Range
Configurable via `start`/`end` function arguments (Python `datetime`). Default bounds: **2024-01-01 00:00:00 to 2024-12-31 23:55:00**.

### Train/Test Split
Chronological, no shuffling. The **last 20% of the requested date range** is test data; the first 80% is training data. This avoids leaking future information into training, which a random split would do for time series data.

### Shared Data Model (`Data` class)

| Attribute | Type | Meaning |
|---|---|---|
| `df` | `pandas.DataFrame` | Cleaned rows for this split (train or test) |
| `regions` | `list[str]` | NEM region IDs included in this `Data` object |
| `start` | `datetime` | Start of the requested date range |
| `end` | `datetime` | End of the requested date range |

`Data.df` columns:

| Column | Type | Meaning |
|---|---|---|
| `settlement_date` | `datetime64` | Timestamp of the 5-minute dispatch interval |
| `region` | `str` | NEM region ID (feature), e.g. `NSW1` |
| `rrp` | `float64` | Regional Reference Price in $/MWh (**target**) |

## Pricing Model References
- [UNSW-CEEM/NEMOSIS git](https://github.com/UNSW-CEEM/NEMOSIS)
- [Multi-Task Graph Adaptive Learning for Multivariate Electricity Price Short-Term Forecasting in Australia's National Electricity Market paper](https://ieeexplore.ieee.org/document/10486198)
