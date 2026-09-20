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

## Running an Experiment

Configure `EXPERIMENT` and `MODEL_REGISTRY` in `main.py`, then run:

```bash
python main.py
```

`ExperimentConfig` selects the NEM region, forecast horizon, target type,
features, and validation fraction. Each `MODEL_REGISTRY` entry contains a
`Model` implementation and its hyperparameter grid. The pipeline prepares and
caches the data, performs a chronological validation search, retrains the best
configuration on all pre-test data, and evaluates reconstructed RRP forecasts
on the held-out test period.

Supported targets are `change`, `price`, and `aemo_error`. Run the checks with:

```bash
python -m pytest -q
```

## Git/Github Conventions

### Branches

Two branches are maintained:

| Branch | Role | Protected |
| --- | --- | --- |
| `production` | deployable and demo-ready. The repository's default branch. Only release and hotfix pull requests land here. | Yes |
| `dev` | Integration branch. All day-to-day work merges here first. The base branch for every feature pull request. | Yes |

Everything else is a short-lived working branch, created from `dev` and deleted once it has merged.

#### Working branch names

```text
type/issue-number-short-summary
```

| Type | Use for | Example |
| --- | --- | --- |
| `feat/` | A new feature or capability | `feat/12-relay-websocket-handshake` |
| `fix/` | A bug fix | `fix/34-tether-empty-workspace` |
| `docs/` | Documentation, ADRs, meeting notes, spec updates | `docs/07-transport-adr` |
| `refactor/` | Restructuring with no change in behaviour | `refactor/28-shared-message-codec` |
| `test/` | Adding or updating tests | `test/19-relay-reconnect-coverage` |
| `chore/` | Tooling, dependencies, repository maintenance | `chore/22-pin-eslint-v10` |
| `spike/` | Throwaway investigation or proof of concept | `spike/31-webrtc-feasibility` |
| `hotfix/` | An urgent fix to `production` | `hotfix/41-relay-crash-on-startup` |

Rules:

- Branch from an up-to-date `dev`. Only Hotfixes are allowed to branch from `production`.
- Each branch **MUST** carry an issue number. It ties the branch to a GitHub issue created, thus clearly showing the work being done.
- The summary is lowercase kebab-case, two to five words, describing the change.

```bash
git switch dev
git pull origin dev
git switch -c feat/12-relay-websocket-handshake
```

### Commits

[Conventional Commits](https://www.conventionalcommits.org/) are the standard for commit messages.

```text
type(scope): subject

Optional body explaining WHY this change was made, not WHAT was changed. 
```

#### Types

| Type | Use for |
| --- | --- |
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `style` | Formatting or whitespace, no code change |
| `refactor` | Restructuring with no change in behaviour |
| `test` | Adding or updating tests |
| `chore` | Tooling, dependencies, repository maintenance |
| `perf` | A performance improvement |
| `ci` | CI/CD configuration |
| `revert` | Reverting a previous commit |

#### Scopes

the package or top-level folder the change touches:

`tether`, `relay`, `shared`, `docs`, `spec`, `design`, `evidence`, `deployment`, `repo`

Omit the scope only when a change genuinely spans the whole repository.

#### Subject line

- Imperative mood: `add`, not `added` or `adds`.
- Lowercase after the colon, no trailing full stop.
- 72 characters maximum.
- Describe the change, not the file you edited.

#### Body

- Optional, but write one whenever the reason isn't obvious from the subject.
- Explain **why** the change was made, not **what** was changed.
- Wrap at 72 characters.

#### Examples

```text
feat(relay): add websocket handshake
```

Avoid: `update`, `fixed stuff`, `WIP`, `final version`, or a subject that names a file (`edit extension.ts`) rather than the change.

### Pull requests

Every change reaches `dev` or `production` through a pull request. These branches should not be pushed directly to.

| Pull request kind | Head branch | Base branch |
| --- | --- | --- |
| Feature, fix, docs, chore, and so on | your working branch | `dev` |
| Release | `dev` | `production` |
| Hotfix | `hotfix/...` | `production`, then back-merge to `dev` |

**Title**:  the same format as a commit subject: `type(scope): subject`.

**Description**:  `.github/PULL_REQUEST_TEMPLATE.md` Complete every section, that the template requires.

| Section | Description |
| --- | --- |
| Summary | What the PR does |
| Related Issues | i.e. `Closes #12` |
| Changes | A short bulleted list of what changed. Short text blocks are permitted |
| How to Test | Steps a reviewer can actually follow to verify it works |
| Evidence | Screenshots or recordings for anything user-facing |
| Checklist | Tick each item in the checkbox before requesting review |

#### Before you request a review

- Open as a **Draft** while you're still working. Mark it *Ready for review* when it's done.
- Read your own diff end to end on the *Files changed* tab first.
- Make sure lint and tests pass. This is especially **non-negotiable** as github actions are disabled for this repo.
- PR's should only tackle 1 issue at a time. More issues are permitted if the solution is small and related, but the PR should still be titled after the main issue.

#### Asking for a review

| Base branch | Approvals required |
| --- | --- |
| `dev` | 1 |
| `production` | 2 |

- Request reviewers explicitly through the **Reviewers** panel. A message in the appropriate chat can also be utilised, however it is not formal evidence.
- Authors **never** approve their own pull request.
- Aim to respond to a review request within 24 hours. Assignees need to communicate if they are unable to review in a timely manner.

#### As a reviewer

Check for the following:

- Does it do what the linked issue asked for?
- Are the edge cases and error paths handled?
- Is it readable by someone who wasn't in your head when you wrote it?
- Are there tests, and do they test behaviour rather than implementation?
- Any secrets, credentials, or generated files committed by accident?
- Use **Request changes** for anything blocking and **Comment** for suggestions. **Approve** only when you'd be comfortable owning the code yourself. You are allowed to push fixes to the branch if you want to. However, this is not required.

#### As an author receiving a review

- Reply to every comment. This enforces clear communication
- Push fixes as new commits.
- Re-request review once you've addressed the feedback.

### Merging and releases

- Use *Create a merge commit* for every pull request. This keeps each teammate's individual commits in the history.
- **Never squash-merge `dev` into `production`.**
- Delete the working branch after merging.

#### Keeping your branch current

Ensure your branch is always up to date via rebasing:

```bash
git switch feat/12-relay-websocket-handshake
git fetch origin
git rebase origin/dev
git push --force-with-lease
```

Once review has started, or if someone else has based work on your branch, merge instead:

```bash
git fetch origin
git merge origin/dev
```

#### Releasing

1. Open a pull request from `dev` into `production`, titled `release: v0.2.0`.
2. Summarise what's shipping in the description.
3. Get two approvals.
4. Merge with a merge commit.
5. Tag the release:

```bash
git switch production
git pull origin production
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

We follow [semantic versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

#### Hot-fixing

```bash
git switch production
git pull origin production
git switch -c hotfix/41-relay-crash-on-startup
```

Open a pull request into `production`, get two approvals, and merge. Then back-merge immediately so the fix isn't lost the next time `dev` is released:

```bash
git switch dev
git pull origin dev
git merge origin/production
git push origin dev
```

## Ingestion & Cleaning Contract

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


## Model Fitting

To be able to evaluate models using the universal `evaluate()` function, it must implement the
abstract `Model` class

## Abstract Model Class

| Abstract Function | Meaning |
|---|---|
| `__init__()` | Initialise your model with specific hyperparameters | 
| `train(training_data: Data) -> None` | Trains your model based on a given training dataset | 
| `predict(test_data: Data) -> np.ndarray` | Makes predictions based on a given test dataset |
| `get_model_name() -> str` | Returns the name of your model |

## Evaluation

Using the implemented `Model` class and your test data `Data` class, you can invoke the following
function to evaluate the model

```python
evaluate(model: Model, test_data: Data) -> None:
```

## Pricing Model References
- [UNSW-CEEM/NEMOSIS git](https://github.com/UNSW-CEEM/NEMOSIS)
- [Multi-Task Graph Adaptive Learning for Multivariate Electricity Price Short-Term Forecasting in Australia's National Electricity Market paper](https://ieeexplore.ieee.org/document/10486198)
