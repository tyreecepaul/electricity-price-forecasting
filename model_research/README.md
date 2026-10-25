# Research workspace

[Project README](../README.md) · [Experiment guide](../docs/experiments.md)

This directory contains individual prototypes and research studies. The
supported experiment entry point is `main.py` at the repository root.
Research scripts may use older data contracts or assume local datasets;
check their imports and settings before running them.

## Working here

- Keep experiments in your assigned directory. Coordinate with its owner
  before editing another person's work.
- Reuse shared preparation and evaluation when comparing with registered
  models. Record the region, horizons, features, split, seed and parameter grid.
- Select settings on validation data and reserve the test period for final
  evaluation. Never include future actual prices in model inputs.
- Keep downloaded data, caches and generated runs out of commits.
- Record the command and environment needed to reproduce a finding.

A promising prototype should be integrated into a production wrapper,
registered in `main.py` and covered by regression tests before it becomes
part of the shared comparison. See the [development guide](../docs/development.md).

Research checks can be run explicitly, for example:

```bash
python -m pytest -q model_research/connor/test_reliability_features.py
```

They are separate from the default `tests/` suite. Other scripts may launch
training or data loading when executed; inspect them before importing them.
