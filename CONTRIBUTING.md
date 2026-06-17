# Contributing

Thanks for helping make differentially private synthetic data easier to use.

## Development Setup

```bash
python -m pip install -e ".[test,datasynthesizer]"
python -m pytest -q
```

Optional backend checks:

```bash
python -m pip install -e ".[smartnoise]"
python -m pip install -e ".[synthcity]"
python -m pip install -e ".[private-pgm]"
```

## Pull Request Expectations

- Keep the dataframe-first `Synthesizer.fit/sample/evaluate` API stable.
- Add adapter contract tests for every backend option that touches epsilon,
  delta, accounting, or output schema.
- Add live smoke tests for optional backends when they are lightweight enough
  for CI.
- Document DP caveats in `docs/dp-guarantees.md`.
- Do not claim a formal guarantee unless the underlying backend and preprocessing
  assumptions are explicit.

## Benchmark Updates

Run:

```bash
python benchmarks/run_benchmark.py
```

Commit the updated files in `benchmarks/results/` when benchmark behavior
changes intentionally.

