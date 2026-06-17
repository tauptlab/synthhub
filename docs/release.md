# Release Process

SynthHub publishes with PyPI trusted publishing from GitHub Releases. No PyPI
API token should be stored in the repository.

## Prerequisites

- PyPI project `synthhub` exists.
- PyPI trusted publisher is configured for:
  - owner: `tauptlab`
  - repository: `synthhub`
  - workflow: `publish.yml`
  - environment: `pypi`
- The `pypi` GitHub environment exists and is protected if required.
- `CHANGELOG.md`, `CITATION.cff`, and `pyproject.toml` agree on the release
  version.

## Local Verification

```bash
python -m pip install -e ".[test,datasynthesizer,release]"
python -m pytest -q
python benchmarks/run_benchmark.py --datasets breast_cancer iris diabetes --methods independent privbayes datasynthesizer-independent --sample-rows 80
python -m build
python -m twine check dist/*
```

Optional backend smoke checks:

```bash
python -m pip install -e ".[test,smartnoise]"
python -m pytest tests/test_optional_backends.py::test_smartnoise_mwem_live_smoke_if_installed -q
```

## Publish

1. Update `version` in `pyproject.toml`.
2. Move the relevant `CHANGELOG.md` entries from `Unreleased` to the release
   version.
3. Commit the release preparation.
4. Create a GitHub Release for the tag, for example `v0.1.0`.
5. Confirm that the `Publish` workflow builds, checks, and uploads the package.
6. Verify installation from PyPI:

```bash
python -m pip install "synthhub[datasynthesizer]"
python - <<'PY'
import synthhub
print(synthhub.__version__ if hasattr(synthhub, "__version__") else "installed")
PY
```
