# Release Process

SynthHub should publish future releases with PyPI trusted publishing from
GitHub Releases. No PyPI API token should be stored in the repository.

## Prerequisites

- PyPI project `synthhub` exists.
- PyPI trusted publisher is configured for tokenless releases:
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
python -m pytest \
  tests/test_optional_backends.py::test_smartnoise_mwem_live_smoke_if_installed \
  tests/test_optional_backends.py::test_smartnoise_aim_live_smoke_if_installed \
  tests/test_optional_backends.py::test_smartnoise_mst_live_smoke_if_installed \
  -q
```

Private-PGM smoke checks require upstream mechanisms:

```bash
python -m pip install -e ".[test,private-pgm]"
git clone --depth 1 https://github.com/ryan112358/private-pgm.git .tmp-private-pgm-src
export PYTHONPATH="$PWD/.tmp-private-pgm-src:$PWD/.tmp-private-pgm-src/mechanisms:$PYTHONPATH"
python -m pytest \
  tests/test_optional_backends.py::test_private_pgm_mst_live_smoke_if_mechanisms_available \
  tests/test_optional_backends.py::test_private_pgm_aim_live_smoke_if_mechanisms_available \
  -q
```

## Publish

1. Confirm `version` in `pyproject.toml`.
2. Confirm `CHANGELOG.md` has the release date and no pending entries.
3. Confirm `docs/releases/<version>.md` is up to date.
4. Commit the release preparation.
5. Create and publish a GitHub Release for the tag, for example `v0.1.1`.
6. Confirm that the `Publish` workflow builds, checks, and uploads the package.
7. Verify installation from PyPI:

```bash
python -m pip install "synthhub[datasynthesizer]"
python - <<'PY'
import synthhub
print(synthhub.__version__ if hasattr(synthhub, "__version__") else "installed")
PY
```

Manual token uploads are only a fallback when trusted publishing is not yet
configured. If used, pass the token through process-local environment variables,
never commit it to the repository, and rotate the token after use.
