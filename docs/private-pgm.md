# Private-PGM AIM/MST Setup

SynthHub supports the upstream
[Private-PGM](https://github.com/ryan112358/private-pgm) AIM and MST mechanism
files without vendoring or reimplementing them. The PyPI package `mbi` provides
the core `Dataset`, `Domain`, and inference primitives. The AIM/MST mechanism
modules are still distributed from the upstream repository's `mechanisms/`
folder.

## Install

```bash
python -m pip install "synthhub[private-pgm]"
git clone --depth 1 https://github.com/ryan112358/private-pgm.git .private-pgm
export PYTHONPATH="$PWD/.private-pgm:$PWD/.private-pgm/mechanisms:$PYTHONPATH"
```

PowerShell:

```powershell
python -m pip install "synthhub[private-pgm]"
git clone --depth 1 https://github.com/ryan112358/private-pgm.git .private-pgm
$env:PYTHONPATH = "$PWD\.private-pgm;$PWD\.private-pgm\mechanisms;$env:PYTHONPATH"
```

## Usage

```python
from synthhub import Synthesizer

synth = Synthesizer(method="mst", epsilon=1.0, random_state=0)
synth.fit(real_df)
synth_df = synth.sample(1000)
```

AIM is workload-aware and can be slower. For a small smoke run:

```python
synth = Synthesizer(
    method="aim",
    epsilon=1.0,
    degree=2,
    rounds=4,
    max_iters=20,
    max_model_size=0.2,
    random_state=0,
)
```

Use larger `rounds`, `max_iters`, and `max_model_size` for real comparisons.

## CI Coverage

SynthHub's CI installs `synthhub[private-pgm]`, clones
`ryan112358/private-pgm`, adds the repo root and `mechanisms/` directory to
`PYTHONPATH`, and runs live smoke tests for both `aim` and `mst`.

## Caveat

The upstream
[Private-PGM PyPI page](https://pypi.org/project/tmlt.private_pgm/) notes that
the research mechanisms use floating-point Gaussian or Laplace noise. SynthHub
exposes backend accounting and adapter wiring, but it does not replace the
upstream implementation's cryptographic or numerical assumptions.
