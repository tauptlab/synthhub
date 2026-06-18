# SynthHub 0.1.0.post1

SynthHub is now available on PyPI:

```bash
python -m pip install "synthhub[datasynthesizer]"
```

SynthHub is a dataframe-first Python library for differentially private
synthetic data. It gives DP synthesizers a small, scikit-learn-like API:

```python
from synthhub import Synthesizer

synth = Synthesizer(method="privbayes", epsilon=1.0, random_state=0)
synth.fit(real_df)
synth_df = synth.sample(1000)
report = synth.evaluate(real_df, synth_df, target="label")
```

## Highlights

- One pandas `DataFrame` API for `fit`, `sample`, and `evaluate`.
- DP-first positioning with explicit requested/spent epsilon reports.
- Live CI smoke coverage for DataSynthesizer PrivBayes, Private-PGM AIM/MST,
  and SmartNoise MWEM/AIM/MST.
- Reproducible benchmark table comparing utility and re-identification risk
  under the same epsilon.
- Colab quickstart and example notebooks.
- Adapter contracts for heavier optional SmartNoise and SynthCity backends.

## Why This Exists

DP synthetic-data tooling is fragmented. Popular synthetic-data products often
separate formal DP guarantees from their default open-source workflow, while
research-oriented implementations expose different schemas, data formats, and
accounting conventions.

SynthHub does not invent a new synthetic-data algorithm. It wraps existing DP
engines behind one API, then makes preprocessing, evaluation, and privacy
accounting visible in a common report.

## Useful Links

- PyPI: https://pypi.org/project/synthhub/
- Quickstart Colab: https://colab.research.google.com/github/tauptlab/synthhub/blob/main/examples/quickstart.ipynb
- DP guarantee notes: https://github.com/tauptlab/synthhub/blob/main/docs/dp-guarantees.md
- Public schema guide: https://github.com/tauptlab/synthhub/blob/main/docs/public-schema.md
- Benchmarks: https://github.com/tauptlab/synthhub/blob/main/benchmarks/results/latest.md

## Caveats

SynthHub is alpha software. Formal DP guarantees require public preprocessing
metadata; use an explicit public `Schema` for formal workflows. The
membership-inference metric is an audit heuristic, not a DP proof.
