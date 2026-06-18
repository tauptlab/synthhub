# Social Launch Posts

## Show HN

Title:

```text
Show HN: SynthHub, a scikit-learn-like API for DP synthetic data
```

Post:

```text
Hi HN, I built SynthHub, a small Python library that wraps differentially
private synthetic-data engines behind one pandas-first API.

The goal is not to invent another synthesizer. The problem I kept running into
is that DP synthetic-data tools are fragmented: PrivBayes, AIM, MST, MWEM, and
GAN-style methods all have different setup, schema, and accounting conventions.

SynthHub gives them a scikit-learn-like interface:

    synth = Synthesizer(method="privbayes", epsilon=1.0).fit(real_df)
    synth_df = synth.sample(1000)
    report = synth.evaluate(real_df, synth_df)

It reports requested/spent epsilon, backend identity, utility metrics, and a
membership-inference audit heuristic. It is alpha, but already on PyPI with CI
smoke coverage for DataSynthesizer, Private-PGM AIM/MST, and SmartNoise
MWEM/AIM/MST.

Repo: https://github.com/tauptlab/synthhub
PyPI: https://pypi.org/project/synthhub/
```

## Reddit

Use for `r/Python`, `r/datascience`, or relevant privacy/ML communities after
checking each community's self-promotion rules.

```text
I released SynthHub, a small Python library for differentially private
synthetic data.

The idea is a scikit-learn-like API over existing DP engines:

    synth = Synthesizer(method="privbayes", epsilon=1.0).fit(df)
    synthetic = synth.sample(1000)
    report = synth.evaluate(df, synthetic)

It is not a new algorithm; it is an integration layer for backend switching,
privacy-accounting reports, utility metrics, and simple benchmark comparison.
Current live smoke coverage includes DataSynthesizer PrivBayes, Private-PGM
AIM/MST, and SmartNoise MWEM/AIM/MST.

I would especially appreciate feedback on API design, DP guarantee wording, and
which backends should be prioritized next.

GitHub: https://github.com/tauptlab/synthhub
PyPI: https://pypi.org/project/synthhub/
```

## LinkedIn

```text
I released SynthHub, a Python library for differentially private synthetic data.

The motivation: DP synthetic-data tooling is fragmented. Different engines use
different APIs, schema assumptions, and accounting conventions, which makes it
hard for non-specialists to compare methods under the same epsilon.

SynthHub wraps existing DP engines behind a pandas-first, scikit-learn-like API:
fit, sample, evaluate. It reports requested/spent epsilon, backend identity,
utility metrics, and a membership-inference audit heuristic.

It is alpha, but already on PyPI with CI coverage for DataSynthesizer,
Private-PGM AIM/MST, and SmartNoise MWEM/AIM/MST.

GitHub: https://github.com/tauptlab/synthhub
PyPI: https://pypi.org/project/synthhub/
```

## X

```text
I released SynthHub: one unified API for differentially private synthetic data.

Switch DP backends with one argument:

Synthesizer(method="privbayes", epsilon=1.0)
Synthesizer(method="aim", epsilon=1.0)
Synthesizer(method="mst", epsilon=1.0)

PyPI + CI + Colab:
https://github.com/tauptlab/synthhub
```
