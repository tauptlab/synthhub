# SynthHub MVP Design

SynthHub is a dataframe-first wrapper around differentially private synthetic
data engines. The goal is a scikit-learn-like interface for comparing methods
without learning every backend's data format, preprocessing rules, and privacy
accounting vocabulary.

## Public API

```python
from synthhub import Synthesizer

synth = Synthesizer(method="aim", epsilon=1.0, random_state=0)
synth.fit(real_df)
synthetic_df = synth.sample(1000)
report = synth.evaluate(real_df, synthetic_df, target="label")
```

Changing `method` should be the main user-facing switch:

- `aim`: Private-PGM AIM adapter. Intended default once optional backend deps
  are installed.
- `mst`: Private-PGM MST adapter.
- `privbayes`: DataSynthesizer correlated-mode adapter for PrivBayes-style
  Bayesian-network synthesis.
- `independent`: built-in one-way marginal baseline for examples, smoke tests,
  and evaluation pipeline checks.
- SmartNoise aliases: `mwem`, `pacsynth`, `dpctgan`, `patectgan`, `pategan`,
  `dpgan`, `quail`, `smartnoise-aim`, `smartnoise-mst`.
- SynthCity aliases: `synthcity-privbayes`, `synthcity-pategan`,
  `synthcity-dpgan`.

## Internal Boundaries

```mermaid
flowchart LR
    A["pandas DataFrame"] --> B["Schema inference or public Schema"]
    B --> C["TabularPreprocessor"]
    C --> D["integer-coded discrete dataframe"]
    D --> E["BackendAdapter"]
    E --> F["FittedBackend"]
    F --> G["encoded synthetic dataframe"]
    G --> H["inverse_transform"]
    H --> I["synthetic pandas DataFrame"]
    I --> J["evaluate utility + privacy"]
```

`Synthesizer` owns validation, schema selection, preprocessing, backend
dispatch, sampling, and report checks. Backends only receive integer-coded data
and a discrete domain.

## Privacy Contract

Every fitted backend must return a `PrivacyReport` with:

- requested epsilon
- epsilon spent
- delta, when applicable
- accountant name
- backend identity
- warnings

`Synthesizer.fit` fails if a backend reports spending more epsilon than the
caller requested. This does not prove the backend's DP guarantee; it locks the
adapter contract and prevents silent budget mismatch.

Formal DP guarantees are conditional on public preprocessing metadata. Automatic
schema inference is convenient, but it derives column types, categories, and
bounds from the input dataframe. SynthHub therefore keeps an explicit warning in
the fitted privacy report unless the caller passes a public `Schema`.

## MVP Stages

1. Implement the public API, schema inference, preprocessing, reporting, and
   evaluation.
2. Register marginal backends with adapter boundaries: AIM, MST, PrivBayes, and
   an executable independent baseline.
3. Add contract tests for epsilon validation, backend wiring, sampling shape,
   and report structure.
4. Add benchmark notebooks that run the same dataframe and epsilon through all
   installed backends.
5. Harden external engine integration with backend-specific tests that mock or
   run the real accountant path in CI.

## Backend Candidate Review

Current implementation status:

| Family | Methods | Status | Notes |
|---|---|---|---|
| Private-PGM | AIM, MST | adapter | Best marginal-first MVP target; external mechanisms must be importable. |
| DataSynthesizer | PrivBayes, independent DP mode | adapter | Lightweight optional PrivBayes path using active-domain categorical synthesis over SynthHub-encoded columns. |
| SmartNoise Synthesizers | AIM, MST, MWEM, PAC-Synth, DP-CTGAN, PATE-CTGAN, PATE-GAN, DP-GAN, QUAIL | adapter | Broadest immediate expansion; SynthHub passes encoded data and sets preprocessing epsilon to 0 by default. |
| SynthCity | PrivBayes, PATEGAN, DPGAN | adapter | Useful for GAN/Bayesian privacy plugins; dependency is heavy and Python-version sensitive. |
| OpenDP contingency tables | AIM, MST, Fixed, Sequential | planned | Strong long-term trust target, but its context/query API needs a dedicated public-schema flow rather than the simple encoded backend contract. |

## Evaluation Scope

Utility:

- per-column distribution similarity
- numeric correlation similarity
- train-on-synthetic, test-on-real score when a target is supplied

Privacy:

- reported epsilon/delta/accountant
- nearest-neighbor membership-inference heuristic

The membership heuristic is an audit signal, not a DP proof.
