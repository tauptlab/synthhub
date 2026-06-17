# DP Guarantees and Caveats

SynthHub is an adapter layer. It does not invent new DP mechanisms. Its job is
to make backend choice, preprocessing, evaluation, and privacy reporting visible
through one dataframe-first API.

## Contract Summary

| Method | Backend | DP accounting source | CI coverage | Main caveat |
|---|---|---|---|---|
| `independent` | SynthHub one-way marginal baseline | Sequential composition over noisy one-way marginals | live unit tests | Baseline only; does not preserve correlations. |
| `privbayes` | DataSynthesizer correlated mode | DataSynthesizer correlated attribute mode | live smoke when dependency is installed | Active-domain categorical DP over SynthHub-encoded columns. |
| `datasynthesizer-independent` | DataSynthesizer independent mode | DataSynthesizer independent attribute mode | live adapter path | Active-domain categorical DP over SynthHub-encoded columns. |
| `aim` | Private-PGM AIM | External Private-PGM mechanism | adapter contract; live only when dependency is available | Requires Private-PGM mechanisms on `PYTHONPATH`. |
| `mst` | Private-PGM MST | External Private-PGM mechanism | adapter contract; live only when dependency is available | Requires Private-PGM mechanisms on `PYTHONPATH`. |
| SmartNoise aliases | SmartNoise Synthesizers | SmartNoise odometer when exposed | mocked adapter contract | Heavy optional dependency; SynthHub sets preprocessing epsilon to zero by default. |
| SynthCity aliases | SynthCity privacy plugins | Plugin-specific epsilon parameter | mocked adapter contract | Plugin accounting varies; use as experimental until live CI is added. |

## Public Metadata Requirement

Formal DP guarantees are conditional on preprocessing metadata being public or
otherwise allowed by the caller's privacy policy. This includes:

- column names
- column types
- categorical domains
- numeric bounds
- binning choices

If SynthHub infers schema metadata from the private dataframe, the fitted
`PrivacyReport` includes a warning. For formal releases, pass an explicit
public `Schema`.

## What SynthHub Verifies

SynthHub tests and runtime checks verify:

- requested epsilon is passed into each backend adapter
- reported `epsilon_spent` does not exceed requested epsilon
- `disabled_dp=True` is rejected for SmartNoise adapters
- sample output returns the original dataframe column order
- missing optional backends fail closed with `BackendNotAvailableError`
- DataSynthesizer PrivBayes runs in a live smoke test when installed

## What SynthHub Does Not Prove

SynthHub does not prove the mathematical correctness of external DP engines. It
records and validates adapter-level contracts, then exposes backend-reported
accounting in a common `PrivacyReport`.

The membership-inference score in `evaluate` is an audit heuristic. It is useful
for comparing synthetic-data behavior, but it is not a formal DP proof.

