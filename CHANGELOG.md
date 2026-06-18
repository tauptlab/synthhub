# Changelog

All notable changes to SynthHub are documented here.

## Unreleased

No unreleased changes.

## 0.1.0.post1 - 2026-06-18

- Corrected PyPI-facing README and install metadata after the first PyPI
  publication.

## 0.1.0 - 2026-06-18

- Added PyPI trusted-publishing workflow and release checklist.
- Added SmartNoise MWEM/AIM/MST live smoke coverage.
- Added Private-PGM AIM/MST live smoke coverage using upstream mechanisms.
- Expanded the public benchmark to breast cancer, iris, and diabetes sklearn datasets.
- Added a public-schema guide for formal DP usage.
- Fixed SmartNoise epsilon-only mechanisms that reject `delta`.
- Added a DataSynthesizer compatibility shim for conditional-probability sampling keys.
- Added the dataframe-first `Synthesizer.fit/sample/evaluate` API.
- Added schema inference, preprocessing, and inverse transforms.
- Added built-in independent marginal baseline.
- Added DataSynthesizer PrivBayes and independent-mode adapters.
- Added adapter contracts for Private-PGM, SmartNoise, and SynthCity backends.
- Added utility and privacy evaluation reports.
