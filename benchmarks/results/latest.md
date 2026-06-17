# SynthHub Benchmark Results

Dataset: `sklearn.datasets.load_breast_cancer` subset. Epsilon: `1.0`. Synthetic rows: `180`.

| Method | Backend | Status | Epsilon spent | Utility similarity | TSTR score | Re-ID risk | Detail |
|---|---|---|---:|---:|---:|---:|---|
| independent | synthhub.independent | ok | 1.000 | 0.795 | 0.619 | 0.000 |  |
| privbayes | datasynthesizer:correlated | ok | 1.000 | 0.576 | 0.432 | 0.000 |  |
| datasynthesizer-independent | datasynthesizer:independent | ok | 1.000 | 0.869 | 0.513 | 0.005 |  |
| aim |  | skipped |  |  |  |  | AIM/MST require Private-PGM. Install the optional dependency and make the mechanisms folder importable, for example: pip install 'synthhub[private-pgm]' on Python 3.10-3.12. |
| mst |  | skipped |  |  |  |  | AIM/MST require Private-PGM. Install the optional dependency and make the mechanisms folder importable, for example: pip install 'synthhub[private-pgm]' on Python 3.10-3.12. |
| mwem |  | skipped |  |  |  |  | SmartNoise backends require smartnoise-synth. Install with pip install 'synthhub[smartnoise]'. |
| smartnoise-aim |  | skipped |  |  |  |  | SmartNoise backends require smartnoise-synth. Install with pip install 'synthhub[smartnoise]'. |
| synthcity-privbayes |  | skipped |  |  |  |  | SynthCity backends require synthcity. Install with pip install 'synthhub[synthcity]' on a supported Python version. |

`Utility similarity` is the mean per-column distribution similarity from `Synthesizer.evaluate`.
`TSTR score` is train-on-synthetic, test-on-real accuracy for `target`.
`Re-ID risk` is a nearest-neighbor membership-inference heuristic, not a DP proof.
