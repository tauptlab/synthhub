# SynthHub Benchmark Results

Datasets: `breast_cancer, iris, diabetes`. Epsilon: `1.0`. Synthetic rows per run: `80`.

| Dataset | Method | Backend | Status | Epsilon spent | Utility similarity | TSTR score | Re-ID risk | Detail |
|---|---|---|---|---:|---:|---:|---:|---|
| breast_cancer | independent | synthhub.independent | ok | 1.000 | 0.788 | 0.636 | 0.086 |  |
| breast_cancer | privbayes | datasynthesizer:correlated | ok | 1.000 | 0.598 | 0.715 | 0.036 |  |
| breast_cancer | datasynthesizer-independent | datasynthesizer:independent | ok | 1.000 | 0.831 | 0.378 | 0.030 |  |
| iris | independent | synthhub.independent | ok | 1.000 | 0.784 | 0.307 | 0.000 |  |
| iris | privbayes | datasynthesizer:correlated | ok | 1.000 | 0.739 | 0.480 | 0.000 |  |
| iris | datasynthesizer-independent | datasynthesizer:independent | ok | 1.000 | 0.762 | 0.167 | 0.000 |  |
| diabetes | independent | synthhub.independent | ok | 1.000 | 0.776 | -0.418 | 0.000 |  |
| diabetes | privbayes | datasynthesizer:correlated | ok | 1.000 | 0.654 | 0.081 | 0.000 |  |
| diabetes | datasynthesizer-independent | datasynthesizer:independent | ok | 1.000 | 0.772 | -0.055 | 0.000 |  |

`Utility similarity` is the mean per-column distribution similarity from `Synthesizer.evaluate`.
`TSTR score` is train-on-synthetic, test-on-real accuracy for `target`.
`Re-ID risk` is a nearest-neighbor membership-inference heuristic, not a DP proof.
