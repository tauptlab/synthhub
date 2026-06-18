# Public Schema Guide

Formal DP guarantees depend on which metadata is treated as public. SynthHub can
infer column types, categories, numeric bounds, and bins for convenience, but
that inference uses the input dataframe. For formal use, pass a `Schema` whose
metadata came from documentation, contracts, public codebooks, or another
approved public source.

## Example

```python
import pandas as pd

from synthhub import ColumnSpec, Schema, Synthesizer

real_df = pd.DataFrame(
    {
        "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
        "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
        "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    }
)

schema = Schema(
    columns=(
        ColumnSpec("age", "continuous", "int64", lower=0, upper=100),
        ColumnSpec("city", "categorical", "object", categories=("A", "B", "C")),
        ColumnSpec("churn", "categorical", "int64", categories=(0, 1)),
    ),
    source="public",
)

synth = Synthesizer(
    method="independent",
    epsilon=1.0,
    schema=schema,
    random_state=0,
)
synth.fit(real_df)

synth_df = synth.sample(100)
report = synth.evaluate(real_df, synth_df, target="churn")

print(synth.privacy_report_.to_dict())
print(report.to_dict())
```

This example is dependency-free. For a stronger marginal backend, install
`synthhub[datasynthesizer]` and change `method="independent"` to
`method="privbayes"`.

With a public schema, `PrivacyReport.warnings` should not contain the inferred
schema warning.

## What Belongs in the Schema

| Metadata | Why it matters |
|---|---|
| Column names | The set of released attributes is part of the query design. |
| Column types | Continuous and categorical columns use different encodings. |
| Categorical domains | Active-domain inference from private rows is not free. |
| Numeric bounds | Bounds control clipping and binning before backend fitting. |
| Nullable flags | Missing-value treatment changes the encoded domain. |

## Practical Guidance

- Prefer broad, policy-approved numeric bounds over min/max inferred from the
  private dataframe.
- Include all allowed categorical values, even if some are absent in the current
  private sample.
- Pass a dataframe whose columns exactly match the schema. If the private source
  has extra attributes, select the intended columns before calling `fit`.
- Treat row filtering, joins, and feature engineering before `fit` as part of
  the private data pipeline unless those rules are public.
- Keep benchmark and production schemas in version control when policy allows.
