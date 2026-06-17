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

print(synth_df.head())
print(synth.privacy_report_.to_dict())
print(report.to_dict())
