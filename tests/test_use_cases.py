import json

import pandas as pd
import pytest

from synthhub import ColumnSpec, Schema, Synthesizer, infer_schema
from synthhub.errors import SchemaError
from synthhub.schema import MISSING_TOKEN


def test_healthcare_mixed_data_with_public_schema_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "age": [71, 64, None, 52, 88, 43, 59, 76],
            "sex": ["F", "M", "F", None, "F", "M", "M", "F"],
            "diagnosis": ["cardio", "ortho", "cardio", "neuro", "neuro", "ortho", "cardio", "neuro"],
            "readmitted": [1, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    schema = Schema(
        columns=(
            ColumnSpec("age", "continuous", "float64", lower=0, upper=100, nullable=True),
            ColumnSpec("sex", "categorical", "object", categories=("F", "M", MISSING_TOKEN), nullable=True),
            ColumnSpec("diagnosis", "categorical", "object", categories=("cardio", "neuro", "ortho")),
            ColumnSpec("readmitted", "categorical", "int64", categories=(0, 1)),
        ),
        source="public",
    )

    synth = Synthesizer(method="independent", epsilon=1.0, schema=schema, random_state=4)
    sample = synth.fit_sample(df, n=30)

    assert list(sample.columns) == list(df.columns)
    assert sample["age"].between(0, 100).all()
    assert set(sample["diagnosis"]).issubset({"cardio", "neuro", "ortho"})
    assert not any("schema was inferred" in warning for warning in synth.privacy_report_.warnings)


def test_finance_regression_scenario_evaluates_train_on_synthetic() -> None:
    df = pd.DataFrame(
        {
            "balance": [1000 + i * 175 for i in range(40)],
            "region": ["north", "south", "west", "east"] * 10,
            "loss": [float(i * 13 + (i % 3) * 7) for i in range(40)],
        }
    )

    synth = Synthesizer(method="independent", epsilon=2.0, random_state=12).fit(df)
    sample = synth.sample(80)
    report = synth.evaluate(df, sample, target="loss").to_dict()

    assert report["utility"]["train_on_synthetic"]["task"] == "regression"
    assert report["privacy"]["accounting"]["epsilon_spent"] == pytest.approx(2.0)


def test_datetime_boolean_and_categorical_columns_round_trip() -> None:
    df = pd.DataFrame(
        {
            "event_day": pd.date_range("2026-01-01", periods=12, freq="D"),
            "was_active": [True, False] * 6,
            "tier": ["free", "pro", "team"] * 4,
        }
    )

    sample = Synthesizer(method="independent", epsilon=1.0, random_state=8).fit_sample(df, n=20)

    assert list(sample.columns) == list(df.columns)
    assert set(sample["was_active"].dropna()).issubset({True, False})
    assert set(sample["tier"]).issubset({"free", "pro", "team"})


def test_numeric_age_like_column_infers_continuous_even_on_small_sample() -> None:
    df = pd.DataFrame(
        {
            "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )

    schema = infer_schema(df)

    assert schema.get("age").kind == "continuous"
    assert schema.get("churn").kind == "categorical"


def test_high_cardinality_categorical_requires_explicit_schema() -> None:
    df = pd.DataFrame({"user_id": [f"user-{idx}" for idx in range(101)]})

    with pytest.raises(SchemaError, match="pass an explicit Schema"):
        infer_schema(df, max_categories=100)


def test_public_schema_rejects_unexpected_private_columns() -> None:
    df = pd.DataFrame(
        {
            "age": [21, 34, 45, 52],
            "city": ["A", "B", "A", "C"],
            "ssn": ["111", "222", "333", "444"],
        }
    )
    schema = Schema(
        columns=(
            ColumnSpec("age", "continuous", "int64", lower=0, upper=100),
            ColumnSpec("city", "categorical", "object", categories=("A", "B", "C")),
        ),
        source="public",
    )

    with pytest.raises(SchemaError, match="not present in schema"):
        Synthesizer(method="independent", epsilon=1.0, schema=schema).fit(df)


def test_all_missing_and_constant_columns_still_produce_serializable_report() -> None:
    df = pd.DataFrame(
        {
            "constant_score": [5.0] * 12,
            "empty_note": [None] * 12,
            "flag": [True] * 12,
        }
    )

    synth = Synthesizer(method="independent", epsilon=1.0, random_state=9).fit(df)
    sample = synth.sample(20)
    report = synth.evaluate(df, sample).to_dict()

    assert list(sample.columns) == list(df.columns)
    assert set(sample["constant_score"].dropna()).issubset({5.0})
    assert sample["empty_note"].isna().all()
    assert set(sample["flag"].dropna()).issubset({True})
    json.dumps(report)
