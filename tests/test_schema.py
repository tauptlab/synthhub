import pandas as pd
import pytest

from synthhub import ColumnSpec, Schema
from synthhub.preprocessing import TabularPreprocessor
from synthhub.schema import MISSING_TOKEN, infer_schema
from synthhub.errors import SchemaError


def test_infer_schema_splits_categorical_and_continuous_columns() -> None:
    df = pd.DataFrame(
        {
            "score": list(range(30)),
            "group": ["a", "b", None] * 10,
        }
    )

    schema = infer_schema(df, categorical_threshold=5)

    assert schema.get("score").kind == "continuous"
    assert schema.get("group").kind == "categorical"
    assert MISSING_TOKEN in schema.get("group").categories
    assert schema.warnings


def test_preprocessor_round_trips_shape_and_columns() -> None:
    df = pd.DataFrame(
        {
            "score": list(range(30)),
            "group": ["a", "b", "c"] * 10,
        }
    )
    schema = infer_schema(df, categorical_threshold=2)
    preprocessor = TabularPreprocessor(schema, continuous_bins=5)

    encoded = preprocessor.fit_transform(df)
    decoded = preprocessor.inverse_transform(encoded)

    assert list(encoded.columns) == ["score", "group"]
    assert encoded["score"].between(0, 4).all()
    assert list(decoded.columns) == list(df.columns)
    assert len(decoded) == len(df)
    assert preprocessor.domain == {"score": 5, "group": 3}


def test_explicit_schema_rejects_duplicate_or_blank_columns() -> None:
    with pytest.raises(SchemaError, match="unique"):
        Schema(
            columns=(
                ColumnSpec("x", "categorical", "object", categories=("a",)),
                ColumnSpec("x", "categorical", "object", categories=("b",)),
            )
        )

    with pytest.raises(SchemaError, match="non-empty"):
        ColumnSpec("", "categorical", "object", categories=("a",))


def test_continuous_bins_must_be_positive_integer() -> None:
    schema = Schema(
        columns=(ColumnSpec("x", "continuous", "float64", lower=0.0, upper=1.0),),
        source="public",
    )

    for bad_bins in (0, -1, 1.5, True):
        with pytest.raises(SchemaError, match="positive integer"):
            TabularPreprocessor(schema, continuous_bins=bad_bins)  # type: ignore[arg-type]


def test_schema_inference_options_are_validated() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})

    for bad_threshold in (-1, 1.5, True):
        with pytest.raises(SchemaError, match="categorical_threshold"):
            infer_schema(df, categorical_threshold=bad_threshold)  # type: ignore[arg-type]

    for bad_max_categories in (0, -1, 1.5, True):
        with pytest.raises(SchemaError, match="max_categories"):
            infer_schema(df, max_categories=bad_max_categories)  # type: ignore[arg-type]
