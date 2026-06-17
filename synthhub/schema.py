"""Schema inference for pandas dataframes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from synthhub.errors import SchemaError

MISSING_TOKEN = "__synthhub_missing__"


@dataclass(frozen=True)
class ColumnSpec:
    """Column-level schema used by SynthHub preprocessors and adapters."""

    name: str
    kind: str
    dtype: str
    categories: tuple[Any, ...] = field(default_factory=tuple)
    lower: float | None = None
    upper: float | None = None
    nullable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError("column names must be non-empty strings")
        if self.kind not in {"categorical", "continuous"}:
            raise SchemaError(f"unsupported column kind for {self.name!r}: {self.kind!r}")
        if self.kind == "categorical" and not self.categories:
            raise SchemaError(f"categorical column {self.name!r} must define categories")
        if self.kind == "continuous" and (self.lower is None or self.upper is None):
            raise SchemaError(f"continuous column {self.name!r} must define lower and upper")


@dataclass(frozen=True)
class Schema:
    """Tabular schema.

    Formal DP guarantees are conditional on this schema being public or otherwise
    allowed by the caller's privacy policy. Inferred schemas are convenient but
    should be treated as an explicit trust decision.
    """

    columns: tuple[ColumnSpec, ...]
    source: str = "inferred"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.columns:
            raise SchemaError("schema must contain at least one column")
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise SchemaError("schema column names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def get(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise SchemaError(f"unknown schema column: {name}")

    def as_public(self) -> "Schema":
        return Schema(columns=self.columns, source="public", warnings=())


def infer_schema(
    df: pd.DataFrame,
    *,
    categorical_threshold: int = 20,
    max_categories: int = 100,
) -> Schema:
    """Infer a pragmatic public-facing schema from a pandas dataframe."""

    _validate_dataframe(df)
    warnings = [
        "schema was inferred from input data; formal DP is conditional on treating "
        "column types, categories, and bounds as public metadata"
    ]
    columns: list[ColumnSpec] = []

    for name in df.columns:
        series = df[name]
        nullable = bool(series.isna().any())
        if _should_be_categorical(series, categorical_threshold):
            categories = _infer_categories(series, max_categories=max_categories)
            columns.append(
                ColumnSpec(
                    name=str(name),
                    kind="categorical",
                    dtype=str(series.dtype),
                    categories=categories,
                    nullable=nullable,
                )
            )
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
        finite = finite.dropna()
        if finite.empty:
            categories = _infer_categories(series, max_categories=max_categories)
            columns.append(
                ColumnSpec(
                    name=str(name),
                    kind="categorical",
                    dtype=str(series.dtype),
                    categories=categories,
                    nullable=nullable,
                )
            )
            continue

        lower = float(finite.min())
        upper = float(finite.max())
        if lower == upper:
            upper = lower + 1.0
            warnings.append(f"continuous column {name!r} had one value; expanded upper bound by 1.0")
        if nullable:
            warnings.append(f"missing numeric values in {name!r} are encoded into the first bin")
        columns.append(
            ColumnSpec(
                name=str(name),
                kind="continuous",
                dtype=str(series.dtype),
                lower=lower,
                upper=upper,
                nullable=nullable,
            )
        )

    return Schema(columns=tuple(columns), source="inferred", warnings=tuple(warnings))


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("expected a pandas DataFrame")
    if df.empty:
        raise SchemaError("cannot fit an empty dataframe")
    names = [str(name) for name in df.columns]
    if any(not isinstance(name, str) for name in df.columns):
        raise SchemaError("dataframe column names must be strings")
    if any(not name for name in names):
        raise SchemaError("dataframe column names must be non-empty")
    if len(set(names)) != len(names):
        raise SchemaError("dataframe column names must be unique")


def _should_be_categorical(series: pd.Series, categorical_threshold: int) -> bool:
    if is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if not is_numeric_dtype(series):
        return True
    unique = int(series.nunique(dropna=True))
    if unique <= 2:
        return True
    non_null = int(series.notna().sum())
    if non_null == 0:
        return True
    unique_ratio = unique / non_null
    return unique <= categorical_threshold and unique_ratio <= 0.2


def _infer_categories(series: pd.Series, *, max_categories: int) -> tuple[Any, ...]:
    values: Iterable[Any] = pd.unique(series.dropna())
    categories = tuple(_stable_sort(values))
    if series.isna().any():
        categories = categories + (MISSING_TOKEN,)
    if len(categories) > max_categories:
        raise SchemaError(
            f"column {series.name!r} has {len(categories)} categories; "
            f"pass an explicit Schema or raise max_categories"
        )
    if not categories:
        return (MISSING_TOKEN,)
    return categories


def _stable_sort(values: Iterable[Any]) -> list[Any]:
    return sorted(list(values), key=lambda value: (str(type(value)), str(value)))
