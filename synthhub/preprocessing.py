"""Dataframe encoding and decoding for discrete marginal backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from synthhub.errors import SchemaError
from synthhub.schema import MISSING_TOKEN, ColumnSpec, Schema


@dataclass(frozen=True)
class EncodedColumn:
    name: str
    kind: str
    domain_size: int
    categories: tuple[Any, ...] = ()
    bin_edges: tuple[float, ...] = ()


class TabularPreprocessor:
    """Map pandas dataframes to integer-coded discrete domains."""

    def __init__(self, schema: Schema, *, continuous_bins: int = 20):
        if continuous_bins < 1:
            raise SchemaError("continuous_bins must be >= 1")
        self.schema = schema
        self.continuous_bins = continuous_bins
        self.columns_: tuple[EncodedColumn, ...] | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)
        encoded: dict[str, np.ndarray] = {}
        encoded_columns: list[EncodedColumn] = []

        for spec in self.schema.columns:
            if spec.kind == "categorical":
                encoded[spec.name], column = self._encode_categorical(df[spec.name], spec)
            else:
                encoded[spec.name], column = self._encode_continuous(df[spec.name], spec)
            encoded_columns.append(column)

        self.columns_ = tuple(encoded_columns)
        return pd.DataFrame(encoded, index=df.index).astype(int)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise SchemaError("preprocessor is not fitted")
        self._validate_columns(df)
        encoded: dict[str, np.ndarray] = {}
        for spec in self.schema.columns:
            if spec.kind == "categorical":
                encoded[spec.name], _ = self._encode_categorical(df[spec.name], spec)
            else:
                encoded[spec.name], _ = self._encode_continuous(df[spec.name], spec)
        return pd.DataFrame(encoded, index=df.index).astype(int)

    def inverse_transform(self, encoded_df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise SchemaError("preprocessor is not fitted")
        decoded: dict[str, Any] = {}
        transforms = {column.name: column for column in self.columns_}
        self._validate_columns(encoded_df, expected_names=transforms.keys(), label="encoded dataframe")
        for spec in self.schema.columns:
            values = encoded_df[spec.name].to_numpy(dtype=int)
            column = transforms[spec.name]
            if spec.kind == "categorical":
                labels = []
                for value in values:
                    if value < 0 or value >= len(column.categories):
                        raise SchemaError(f"encoded value out of range for {spec.name!r}: {value}")
                    label = column.categories[value]
                    labels.append(pd.NA if label == MISSING_TOKEN else label)
                decoded[spec.name] = labels
            else:
                edges = np.asarray(column.bin_edges, dtype=float)
                mids = (edges[:-1] + edges[1:]) / 2.0
                clipped = np.clip(values, 0, len(mids) - 1)
                decoded[spec.name] = mids[clipped]
        return pd.DataFrame(decoded, columns=self.schema.names)

    @property
    def domain(self) -> dict[str, int]:
        if self.columns_ is None:
            raise SchemaError("preprocessor is not fitted")
        return {column.name: column.domain_size for column in self.columns_}

    def _validate_columns(
        self,
        df: pd.DataFrame,
        *,
        expected_names: Iterable[str] | None = None,
        label: str = "dataframe",
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise SchemaError(f"expected a pandas DataFrame for {label}")
        names = list(df.columns)
        if any(not isinstance(name, str) or not name for name in names):
            raise SchemaError(f"{label} column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise SchemaError(f"{label} column names must be unique")

        expected = tuple(expected_names or self.schema.names)
        missing = [name for name in expected if name not in df.columns]
        if missing:
            raise SchemaError(f"{label} is missing schema columns: {missing}")
        extra = [name for name in names if name not in expected]
        if extra:
            raise SchemaError(f"{label} has columns not present in schema: {extra}")

    def _encode_categorical(
        self, series: pd.Series, spec: ColumnSpec
    ) -> tuple[np.ndarray, EncodedColumn]:
        categories = tuple(spec.categories)
        mapping = {_category_key(value): idx for idx, value in enumerate(categories)}
        values = []
        for raw in series:
            value = MISSING_TOKEN if pd.isna(raw) else raw
            key = _category_key(value)
            if key not in mapping:
                raise SchemaError(
                    f"value {value!r} in column {spec.name!r} is not in the schema categories"
                )
            values.append(mapping[key])
        return (
            np.asarray(values, dtype=int),
            EncodedColumn(
                name=spec.name,
                kind=spec.kind,
                domain_size=len(categories),
                categories=categories,
            ),
        )

    def _encode_continuous(
        self, series: pd.Series, spec: ColumnSpec
    ) -> tuple[np.ndarray, EncodedColumn]:
        if spec.lower is None or spec.upper is None:
            raise SchemaError(f"continuous column {spec.name!r} is missing bounds")
        lower = float(spec.lower)
        upper = float(spec.upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise SchemaError(f"invalid bounds for column {spec.name!r}: {lower}, {upper}")

        edges = np.linspace(lower, upper, self.continuous_bins + 1)
        numeric = pd.to_numeric(series, errors="coerce").fillna(lower)
        clipped = np.clip(numeric.to_numpy(dtype=float), lower, upper)
        encoded = np.digitize(clipped, edges[1:-1], right=False)
        return (
            encoded.astype(int),
            EncodedColumn(
                name=spec.name,
                kind=spec.kind,
                domain_size=self.continuous_bins,
                bin_edges=tuple(float(edge) for edge in edges),
            ),
        )


def _category_key(value: Any) -> Any:
    if isinstance(value, str) and value == MISSING_TOKEN:
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_datetime64()
    return value
