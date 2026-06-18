"""Utility metrics for synthetic tabular data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


def utility_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    target: str | None = None,
    random_state=None,
) -> dict[str, Any]:
    _validate_compatible_frames(real_df, synth_df)
    column_scores = {
        column: _column_similarity(real_df[column], synth_df[column]) for column in real_df.columns
    }
    utility: dict[str, Any] = {
        "shape": {
            "real_rows": int(len(real_df)),
            "synthetic_rows": int(len(synth_df)),
            "columns": int(len(real_df.columns)),
        },
        "column_similarity": column_scores,
        "mean_column_similarity": float(np.mean(list(column_scores.values()))),
        "correlation_similarity": _correlation_similarity(real_df, synth_df),
    }
    if target is not None:
        utility["train_on_synthetic"] = _train_on_synthetic_score(
            real_df,
            synth_df,
            target=target,
            random_state=random_state,
        )
    return utility


def _validate_compatible_frames(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    if list(real_df.columns) != list(synth_df.columns):
        raise ValueError("real_df and synth_df must have the same columns in the same order")
    if real_df.empty or synth_df.empty:
        raise ValueError("real_df and synth_df must be non-empty")


def _column_similarity(real: pd.Series, synth: pd.Series) -> float:
    if is_numeric_dtype(real):
        return _numeric_histogram_similarity(real, synth)
    return _categorical_similarity(real, synth)


def _categorical_similarity(real: pd.Series, synth: pd.Series) -> float:
    real_counts = _categorical_values(real).value_counts(normalize=True)
    synth_counts = _categorical_values(synth).value_counts(normalize=True)
    labels = sorted(set(real_counts.index) | set(synth_counts.index), key=str)
    real_p = np.asarray([real_counts.get(label, 0.0) for label in labels])
    synth_p = np.asarray([synth_counts.get(label, 0.0) for label in labels])
    tvd = 0.5 * np.abs(real_p - synth_p).sum()
    return float(max(0.0, 1.0 - tvd))


def _numeric_histogram_similarity(real: pd.Series, synth: pd.Series, bins: int = 10) -> float:
    real_numeric = pd.to_numeric(real, errors="coerce").dropna().to_numpy(dtype=float)
    synth_numeric = pd.to_numeric(synth, errors="coerce").dropna().to_numpy(dtype=float)
    if real_numeric.size == 0 or synth_numeric.size == 0:
        return 0.0
    lower = float(np.min(real_numeric))
    upper = float(np.max(real_numeric))
    if lower == upper:
        return 1.0 if np.allclose(synth_numeric, lower) else 0.0
    edges = np.linspace(lower, upper, bins + 1)
    real_hist, _ = np.histogram(np.clip(real_numeric, lower, upper), bins=edges)
    synth_hist, _ = np.histogram(np.clip(synth_numeric, lower, upper), bins=edges)
    real_p = real_hist / max(real_hist.sum(), 1)
    synth_p = synth_hist / max(synth_hist.sum(), 1)
    tvd = 0.5 * np.abs(real_p - synth_p).sum()
    return float(max(0.0, 1.0 - tvd))


def _correlation_similarity(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict[str, Any]:
    numeric = [column for column in real_df.columns if is_numeric_dtype(real_df[column])]
    if len(numeric) < 2:
        return {"status": "skipped", "reason": "fewer than two numeric columns"}
    real_corr = real_df[numeric].corr(numeric_only=True).fillna(0.0).to_numpy()
    synth_corr = synth_df[numeric].corr(numeric_only=True).fillna(0.0).to_numpy()
    mae = float(np.mean(np.abs(real_corr - synth_corr)))
    return {"mean_absolute_error": mae, "similarity": float(max(0.0, 1.0 - min(mae / 2.0, 1.0)))}


def _train_on_synthetic_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    target: str,
    random_state=None,
) -> dict[str, Any]:
    if target not in real_df.columns:
        raise ValueError(f"target column not found: {target}")
    if target not in synth_df.columns:
        raise ValueError(f"target column not found in synthetic data: {target}")

    x_synth = synth_df.drop(columns=[target])
    y_synth = synth_df[target]
    x_real = real_df.drop(columns=[target])
    y_real = real_df[target]
    x_synth_enc, x_real_enc = _align_features(x_synth, x_real)

    if _is_classification(y_real):
        model = RandomForestClassifier(n_estimators=50, random_state=random_state)
        model.fit(x_synth_enc, y_synth.astype(str))
        pred = model.predict(x_real_enc)
        return {"task": "classification", "accuracy": float(accuracy_score(y_real.astype(str), pred))}

    model = RandomForestRegressor(n_estimators=50, random_state=random_state)
    model.fit(x_synth_enc, pd.to_numeric(y_synth, errors="coerce").fillna(0.0))
    pred = model.predict(x_real_enc)
    score = r2_score(pd.to_numeric(y_real, errors="coerce").fillna(0.0), pred)
    return {"task": "regression", "r2": float(score)}


def _align_features(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_encoded = pd.get_dummies(_normalize_categorical_missing(left), dummy_na=False)
    right_encoded = pd.get_dummies(_normalize_categorical_missing(right), dummy_na=False)
    left_aligned, right_aligned = left_encoded.align(right_encoded, join="outer", axis=1, fill_value=0)
    return left_aligned.fillna(0.0).astype(float), right_aligned.fillna(0.0).astype(float)


def _is_classification(series: pd.Series) -> bool:
    if not is_numeric_dtype(series):
        return True
    return int(series.nunique(dropna=True)) <= 20


def _normalize_categorical_missing(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series):
            normalized[column] = series
            continue
        normalized[column] = _categorical_values(series)
    return pd.DataFrame(normalized, index=frame.index)


def _categorical_values(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    values = values.where(values.notna(), "__missing__")
    return values.map(_feature_key)


def _feature_key(value: Any) -> str:
    if value == "__missing__":
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
