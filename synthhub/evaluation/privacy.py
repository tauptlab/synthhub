"""Privacy-oriented evaluation heuristics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances

from synthhub.reports import PrivacyReport


def privacy_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    privacy_report: PrivacyReport | None = None,
    random_state=None,
    max_rows: int = 1_000,
) -> dict[str, Any]:
    report = privacy_report.to_dict() if privacy_report is not None else None
    risk = membership_inference_risk(
        real_df,
        synth_df,
        random_state=random_state,
        max_rows=max_rows,
    )
    return {
        "accounting": report,
        "membership_inference": risk,
    }


def membership_inference_risk(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    random_state=None,
    max_rows: int = 1_000,
) -> dict[str, Any]:
    if len(real_df) < 4 or len(synth_df) < 2:
        return {"status": "skipped", "reason": "not enough rows"}

    rng = np.random.default_rng(random_state)
    real_sample = real_df.sample(
        n=min(len(real_df), max_rows),
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)
    synth_sample = synth_df.sample(
        n=min(len(synth_df), max_rows),
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)

    split = len(real_sample) // 2
    members = real_sample.iloc[:split]
    nonmembers = real_sample.iloc[split : split * 2]
    if members.empty or nonmembers.empty:
        return {"status": "skipped", "reason": "not enough rows after split"}

    member_scores = -_nearest_distances(members, synth_sample)
    nonmember_scores = -_nearest_distances(nonmembers, synth_sample)
    y_true = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    y_score = np.concatenate([member_scores, nonmember_scores])
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "status": "ok",
        "attack": "nearest_neighbor_shadow_split",
        "auc": auc,
        "risk_score": float(max(0.0, (auc - 0.5) * 2.0)),
        "member_mean_distance": float((-member_scores).mean()),
        "nonmember_mean_distance": float((-nonmember_scores).mean()),
        "note": "heuristic audit metric, not a formal privacy guarantee",
    }


def _nearest_distances(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    left_encoded, right_encoded = _align_numeric(left, right)
    distances = pairwise_distances(left_encoded, right_encoded, metric="euclidean")
    return distances.min(axis=1)


def _align_numeric(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([left, right], axis=0, ignore_index=True)
    encoded = pd.get_dummies(_normalize_categorical_missing(combined), dummy_na=False)
    numeric = encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    std = numeric.std(axis=0).replace(0.0, 1.0)
    numeric = (numeric - numeric.mean(axis=0)) / std
    return numeric.iloc[: len(left)], numeric.iloc[len(left) :]


def _normalize_categorical_missing(frame: pd.DataFrame) -> pd.DataFrame:
    normalized: dict[str, pd.Series] = {}
    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series):
            normalized[column] = series
            continue
        values = series.astype("object")
        values = values.where(values.notna(), "__missing__")
        normalized[column] = values.map(_feature_key)
    return pd.DataFrame(normalized, index=frame.index)


def _feature_key(value: Any) -> str:
    if value == "__missing__":
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
