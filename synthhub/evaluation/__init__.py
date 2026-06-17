"""Evaluation entry points."""

from __future__ import annotations

import pandas as pd

from synthhub.evaluation.privacy import membership_inference_risk, privacy_metrics
from synthhub.evaluation.utility import utility_metrics
from synthhub.reports import EvaluationReport, PrivacyReport


def evaluate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    privacy_report: PrivacyReport | None = None,
    target: str | None = None,
    random_state=None,
) -> EvaluationReport:
    """Evaluate synthetic data utility and privacy risk heuristics."""

    utility = utility_metrics(real_df, synth_df, target=target, random_state=random_state)
    privacy = privacy_metrics(
        real_df,
        synth_df,
        privacy_report=privacy_report,
        random_state=random_state,
    )
    return EvaluationReport(
        utility=utility,
        privacy=privacy,
        metadata={
            "real_rows": int(len(real_df)),
            "synthetic_rows": int(len(synth_df)),
            "target": target,
        },
    )


__all__ = ["evaluate", "membership_inference_risk", "privacy_metrics", "utility_metrics"]

