"""Validation helpers for privacy parameters."""

from __future__ import annotations

import math

from synthhub.errors import PrivacyBudgetError


def validate_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool):
        raise PrivacyBudgetError("epsilon must be a finite positive number")
    try:
        value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise PrivacyBudgetError("epsilon must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise PrivacyBudgetError("epsilon must be a finite positive number")
    return value


def validate_delta(delta: float | None) -> float | None:
    if delta is None:
        return None
    if isinstance(delta, bool):
        raise PrivacyBudgetError("delta must be None or a finite number in [0, 1)")
    try:
        value = float(delta)
    except (TypeError, ValueError) as exc:
        raise PrivacyBudgetError("delta must be None or a finite number in [0, 1)") from exc
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise PrivacyBudgetError("delta must be None or a finite number in [0, 1)")
    return value
