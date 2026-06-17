"""Backend adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from synthhub.reports import PrivacyReport


@dataclass(frozen=True)
class FitContext:
    """Information every backend receives after preprocessing."""

    method: str
    epsilon: float
    delta: float | None
    domain: dict[str, int]
    warnings: tuple[str, ...] = ()


class FittedBackend(Protocol):
    privacy_report: PrivacyReport

    def sample(self, n: int) -> pd.DataFrame:
        """Return integer-coded synthetic rows."""


class BackendAdapter(Protocol):
    name: str

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> FittedBackend:
        """Fit a backend on integer-coded data."""

