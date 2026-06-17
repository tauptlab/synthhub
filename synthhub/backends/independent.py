"""A small DP one-way marginal baseline used for smoke tests and examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.errors import PrivacyBudgetError
from synthhub.reports import PrivacyReport


class IndependentMarginalAdapter:
    """Fit independent noisy one-way marginals.

    This backend exists so the public API, evaluation, and accounting contracts
    are executable without heavyweight optional research dependencies. It is not
    intended to replace AIM, MST, or PrivBayes for real benchmark claims.
    """

    name = "independent"

    def __init__(self, *, epsilon: float, delta: float | None = None, random_state=None, **_: object):
        if epsilon <= 0:
            raise PrivacyBudgetError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.delta = delta
        self.random_state = random_state

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> "FittedIndependentMarginal":
        rng = np.random.default_rng(self.random_state)
        column_count = max(len(context.domain), 1)
        epsilon_per_column = self.epsilon / column_count
        probabilities: dict[str, np.ndarray] = {}

        for column, domain_size in context.domain.items():
            counts = np.bincount(encoded_df[column].to_numpy(dtype=int), minlength=domain_size)
            noise = rng.laplace(loc=0.0, scale=2.0 / epsilon_per_column, size=domain_size)
            noisy = np.maximum(counts.astype(float) + noise, 0.0)
            if noisy.sum() == 0:
                probs = np.full(domain_size, 1.0 / domain_size)
            else:
                probs = noisy / noisy.sum()
            probabilities[column] = probs

        warnings = context.warnings + (
            "independent backend is a one-way marginal baseline; it does not preserve correlations",
        )
        report = PrivacyReport(
            method=context.method,
            requested_epsilon=self.epsilon,
            epsilon_spent=self.epsilon,
            delta=self.delta,
            accountant="sequential_laplace_one_way",
            backend="synthhub.independent",
            warnings=warnings,
        )
        return FittedIndependentMarginal(
            probabilities=probabilities,
            domain=context.domain,
            random_state=self.random_state,
            privacy_report=report,
        )


@dataclass
class FittedIndependentMarginal:
    probabilities: dict[str, np.ndarray]
    domain: dict[str, int]
    random_state: object
    privacy_report: PrivacyReport

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    def sample(self, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        data: dict[str, np.ndarray] = {}
        for column, domain_size in self.domain.items():
            data[column] = self._rng.choice(domain_size, size=n, p=self.probabilities[column])
        return pd.DataFrame(data).astype(int)

