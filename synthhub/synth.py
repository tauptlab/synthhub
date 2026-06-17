"""Public Synthesizer class."""

from __future__ import annotations

import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.backends.registry import available_methods, create_backend
from synthhub.errors import NotFittedError, PrivacyBudgetError
from synthhub.evaluation import evaluate as evaluate_frames
from synthhub.preprocessing import TabularPreprocessor
from synthhub.reports import EvaluationReport, PrivacyReport
from synthhub.schema import Schema, infer_schema


class Synthesizer:
    """Unified dataframe-first interface for DP synthetic data backends."""

    def __init__(
        self,
        *,
        method: str = "aim",
        epsilon: float = 1.0,
        delta: float | None = 1e-9,
        schema: Schema | None = None,
        continuous_bins: int = 20,
        random_state=None,
        **backend_options: object,
    ):
        if epsilon <= 0:
            raise PrivacyBudgetError("epsilon must be positive")
        self.method = method.lower()
        self.epsilon = float(epsilon)
        self.delta = delta
        self.schema = schema
        self.continuous_bins = continuous_bins
        self.random_state = random_state
        self.backend_options = backend_options

        self.schema_: Schema | None = None
        self.preprocessor_: TabularPreprocessor | None = None
        self.backend_: object | None = None
        self.fitted_backend_: object | None = None
        self.privacy_report_: PrivacyReport | None = None

    @classmethod
    def available_methods(cls) -> tuple[str, ...]:
        return available_methods()

    def fit(self, df: pd.DataFrame) -> "Synthesizer":
        schema = self.schema or infer_schema(df)
        preprocessor = TabularPreprocessor(schema, continuous_bins=self.continuous_bins)
        encoded = preprocessor.fit_transform(df)

        backend = create_backend(
            self.method,
            epsilon=self.epsilon,
            delta=self.delta,
            random_state=self.random_state,
            **self.backend_options,
        )
        context = FitContext(
            method=self.method,
            epsilon=self.epsilon,
            delta=self.delta,
            domain=preprocessor.domain,
            warnings=schema.warnings,
        )
        fitted = backend.fit(encoded, context)

        self.schema_ = schema
        self.preprocessor_ = preprocessor
        self.backend_ = backend
        self.fitted_backend_ = fitted
        self.privacy_report_ = fitted.privacy_report
        self._validate_privacy_report()
        return self

    def fit_sample(self, df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
        """Fit the synthesizer and return synthetic rows."""

        self.fit(df)
        return self.sample(len(df) if n is None else n)

    def sample(self, n: int) -> pd.DataFrame:
        if self.fitted_backend_ is None or self.preprocessor_ is None:
            raise NotFittedError("call fit(df) before sample(n)")
        encoded = self.fitted_backend_.sample(n)
        return self.preprocessor_.inverse_transform(encoded)

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        *,
        target: str | None = None,
    ) -> EvaluationReport:
        return evaluate_frames(
            real_df,
            synth_df,
            privacy_report=self.privacy_report_,
            target=target,
            random_state=self.random_state,
        )

    def _validate_privacy_report(self) -> None:
        if self.privacy_report_ is None:
            raise PrivacyBudgetError("backend did not return a privacy report")
        spent = float(self.privacy_report_.epsilon_spent)
        requested = float(self.epsilon)
        if spent < 0 or spent - requested > 1e-9:
            raise PrivacyBudgetError(
                f"backend reported epsilon_spent={spent}, exceeding requested epsilon={requested}"
            )
