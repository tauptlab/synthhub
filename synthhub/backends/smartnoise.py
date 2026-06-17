"""Adapters for SmartNoise Synthesizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.errors import BackendNotAvailableError, PrivacyBudgetError
from synthhub.reports import PrivacyReport


class SmartNoiseAdapter:
    """Wrap `snsynth.Synthesizer.create` behind the SynthHub backend contract."""

    name = "smartnoise"

    def __init__(
        self,
        *,
        synth: str,
        epsilon: float,
        delta: float | None = None,
        random_state=None,
        preprocessor_eps: float = 0.0,
        nullable: bool = False,
        **options: object,
    ):
        if epsilon <= 0:
            raise PrivacyBudgetError("epsilon must be positive")
        if preprocessor_eps < 0:
            raise PrivacyBudgetError("preprocessor_eps must be non-negative")
        if preprocessor_eps > epsilon:
            raise PrivacyBudgetError("preprocessor_eps cannot exceed epsilon")
        if options.get("disabled_dp") is True:
            raise PrivacyBudgetError("disabled_dp=True is not allowed for DP SynthHub backends")
        self.synth = synth
        self.epsilon = float(epsilon)
        self.delta = delta
        self.random_state = random_state
        self.preprocessor_eps = float(preprocessor_eps)
        self.nullable = nullable
        self.options = options

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> "FittedSmartNoise":
        try:
            from snsynth import Synthesizer as SmartNoiseSynthesizer
        except Exception as exc:
            raise BackendNotAvailableError(
                "SmartNoise backends require smartnoise-synth. Install with "
                "pip install 'synthhub[smartnoise]'."
            ) from exc

        if self.random_state is not None:
            np.random.seed(int(self.random_state))

        create_kwargs = dict(self.options)
        if self.delta is not None and "delta" not in create_kwargs:
            create_kwargs["delta"] = self.delta

        try:
            model = SmartNoiseSynthesizer.create(
                self.synth,
                epsilon=self.epsilon,
                **create_kwargs,
            )
            model.fit(
                encoded_df,
                categorical_columns=list(encoded_df.columns),
                continuous_columns=[],
                ordinal_columns=[],
                preprocessor_eps=self.preprocessor_eps,
                nullable=self.nullable,
            )
        except Exception as exc:
            raise BackendNotAvailableError(
                f"SmartNoise synthesizer {self.synth!r} could not fit the data: {exc}"
            ) from exc

        spent_epsilon, spent_delta = _read_smartnoise_spend(
            model,
            fallback_epsilon=self.epsilon,
            fallback_delta=self.delta,
        )
        warnings = context.warnings
        if self.preprocessor_eps == 0:
            warnings = warnings + (
                "SmartNoise preprocessing budget set to 0.0; SynthHub encoded the data before fitting",
            )

        report = PrivacyReport(
            method=context.method,
            requested_epsilon=self.epsilon,
            epsilon_spent=spent_epsilon,
            delta=spent_delta,
            accountant="smartnoise_odometer",
            backend=f"smartnoise:{self.synth}",
            warnings=warnings,
        )
        return FittedSmartNoise(model=model, domain=context.domain, privacy_report=report)


@dataclass
class FittedSmartNoise:
    model: Any
    domain: dict[str, int]
    privacy_report: PrivacyReport

    def sample(self, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        try:
            sampled = self.model.sample(n)
        except Exception as exc:
            raise BackendNotAvailableError(f"SmartNoise sample failed: {exc}") from exc
        return _coerce_encoded_frame(sampled, self.domain)


def _read_smartnoise_spend(
    model: Any,
    *,
    fallback_epsilon: float,
    fallback_delta: float | None,
) -> tuple[float, float | None]:
    odometer = getattr(model, "odometer", None)
    spent = getattr(odometer, "spent", None)
    if isinstance(spent, tuple) and spent:
        epsilon = float(spent[0])
        delta = float(spent[1]) if len(spent) > 1 and spent[1] is not None else fallback_delta
        if epsilon > 0:
            return epsilon, delta
    if isinstance(spent, (int, float)) and float(spent) > 0:
        return float(spent), fallback_delta
    return float(fallback_epsilon), fallback_delta


def _coerce_encoded_frame(sampled: Any, domain: dict[str, int]) -> pd.DataFrame:
    columns = list(domain)
    frame = pd.DataFrame(sampled)
    if set(columns).issubset(frame.columns):
        frame = frame.loc[:, columns]
    elif len(frame.columns) == len(columns):
        frame.columns = columns
    else:
        raise BackendNotAvailableError(
            f"backend returned columns {list(frame.columns)!r}, expected {columns!r}"
        )

    coerced: dict[str, np.ndarray] = {}
    for column, size in domain.items():
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).round().astype(int)
        coerced[column] = np.clip(values.to_numpy(dtype=int), 0, size - 1)
    return pd.DataFrame(coerced, columns=columns)
