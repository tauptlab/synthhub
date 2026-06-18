"""Adapters for SmartNoise Synthesizers."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
import math
from typing import Any

import numpy as np
import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.errors import BackendNotAvailableError, PrivacyBudgetError
from synthhub.reports import PrivacyReport
from synthhub.validation import validate_delta, validate_epsilon


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
        epsilon_value = validate_epsilon(epsilon)
        delta_value = validate_delta(delta)
        try:
            preprocessor_eps_value = float(preprocessor_eps)
        except (TypeError, ValueError) as exc:
            raise PrivacyBudgetError("preprocessor_eps must be non-negative") from exc
        if not math.isfinite(preprocessor_eps_value) or preprocessor_eps_value < 0:
            raise PrivacyBudgetError("preprocessor_eps must be non-negative")
        if preprocessor_eps_value > epsilon_value:
            raise PrivacyBudgetError("preprocessor_eps cannot exceed epsilon")
        if options.get("disabled_dp") is True:
            raise PrivacyBudgetError("disabled_dp=True is not allowed for DP SynthHub backends")
        self.synth = synth
        self.epsilon = epsilon_value
        self.delta = delta_value
        self.random_state = random_state
        self.preprocessor_eps = preprocessor_eps_value
        self.nullable = nullable
        self.verbose = bool(options.get("verbose", False))
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

        try:
            with _output_context(self.verbose):
                model, spent_delta_fallback, delta_warning = _create_smartnoise_model(
                    SmartNoiseSynthesizer,
                    self.synth,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    options=create_kwargs,
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
            fallback_delta=spent_delta_fallback,
        )
        warnings = context.warnings + delta_warning
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


def _create_smartnoise_model(
    smartnoise_synthesizer: Any,
    synth: str,
    *,
    epsilon: float,
    delta: float | None,
    options: dict[str, object],
) -> tuple[Any, float | None, tuple[str, ...]]:
    create_kwargs = dict(options)
    injected_delta = delta is not None and "delta" not in create_kwargs
    if injected_delta:
        create_kwargs["delta"] = delta

    try:
        return (
            smartnoise_synthesizer.create(synth, epsilon=epsilon, **create_kwargs),
            delta,
            (),
        )
    except TypeError as exc:
        if not injected_delta or "delta" not in str(exc):
            raise

    create_kwargs.pop("delta", None)
    warning = (
        f"SmartNoise synthesizer {synth!r} does not accept delta; fitted as an epsilon-only mechanism",
    )
    return (
        smartnoise_synthesizer.create(synth, epsilon=epsilon, **create_kwargs),
        None,
        warning,
    )


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


def _output_context(verbose: bool):
    if verbose:
        return nullcontext()
    return redirect_stdout(io.StringIO())
