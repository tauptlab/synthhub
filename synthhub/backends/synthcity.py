"""Adapters for SynthCity privacy plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.backends.smartnoise import _coerce_encoded_frame
from synthhub.errors import BackendNotAvailableError
from synthhub.reports import PrivacyReport
from synthhub.validation import validate_delta, validate_epsilon


class SynthCityAdapter:
    """Wrap SynthCity plugins behind the SynthHub backend contract."""

    name = "synthcity"

    def __init__(
        self,
        *,
        plugin: str,
        epsilon: float,
        delta: float | None = None,
        random_state=None,
        **options: object,
    ):
        self.plugin = plugin
        self.epsilon = validate_epsilon(epsilon)
        self.delta = validate_delta(delta)
        self.random_state = random_state
        self.options = options

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> "FittedSynthCity":
        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
        except Exception as exc:
            raise BackendNotAvailableError(
                "SynthCity backends require synthcity. Install with "
                "pip install 'synthhub[synthcity]' on a supported Python version."
            ) from exc

        options = dict(self.options)
        options.setdefault("epsilon", self.epsilon)
        if self.delta is not None:
            options.setdefault("delta", self.delta)
        if self.random_state is not None:
            options.setdefault("random_state", self.random_state)

        try:
            loader = GenericDataLoader(encoded_df)
            model = Plugins().get(self.plugin, **options)
            model.fit(loader)
        except Exception as exc:
            raise BackendNotAvailableError(
                f"SynthCity plugin {self.plugin!r} could not fit the data: {exc}"
            ) from exc

        report = PrivacyReport(
            method=context.method,
            requested_epsilon=self.epsilon,
            epsilon_spent=self.epsilon,
            delta=self.delta,
            accountant="synthcity_plugin",
            backend=f"synthcity:{self.plugin}",
            warnings=context.warnings
            + ("SynthCity epsilon accounting is plugin-specific; SynthHub validates adapter wiring",),
        )
        return FittedSynthCity(model=model, domain=context.domain, privacy_report=report)


@dataclass
class FittedSynthCity:
    model: Any
    domain: dict[str, int]
    privacy_report: PrivacyReport

    def sample(self, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        try:
            generated = self.model.generate(count=n)
            if hasattr(generated, "dataframe"):
                generated = generated.dataframe()
        except Exception as exc:
            raise BackendNotAvailableError(f"SynthCity sample failed: {exc}") from exc
        return _coerce_encoded_frame(generated, self.domain)
