"""Adapters for DataSynthesizer."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
import io
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.backends.smartnoise import _coerce_encoded_frame
from synthhub.errors import BackendNotAvailableError, PrivacyBudgetError
from synthhub.reports import PrivacyReport


class DataSynthesizerAdapter:
    """Wrap DataSynthesizer's DP independent and correlated modes."""

    name = "datasynthesizer"

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float | None = None,
        random_state=None,
        mode: str = "correlated",
        max_parents: int = 2,
        histogram_bins: int = 20,
        verbose: bool = False,
        **_: object,
    ):
        if epsilon <= 0:
            raise PrivacyBudgetError("epsilon must be positive")
        if max_parents < 0:
            raise ValueError("max_parents must be non-negative")
        if mode not in {"correlated", "independent"}:
            raise ValueError("mode must be 'correlated' or 'independent'")
        self.epsilon = float(epsilon)
        self.delta = delta
        self.random_state = random_state
        self.mode = mode
        self.max_parents = max_parents
        self.histogram_bins = histogram_bins
        self.verbose = verbose

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> "FittedDataSynthesizer":
        try:
            from DataSynthesizer.DataDescriber import DataDescriber
        except Exception as exc:
            raise BackendNotAvailableError(
                "DataSynthesizer backends require DataSynthesizer. Install with "
                "pip install 'synthhub[datasynthesizer]'."
            ) from exc

        tmpdir = tempfile.TemporaryDirectory(prefix="synthhub-datasynthesizer-")
        tmp_path = Path(tmpdir.name)
        input_file = tmp_path / "encoded.csv"
        domain_file = tmp_path / "domain.json"
        description_file = tmp_path / "description.json"

        encoded_df.to_csv(input_file, index=False)
        domain_file.write_text(
            json.dumps({column: list(range(size)) for column, size in context.domain.items()}),
            encoding="utf-8",
        )

        attr_to_datatype = {column: "Integer" for column in context.domain}
        attr_to_is_categorical = {column: True for column in context.domain}
        attr_to_is_candidate_key = {column: False for column in context.domain}
        seed = 0 if self.random_state is None else int(self.random_state)
        mode_used = self.mode
        if mode_used == "correlated" and len(context.domain) < 2:
            mode_used = "independent"

        try:
            describer = DataDescriber(histogram_bins=self.histogram_bins)
            with _output_context(self.verbose):
                if mode_used == "correlated":
                    describer.describe_dataset_in_correlated_attribute_mode(
                        str(input_file),
                        k=self.max_parents,
                        epsilon=self.epsilon,
                        attribute_to_datatype=attr_to_datatype,
                        attribute_to_is_categorical=attr_to_is_categorical,
                        attribute_to_is_candidate_key=attr_to_is_candidate_key,
                        categorical_attribute_domain_file=str(domain_file),
                        seed=seed,
                    )
                else:
                    describer.describe_dataset_in_independent_attribute_mode(
                        str(input_file),
                        epsilon=self.epsilon,
                        attribute_to_datatype=attr_to_datatype,
                        attribute_to_is_categorical=attr_to_is_categorical,
                        attribute_to_is_candidate_key=attr_to_is_candidate_key,
                        categorical_attribute_domain_file=str(domain_file),
                        seed=seed,
                    )
                describer.save_dataset_description_to_file(str(description_file))
        except Exception as exc:
            tmpdir.cleanup()
            raise BackendNotAvailableError(f"DataSynthesizer fit failed: {exc}") from exc

        warnings = context.warnings + (
            "DataSynthesizer uses active-domain categorical DP synthesis over SynthHub-encoded columns",
        )
        if mode_used != self.mode:
            warnings = warnings + ("DataSynthesizer correlated mode requires at least two columns; used independent mode",)

        report = PrivacyReport(
            method=context.method,
            requested_epsilon=self.epsilon,
            epsilon_spent=self.epsilon,
            delta=self.delta,
            accountant=f"datasynthesizer_{mode_used}",
            backend=f"datasynthesizer:{mode_used}",
            warnings=warnings,
        )
        return FittedDataSynthesizer(
            description_file=description_file,
            tmpdir=tmpdir,
            domain=context.domain,
            mode=mode_used,
            random_state=self.random_state,
            verbose=self.verbose,
            privacy_report=report,
        )


@dataclass
class FittedDataSynthesizer:
    description_file: Path
    tmpdir: tempfile.TemporaryDirectory[str]
    domain: dict[str, int]
    mode: str
    random_state: object
    verbose: bool
    privacy_report: PrivacyReport

    def sample(self, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        try:
            from DataSynthesizer.DataGenerator import DataGenerator
        except Exception as exc:
            raise BackendNotAvailableError(
                "DataSynthesizer generator is unavailable after fitting"
            ) from exc

        seed = 0 if self.random_state is None else int(self.random_state)
        try:
            generator = DataGenerator()
            with _output_context(self.verbose):
                if self.mode == "correlated":
                    generator.generate_dataset_in_correlated_attribute_mode(
                        n,
                        str(self.description_file),
                        seed=seed,
                    )
                else:
                    generator.generate_dataset_in_independent_mode(
                        n,
                        str(self.description_file),
                        seed=seed,
                    )
        except Exception as exc:
            raise BackendNotAvailableError(f"DataSynthesizer sample failed: {exc}") from exc
        return _coerce_encoded_frame(generator.synthetic_dataset, self.domain)

    def __del__(self) -> None:
        try:
            self.tmpdir.cleanup()
        except Exception:
            pass


@contextmanager
def _output_context(verbose: bool):
    if verbose:
        with nullcontext():
            yield
        return
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"DataSynthesizer\..*")
        yield
