"""Adapters for Private-PGM mechanisms such as AIM and MST."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from synthhub.backends.base import FitContext
from synthhub.errors import BackendNotAvailableError
from synthhub.reports import PrivacyReport
from synthhub.validation import validate_delta, validate_epsilon


class PrivatePGMAdapter:
    """Wrap AIM or MST from the Private-PGM mechanisms folder."""

    name = "private-pgm"

    def __init__(
        self,
        *,
        mechanism: str,
        epsilon: float,
        delta: float | None = 1e-9,
        random_state=None,
        degree: int = 2,
        rounds: int | None = None,
        max_cells: int = 10_000,
        max_model_size: float = 1.0,
        max_iters: int = 1_000,
        verbose: bool = False,
        **_: object,
    ):
        self.mechanism = mechanism.lower()
        self.epsilon = validate_epsilon(epsilon)
        self.delta = validate_delta(1e-9 if delta is None else delta)
        self.random_state = random_state
        self.degree = degree
        self.rounds = rounds
        self.max_cells = max_cells
        self.max_model_size = max_model_size
        self.max_iters = max_iters
        self.verbose = verbose

    def fit(self, encoded_df: pd.DataFrame, context: FitContext) -> "FittedPrivatePGM":
        imports = _load_private_pgm(self.mechanism)
        domain = _make_domain(imports.Domain, context.domain)
        data = {
            column: encoded_df[column].to_numpy(dtype=int)
            for column in context.domain
        }
        dataset = imports.Dataset(data, domain)

        if self.random_state is not None:
            np.random.seed(int(self.random_state))

        with _output_context(self.verbose):
            if self.mechanism == "aim":
                workload = _build_workload(domain, context.domain, self.degree, self.max_cells)
                mechanism = imports.AIM(
                    self.epsilon,
                    self.delta,
                    rounds=self.rounds,
                    max_model_size=self.max_model_size,
                    max_iters=self.max_iters,
                )
                model, synth = mechanism.run(dataset, workload, num_synth_rows=len(encoded_df))
                synthetic = _dataset_to_frame(synth, context.domain)
                fitted_model = model
            elif self.mechanism == "mst":
                synth = imports.MST(dataset, self.epsilon, self.delta)
                synthetic = _dataset_to_frame(synth, context.domain)
                fitted_model = None
            else:
                raise BackendNotAvailableError(f"unsupported Private-PGM mechanism: {self.mechanism}")

        report = PrivacyReport(
            method=context.method,
            requested_epsilon=self.epsilon,
            epsilon_spent=self.epsilon,
            delta=self.delta,
            accountant="private_pgm_mechanism",
            backend=f"private-pgm:{self.mechanism}",
            warnings=context.warnings,
        )
        return FittedPrivatePGM(
            synthetic=synthetic,
            domain=context.domain,
            privacy_report=report,
            model=fitted_model,
        )


@dataclass
class FittedPrivatePGM:
    synthetic: pd.DataFrame
    domain: dict[str, int]
    privacy_report: PrivacyReport
    model: Any = None

    def sample(self, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        if self.model is not None and hasattr(self.model, "synthetic_data"):
            try:
                return _dataset_to_frame(self.model.synthetic_data(rows=n), self.domain)
            except TypeError:
                pass
        replace = len(self.synthetic) < n
        return self.synthetic.sample(n=n, replace=replace, random_state=None).reset_index(drop=True)


@dataclass(frozen=True)
class _PrivatePGMImports:
    Dataset: Any
    Domain: Any
    AIM: Any | None = None
    MST: Any | None = None


def _load_private_pgm(mechanism: str) -> _PrivatePGMImports:
    try:
        from mbi import Dataset, Domain
    except Exception as exc:
        raise BackendNotAvailableError(
            "AIM/MST require Private-PGM's mbi package. Install with "
            "pip install 'synthhub[private-pgm]'."
        ) from exc

    if mechanism == "aim":
        try:
            from mechanisms.aim import AIM
        except Exception as exc:
            raise BackendNotAvailableError(
                "AIM requires Private-PGM's mechanisms/aim.py to be importable. "
                "Clone https://github.com/ryan112358/private-pgm and add the repo root "
                "and mechanisms folder to PYTHONPATH."
            ) from exc
        return _PrivatePGMImports(Dataset=Dataset, Domain=Domain, AIM=AIM)

    if mechanism == "mst":
        try:
            from mechanisms.mst import MST
        except Exception as exc:
            raise BackendNotAvailableError(
                "MST requires Private-PGM's mechanisms/mst.py to be importable. "
                "Clone https://github.com/ryan112358/private-pgm and add the repo root "
                "and mechanisms folder to PYTHONPATH."
            ) from exc
        return _PrivatePGMImports(Dataset=Dataset, Domain=Domain, MST=MST)

    raise BackendNotAvailableError(f"unknown Private-PGM mechanism: {mechanism}")


def _make_domain(domain_cls: Any, domain: dict[str, int]) -> Any:
    try:
        return domain_cls.fromdict(domain)
    except AttributeError:
        return domain_cls(list(domain.keys()), list(domain.values()))


def _build_workload(domain_obj: Any, domain: dict[str, int], degree: int, max_cells: int) -> list[tuple[tuple[str, ...], float]]:
    attrs = list(domain.keys())
    cliques = list(combinations(attrs, max(1, degree)))
    workload: list[tuple[tuple[str, ...], float]] = []
    for clique in cliques:
        try:
            cells = int(domain_obj.size(clique))
        except Exception:
            cells = int(np.prod([domain[column] for column in clique]))
        if cells <= max_cells:
            workload.append((tuple(clique), 1.0))
    if not workload:
        workload = [((column,), 1.0) for column in attrs]
    return workload


def _dataset_to_frame(dataset: Any, domain: dict[str, int]) -> pd.DataFrame:
    columns = list(domain.keys())
    if hasattr(dataset, "df"):
        frame = pd.DataFrame(dataset.df)
    elif hasattr(dataset, "to_dict"):
        frame = pd.DataFrame(dataset.to_dict(), columns=columns)
    else:
        frame = pd.DataFrame(np.asarray(dataset), columns=columns)
    frame = frame.loc[:, columns]
    return frame.astype(int).reset_index(drop=True)


def _output_context(verbose: bool):
    if verbose:
        return nullcontext()
    return redirect_stdout(io.StringIO())
