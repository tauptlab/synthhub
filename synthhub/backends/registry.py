"""Backend registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synthhub.backends.datasynthesizer import DataSynthesizerAdapter
from synthhub.backends.independent import IndependentMarginalAdapter
from synthhub.backends.private_pgm import PrivatePGMAdapter
from synthhub.backends.smartnoise import SmartNoiseAdapter
from synthhub.backends.synthcity import SynthCityAdapter
from synthhub.errors import BackendNotAvailableError

BackendFactory = Callable[..., Any]

_BACKENDS: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    key = name.lower()
    if key in _BACKENDS and not replace:
        raise ValueError(f"backend already registered: {name}")
    _BACKENDS[key] = factory


def create_backend(name: str, **kwargs: Any) -> Any:
    key = name.lower()
    if key not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS))
        raise BackendNotAvailableError(f"unknown backend {name!r}; available: {available}")
    return _BACKENDS[key](**kwargs)


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def _register_builtins() -> None:
    register_backend("independent", IndependentMarginalAdapter, replace=True)
    register_backend(
        "aim",
        lambda **kwargs: PrivatePGMAdapter(mechanism="aim", **kwargs),
        replace=True,
    )
    register_backend(
        "mst",
        lambda **kwargs: PrivatePGMAdapter(mechanism="mst", **kwargs),
        replace=True,
    )
    register_backend(
        "privbayes",
        lambda **kwargs: DataSynthesizerAdapter(mode="correlated", **kwargs),
        replace=True,
    )
    register_backend(
        "datasynthesizer-privbayes",
        lambda **kwargs: DataSynthesizerAdapter(mode="correlated", **kwargs),
        replace=True,
    )
    register_backend(
        "datasynthesizer-independent",
        lambda **kwargs: DataSynthesizerAdapter(mode="independent", **kwargs),
        replace=True,
    )
    for method in ("mwem", "pacsynth", "dpctgan", "patectgan", "pategan", "dpgan", "quail"):
        register_backend(
            method,
            lambda synth=method, **kwargs: SmartNoiseAdapter(synth=synth, **kwargs),
            replace=True,
        )
    register_backend(
        "smartnoise-aim",
        lambda **kwargs: SmartNoiseAdapter(synth="aim", **kwargs),
        replace=True,
    )
    register_backend(
        "smartnoise-mst",
        lambda **kwargs: SmartNoiseAdapter(synth="mst", **kwargs),
        replace=True,
    )
    for method in ("synthcity-privbayes", "synthcity-pategan", "synthcity-dpgan"):
        plugin = method.removeprefix("synthcity-")
        register_backend(
            method,
            lambda plugin=plugin, **kwargs: SynthCityAdapter(plugin=plugin, **kwargs),
            replace=True,
        )


_register_builtins()
