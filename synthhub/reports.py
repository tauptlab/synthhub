"""Report dataclasses returned by SynthHub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrivacyReport:
    """Privacy accounting information for a fitted synthesizer."""

    method: str
    requested_epsilon: float
    epsilon_spent: float
    delta: float | None = None
    accountant: str = "backend"
    guarantee: str = "conditional_on_public_schema"
    backend: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "requested_epsilon": self.requested_epsilon,
            "epsilon_spent": self.epsilon_spent,
            "delta": self.delta,
            "accountant": self.accountant,
            "guarantee": self.guarantee,
            "backend": self.backend,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Utility and privacy evaluation output."""

    utility: dict[str, Any]
    privacy: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utility": self.utility,
            "privacy": self.privacy,
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }

