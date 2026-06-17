"""SynthHub exception types."""


class SynthHubError(Exception):
    """Base class for SynthHub errors."""


class BackendNotAvailableError(SynthHubError):
    """Raised when a requested optional backend cannot be imported or used."""


class NotFittedError(SynthHubError):
    """Raised when sampling is requested before fitting."""


class PrivacyBudgetError(SynthHubError):
    """Raised when privacy parameters are invalid or inconsistent."""


class SchemaError(SynthHubError):
    """Raised when a dataframe cannot be mapped to a SynthHub schema."""

