import pandas as pd
import pytest

from synthhub.backends.registry import register_backend
from synthhub.reports import PrivacyReport
from synthhub.synth import Synthesizer
from synthhub.errors import PrivacyBudgetError


class ContractBackend:
    def __init__(self, *, epsilon, delta=None, random_state=None, overspend=False, **kwargs):
        self.epsilon = epsilon
        self.delta = delta
        self.overspend = overspend

    def fit(self, encoded_df, context):
        spent = self.epsilon + 0.1 if self.overspend else self.epsilon

        class Fitted:
            privacy_report = PrivacyReport(
                method=context.method,
                requested_epsilon=context.epsilon,
                epsilon_spent=spent,
                delta=context.delta,
                accountant="contract-test",
            )

            def sample(self, n):
                return pd.DataFrame({column: [0] * n for column in context.domain})

        return Fitted()


def test_custom_backend_receives_epsilon_and_domain() -> None:
    register_backend("contract-ok", ContractBackend, replace=True)
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})

    synth = Synthesizer(method="contract-ok", epsilon=0.7).fit(df)

    assert synth.privacy_report_.requested_epsilon == pytest.approx(0.7)
    assert synth.privacy_report_.epsilon_spent == pytest.approx(0.7)


def test_backend_cannot_report_more_epsilon_than_requested() -> None:
    register_backend(
        "contract-overspend",
        lambda **kwargs: ContractBackend(overspend=True, **kwargs),
        replace=True,
    )
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})

    with pytest.raises(PrivacyBudgetError, match="exceeding requested epsilon"):
        Synthesizer(method="contract-overspend", epsilon=0.7).fit(df)

