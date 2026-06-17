import sys

import pandas as pd
import pytest

from synthhub import Synthesizer
from synthhub.errors import BackendNotAvailableError, NotFittedError, PrivacyBudgetError


def demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
            "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )


def test_independent_backend_fits_samples_and_evaluates() -> None:
    df = demo_df()
    synth = Synthesizer(method="independent", epsilon=1.0, random_state=7)

    synth.fit(df)
    sample = synth.sample(25)
    report = synth.evaluate(df, sample, target="churn").to_dict()

    assert list(sample.columns) == list(df.columns)
    assert len(sample) == 25
    assert synth.privacy_report_.epsilon_spent == pytest.approx(1.0)
    assert report["privacy"]["accounting"]["epsilon_spent"] == pytest.approx(1.0)
    assert 0.0 <= report["utility"]["mean_column_similarity"] <= 1.0
    assert "membership_inference" in report["privacy"]


def test_sample_before_fit_fails() -> None:
    synth = Synthesizer(method="independent", epsilon=1.0)

    with pytest.raises(NotFittedError):
        synth.sample(10)


def test_epsilon_must_be_positive() -> None:
    with pytest.raises(PrivacyBudgetError):
        Synthesizer(method="independent", epsilon=0)


def test_default_method_is_aim() -> None:
    synth = Synthesizer(epsilon=1.0)

    assert synth.method == "aim"
    assert synth.delta is None


def test_privbayes_missing_dependency_is_explicit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "DataSynthesizer", None)
    monkeypatch.setitem(sys.modules, "DataSynthesizer.DataDescriber", None)
    synth = Synthesizer(method="privbayes", epsilon=1.0)

    with pytest.raises(BackendNotAvailableError, match="DataSynthesizer"):
        synth.fit(demo_df())
