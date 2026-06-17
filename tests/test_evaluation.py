import pandas as pd

from synthhub.evaluation import evaluate
from synthhub.reports import PrivacyReport


def test_evaluate_returns_utility_and_privacy_sections() -> None:
    real = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "segment": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 0, 1, 1, 1, 0],
        }
    )
    synth = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "segment": ["a", "b", "b", "c", "c", "a"],
            "label": [0, 1, 1, 1, 0, 0],
        }
    )
    privacy = PrivacyReport(
        method="independent",
        requested_epsilon=1.0,
        epsilon_spent=1.0,
        accountant="test",
    )

    report = evaluate(real, synth, privacy_report=privacy, target="label", random_state=0).to_dict()

    assert "utility" in report
    assert "privacy" in report
    assert report["privacy"]["accounting"]["method"] == "independent"
    assert 0.0 <= report["utility"]["mean_column_similarity"] <= 1.0

