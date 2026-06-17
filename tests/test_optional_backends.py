import sys
import types

import pandas as pd
import pytest

from synthhub import Synthesizer
from synthhub.errors import PrivacyBudgetError


def test_smartnoise_adapter_passes_encoded_dataframe_and_budget(monkeypatch) -> None:
    calls = {}

    class FakeSmartNoiseModel:
        def __init__(self):
            self.odometer = types.SimpleNamespace(spent=(0.9, 1e-9))
            self.columns = []

        def fit(self, data, **kwargs):
            calls["fit_data"] = data.copy()
            calls["fit_kwargs"] = kwargs
            self.columns = list(data.columns)

        def sample(self, n):
            return pd.DataFrame({column: [0] * n for column in self.columns})

    class FakeSmartNoiseSynthesizer:
        @classmethod
        def create(cls, synth, epsilon=None, **kwargs):
            calls["create"] = {"synth": synth, "epsilon": epsilon, "kwargs": kwargs}
            return FakeSmartNoiseModel()

    module = types.ModuleType("snsynth")
    module.Synthesizer = FakeSmartNoiseSynthesizer
    monkeypatch.setitem(sys.modules, "snsynth", module)

    df = pd.DataFrame({"age": [20, 30, 40, 50, 60], "city": ["a", "b", "a", "c", "b"]})
    synth = Synthesizer(method="mwem", epsilon=0.9, delta=1e-9, random_state=0).fit(df)
    sample = synth.sample(4)

    assert calls["create"]["synth"] == "mwem"
    assert calls["create"]["epsilon"] == pytest.approx(0.9)
    assert calls["fit_kwargs"]["preprocessor_eps"] == 0.0
    assert calls["fit_kwargs"]["categorical_columns"] == list(calls["fit_data"].columns)
    assert sample.shape == (4, 2)
    assert synth.privacy_report_.backend == "smartnoise:mwem"
    assert synth.privacy_report_.epsilon_spent == pytest.approx(0.9)


def test_smartnoise_adapter_rejects_preprocessor_overspend() -> None:
    with pytest.raises(PrivacyBudgetError, match="preprocessor_eps"):
        Synthesizer(method="mwem", epsilon=0.5, preprocessor_eps=0.6).fit(
            pd.DataFrame({"x": [0, 1, 2], "y": ["a", "b", "a"]})
        )


def test_smartnoise_adapter_rejects_disabled_dp_mode() -> None:
    with pytest.raises(PrivacyBudgetError, match="disabled_dp"):
        Synthesizer(method="dpctgan", epsilon=1.0, disabled_dp=True).fit(
            pd.DataFrame({"x": [0, 1, 2], "y": ["a", "b", "a"]})
        )


def test_synthcity_adapter_passes_plugin_and_epsilon(monkeypatch) -> None:
    calls = {}

    class FakeGenericDataLoader:
        def __init__(self, data):
            self.data = data.copy()

    class FakeGenerated:
        def dataframe(self):
            return pd.DataFrame({"x": [0, 1, 0], "y": [0, 0, 0]})

    class FakeSynthCityModel:
        def fit(self, loader):
            calls["fit_loader"] = loader

        def generate(self, count=None):
            calls["count"] = count
            return FakeGenerated()

    class FakePlugins:
        def get(self, plugin, **options):
            calls["get"] = {"plugin": plugin, "options": options}
            return FakeSynthCityModel()

    synthcity_pkg = types.ModuleType("synthcity")
    plugins_mod = types.ModuleType("synthcity.plugins")
    plugins_mod.Plugins = FakePlugins
    core_mod = types.ModuleType("synthcity.plugins.core")
    dataloader_mod = types.ModuleType("synthcity.plugins.core.dataloader")
    dataloader_mod.GenericDataLoader = FakeGenericDataLoader

    monkeypatch.setitem(sys.modules, "synthcity", synthcity_pkg)
    monkeypatch.setitem(sys.modules, "synthcity.plugins", plugins_mod)
    monkeypatch.setitem(sys.modules, "synthcity.plugins.core", core_mod)
    monkeypatch.setitem(sys.modules, "synthcity.plugins.core.dataloader", dataloader_mod)

    df = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": ["a", "b", "a", "b", "a"]})
    synth = Synthesizer(method="synthcity-privbayes", epsilon=1.2, delta=1e-9).fit(df)
    sample = synth.sample(3)

    assert calls["get"]["plugin"] == "privbayes"
    assert calls["get"]["options"]["epsilon"] == pytest.approx(1.2)
    assert calls["get"]["options"]["delta"] == pytest.approx(1e-9)
    assert calls["count"] == 3
    assert sample.shape == (3, 2)
    assert synth.privacy_report_.backend == "synthcity:privbayes"


def test_new_backend_names_are_registered() -> None:
    methods = set(Synthesizer.available_methods())

    assert {
        "mwem",
        "dpctgan",
        "smartnoise-aim",
        "synthcity-privbayes",
        "datasynthesizer-privbayes",
        "datasynthesizer-independent",
    }.issubset(methods)
    assert "quail" not in methods


def test_datasynthesizer_privbayes_adapter_uses_correlated_mode(monkeypatch) -> None:
    calls = {}

    class FakeDataDescriber:
        def __init__(self, histogram_bins=20):
            calls["histogram_bins"] = histogram_bins

        def describe_dataset_in_correlated_attribute_mode(self, dataset_file, **kwargs):
            calls["correlated"] = {"dataset_file": dataset_file, "kwargs": kwargs}

        def save_dataset_description_to_file(self, description_file):
            calls["description_file"] = description_file
            with open(description_file, "w", encoding="utf-8") as output:
                output.write("{}")

    class FakeDataGenerator:
        def generate_dataset_in_correlated_attribute_mode(self, n, description_file, seed=0):
            calls["generate"] = {"n": n, "description_file": description_file, "seed": seed}
            self.synthetic_dataset = pd.DataFrame({"age": [0, 1, 0], "city": [0, 0, 0]})

    describer_mod = types.ModuleType("DataSynthesizer.DataDescriber")
    describer_mod.DataDescriber = FakeDataDescriber
    generator_mod = types.ModuleType("DataSynthesizer.DataGenerator")
    generator_mod.DataGenerator = FakeDataGenerator
    package_mod = types.ModuleType("DataSynthesizer")

    monkeypatch.setitem(sys.modules, "DataSynthesizer", package_mod)
    monkeypatch.setitem(sys.modules, "DataSynthesizer.DataDescriber", describer_mod)
    monkeypatch.setitem(sys.modules, "DataSynthesizer.DataGenerator", generator_mod)

    df = pd.DataFrame({"age": [20, 30, 40, 50, 60], "city": ["a", "b", "a", "c", "b"]})
    synth = Synthesizer(method="privbayes", epsilon=1.5, random_state=123).fit(df)
    sample = synth.sample(3)

    assert calls["correlated"]["kwargs"]["epsilon"] == pytest.approx(1.5)
    assert calls["correlated"]["kwargs"]["k"] == 2
    assert calls["correlated"]["kwargs"]["attribute_to_is_categorical"] == {"age": True, "city": True}
    assert calls["generate"]["n"] == 3
    assert calls["generate"]["seed"] == 123
    assert sample.shape == (3, 2)
    assert synth.privacy_report_.backend == "datasynthesizer:correlated"


def test_datasynthesizer_single_column_falls_back_to_independent_mode(monkeypatch) -> None:
    calls = {}

    class FakeDataDescriber:
        def __init__(self, histogram_bins=20):
            pass

        def describe_dataset_in_independent_attribute_mode(self, dataset_file, **kwargs):
            calls["independent"] = kwargs

        def save_dataset_description_to_file(self, description_file):
            with open(description_file, "w", encoding="utf-8") as output:
                output.write("{}")

    class FakeDataGenerator:
        def generate_dataset_in_independent_mode(self, n, description_file, seed=0):
            calls["generate_independent"] = {"n": n, "seed": seed}
            self.synthetic_dataset = pd.DataFrame({"flag": [0, 1]})

    describer_mod = types.ModuleType("DataSynthesizer.DataDescriber")
    describer_mod.DataDescriber = FakeDataDescriber
    generator_mod = types.ModuleType("DataSynthesizer.DataGenerator")
    generator_mod.DataGenerator = FakeDataGenerator
    package_mod = types.ModuleType("DataSynthesizer")

    monkeypatch.setitem(sys.modules, "DataSynthesizer", package_mod)
    monkeypatch.setitem(sys.modules, "DataSynthesizer.DataDescriber", describer_mod)
    monkeypatch.setitem(sys.modules, "DataSynthesizer.DataGenerator", generator_mod)

    synth = Synthesizer(method="privbayes", epsilon=1.0, random_state=9).fit(
        pd.DataFrame({"flag": [0, 1, 0, 1]})
    )
    sample = synth.sample(2)

    assert "independent" in calls
    assert "correlated mode requires at least two columns" in " ".join(synth.privacy_report_.warnings)
    assert sample.shape == (2, 1)


def test_datasynthesizer_live_smoke_if_installed() -> None:
    pytest.importorskip("DataSynthesizer")
    df = pd.DataFrame(
        {
            "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
            "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )

    synth = Synthesizer(method="privbayes", epsilon=1.0, random_state=0).fit(df)
    sample = synth.sample(8)

    assert sample.shape == (8, 3)
    assert list(sample.columns) == list(df.columns)
    assert synth.privacy_report_.backend == "datasynthesizer:correlated"
    assert synth.privacy_report_.epsilon_spent == pytest.approx(1.0)


def test_private_pgm_mst_live_smoke_if_mechanisms_available() -> None:
    pytest.importorskip("mbi")
    pytest.importorskip("mechanisms.mst")
    df = pd.DataFrame(
        {
            "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
            "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )

    synth = Synthesizer(method="mst", epsilon=1.0, random_state=0).fit(df)
    sample = synth.sample(8)

    assert sample.shape == (8, 3)
    assert list(sample.columns) == list(df.columns)
    assert synth.privacy_report_.backend == "private-pgm:mst"
    assert synth.privacy_report_.epsilon_spent == pytest.approx(1.0)


def test_private_pgm_aim_live_smoke_if_mechanisms_available() -> None:
    pytest.importorskip("mbi")
    pytest.importorskip("mechanisms.aim")
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"] * 2,
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0] * 2,
        }
    )

    synth = Synthesizer(
        method="aim",
        epsilon=1.0,
        random_state=0,
        degree=2,
        rounds=4,
        max_iters=20,
        max_model_size=0.2,
    ).fit(df)
    sample = synth.sample(8)

    assert sample.shape == (8, 2)
    assert list(sample.columns) == list(df.columns)
    assert synth.privacy_report_.backend == "private-pgm:aim"
    assert synth.privacy_report_.epsilon_spent == pytest.approx(1.0)


def test_smartnoise_mwem_live_smoke_if_installed() -> None:
    pytest.importorskip("snsynth")
    df = pd.DataFrame(
        {
            "age": [21, 34, 37, 45, 52, 23, 41, 29, 62, 31],
            "city": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )

    synth = Synthesizer(method="mwem", epsilon=1.0, delta=1e-9, random_state=0).fit(df)
    sample = synth.sample(8)

    assert sample.shape == (8, 3)
    assert list(sample.columns) == list(df.columns)
    assert synth.privacy_report_.backend == "smartnoise:mwem"
    assert synth.privacy_report_.epsilon_spent == pytest.approx(1.0)
    assert synth.privacy_report_.delta is None
    assert any("does not accept delta" in warning for warning in synth.privacy_report_.warnings)
