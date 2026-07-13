import os
import sys
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest


CODEBOOK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "CODEBOOK.md")
)


def test_color_criteria_from_codebook_covers_all_labels():
    from lars.nepho.inference import color_criteria_from_codebook

    rules = color_criteria_from_codebook(CODEBOOK_PATH)
    assert set(rules) == {
        "No Precipitation",
        "Stratiform Precipitation",
        "Isolated Convection",
        "Mesoscale Convective System",
    }


def test_color_criteria_no_pink_on_stratiform():
    from lars.nepho.inference import color_criteria_from_codebook

    rules = color_criteria_from_codebook(CODEBOOK_PATH)
    strat = rules["Stratiform Precipitation"]
    no_pink = [
        r for r in strat
        if r["kind"] == "max_pct_above"
        and r["field"] == "pct_gates_50dbz"
        and "pink" in r["colors"]
    ]
    assert len(no_pink) == 1


def test_color_criteria_dominance_for_isolated():
    from lars.nepho.inference import color_criteria_from_codebook

    rules = color_criteria_from_codebook(CODEBOOK_PATH)
    iso = rules["Isolated Convection"]
    dom = [
        r for r in iso
        if r["kind"] == "max_pct_above"
        and r["field"] == "pct_gates_10dbz"
        and r["value"] == 50.0
    ]
    assert len(dom) == 1
    assert "blue" in dom[0]["colors"] and "black" in dom[0]["colors"]


def test_color_criteria_exclusivity_for_no_precip():
    from lars.nepho.inference import color_criteria_from_codebook

    rules = color_criteria_from_codebook(CODEBOOK_PATH)
    np_rules = rules["No Precipitation"]
    excl = [
        r for r in np_rules
        if r["kind"] == "max_pct_above" and r["field"] == "pct_gates_10dbz"
    ]
    assert len(excl) == 1


def test_colormap_from_codebook_parses_name_and_bounds():
    from lars.nepho.inference import colormap_from_codebook

    cmap = colormap_from_codebook(CODEBOOK_PATH)
    assert cmap["colormap"] == "ChaseSpectral"
    assert cmap["vmin"] == -10
    assert cmap["vmax"] == 60


def test_colormap_from_codebook_falls_back_to_defaults():
    from lars.nepho.inference import (
        colormap_from_codebook,
        DEFAULT_VMIN,
        DEFAULT_VMAX,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write("# Codebook\n\nNo color scale specified here.\n")
        path = f.name
    try:
        cmap = colormap_from_codebook(path)
        assert cmap["colormap"] is None
        assert cmap["vmin"] == DEFAULT_VMIN == -20
        assert cmap["vmax"] == DEFAULT_VMAX == 60
    finally:
        os.unlink(path)


def test_compute_validation_metrics_counts_violations():
    from lars.nepho.tracking import compute_validation_metrics

    criteria = {
        "Stratiform Precipitation": [{
            "field": "pct_gates_50dbz", "kind": "pct",
            "threshold_dbz": 50, "max_value": 0.02,
            "reclassify_as": "Mesoscale Convective System",
        }],
    }
    color_criteria = {
        "Stratiform Precipitation": [{
            "kind": "max_pct_above", "field": "pct_gates_50dbz",
            "value": 0.5, "colors": ["pink"],
            "phrase": "must have no pink colors",
        }],
    }
    df = pd.DataFrame({
        "label":     ["Stratiform Precipitation"] * 3,
        "llm_label": ["Stratiform Precipitation",
                      "Stratiform Precipitation",
                      "Mesoscale Convective System"],
        "pct_gates_50dbz": [0.0, 1.0, 5.0],
    })
    m = compute_validation_metrics(df, criteria=criteria,
                                   color_criteria=color_criteria)

    refl_keys = [k for k in m if k.startswith("reflectivity_violations/llm")
                 and k.endswith("/count")]
    assert refl_keys, f"expected reflectivity llm count metric, got: {list(m)}"
    assert m[refl_keys[0]] == 1

    color_keys = [k for k in m if k.startswith("color_violations/llm")
                  and k.endswith("/count")]
    assert color_keys
    assert m[color_keys[0]] == 1

    assert m["agreement/n_compared"] == 3
    assert m["agreement/overall_accuracy"] == pytest.approx(2 / 3)


def test_compute_validation_metrics_skips_unknown_hand_labels():
    from lars.nepho.tracking import compute_validation_metrics

    df = pd.DataFrame({
        "label":     ["UNKNOWN", "Stratiform Precipitation"],
        "llm_label": ["Stratiform Precipitation", "Stratiform Precipitation"],
    })
    m = compute_validation_metrics(df)
    assert m["agreement/n_compared"] == 1
    assert m["agreement/overall_accuracy"] == 1.0


def test_compute_validation_metrics_handles_missing_columns():
    from lars.nepho.tracking import compute_validation_metrics

    df = pd.DataFrame({"file_path": ["a", "b"], "llm_label": ["x", "y"]})
    m = compute_validation_metrics(df)
    assert m == {}


def test_log_run_to_mlflow_raises_clear_error_without_mlflow(monkeypatch):
    from lars.nepho import tracking

    monkeypatch.setitem(sys.modules, "mlflow", None)
    df = pd.DataFrame({"label": ["x"], "llm_label": ["x"]})
    with pytest.raises(ImportError, match="mlflow is required"):
        tracking.log_run_to_mlflow(df, experiment="exp")


def test_codebook_hash_is_stable():
    from lars.nepho.tracking import codebook_hash

    assert codebook_hash(CODEBOOK_PATH) == codebook_hash(CODEBOOK_PATH)
    assert codebook_hash("/no/such/path") is None


def test_log_run_to_mlflow_calls_expected_apis(monkeypatch):
    from lars.nepho import tracking

    fake = MagicMock()
    fake_run = MagicMock()
    fake_run.info.run_id = "rid-123"
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=fake_run)
    cm.__exit__ = MagicMock(return_value=False)
    fake.start_run.return_value = cm

    monkeypatch.setitem(sys.modules, "mlflow", fake)

    with tempfile.TemporaryDirectory() as outputs:
        with open(os.path.join(outputs, "a.txt"), "w") as f:
            f.write("hello")
        df = pd.DataFrame({
            "label":     ["Stratiform Precipitation"],
            "llm_label": ["Stratiform Precipitation"],
            "pct_gates_50dbz": [0.0],
        })
        run_id = tracking.log_run_to_mlflow(
            df, experiment="exp", run_name="r1",
            params={"model": "test"},
            model_output_dir=outputs,
        )

    assert run_id == "rid-123"
    fake.set_experiment.assert_called_once_with("exp")
    fake.start_run.assert_called_once()
    assert fake.log_params.called
    fake.log_artifacts.assert_called_with(
        outputs, artifact_path="model_outputs"
    )
