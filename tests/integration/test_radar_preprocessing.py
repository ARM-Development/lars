"""
Integration tests for lars.preprocessing.preprocess_radar_data.

Downloads three GUC XPRECIP CMAC PPI radar files from open-radar-data,
runs the full preprocessing workflow, validates the returned DataFrame,
and compares each generated PNG against a stored baseline image.

Generating baselines (first-time setup or after intentional changes):
    pytest tests/integration/ --generate-baseline

Running the tests normally:
    pytest tests/integration/
"""

import os
import shutil

import matplotlib.image as mpimg
import numpy as np
import pytest

open_radar_data = pytest.importorskip("open_radar_data")
xradar = pytest.importorskip("xradar")

RADAR_FILES = [
    "gucxprecipradarcmacppiS2.c1.20220314.021559.nc",
    "gucxprecipradarcmacppiS2.c1.20220314.024239.nc",
    "gucxprecipradarcmacppiS2.c1.20220314.025840.nc",
]

BASELINE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "baseline", "preprocessing"
)

# Pixel-value tolerance for image comparison (values are float in [0, 1]).
IMAGE_TOLERANCE = 5 / 255


@pytest.fixture(scope="module")
def radar_data_dir(tmp_path_factory):
    """Download the three test radar files into an isolated temp directory."""
    from open_radar_data import DATASETS

    tmp_dir = tmp_path_factory.mktemp("radar_data")
    for fname in RADAR_FILES:
        src = DATASETS.fetch(fname)
        shutil.copy(src, tmp_dir / fname)
    return str(tmp_dir)


@pytest.fixture(scope="module")
def preprocessing_output(tmp_path_factory, radar_data_dir):
    """Run preprocessing once and share the output across all tests."""
    from lars.preprocessing import preprocess_radar_data

    out_dir = str(tmp_path_factory.mktemp("preprocessing_output"))
    label_df = preprocess_radar_data(radar_data_dir, out_dir)
    return out_dir, label_df


# ---------------------------------------------------------------------------
# DataFrame tests
# ---------------------------------------------------------------------------


def test_dataframe_row_count(preprocessing_output):
    _, label_df = preprocessing_output
    assert len(label_df) == 3


def test_dataframe_columns(preprocessing_output):
    _, label_df = preprocessing_output
    assert set(label_df.columns) == {
        "file_path", "label", "ref_min", "ref_max",
        "n_gates_10dbz", "n_gates_20dbz", "n_gates_30dbz", "n_gates_40dbz", "n_gates_50dbz",
        "pct_gates_10dbz", "pct_gates_20dbz", "pct_gates_30dbz", "pct_gates_40dbz", "pct_gates_50dbz",
    }


def test_labels_are_unknown(preprocessing_output):
    _, label_df = preprocessing_output
    assert (label_df["label"] == "UNKNOWN").all()


def test_reflectivity_bounds(preprocessing_output):
    _, label_df = preprocessing_output
    assert (label_df["ref_min"] <= label_df["ref_max"]).all()


def test_timestamps_are_on_correct_date(preprocessing_output):
    _, label_df = preprocessing_output
    assert all("2022-03-14" in str(idx) for idx in label_df.index)


def test_index_is_sorted(preprocessing_output):
    _, label_df = preprocessing_output
    assert label_df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Image file tests
# ---------------------------------------------------------------------------


def test_png_files_created(preprocessing_output):
    out_dir, _ = preprocessing_output
    for fname in RADAR_FILES:
        assert os.path.exists(os.path.join(out_dir, fname.replace(".nc", ".png")))


# ---------------------------------------------------------------------------
# Baseline image-comparison tests
# ---------------------------------------------------------------------------


def _compare_to_baseline(generated_path, baseline_path, tolerance):
    generated = mpimg.imread(generated_path).astype(np.float32)
    baseline = mpimg.imread(baseline_path).astype(np.float32)

    assert generated.shape == baseline.shape, (
        f"Shape mismatch: generated {generated.shape} vs baseline {baseline.shape}"
    )
    max_diff = np.max(np.abs(generated - baseline))
    assert max_diff <= tolerance, (
        f"Max pixel difference {max_diff:.4f} exceeds tolerance {tolerance:.4f} "
        f"({os.path.basename(generated_path)})"
    )


@pytest.mark.parametrize("fname", RADAR_FILES)
def test_image_matches_baseline(request, preprocessing_output, fname):
    out_dir, _ = preprocessing_output
    png_name = fname.replace(".nc", ".png")
    generated_path = os.path.join(out_dir, png_name)
    baseline_path = os.path.join(BASELINE_DIR, png_name)

    if request.config.getoption("--generate-baseline"):
        os.makedirs(BASELINE_DIR, exist_ok=True)
        shutil.copy(generated_path, baseline_path)
        pytest.skip(f"Baseline written to {baseline_path}")

    if not os.path.exists(baseline_path):
        pytest.skip(
            f"No baseline found at {baseline_path}. "
            "Run with --generate-baseline to create it."
        )

    _compare_to_baseline(generated_path, baseline_path, IMAGE_TOLERANCE)
