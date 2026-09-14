import os

import numpy as np
import pytest
from PIL import Image

CODEBOOK = os.path.join(
    os.path.dirname(__file__), "..", "example_codebooks", "CODEBOOK_SVRIMG.md"
)

CLASS_CSV = (
    "UNID,Class Code,Class Name\n"
    "199604200208z000000206,0,Cellular\n"
    "200210291000z000000352,1,QLCS\n"
    "201304011200z000000111,0,Cellular\n"
    "201504011200z000000222,2,Tropical\n"
)

INDEX_CSV = (
    "unid,area,convection_area,intense_area,intensity_variance,kind,"
    "max_intensity,mean_intensity,radar_time,report_time\n"
    "199604200208z000000206,1,2,3,4,tor,63,5,4/20/1996 2:00,4/20/1996 2:08\n"
    "200210291000z000000352,1,2,3,4,tor,65,5,10/29/2002 10:00,10/29/2002 10:00\n"
    "201304011200z000000111,1,2,3,4,tor,55,5,4/1/2013 12:00,4/1/2013 12:00\n"
    "201504011200z000000222,1,2,3,4,tor,55,5,4/1/2015 12:00,4/1/2015 12:00\n"
)


def _dummy_ramp_palette():
    """The meaningless linear palette SVRIMG PNGs carry."""
    return [v % 256 for i in range(256) for v in (3 * i, 3 * i + 1, 3 * i + 2)]


def _write_palette_png(path, array):
    """Write ``array`` as a palette-mode PNG, the way SVRIMG distributes them."""
    im = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="P")
    im.putpalette(_dummy_ramp_palette())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return path


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test tries to fetch from the SVRIMG server."""
    import lars.preprocessing.svrimg as svrimg

    def _offline(url, dest):
        if os.path.exists(dest):
            return dest
        raise AssertionError(f"test attempted a network fetch: {url}")

    monkeypatch.setattr(svrimg, "_download", _offline)


def _seed_cache(data_dir):
    """Pre-populate the download cache so no test touches the network."""
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "sample_classifications_96-17.csv"), "w") as f:
        f.write(CLASS_CSV)
    with open(os.path.join(data_dir, "96-17_tor_utc_svrimg_index.csv"), "w") as f:
        f.write(INDEX_CSV)
    return data_dir


# --------------------------------------------------------------------------
# Palette decoding -- the regression guard for the greyscale-looking PNGs
# --------------------------------------------------------------------------


def test_read_svrimg_png_returns_palette_indices_not_rgb(tmp_path):
    from lars.preprocessing.svrimg import read_svrimg_png

    expected = np.zeros((136, 136), dtype=np.uint8)
    expected[10:20, 10:20] = 55
    expected[30, 30] = 72
    path = _write_palette_png(str(tmp_path / "scene.png"), expected)

    np.testing.assert_array_equal(read_svrimg_png(path), expected)


def test_read_svrimg_png_rejects_non_palette_image(tmp_path):
    from lars.preprocessing.svrimg import read_svrimg_png

    path = str(tmp_path / "rgb.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(path)

    with pytest.raises(ValueError, match="palette-mode"):
        read_svrimg_png(path)


# --------------------------------------------------------------------------
# Coverage statistics
# --------------------------------------------------------------------------


def test_reflectivity_stats_against_hand_computed_array():
    from lars.preprocessing.svrimg import reflectivity_stats

    arr = np.zeros((10, 10), dtype=np.uint8)   # 100 gates
    arr[0, :10] = 35                           # 10 gates > 30
    arr[1, :5] = 55                            # 5 gates > 50
    arr[2, :20] = 15                           # 10 gates > 10

    stats = reflectivity_stats(arr, (10, 30, 50))

    assert stats["n_gates_10dbz"] == 25        # 10 + 5 + 10
    assert stats["n_gates_30dbz"] == 15        # 10 + 5
    assert stats["n_gates_50dbz"] == 5
    assert stats["pct_gates_30dbz"] == 15.0
    assert stats["ref_min"] == 15.0            # echoing gates only, zeros excluded
    assert stats["ref_max"] == 55.0


def test_reflectivity_stats_handles_echo_free_scene():
    from lars.preprocessing.svrimg import reflectivity_stats

    stats = reflectivity_stats(np.zeros((5, 5), dtype=np.uint8), (10,))

    assert np.isnan(stats["ref_min"]) and np.isnan(stats["ref_max"])
    assert stats["n_gates_10dbz"] == 0
    assert stats["pct_gates_10dbz"] == 0.0


def test_split_for_year_matches_published_split():
    from lars.preprocessing.svrimg import split_for_year

    assert split_for_year(1996) == "train"
    assert split_for_year(2011) == "train"
    assert split_for_year(2012) == "val"
    assert split_for_year(2013) == "val"
    assert split_for_year(2014) == "test"
    assert split_for_year(2017) == "test"
    assert split_for_year(2018) is None


# --------------------------------------------------------------------------
# Label fetching
# --------------------------------------------------------------------------


def test_fetch_svrimg_labels_collapses_to_cellular_and_qlcs(tmp_path):
    from lars.preprocessing.svrimg import fetch_svrimg_labels

    labels = fetch_svrimg_labels(_seed_cache(str(tmp_path)))

    assert set(labels["label"]) == {"Cellular", "QLCS"}
    assert "201504011200z000000222" not in labels.index      # Tropical dropped
    assert len(labels) == 3


def test_fetch_svrimg_labels_assigns_splits_and_parses_times(tmp_path):
    from lars.preprocessing.svrimg import fetch_svrimg_labels

    labels = fetch_svrimg_labels(_seed_cache(str(tmp_path)))

    assert labels.loc["199604200208z000000206", "split"] == "train"
    assert labels.loc["201304011200z000000111", "split"] == "val"
    assert labels.loc["200210291000z000000352", "radar_time"].hour == 10
    assert labels.loc["199604200208z000000206", "max_intensity"] == 63


def test_fetch_svrimg_labels_can_keep_all_classes(tmp_path):
    from lars.preprocessing.svrimg import fetch_svrimg_labels

    labels = fetch_svrimg_labels(_seed_cache(str(tmp_path)), classes=None)

    assert "Tropical" in set(labels["label"])
    assert len(labels) == 4


# --------------------------------------------------------------------------
# End-to-end preprocessing
# --------------------------------------------------------------------------


def test_preprocess_svrimg_data_matches_radar_schema(tmp_path):
    from lars.preprocessing.svrimg import preprocess_svrimg_data

    data_dir = _seed_cache(str(tmp_path / "cache"))
    scene = np.zeros((136, 136), dtype=np.uint8)
    scene[:40, :40] = 45                        # 1600 gates > 30 of 18496
    _write_palette_png(
        os.path.join(data_dir, "tor", "1996", "04",
                     "199604200208z000000206.png"),
        scene,
    )
    _write_palette_png(
        os.path.join(data_dir, "tor", "2002", "10",
                     "200210291000z000000352.png"),
        np.zeros((136, 136), dtype=np.uint8),
    )

    df = preprocess_svrimg_data(
        output_path=str(tmp_path / "out"),
        data_dir=data_dir,
        splits=("train",),
        cmap="viridis",                         # avoid the cmweather dependency
    )

    assert len(df) == 2                         # both 1996 and 2002 are train
    row = df[df["unid"] == "199604200208z000000206"].iloc[0]
    assert row["unid"] == "199604200208z000000206"
    assert row["label"] == "Cellular"
    assert row["split"] == "train"
    assert row["n_gates_30dbz"] == 1600
    assert row["pct_gates_30dbz"] == round(1600 / 18496 * 100, 4)
    assert os.path.exists(row["file_path"])
    assert df.index.name == "time"

    # Same schema as preprocess_radar_data, plus unid/split.
    base = {"file_path", "label", "ref_min", "ref_max"}
    base |= {f"n_gates_{t}dbz" for t in (10, 20, 30, 40, 50)}
    base |= {f"pct_gates_{t}dbz" for t in (10, 20, 30, 40, 50)}
    assert base <= set(df.columns)
    assert {"unid", "split"} <= set(df.columns)


def test_preprocess_svrimg_data_rejects_axis_kwarg(tmp_path):
    from lars.preprocessing.svrimg import preprocess_svrimg_data

    with pytest.raises(ValueError, match="Do not pass in an axis"):
        preprocess_svrimg_data(
            output_path=str(tmp_path / "out"),
            data_dir=_seed_cache(str(tmp_path / "cache")),
            ax="something",
        )


# --------------------------------------------------------------------------
# Codebook wiring
# --------------------------------------------------------------------------


def test_codebook_exposes_binary_convective_mode_classes():
    from lars.nepho.inference import categories_from_codebook

    assert set(categories_from_codebook(CODEBOOK)) == {
        "Cellular", "QLCS", "Ambiguous / Uncertain",
    }


def test_codebook_colormap_uses_zero_floor():
    from lars.nepho.inference import colormap_from_codebook

    # SVRIMG reflectivity is floored at 0, unlike the -20 used for CSAPR2.
    assert colormap_from_codebook(CODEBOOK) == {
        "colormap": "NWSRef", "vmin": 0, "vmax": 80,
    }


def test_codebook_criterion_targets_the_emitted_column():
    from lars.nepho.inference import criteria_from_codebook

    criteria = criteria_from_codebook(CODEBOOK)

    assert set(criteria) == {"Cellular"}
    criterion = criteria["Cellular"][0]
    assert criterion["field"] == "pct_gates_30dbz"
    assert criterion["max_value"] == 6.5
    assert criterion["reclassify_as"] == "QLCS"


def test_codebook_guidelines_are_non_empty():
    from lars.nepho.inference import guidelines_from_codebook

    assert len(guidelines_from_codebook(CODEBOOK)) >= 3


def test_standardize_labels_canonicalises_svrimg_classes():
    import pandas as pd
    from lars.preprocessing.labels import standardize_labels

    df = pd.DataFrame({"label": ["cellular", "QLCS", "qlcs", "Cellular"]})

    assert list(standardize_labels(df)["label"]) == [
        "Cellular", "QLCS", "QLCS", "Cellular",
    ]
