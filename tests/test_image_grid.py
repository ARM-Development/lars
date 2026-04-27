import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png(path):
    """Save a tiny 4x4 solid-colour PNG to *path* and return the path."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.imshow(np.zeros((4, 4, 3), dtype=np.uint8))
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    return path


@pytest.fixture()
def image_df(tmp_path):
    """DataFrame with 8 labelled images backed by real PNG files."""
    rows = [
        # hand_label,      llm_label,        filename
        ("stratiform",     "stratiform",      "img_00.png"),
        ("stratiform",     "stratiform",      "img_01.png"),
        ("stratiform",     "stratiform",      "img_02.png"),
        ("stratiform",     "convective",      "img_03.png"),
        ("convective",     "convective",      "img_04.png"),
        ("convective",     "convective",      "img_05.png"),
        ("convective",     "stratiform",      "img_06.png"),
        ("no precipitation", "no precipitation", "img_07.png"),
    ]
    records = []
    for hand, llm, fname in rows:
        path = _make_png(tmp_path / fname)
        records.append({"label": hand, "llm_label": llm, "file_path": str(path)})
    return pd.DataFrame(records)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_file_is_created(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    out = tmp_path / "grid.png"
    plot_label_images(image_df, "stratiform", "stratiform", n=2, output_path=str(out), seed=0)
    assert out.exists()


def test_output_is_valid_png(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    out = tmp_path / "grid.png"
    plot_label_images(image_df, "stratiform", "stratiform", n=2, output_path=str(out), seed=0)
    with open(out, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_correct_number_of_visible_axes(image_df, tmp_path, monkeypatch):
    from lars.util import image_grid

    captured = {}

    original_savefig = matplotlib.figure.Figure.savefig

    def mock_savefig(self, *args, **kwargs):
        captured["fig"] = self
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", mock_savefig)

    out = tmp_path / "grid.png"
    plot_label_images = image_grid.plot_label_images
    plot_label_images(image_df, "stratiform", "stratiform", n=3, output_path=str(out), seed=0)

    fig = captured["fig"]
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3


def test_suptitle_contains_labels(image_df, tmp_path, monkeypatch):
    from lars.util import image_grid

    captured = {}

    original_savefig = matplotlib.figure.Figure.savefig

    def mock_savefig(self, *args, **kwargs):
        captured["fig"] = self
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", mock_savefig)

    out = tmp_path / "grid.png"
    image_grid.plot_label_images(
        image_df, "stratiform", "stratiform", n=2, output_path=str(out), seed=0
    )

    title = captured["fig"].texts[0].get_text()
    assert "stratiform" in title.lower()


def test_raises_when_no_matching_rows(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    with pytest.raises(ValueError, match="No images found"):
        plot_label_images(
            image_df, "anvil", "anvil", n=1, output_path=str(tmp_path / "out.png")
        )


def test_raises_when_not_enough_images(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    # Only 3 stratiform/stratiform images exist
    with pytest.raises(ValueError, match="only 3 match"):
        plot_label_images(
            image_df, "stratiform", "stratiform", n=10,
            output_path=str(tmp_path / "out.png"),
        )


def test_case_insensitive_matching(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    out = tmp_path / "grid.png"
    plot_label_images(image_df, "Stratiform", "STRATIFORM", n=2, output_path=str(out), seed=0)
    assert out.exists()


def test_custom_column_names(tmp_path):
    from lars.util.image_grid import plot_label_images

    rows = []
    for i in range(3):
        path = _make_png(tmp_path / f"custom_{i}.png")
        rows.append({"true": "stratiform", "pred": "stratiform", "file_path": str(path)})
    df = pd.DataFrame(rows)

    out = tmp_path / "grid.png"
    plot_label_images(
        df, "stratiform", "stratiform", n=2, output_path=str(out),
        label_col="true", pred_col="pred", seed=0,
    )
    assert out.exists()


def test_seed_reproducibility(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    out1 = tmp_path / "g1.png"
    out2 = tmp_path / "g2.png"
    plot_label_images(image_df, "stratiform", "stratiform", n=2, output_path=str(out1), seed=42)
    plot_label_images(image_df, "stratiform", "stratiform", n=2, output_path=str(out2), seed=42)

    import hashlib
    h1 = hashlib.md5(out1.read_bytes()).hexdigest()
    h2 = hashlib.md5(out2.read_bytes()).hexdigest()
    assert h1 == h2


def test_n_equals_one(image_df, tmp_path):
    from lars.util.image_grid import plot_label_images

    out = tmp_path / "single.png"
    plot_label_images(image_df, "no precipitation", "no precipitation", n=1, output_path=str(out), seed=0)
    assert out.exists()
