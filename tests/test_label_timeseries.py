import io
import pytest
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

CSV_DATA = """\
time,label,llm_label
3/4/25 0:00,No Precipitation,No Precipitation
3/4/25 0:12,No Precipitation,Stratiform Precipitation
3/4/25 0:24,Stratiform Precipitation,Stratiform Precipitation
3/4/25 0:36,Stratiform Precipitation,Isolated Convection
3/4/25 0:48,Isolated Convection,Isolated Convection
3/4/25 1:00,Mesoscale Convective System,Mesoscale Convective System
"""


@pytest.fixture
def sample_df():
    return pd.read_csv(io.StringIO(CSV_DATA))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.mpl_image_compare(tolerance=60)
def test_plotting(sample_df):
    from lars.util.label_timeseries import plot_label_timeseries

    fig, ax = plt.subplots()
    plot_label_timeseries(sample_df, ax=ax)
    return fig


def test_yticklabels_use_codebook_order(sample_df):
    from lars.util.label_timeseries import plot_label_timeseries

    _, ax = plt.subplots()
    plot_label_timeseries(sample_df, ax=ax)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [
        "No Precipitation",
        "Stratiform Precipitation",
        "Isolated Convection",
        "Mesoscale Convective System",
    ]


def test_overlaying_predictions_adds_legend(sample_df):
    from lars.util.label_timeseries import plot_label_timeseries

    _, ax = plt.subplots()
    plot_label_timeseries(sample_df, pred_col="llm_label", ax=ax)
    assert ax.get_legend() is not None
    assert len(ax.lines) == 2


def test_saves_output(tmp_path, sample_df):
    from lars.util.label_timeseries import plot_label_timeseries

    out = tmp_path / "ts.png"
    plot_label_timeseries(sample_df, output_path=str(out))
    assert out.exists()


def test_missing_column_raises(sample_df):
    from lars.util.label_timeseries import plot_label_timeseries

    with pytest.raises(ValueError):
        plot_label_timeseries(sample_df, label_col="does_not_exist")


def test_empty_dataframe_raises():
    from lars.util.label_timeseries import plot_label_timeseries

    with pytest.raises(ValueError):
        plot_label_timeseries(pd.DataFrame({"time": [], "label": []}))
