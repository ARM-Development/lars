import io
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DATA = """\
rater_a,rater_b,rater_c
stratiform,stratiform,stratiform
convective,convective,stratiform
stratiform,convective,convective
convective,stratiform,convective
anvil,anvil,anvil
stratiform,stratiform,convective
"""


@pytest.fixture
def sample_df():
    return pd.read_csv(io.StringIO(CSV_DATA))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_matrix_is_square_and_symmetric(sample_df):
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    matrix = calculate_label_rate_diff_matrix(sample_df, "stratiform")
    assert list(matrix.columns) == ["rater_a", "rater_b", "rater_c"]
    assert list(matrix.index) == ["rater_a", "rater_b", "rater_c"]
    np.testing.assert_array_almost_equal(matrix.values, matrix.values.T)


def test_diagonal_is_zero(sample_df):
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    matrix = calculate_label_rate_diff_matrix(sample_df, "stratiform")
    np.testing.assert_array_almost_equal(np.diag(matrix.values), np.zeros(3))


def test_usage_rate_diff_values(sample_df):
    """rater_a: 3/6=50% stratiform, rater_b: 3/6=50%, rater_c: 2/6=33.33%."""
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    matrix = calculate_label_rate_diff_matrix(sample_df, "stratiform")
    assert matrix.loc["rater_a", "rater_b"] == pytest.approx(0.0)
    assert matrix.loc["rater_a", "rater_c"] == pytest.approx(100 / 6)
    assert matrix.loc["rater_b", "rater_c"] == pytest.approx(100 / 6)


def test_label_is_case_insensitive(sample_df):
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    lower = calculate_label_rate_diff_matrix(sample_df, "stratiform")
    upper = calculate_label_rate_diff_matrix(sample_df, "STRATIFORM")
    np.testing.assert_array_almost_equal(lower.values, upper.values)


def test_subset_of_columns(sample_df):
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    matrix = calculate_label_rate_diff_matrix(sample_df, "stratiform", columns=["rater_a", "rater_c"])
    assert list(matrix.columns) == ["rater_a", "rater_c"]


def test_ignores_missing_values():
    from lars.util.label_rate_matrix import calculate_label_rate_diff_matrix

    df = pd.DataFrame({
        "a": ["x", "x", None],
        "b": ["x", "y", "y"],
    })
    matrix = calculate_label_rate_diff_matrix(df, "x")
    # a: 2/2 non-missing are 'x' -> 100%; b: 1/3 are 'x' -> 33.33%
    assert matrix.loc["a", "b"] == pytest.approx(100.0 - 100.0 / 3)


def test_plot_returns_axes(sample_df):
    from lars.util.label_rate_matrix import plot_label_rate_diff_matrix

    _, ax = plt.subplots()
    result = plot_label_rate_diff_matrix(sample_df, "stratiform", ax=ax)
    assert result is ax


def test_plot_title_includes_label(sample_df):
    from lars.util.label_rate_matrix import plot_label_rate_diff_matrix

    _, ax = plt.subplots()
    plot_label_rate_diff_matrix(sample_df, "stratiform", ax=ax)
    assert ax.get_title() == "Usage Rate Difference: 'stratiform'"


def test_plot_tick_labels_match_columns(sample_df):
    from lars.util.label_rate_matrix import plot_label_rate_diff_matrix

    _, ax = plt.subplots()
    plot_label_rate_diff_matrix(sample_df, "stratiform", ax=ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["rater_a", "rater_b", "rater_c"]


def test_plot_custom_labels(sample_df):
    from lars.util.label_rate_matrix import plot_label_rate_diff_matrix

    _, ax = plt.subplots()
    plot_label_rate_diff_matrix(sample_df, "stratiform", ax=ax, labels=["Alice", "Bob", "LLM"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Alice", "Bob", "LLM"]


def test_plot_uses_gca_when_no_ax(sample_df):
    from lars.util.label_rate_matrix import plot_label_rate_diff_matrix

    fig, ax = plt.subplots()
    plt.sca(ax)
    plot_label_rate_diff_matrix(sample_df, "stratiform")
    assert ax.get_title() == "Usage Rate Difference: 'stratiform'"
