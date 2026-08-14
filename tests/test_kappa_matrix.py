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
    from lars.util.kappa_matrix import calculate_kappa_matrix

    matrix = calculate_kappa_matrix(sample_df)
    assert list(matrix.columns) == ["rater_a", "rater_b", "rater_c"]
    assert list(matrix.index) == ["rater_a", "rater_b", "rater_c"]
    np.testing.assert_array_almost_equal(matrix.values, matrix.values.T)


def test_diagonal_is_one(sample_df):
    from lars.util.kappa_matrix import calculate_kappa_matrix

    matrix = calculate_kappa_matrix(sample_df)
    np.testing.assert_array_almost_equal(np.diag(matrix.values), np.ones(3))


def test_matrix_matches_pairwise_kappa(sample_df):
    from lars.util.kappa_matrix import calculate_kappa_matrix
    from lars.util.confusion_matrix import calculate_cohen_kappa

    matrix = calculate_kappa_matrix(sample_df)
    expected = calculate_cohen_kappa(
        sample_df.rename(columns={"rater_a": "label", "rater_b": "llm_label"})
    )
    assert matrix.loc["rater_a", "rater_b"] == pytest.approx(expected)


def test_subset_of_columns(sample_df):
    from lars.util.kappa_matrix import calculate_kappa_matrix

    matrix = calculate_kappa_matrix(sample_df, columns=["rater_a", "rater_c"])
    assert list(matrix.columns) == ["rater_a", "rater_c"]


def test_excludes_rows_with_missing_values():
    from lars.util.kappa_matrix import calculate_kappa_matrix

    df = pd.DataFrame({
        "a": ["x", "y", "x", None],
        "b": ["x", "y", "y", "x"],
    })
    matrix = calculate_kappa_matrix(df)
    assert not np.isnan(matrix.loc["a", "b"])


def test_plot_returns_axes(sample_df):
    from lars.util.kappa_matrix import plot_kappa_matrix

    _, ax = plt.subplots()
    result = plot_kappa_matrix(sample_df, ax=ax)
    assert result is ax


def test_plot_title_is_set(sample_df):
    from lars.util.kappa_matrix import plot_kappa_matrix

    _, ax = plt.subplots()
    plot_kappa_matrix(sample_df, ax=ax)
    assert ax.get_title() == "Cohen's Kappa Matrix"


def test_plot_tick_labels_match_columns(sample_df):
    from lars.util.kappa_matrix import plot_kappa_matrix

    _, ax = plt.subplots()
    plot_kappa_matrix(sample_df, ax=ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["rater_a", "rater_b", "rater_c"]


def test_plot_custom_labels(sample_df):
    from lars.util.kappa_matrix import plot_kappa_matrix

    _, ax = plt.subplots()
    plot_kappa_matrix(sample_df, ax=ax, labels=["Alice", "Bob", "LLM"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Alice", "Bob", "LLM"]


def test_plot_uses_gca_when_no_ax(sample_df):
    from lars.util.kappa_matrix import plot_kappa_matrix

    fig, ax = plt.subplots()
    plt.sca(ax)
    plot_kappa_matrix(sample_df)
    assert ax.get_title() == "Cohen's Kappa Matrix"
