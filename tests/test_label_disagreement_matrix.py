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
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    matrix = calculate_label_disagreement_matrix(sample_df, "stratiform")
    assert list(matrix.columns) == ["rater_a", "rater_b", "rater_c"]
    assert list(matrix.index) == ["rater_a", "rater_b", "rater_c"]
    np.testing.assert_array_almost_equal(matrix.values, matrix.values.T)


def test_diagonal_is_zero(sample_df):
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    matrix = calculate_label_disagreement_matrix(sample_df, "stratiform")
    np.testing.assert_array_almost_equal(np.diag(matrix.values), np.zeros(3))


def test_disagreement_values(sample_df):
    """Binarized 'stratiform' (T/F) per row:
    a: T F T F F T
    b: T F F T F T
    c: T T F F F F
    a vs b disagree at rows 2,3 -> 2/6
    a vs c disagree at rows 1,2,5 -> 3/6
    b vs c disagree at rows 1,3,5 -> 3/6
    """
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    matrix = calculate_label_disagreement_matrix(sample_df, "stratiform")
    assert matrix.loc["rater_a", "rater_b"] == pytest.approx(100 / 3)
    assert matrix.loc["rater_a", "rater_c"] == pytest.approx(50.0)
    assert matrix.loc["rater_b", "rater_c"] == pytest.approx(50.0)


def test_label_is_case_insensitive(sample_df):
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    lower = calculate_label_disagreement_matrix(sample_df, "stratiform")
    upper = calculate_label_disagreement_matrix(sample_df, "STRATIFORM")
    np.testing.assert_array_almost_equal(lower.values, upper.values)


def test_subset_of_columns(sample_df):
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    matrix = calculate_label_disagreement_matrix(sample_df, "stratiform", columns=["rater_a", "rater_c"])
    assert list(matrix.columns) == ["rater_a", "rater_c"]


def test_excludes_rows_with_missing_values():
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    df = pd.DataFrame({
        "a": ["x", "y", "x", None],
        "b": ["x", "y", "y", "x"],
    })
    matrix = calculate_label_disagreement_matrix(df, "x")
    # Row 4 dropped (a missing). Remaining rows: (x,x) agree, (y,y) agree
    # (both non-x -> agree), (x,y) disagree -> 1/3 disagreement.
    assert matrix.loc["a", "b"] == pytest.approx(100 / 3)


def test_nan_when_no_shared_rows():
    from lars.util.label_disagreement_matrix import calculate_label_disagreement_matrix

    df = pd.DataFrame({
        "a": ["x", None],
        "b": [None, "y"],
    })
    matrix = calculate_label_disagreement_matrix(df, "x")
    assert np.isnan(matrix.loc["a", "b"])


def test_plot_returns_axes(sample_df):
    from lars.util.label_disagreement_matrix import plot_label_disagreement_matrix

    _, ax = plt.subplots()
    result = plot_label_disagreement_matrix(sample_df, "stratiform", ax=ax)
    assert result is ax


def test_plot_title_includes_label(sample_df):
    from lars.util.label_disagreement_matrix import plot_label_disagreement_matrix

    _, ax = plt.subplots()
    plot_label_disagreement_matrix(sample_df, "stratiform", ax=ax)
    assert ax.get_title() == "Item-Level Disagreement: 'stratiform'"


def test_plot_tick_labels_match_columns(sample_df):
    from lars.util.label_disagreement_matrix import plot_label_disagreement_matrix

    _, ax = plt.subplots()
    plot_label_disagreement_matrix(sample_df, "stratiform", ax=ax)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["rater_a", "rater_b", "rater_c"]


def test_plot_custom_labels(sample_df):
    from lars.util.label_disagreement_matrix import plot_label_disagreement_matrix

    _, ax = plt.subplots()
    plot_label_disagreement_matrix(sample_df, "stratiform", ax=ax, labels=["Alice", "Bob", "LLM"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Alice", "Bob", "LLM"]


def test_plot_uses_gca_when_no_ax(sample_df):
    from lars.util.label_disagreement_matrix import plot_label_disagreement_matrix

    fig, ax = plt.subplots()
    plt.sca(ax)
    plot_label_disagreement_matrix(sample_df, "stratiform")
    assert ax.get_title() == "Item-Level Disagreement: 'stratiform'"
