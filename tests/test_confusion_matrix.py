import io
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DATA = """\
label,llm_label
stratiform,stratiform
convective,convective
stratiform,convective
convective,stratiform
anvil,anvil
stratiform,stratiform
"""


@pytest.fixture
def sample_df():
    return pd.read_csv(io.StringIO(CSV_DATA))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_title_is_set(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_returns_none(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    result = plot_confusion_matrix(sample_df, ax=ax)
    assert result is None


def test_normalized(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, normalize="true", ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_custom_column_names(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    df = sample_df.rename(columns={"label": "true", "llm_label": "pred"})
    _, ax = plt.subplots()
    plot_confusion_matrix(df, label_col="true", pred_col="pred", ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_uses_gca_when_no_ax(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    fig, ax = plt.subplots()
    plt.sca(ax)
    plot_confusion_matrix(sample_df)
    assert ax.get_title() == "Confusion Matrix"


def test_confusion_matrix_counts(sample_df):
    """Labels sorted alphabetically: anvil=0, convective=1, stratiform=2.

    True \\ Pred   anvil  convective  stratiform
    anvil              1           0           0
    convective         0           1           1
    stratiform         0           1           2
    """
    from lars.util.confusion_matrix import plot_confusion_matrix

    expected = np.array([[1, 0, 0],
                         [0, 1, 1],
                         [0, 1, 2]])

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, ax=ax)
    actual = ax.images[0].get_array()
    np.testing.assert_array_equal(actual, expected)


def test_cohen_kappa(sample_df):
    """κ = (p_o - p_e) / (1 - p_e)
    p_o = 4/6 (4 agreements out of 6 rows)
    p_e = (3/6)^2 + (2/6)^2 + (1/6)^2 = 14/36 = 7/18
    κ   = (12/18 - 7/18) / (11/18) = 5/11
    """
    from lars.util.confusion_matrix import calculate_cohen_kappa

    kappa = calculate_cohen_kappa(sample_df)
    assert kappa == pytest.approx(5 / 11)


def test_normalized_confusion_matrix_values(sample_df):
    """Row-normalized values for each true class."""
    from lars.util.confusion_matrix import plot_confusion_matrix

    expected = np.array([[1.0,       0.0,       0.0      ],
                         [0.0,       0.5,       0.5      ],
                         [0.0,       1.0 / 3.0, 2.0 / 3.0]])

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, normalize="true", ax=ax)
    actual = ax.images[0].get_array()
    np.testing.assert_array_almost_equal(actual, expected)
