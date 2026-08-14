import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@pytest.fixture
def sample_df():
    n_cycles = 10
    true_pattern = ["convective", "stratiform", "anvil"] * n_cycles

    rater_a = list(true_pattern)
    rater_b = list(true_pattern)
    rater_b[0], rater_b[1] = "stratiform", "convective"
    rater_c = ["convective" if label == "anvil" else label for label in true_pattern]

    # An extra row where all three raters pick a different class: genuinely
    # ambiguous, unlike the systematic rater_c bias above (which the model
    # learns to discount and remains confident about).
    rater_a.append("convective")
    rater_b.append("stratiform")
    rater_c.append("anvil")
    exp_good = true_pattern + ["convective"]
    exp_bad = ["anvil"] * len(true_pattern) + ["anvil"]

    return pd.DataFrame({
        "rater_a": rater_a,
        "rater_b": rater_b,
        "rater_c": rater_c,
        "exp_good": exp_good,
        "exp_bad": exp_bad,
    })


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def fitted(sample_df):
    from lars.util.dawid_skene import fit_dawid_skene

    return fit_dawid_skene(sample_df, columns=["rater_a", "rater_b", "rater_c"])


def test_confusion_matrices_are_row_stochastic(fitted):
    for matrix in fitted["confusion_matrices"].values():
        np.testing.assert_array_almost_equal(matrix.sum(axis=1).values, np.ones(len(matrix)))


def test_confusion_matrices_indexed_by_columns(fitted, sample_df):
    assert set(fitted["confusion_matrices"].keys()) == {"rater_a", "rater_b", "rater_c"}
    for matrix in fitted["confusion_matrices"].values():
        assert set(matrix.index) == {"convective", "stratiform", "anvil"}
        assert set(matrix.columns) == {"convective", "stratiform", "anvil"}


def test_noisy_rater_has_lower_diagonal_mass_for_biased_class(fitted):
    clean = fitted["confusion_matrices"]["rater_a"].loc["anvil", "anvil"]
    noisy = fitted["confusion_matrices"]["rater_c"].loc["anvil", "anvil"]
    assert clean > noisy


def test_consensus_recovers_true_pattern(fitted, sample_df):
    cyclic_rows = slice(0, -1)
    assert (fitted["consensus"].values[cyclic_rows] == sample_df["exp_good"].values[cyclic_rows]).all()


def test_unanimous_row_has_near_zero_entropy(fitted):
    assert fitted["item_entropy"].iloc[3] < 0.01


def test_three_way_disagreement_has_higher_entropy_than_unanimous(fitted):
    assert fitted["item_entropy"].iloc[-1] > fitted["item_entropy"].iloc[3]


def test_consensus_proba_rows_sum_to_one(fitted):
    np.testing.assert_array_almost_equal(
        fitted["consensus_proba"].sum(axis=1).values, np.ones(len(fitted["consensus_proba"]))
    )


def test_subset_of_columns(sample_df):
    from lars.util.dawid_skene import fit_dawid_skene

    result = fit_dawid_skene(sample_df, columns=["rater_a", "rater_c"])
    assert result["columns"] == ["rater_a", "rater_c"]
    assert set(result["confusion_matrices"].keys()) == {"rater_a", "rater_c"}


def test_score_against_consensus_ranks_good_above_bad(fitted, sample_df):
    from lars.util.dawid_skene import score_against_consensus

    scores = score_against_consensus(sample_df, fitted, columns=["exp_good", "exp_bad"])
    assert list(scores.index)[0] == "exp_good"
    assert scores.loc["exp_good", "accuracy"] == pytest.approx(1.0)
    assert scores.loc["exp_good", "macro_f1"] > scores.loc["exp_bad", "macro_f1"]


def test_score_against_consensus_sorted_by_macro_f1(fitted, sample_df):
    from lars.util.dawid_skene import score_against_consensus

    scores = score_against_consensus(sample_df, fitted, columns=["exp_bad", "exp_good"])
    assert scores["macro_f1"].is_monotonic_decreasing


def test_plot_returns_axes(fitted):
    from lars.util.dawid_skene import plot_dawid_skene_confusion

    _, ax = plt.subplots()
    result = plot_dawid_skene_confusion(fitted, "rater_a", ax=ax)
    assert result is ax


def test_plot_tick_labels_match_classes(fitted):
    from lars.util.dawid_skene import plot_dawid_skene_confusion

    _, ax = plt.subplots()
    plot_dawid_skene_confusion(fitted, "rater_a", ax=ax)
    labels = set(t.get_text() for t in ax.get_xticklabels())
    assert labels == {"convective", "stratiform", "anvil"}
