import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def _log_sum_exp(log_values, axis):
    max_val = np.max(log_values, axis=axis, keepdims=True)
    max_val = np.where(np.isfinite(max_val), max_val, 0.0)
    summed = np.sum(np.exp(log_values - max_val), axis=axis, keepdims=True)
    return (max_val + np.log(summed)).squeeze(axis)


def fit_dawid_skene(df, columns=None, max_iter=100, tol=1e-4, smoothing=0.1):
    """
    Fit a Dawid-Skene model to estimate a consensus label per item and a
    confusion matrix per rater, without requiring any rater's labels to be
    treated as ground truth.

    Each rater's label is modeled as a noisy observation of an unknown true
    class. The EM algorithm jointly estimates (a) each rater's confusion
    matrix ``P(observed | true)`` and (b) a posterior distribution over the
    true class for every item. The entropy of that posterior is a per-item
    uncertainty estimate: near zero when raters agree, higher when they
    don't.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per rater/source.
        Missing values (NaN) are treated as "this rater did not label this
        item".
    columns (list of str or None): Columns (raters) to fit on. Defaults to
        all columns in ``df``.
    max_iter (int): Maximum number of EM iterations.
    tol (float): Convergence tolerance on the maximum absolute change in the
        item posterior between iterations.
    smoothing (float): Additive (Laplace) smoothing applied to each rater's
        confusion matrix during the M-step, to avoid zero probabilities.

    Returns
    -------
    dict with keys:
        consensus (pd.Series): MAP consensus label per item, indexed like
            ``df``. NaN for items with no non-missing rater.
        consensus_proba (pd.DataFrame): Item x class posterior probabilities.
        item_entropy (pd.Series): Shannon entropy (bits) of each item's
            posterior -- the per-item uncertainty estimate.
        confusion_matrices (dict[str, pd.DataFrame]): Per-rater confusion
            matrix, indexed by true class and labeled by observed class.
            Each row sums to 1.
        class_prior (pd.Series): Estimated prevalence of each true class.
        columns (list of str): Columns used to fit the model.
        n_iter (int): Number of EM iterations actually run.
        log_likelihood (list of float): Observed-data log-likelihood at each
            iteration (should increase monotonically).
    """
    if columns is None:
        columns = list(df.columns)

    labels = df[columns].apply(lambda s: s.astype(str).str.lower())
    labels = labels.where(df[columns].notna())

    classes = sorted(set(labels.values.flatten()) - {None, np.nan})
    classes = [c for c in classes if isinstance(c, str)]
    n_classes = len(classes)
    class_index = {c: k for k, c in enumerate(classes)}

    n_items = len(df)
    n_raters = len(columns)

    observed = np.full((n_items, n_raters), -1, dtype=int)
    for j, col in enumerate(columns):
        for i, value in enumerate(labels[col].values):
            if isinstance(value, str):
                observed[i, j] = class_index[value]

    has_any_label = (observed >= 0).any(axis=1)

    q = np.full((n_items, n_classes), 1.0 / n_classes)
    for i in range(n_items):
        if not has_any_label[i]:
            continue
        votes = np.zeros(n_classes)
        for j in range(n_raters):
            if observed[i, j] >= 0:
                votes[observed[i, j]] += 1
        q[i] = votes / votes.sum()

    log_likelihood_history = []
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        # M-step
        class_prior = q[has_any_label].mean(axis=0)
        confusion = np.zeros((n_raters, n_classes, n_classes))
        for j in range(n_raters):
            mask = observed[:, j] >= 0
            numer = np.full((n_classes, n_classes), smoothing)
            denom = np.full(n_classes, smoothing * n_classes)
            for k in range(n_classes):
                weight = q[mask, k]
                denom[k] += weight.sum()
                for l in range(n_classes):
                    numer[k, l] += weight[observed[mask, j] == l].sum()
            confusion[j] = numer / denom[:, None]

        # E-step
        log_confusion = np.log(confusion)
        log_prior = np.log(class_prior)
        log_q_unnorm = np.tile(log_prior, (n_items, 1))
        for j in range(n_raters):
            mask = observed[:, j] >= 0
            log_q_unnorm[mask] += log_confusion[j, :, observed[mask, j]]

        log_norm = _log_sum_exp(log_q_unnorm, axis=1)
        log_likelihood_history.append(log_norm[has_any_label].sum())

        new_q = np.exp(log_q_unnorm - log_norm[:, None])
        new_q[~has_any_label] = 1.0 / n_classes

        delta = np.max(np.abs(new_q - q))
        q = new_q
        if delta < tol:
            break

    index = df.index
    consensus_proba = pd.DataFrame(q, index=index, columns=classes)
    consensus_proba.loc[~has_any_label, :] = np.nan

    consensus = pd.Series(
        [classes[k] for k in np.argmax(q, axis=1)], index=index
    )
    consensus[~has_any_label] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(q > 0, q * np.log2(q), 0.0)
    item_entropy = pd.Series(-entropy_terms.sum(axis=1), index=index)
    item_entropy[~has_any_label] = np.nan

    confusion_matrices = {
        col: pd.DataFrame(confusion[j], index=classes, columns=classes)
        for j, col in enumerate(columns)
    }

    return {
        "consensus": consensus,
        "consensus_proba": consensus_proba,
        "item_entropy": item_entropy,
        "confusion_matrices": confusion_matrices,
        "class_prior": pd.Series(class_prior, index=classes),
        "columns": columns,
        "n_iter": n_iter,
        "log_likelihood": log_likelihood_history,
    }


def score_against_consensus(df, result, columns=None):
    """
    Score one or more raters/experiments against a Dawid-Skene consensus
    label, e.g. to rank several LLM labelling experiments without treating
    any single human rater as ground truth.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the columns to score. Must share
        an index with the DataFrame ``result`` was fit on.
    result (dict): Return value of ``fit_dawid_skene``.
    columns (list of str or None): Columns to score. Defaults to the columns
        the model was fit on (``result['columns']``).

    Returns
    -------
    pd.DataFrame
        Indexed by column name, with columns ``accuracy``, ``macro_f1``, and
        ``kappa`` computed against the consensus label (rows missing either
        the consensus or the rater's label are excluded per column). Sorted
        by ``macro_f1`` descending, so the top row is the best-agreeing
        experiment.
    """
    if columns is None:
        columns = result["columns"]

    consensus = result["consensus"]
    rows = []
    for col in columns:
        pair = pd.DataFrame({"consensus": consensus, "rater": df[col]}).dropna()
        true_values = pair["consensus"]
        pred_values = pair["rater"].astype(str).str.lower()
        rows.append(
            {
                "column": col,
                "accuracy": accuracy_score(true_values, pred_values),
                "macro_f1": f1_score(
                    true_values, pred_values, average="macro", zero_division=0
                ),
                "kappa": cohen_kappa_score(true_values, pred_values),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("column")
        .sort_values("macro_f1", ascending=False)
    )


def plot_dawid_skene_confusion(result, column, ax=None, cmap=None, annot=True):
    """
    Plot one rater's estimated Dawid-Skene confusion matrix as a heatmap.

    Parameters
    ----------
    result (dict): Return value of ``fit_dawid_skene``.
    column (str): Which rater's confusion matrix to plot (a key of
        ``result['confusion_matrices']``).
    ax (matplotlib axis handle): The axis handle to plot on. Set to None to
        use the current axis.
    cmap (matplotlib colormap or None): Colormap for the heatmap. Defaults
        to ``plt.cm.Blues``.
    annot (bool): Whether to annotate each cell with its value.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the confusion matrix was drawn on.
    """
    matrix = result["confusion_matrices"][column]
    classes = list(matrix.columns)

    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = plt.cm.Blues

    im = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=1)

    n = len(classes)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    if annot:
        for i in range(n):
            for j in range(n):
                value = matrix.values[i, j]
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black",
                )

    ax.figure.colorbar(im, ax=ax, label="P(observed | true)")
    ax.set_xlabel("Observed label")
    ax.set_ylabel("True label (estimated)")
    ax.set_title(f"Dawid-Skene Confusion Matrix: {column}")

    return ax
