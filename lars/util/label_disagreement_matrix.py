import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_label_disagreement_matrix(df, label, columns=None):
    """
    Compute a matrix of pairwise item-level disagreement rates for a single
    label, treated as a one-vs-rest binary classification.

    For each pair of labellers, only rows where both columns have a
    non-missing value are considered. Each value is binarized to whether it
    equals ``label`` (case-insensitively), and entry (i, j) of the returned
    matrix is the percentage of those rows where the two labellers'
    binarized values differ.

    Unlike ``calculate_label_rate_diff_matrix`` (see ``label_rate_matrix.py``),
    which only compares how often each labeller uses ``label`` overall, this
    compares labellers item-by-item, so it can be high even when two
    labellers use ``label`` at the same overall rate but disagree on which
    items it applies to.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeller/source.
    label (str): The label value to treat as the positive class. Compared
        case-insensitively.
    columns (list of str or None): Columns (labellers) to compare. Defaults
        to all columns in ``df``.

    Returns
    -------
    pd.DataFrame
        Square DataFrame indexed and labeled by ``columns``, where entry
        (i, j) is the item-level disagreement rate (0-100) between columns
        i and j (0.0 on the diagonal). NaN if a pair shares no non-missing
        rows.
    """
    if columns is None:
        columns = list(df.columns)

    label_lower = str(label).lower()
    matrix = pd.DataFrame(0.0, index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1:]:
            pair = df[[col_i, col_j]].dropna()
            if len(pair) == 0:
                diff = np.nan
            else:
                a = pair[col_i].astype(str).str.lower() == label_lower
                b = pair[col_j].astype(str).str.lower() == label_lower
                diff = 100.0 * (a != b).mean()
            matrix.loc[col_i, col_j] = diff
            matrix.loc[col_j, col_i] = diff

    return matrix


def plot_label_disagreement_matrix(df, label, columns=None, labels=None, ax=None,
                                   cmap=None, annot=True):
    """
    Plot a matrix of pairwise item-level disagreement rates for a single
    label as a heatmap.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeller/source.
    label (str): The label value to treat as the positive class.
    columns (list of str or None): Columns to compare. Defaults to all
        columns in ``df``.
    labels (list of str or None): Display names for the tick labels, in the
        same order as ``columns``. Defaults to the column names.
    ax (matplotlib axis handle): The axis handle to plot on. Set to None to
        use the current axis.
    cmap (matplotlib colormap or None): Colormap for the heatmap. Defaults
        to ``plt.cm.Reds``.
    annot (bool): Whether to annotate each cell with its value.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the matrix was drawn on.
    """
    matrix = calculate_label_disagreement_matrix(df, label, columns=columns)
    display_labels = labels if labels is not None else list(matrix.columns)

    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = plt.cm.Reds

    finite_values = matrix.values[~np.isnan(matrix.values)]
    vmax = finite_values.max() if finite_values.size > 0 and finite_values.max() > 0 else 100.0

    im = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=vmax)

    n = len(display_labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)

    if annot:
        for i in range(n):
            for j in range(n):
                value = matrix.values[i, j]
                text = "" if np.isnan(value) else f"{value:.1f}"
                color = "white" if (not np.isnan(value)) and value > vmax / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)

    ax.figure.colorbar(im, ax=ax, label="Disagreement rate (%)")
    ax.set_title(f"Item-Level Disagreement: '{label}'")

    return ax
