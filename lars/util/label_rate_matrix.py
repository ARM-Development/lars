import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_label_rate_diff_matrix(df, label, columns=None):
    """
    Compute a matrix of pairwise usage-rate differences for a single label
    across multiple labellers.

    For each column (labeller), the usage rate is the percentage of its
    non-missing values that equal ``label``. Entry (i, j) of the returned
    matrix is the absolute difference, in percentage points, between the
    usage rates of labellers i and j.

    Unlike Cohen's kappa (see ``kappa_matrix.py``), this does not require
    labellers to agree item-by-item -- it only compares how often each
    labeller applies ``label`` overall, so it can flag labellers that are
    systematically biased toward or away from a class even when they are
    scored on different items.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeller/source.
    label (str): The label value to compute usage rates for. Compared
        case-insensitively.
    columns (list of str or None): Columns (labellers) to compare. Defaults
        to all columns in ``df``.

    Returns
    -------
    pd.DataFrame
        Square DataFrame indexed and labeled by ``columns``, where entry
        (i, j) is the absolute usage-rate difference (0-100) between
        columns i and j (0.0 on the diagonal).
    """
    if columns is None:
        columns = list(df.columns)

    label_lower = str(label).lower()
    rates = {}
    for col in columns:
        values = df[col].dropna().astype(str).str.lower()
        rates[col] = 100.0 * (values == label_lower).mean() if len(values) > 0 else np.nan

    matrix = pd.DataFrame(0.0, index=columns, columns=columns)
    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1:]:
            diff = abs(rates[col_i] - rates[col_j])
            matrix.loc[col_i, col_j] = diff
            matrix.loc[col_j, col_i] = diff

    return matrix


def plot_label_rate_diff_matrix(df, label, columns=None, labels=None, ax=None,
                                cmap=None, annot=True):
    """
    Plot a matrix of pairwise usage-rate differences for a single label
    across multiple labellers as a heatmap.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeller/source.
    label (str): The label value to compute usage rates for.
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
    matrix = calculate_label_rate_diff_matrix(df, label, columns=columns)
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

    ax.figure.colorbar(im, ax=ax, label="Usage rate difference (pp)")
    ax.set_title(f"Usage Rate Difference: '{label}'")

    return ax
