import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score


def calculate_kappa_matrix(df, columns=None):
    """
    Compute pairwise Cohen's kappa scores between multiple label columns.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeler/source.
    columns (list of str or None): Columns to compare. Defaults to all
        columns in ``df``.

    Returns
    -------
    pd.DataFrame
        Square DataFrame indexed and labeled by ``columns``, where entry
        (i, j) is the Cohen's kappa score between columns i and j (1.0 on
        the diagonal). Rows with a missing value in either of the two
        compared columns are excluded from that pair's calculation.
    """
    if columns is None:
        columns = list(df.columns)

    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1:]:
            pair = df[[col_i, col_j]].dropna()
            a = pair[col_i].astype(str).str.lower()
            b = pair[col_j].astype(str).str.lower()
            kappa = cohen_kappa_score(a, b) if len(pair) > 0 else np.nan
            matrix.loc[col_i, col_j] = kappa
            matrix.loc[col_j, col_i] = kappa

    return matrix


def plot_kappa_matrix(df, columns=None, labels=None, ax=None, cmap=None, annot=True, vmin=-1, vmax=1,
                      matrix=None):
    """
    Plot a matrix of pairwise Cohen's kappa scores between multiple label
    columns as a heatmap.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing one column per labeler/source.
    columns (list of str or None): Columns to compare. Defaults to all
        columns in ``df``.
    labels (list of str or None): Display names for the tick labels, in the
        same order as ``columns``. Defaults to the column names.
    ax (matplotlib axis handle): The axis handle to plot on. Set to None to
        use the current axis.
    cmap (matplotlib colormap or None): Colormap for the heatmap. Defaults
        to ``plt.cm.RdYlGn``.
    annot (bool): Whether to annotate each cell with its kappa value.
    vmin (float): Minimum value for the colormap. Defaults to -1.
    vmax (float): Maximum value for the colormap. Defaults to 1.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the kappa matrix was drawn on.
    """
    if matrix is None:
        matrix = calculate_kappa_matrix(df, columns=columns)
    display_labels = labels if labels is not None else list(matrix.columns)

    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = plt.cm.RdYlGn

    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax)

    n = len(display_labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)

    if annot:
        for i in range(n):
            for j in range(n):
                value = matrix.values[i, j]
                text = "" if np.isnan(value) else f"{value:.2f}"
                color = "white" if (not np.isnan(value)) and value < 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)

    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Cohen's Kappa Matrix")

    return ax
