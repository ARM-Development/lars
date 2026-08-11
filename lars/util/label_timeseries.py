import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Codebook label order, from clear sky to organised convection. Used to order
# the categorical y-axis when the labels present in the data are a subset of
# these. Any labels not listed here are appended afterwards in the order they
# first appear.
DEFAULT_LABEL_ORDER = [
    "No Precipitation",
    "Stratiform Precipitation",
    "Isolated Convection",
    "Mesoscale Convective System",
    "Ambiguous",
    "UNKNOWN",
]


def _ordered_categories(values, order):
    """Return the unique *values* ordered by *order*, appending extras at end."""
    present = list(pd.unique(values.dropna()))
    ordered = [c for c in order if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def plot_label_timeseries(
    df,
    time_col="time",
    label_col="label",
    pred_col=None,
    order=None,
    time_format="%m/%d/%y %H:%M",
    ax=None,
    output_path=None,
):
    """
    Plot a time series of categorical labels.

    Each label class occupies a row on the y-axis and the sequence of labels is
    drawn as a stepped line with markers, so runs of the same class and
    transitions between classes are both easy to read. Optionally a second
    label column (e.g. model predictions) can be overlaid for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``time_col`` and ``label_col``.
    time_col : str
        Column holding the timestamps. Parsed with ``pd.to_datetime``.
        Default is ``"time"``.
    label_col : str
        Column holding the (categorical) labels to plot. Default is ``"label"``.
    pred_col : str or None
        Optional second label column to overlay on the same axes, useful for
        comparing hand labels against model predictions. Default is ``None``.
    order : list of str or None
        Explicit top-to-bottom ordering of the label classes on the y-axis.
        When ``None``, classes are ordered using the codebook order in
        ``DEFAULT_LABEL_ORDER`` with any unlisted classes appended.
    time_format : str or None
        ``strptime`` format string passed to ``pd.to_datetime`` for parsing
        ``time_col``. Defaults to ``"%m/%d/%y %H:%M"`` (the label CSV format).
        Set to ``None`` to let pandas infer the format.
    ax : matplotlib axis handle or None
        Axis to plot on. When ``None`` the current axis is used.
    output_path : str or None
        When given, the figure is saved to this path (and left open for
        further use). Default is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the time series was drawn on.

    Raises
    ------
    ValueError
        If ``df`` is empty or a requested column is missing.
    """
    for col in (time_col, label_col) + ((pred_col,) if pred_col else ()):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if len(df) == 0:
        raise ValueError("Cannot plot an empty DataFrame.")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], format=time_format)
    df = df.sort_values(time_col)

    if order is None:
        label_values = df[label_col]
        if pred_col:
            label_values = pd.concat([label_values, df[pred_col]])
        order = _ordered_categories(label_values, DEFAULT_LABEL_ORDER)

    positions = {label: i for i, label in enumerate(order)}

    if ax is None:
        ax = plt.gca()

    def _plot(col, **kwargs):
        y = df[col].map(positions)
        ax.plot(df[time_col], y, drawstyle="steps-post", marker="o",
                markersize=4, **kwargs)

    _plot(label_col, label=label_col, color="tab:blue")
    if pred_col:
        _plot(pred_col, label=pred_col, color="tab:orange", alpha=0.7)
        ax.legend(loc="best")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlabel("Time")
    ax.set_title("Label time series")
    ax.grid(True, axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")

    if output_path is not None:
        ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")

    return ax
