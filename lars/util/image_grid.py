import math
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_label_images(
    df,
    hand_label,
    llm_label,
    n,
    output_path,
    label_col="label",
    pred_col="llm_label",
    seed=None,
):
    """
    Randomly sample images where hand label and LLM label match the given values,
    plot them in a single grid figure, and save to a PNG file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at minimum columns for file paths, hand labels, and LLM labels.
    hand_label : str
        The human-annotated label to filter on.
    llm_label : str
        The LLM-generated label to filter on.
    n : int
        Number of images to sample and display.
    output_path : str
        File path for the saved PNG (e.g. "output/grid.png").
    label_col : str
        Column name for hand labels. Default is "label".
    pred_col : str
        Column name for LLM labels. Default is "llm_label".
    seed : int or None
        Random seed for reproducibility. Default is None.

    Raises
    ------
    ValueError
        If fewer matching images exist than requested.
    """
    mask = (df[label_col].str.lower() == hand_label.lower()) & (
        df[pred_col].str.lower() == llm_label.lower()
    )
    subset = df[mask]

    if len(subset) == 0:
        raise ValueError(
            f"No images found where {label_col}='{hand_label}' and {pred_col}='{llm_label}'."
        )
    if len(subset) < n:
        raise ValueError(
            f"Requested {n} images but only {len(subset)} match the given labels."
        )

    rng = random.Random(seed)
    sampled = subset.sample(n=n, random_state=seed if seed is not None else rng.randint(0, 2**31))

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Normalise axes to always be a flat list
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, (idx, row) in zip(axes, sampled.iterrows()):
        img = mpimg.imread(row["file_path"])
        ax.imshow(img)
        ax.set_title(str(idx), fontsize=8)
        ax.axis("off")

    # Hide any unused subplot panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Hand label: '{hand_label}'  |  LLM label: '{llm_label}'  |  n={n}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
