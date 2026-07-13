import pandas as pd
import os

def change_file_path(radar_df, new_path):
    """
    Change the file paths in the radar DataFrame to a new path.

    Parameters
    ----------
    radar_df (pd.DataFrame): DataFrame containing radar data with file paths.
    new_path (str): New base path to replace in the file paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated file paths.
    """
    radar_df = radar_df.copy()
    radar_df['file_path'] = radar_df['file_path'].apply(
        lambda x: os.path.join(new_path, os.path.basename(x))
    )
    return radar_df

def load_labels(label_file):
    """
    Load labels from a CSV file.

    Parameters
    ----------
    label_file (str): Path to the CSV file containing labels.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the labels.
    """
    return pd.read_csv(label_file)

def copy_labels(source_df, target_df, match_on='time', label_column='label'):
    """
    Copy labels from a source DataFrame to a target DataFrame.

    Matches rows either on the time index or on the file name (basename only,
    so the directory portion of the path does not need to match).

    Parameters
    ----------
    source_df (pd.DataFrame): DataFrame containing the labels to copy from.
    target_df (pd.DataFrame): DataFrame to copy the labels into.
    match_on (str): Either 'time' (match on the DataFrame index) or
        'file_path' (match on the basename of the 'file_path' column).
    label_column (str): Name of the column containing labels. Default 'label'.

    Returns
    -------
    pd.DataFrame
        A copy of ``target_df`` with labels filled in from ``source_df`` where
        a match was found. Rows with no match keep their existing label value.
    """
    if match_on not in ('time', 'file_path'):
        raise ValueError("match_on must be either 'time' or 'file_path'")

    target_df = target_df.copy()

    if match_on == 'time':
        lookup = source_df[label_column]
        new_labels = target_df.index.map(lookup)
    else:
        source_keys = source_df['file_path'].apply(os.path.basename)
        lookup = pd.Series(source_df[label_column].values, index=source_keys)
        target_keys = target_df['file_path'].apply(os.path.basename)
        new_labels = target_keys.map(lookup)

    new_labels = pd.Series(new_labels, index=target_df.index)
    if label_column in target_df.columns:
        target_df[label_column] = new_labels.where(new_labels.notna(),
                                                   target_df[label_column])
    else:
        target_df[label_column] = new_labels
    return target_df


def apply_criteria_to_labels(df, criteria, label_column='label',
                             original_column=None, violation_column=None):
    """
    Reclassify rows whose label violates a hard criterion from the codebook.

    For each rule attached to a label in ``criteria``, every row currently
    carrying that label is checked against the rule's ``field`` and
    ``max_value``. Rows whose value is strictly greater than ``max_value``
    have their label overwritten with the rule's ``reclassify_as`` target
    (or kept unchanged if the target is missing). The pre-override label
    and a short description of the rule that fired are recorded in
    ``original_column`` and ``violation_column`` for auditability.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``label_column`` and every ``field`` referenced by the
        criteria (e.g. ``pct_gates_50dbz``, ``n_gates_30dbz``). Rules
        referencing missing columns are skipped.
    criteria : dict
        Mapping of label → list of criterion dicts as returned by
        ``lars.nepho.inference.criteria_from_codebook``.
    label_column : str, optional
        Column to read and update. Default ``'label'``. Use ``'llm_label'``
        to enforce hard criteria on model output.
    original_column : str, optional
        Column to record the pre-override label.
        Defaults to ``f"{label_column}_original"``.
    violation_column : str, optional
        Column to record the rule that fired.
        Defaults to ``f"{label_column}_criteria_violation"``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with possibly updated labels and the two audit
        columns populated for any rows that were reclassified.
    """
    if original_column is None:
        original_column = f"{label_column}_original"
    if violation_column is None:
        violation_column = f"{label_column}_criteria_violation"

    df = df.copy()
    if original_column not in df.columns:
        df[original_column] = pd.NA
    if violation_column not in df.columns:
        df[violation_column] = pd.NA

    for label, rules in criteria.items():
        label_mask = df[label_column] == label
        if not label_mask.any():
            continue
        for rule in rules:
            field = rule["field"]
            if field not in df.columns:
                continue
            violated = label_mask & (df[field] > rule["max_value"])
            if not violated.any():
                continue
            new_label = rule["reclassify_as"] or label
            unit = "percent" if rule["kind"] == "pct" else "gates"
            note = (f"{label}: {field} > {rule['max_value']} {unit}"
                    f" -> {new_label}")
            no_original = df[original_column].isna()
            df.loc[violated & no_original, original_column] = label
            df.loc[violated, violation_column] = note
            df.loc[violated, label_column] = new_label
            label_mask = label_mask & ~violated

    return df


def save_labels(label_df, output_file):
    """
    Save labels to a CSV file.

    Parameters
    ----------
    label_df (pd.DataFrame): DataFrame containing the labels.
    output_file (str): Path to save the CSV file.
    """
    label_df.to_csv(output_file, index=False)