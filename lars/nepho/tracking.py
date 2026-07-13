"""MLflow experiment tracking for LARS inference runs.

Provides validation-metric computation (reflectivity-criteria violations,
color-criteria violations, label agreement vs. hand labels) and a single
``log_run_to_mlflow`` entry point that lazily imports mlflow.
"""
import hashlib
import os
import tempfile

import pandas as pd


def _lazy_mlflow():
    try:
        import mlflow
    except ImportError as e:
        raise ImportError(
            "mlflow is required for experiment tracking. "
            "Install it with `pip install mlflow`."
        ) from e
    return mlflow


def codebook_hash(path):
    """Return a short SHA-256 of the codebook file at ``path``, or None."""
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _safe(name):
    """Sanitize a string for use in mlflow metric / artifact names."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def _reflectivity_violation_metrics(df, criteria, label_col, prefix):
    """Per-rule violation counts and rates against the reflectivity criteria."""
    out = {}
    if criteria is None or label_col not in df.columns:
        return out
    total = len(df)
    for label, rules in criteria.items():
        label_mask = df[label_col] == label
        n_label = int(label_mask.sum())
        for rule in rules:
            field = rule["field"]
            if field not in df.columns:
                continue
            violated = label_mask & (df[field] > rule["max_value"])
            n_viol = int(violated.sum())
            base = (f"{prefix}/{_safe(label)}/"
                    f"{_safe(field)}_gt_{rule['max_value']}")
            out[f"{base}/count"] = n_viol
            out[f"{base}/rate_of_label"] = (
                n_viol / n_label if n_label else 0.0
            )
            out[f"{base}/rate_of_total"] = (
                n_viol / total if total else 0.0
            )
    return out


def _color_violation_metrics(df, color_criteria, label_col, prefix):
    """Per-rule violation counts and rates against the color criteria."""
    out = {}
    if color_criteria is None or label_col not in df.columns:
        return out
    total = len(df)
    for label, rules in color_criteria.items():
        label_mask = df[label_col] == label
        n_label = int(label_mask.sum())
        for rule in rules:
            field = rule["field"]
            if field not in df.columns:
                continue
            kind = rule["kind"]
            value = rule["value"]
            if kind == "max_pct_above":
                violated = label_mask & (df[field] > value)
                op = "gt"
            elif kind == "min_pct_above":
                violated = label_mask & (df[field] < value)
                op = "lt"
            else:
                continue
            n_viol = int(violated.sum())
            colors = "_".join(rule.get("colors", [])) or "color"
            base = (f"{prefix}/{_safe(label)}/{_safe(colors)}/"
                    f"{_safe(field)}_{op}_{value}")
            out[f"{base}/count"] = n_viol
            out[f"{base}/rate_of_label"] = (
                n_viol / n_label if n_label else 0.0
            )
            out[f"{base}/rate_of_total"] = (
                n_viol / total if total else 0.0
            )
    return out


def _agreement_metrics(df, hand_col, llm_col):
    """Overall accuracy, per-class recall, and Cohen's kappa vs hand labels."""
    out = {}
    if hand_col not in df.columns or llm_col not in df.columns:
        return out
    valid = df[[hand_col, llm_col]].dropna()
    valid = valid[(valid[hand_col].astype(str) != "")
                  & (valid[llm_col].astype(str) != "")]
    valid = valid[valid[hand_col].astype(str).str.upper() != "UNKNOWN"]
    n = len(valid)
    if n == 0:
        return out
    hand_norm = valid[hand_col].astype(str).str.lower()
    llm_norm = valid[llm_col].astype(str).str.lower()
    out["agreement/n_compared"] = int(n)
    out["agreement/overall_accuracy"] = float((hand_norm == llm_norm).mean())
    try:
        from sklearn.metrics import cohen_kappa_score
        out["agreement/cohen_kappa"] = float(
            cohen_kappa_score(hand_norm, llm_norm)
        )
    except Exception:
        pass
    for cls in sorted(hand_norm.unique()):
        mask = hand_norm == cls
        if mask.any():
            out[f"agreement/per_class/{_safe(cls)}/recall"] = float(
                (llm_norm[mask] == cls).mean()
            )
            out[f"agreement/per_class/{_safe(cls)}/n"] = int(mask.sum())
    return out


def compute_validation_metrics(df, criteria=None, color_criteria=None,
                               hand_label_col="label",
                               llm_label_col="llm_label"):
    """
    Compute a flat dict of validation metrics for an inference run.

    Reflectivity-criteria violations and color-criteria violations are
    computed twice — once against ``llm_label_col`` and once against
    ``hand_label_col`` (when present) — so a run can be evaluated both as
    "did the model break the codebook" and "did the codebook itself break
    on hand labels". Label-agreement metrics (accuracy, per-class recall,
    Cohen's kappa) compare ``hand_label_col`` to ``llm_label_col``,
    skipping rows whose hand label is missing or ``"UNKNOWN"``.

    Parameters
    ----------
    df : pd.DataFrame
    criteria : dict, optional
        Reflectivity criteria as returned by
        ``lars.nepho.inference.criteria_from_codebook``.
    color_criteria : dict, optional
        Color criteria as returned by
        ``lars.nepho.inference.color_criteria_from_codebook``.
    hand_label_col : str, optional
    llm_label_col : str, optional

    Returns
    -------
    dict
        Flat ``{metric_name: numeric_value}`` mapping safe to pass to
        ``mlflow.log_metrics``.
    """
    metrics = {}
    metrics.update(_reflectivity_violation_metrics(
        df, criteria, llm_label_col, "reflectivity_violations/llm"
    ))
    metrics.update(_reflectivity_violation_metrics(
        df, criteria, hand_label_col, "reflectivity_violations/hand"
    ))
    metrics.update(_color_violation_metrics(
        df, color_criteria, llm_label_col, "color_violations/llm"
    ))
    metrics.update(_color_violation_metrics(
        df, color_criteria, hand_label_col, "color_violations/hand"
    ))
    metrics.update(_agreement_metrics(df, hand_label_col, llm_label_col))
    return metrics


def _log_confusion_matrix(mlflow, df, tmpdir, hand_col, llm_col):
    if hand_col not in df.columns or llm_col not in df.columns:
        return
    valid = df[[hand_col, llm_col]].dropna()
    valid = valid[(valid[hand_col].astype(str) != "")
                  & (valid[llm_col].astype(str) != "")]
    valid = valid[valid[hand_col].astype(str).str.upper() != "UNKNOWN"]
    if len(valid) == 0:
        return

    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    hand_norm = valid[hand_col].astype(str).str.lower()
    llm_norm = valid[llm_col].astype(str).str.lower()
    labels = sorted(set(hand_norm) | set(llm_norm))
    cm = confusion_matrix(hand_norm, llm_norm, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    csv_path = os.path.join(tmpdir, "confusion_matrix.csv")
    cm_df.to_csv(csv_path)
    mlflow.log_artifact(csv_path, artifact_path="metrics")

    from lars.util.confusion_matrix import plot_confusion_matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(valid, label_col=hand_col, pred_col=llm_col, ax=ax)
    fig.tight_layout()
    png_path = os.path.join(tmpdir, "confusion_matrix.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(png_path, artifact_path="metrics")


def log_run_to_mlflow(radar_df, *, experiment, run_name=None,
                      tracking_uri=None, params=None,
                      criteria=None, color_criteria=None,
                      codebook_path=None, model_output_dir=None,
                      hand_label_col="label", llm_label_col="llm_label"):
    """
    Open an MLflow run, log params + validation metrics + artifacts, close it.

    Artifacts logged
    ----------------
    * ``data/labelled.csv`` — the labelled DataFrame including any
      ``*_original`` / ``*_criteria_violation`` audit columns.
    * ``metrics/confusion_matrix.csv`` and ``.png`` — vs. hand labels (if
      enough rows are present).
    * ``model_outputs/`` — raw text outputs from the LLM, if
      ``model_output_dir`` is provided.

    Parameters
    ----------
    radar_df : pd.DataFrame
        Labelled DataFrame (post-inference, post-criteria).
    experiment : str
        MLflow experiment name. Created if missing.
    run_name : str, optional
    tracking_uri : str, optional
        Forwarded to ``mlflow.set_tracking_uri`` if provided.
    params : dict, optional
        Extra params merged with the standard ``n_rows`` /
        ``codebook_hash`` set.
    criteria, color_criteria : dict, optional
        Passed to ``compute_validation_metrics``.
    codebook_path : str, optional
        Hashed and logged as ``codebook_hash`` for traceability.
    model_output_dir : str, optional
        Logged as the ``model_outputs`` artifact directory.

    Returns
    -------
    str
        The MLflow run ID.
    """
    mlflow = _lazy_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        all_params = {"n_rows": len(radar_df)}
        ch = codebook_hash(codebook_path)
        if ch:
            all_params["codebook_hash"] = ch
            all_params["codebook_path"] = codebook_path
        if params:
            all_params.update(params)
        mlflow.log_params({
            k: str(v)[:500] for k, v in all_params.items() if v is not None
        })

        metrics = compute_validation_metrics(
            radar_df,
            criteria=criteria,
            color_criteria=color_criteria,
            hand_label_col=hand_label_col,
            llm_label_col=llm_label_col,
        )
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "labelled.csv")
            radar_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="data")
            try:
                _log_confusion_matrix(
                    mlflow, radar_df, td, hand_label_col, llm_label_col
                )
            except Exception as e:
                mlflow.log_param(
                    "confusion_matrix_error", str(e)[:500]
                )

        if model_output_dir and os.path.isdir(model_output_dir):
            mlflow.log_artifacts(
                model_output_dir, artifact_path="model_outputs"
            )

        return run.info.run_id
