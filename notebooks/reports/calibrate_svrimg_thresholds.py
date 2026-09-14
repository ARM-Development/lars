"""Calibrate reflectivity-coverage thresholds for CODEBOOK_SVRIMG.md.

Cellular vs QLCS is a morphological distinction -- discrete, mutually
separated cores versus a single connected quasi-linear band -- so it is an
open question whether scalar reflectivity coverage fractions discriminate the
two classes at all. This script answers that empirically rather than assuming
either way.

For each candidate dBZ threshold it computes the coverage fraction over the
SVRIMG *training* split only (1996-2011), then reports ROC AUC and the
Youden-optimal cut point for Cellular vs QLCS. A threshold earns a place in
the codebook only if it clears AUC_GATE; everything else is reported here and
recorded in the codebook changelog as a measured null result.

Run directly:

    python notebooks/reports/calibrate_svrimg_thresholds.py

The first run downloads roughly 1,000 scenes (~11 MB) and caches them.
"""
import sys
from unittest.mock import MagicMock
sys.modules.setdefault("pip_system_certs", MagicMock())
sys.modules.setdefault("pip_system_certs.wrapt_requests", MagicMock())

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lars.preprocessing.svrimg import (  # noqa: E402
    fetch_svrimg_labels, read_svrimg_png, reflectivity_stats,
    svrimg_image_url, _download,
)

SVRIMG_DIR = os.environ.get(
    "SVRIMG_DATA_DIR", os.path.expanduser("~/lars_examples/svrimg")
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "svrimg_threshold_calibration.csv")

DBZ_THRESHOLDS = (10, 20, 30, 40, 50, 60)

#: Minimum ROC AUC for a threshold to be written into the codebook.
AUC_GATE = 0.65

#: QLCS is the positive class throughout.
POSITIVE = "QLCS"


def build_feature_table(split="train", haz_type="tor"):
    """
    Decode every scene in a split and return its coverage fractions.

    Parameters
    ----------
    split (str): Published SVRIMG split to use. Default 'train'.
    haz_type (str): Hazard index. Default 'tor'.

    Returns
    -------
    pd.DataFrame
        One row per scene, with a 'label' column and one
        'pct_gates_<T>dbz' column per threshold in DBZ_THRESHOLDS.
    """
    labels = fetch_svrimg_labels(SVRIMG_DIR, haz_type=haz_type)
    labels = labels[labels["split"] == split]
    print(f"{split}: {len(labels)} scenes "
          f"({labels['label'].value_counts().to_dict()})")

    rows = []
    for n, (unid, row) in enumerate(labels.iterrows(), start=1):
        dest = os.path.join(
            SVRIMG_DIR, haz_type, unid[:4], unid[4:6], f"{unid}.png"
        )
        try:
            _download(svrimg_image_url(unid, haz_type), dest)
            reflectivity = read_svrimg_png(dest)
        except Exception as err:                      # noqa: BLE001
            print(f"  skipping {unid}: {err}")
            continue

        stats = reflectivity_stats(reflectivity, DBZ_THRESHOLDS)
        rows.append({"unid": unid, "label": row["label"], **stats})

        if n % 200 == 0:
            print(f"  ...{n}/{len(labels)}")

    return pd.DataFrame(rows).set_index("unid")


def evaluate(features):
    """
    Score each coverage fraction as a Cellular/QLCS discriminator.

    Parameters
    ----------
    features (pd.DataFrame): Output of :func:`build_feature_table`.

    Returns
    -------
    pd.DataFrame
        One row per threshold with ROC AUC, the orientation of the
        relationship, the Youden-optimal cut point and its sensitivity and
        specificity, and whether the threshold clears AUC_GATE.
    """
    y = (features["label"] == POSITIVE).astype(int).values
    results = []

    for threshold in DBZ_THRESHOLDS:
        column = f"pct_gates_{threshold}dbz"
        score = features[column].values

        auc = roc_auc_score(y, score)
        # An AUC below 0.5 just means coverage runs the other way (higher
        # coverage implies Cellular). Orient it so magnitudes compare.
        oriented = max(auc, 1 - auc)
        direction = "higher=QLCS" if auc >= 0.5 else "higher=Cellular"

        fpr, tpr, cuts = roc_curve(y, score if auc >= 0.5 else -score)
        best = int(np.argmax(tpr - fpr))
        cut = cuts[best] if auc >= 0.5 else -cuts[best]

        results.append({
            "threshold_dbz": threshold,
            "column": column,
            "auc": round(auc, 4),
            "oriented_auc": round(oriented, 4),
            "direction": direction,
            "youden_cut_pct": round(float(cut), 4),
            "sensitivity": round(float(tpr[best]), 4),
            "specificity": round(float(1 - fpr[best]), 4),
            "keep_in_codebook": bool(oriented >= AUC_GATE),
        })

    return pd.DataFrame(results)


def main():
    features = build_feature_table()
    results = evaluate(features)

    pd.set_option("display.width", 160)
    print("\nCellular vs QLCS discrimination on the training split "
          f"(n={len(features)}, positive class = {POSITIVE}):\n")
    print(results.to_string(index=False))

    kept = results[results["keep_in_codebook"]]
    print(f"\nAUC gate = {AUC_GATE}. "
          f"{len(kept)} of {len(results)} thresholds clear it.")
    if kept.empty:
        print("=> No quantitative coverage criteria belong in the codebook; "
              "rely on the topological description.")
    else:
        for _, r in kept.iterrows():
            print(f"=> {r['column']}: cut at {r['youden_cut_pct']}% "
                  f"({r['direction']}, AUC {r['oriented_auc']})")

    results.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
