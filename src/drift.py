"""
drift.py
--------
Simple, explainable DATA DRIFT detection using the Population Stability
Index (PSI).

IDEA
    1. At training time we remember how each feature was distributed
       (the "baseline"): we cut the training values into 10 quantile bins
       and store the fraction of training rows in each bin.
    2. At serving time we put recent incoming requests into the SAME bins.
    3. PSI measures how different the two bin-fraction lists are:

          PSI = sum( (actual% - expected%) * ln(actual% / expected%) )

    Common rule of thumb:
          PSI < 0.10        -> Normal (no meaningful shift)
          0.10 <= PSI < 0.25 -> Moderate shift (watch)
          PSI >= 0.25       -> Warning (investigate)

IMPORTANT (say this in the viva):
    Drift is a WARNING SIGNAL that needs investigation. It does NOT prove
    that the model has failed. That would need labelled data (MAE increase).
"""
import json
import numpy as np
import pandas as pd
from src import config as C

EPS = 1e-4   # avoids log(0) / division by zero when a bin is empty


def build_baseline(X_train: pd.DataFrame) -> dict:
    """Compute and return the baseline statistics from the TRAINING features."""
    baseline = {}
    for col in C.DRIFT_FEATURES:
        values = X_train[col].to_numpy(dtype=float)
        # interior bin edges = 10%,20%,...,90% quantiles (duplicates removed)
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        counts = np.bincount(np.searchsorted(edges, values, side="right"),
                             minlength=len(edges) + 1)
        props = counts / counts.sum()
        baseline[col] = {
            "edges": edges.tolist(),
            "props": props.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return baseline


def save_baseline(baseline: dict) -> None:
    C.BASELINE_PATH.write_text(json.dumps(baseline, indent=2))


def load_baseline() -> dict:
    return json.loads(C.BASELINE_PATH.read_text())


def psi_for_feature(base: dict, values) -> float:
    """PSI between the stored baseline of one feature and a window of new values."""
    values = np.asarray(values, dtype=float)
    edges = np.array(base["edges"])
    counts = np.bincount(np.searchsorted(edges, values, side="right"),
                         minlength=len(edges) + 1)
    actual = np.clip(counts / counts.sum(), EPS, None)
    expected = np.clip(np.array(base["props"]), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status_from_psi(psi: float) -> str:
    if psi < 0.10:
        return "Normal"
    if psi < 0.25:
        return "Moderate"
    return "Warning"


def drift_report(baseline: dict, window: pd.DataFrame) -> dict:
    """
    Compare a window of incoming requests with the training baseline.
    Returns per-feature PSI + an overall status (worst feature wins).
    """
    per_feature = {}
    for col in C.DRIFT_FEATURES:
        p = psi_for_feature(baseline[col], window[col].to_numpy())
        per_feature[col] = {"psi": round(p, 4), "status": status_from_psi(p)}
    worst = max(v["psi"] for v in per_feature.values())
    return {
        "n_requests_in_window": int(len(window)),
        "overall_status": status_from_psi(worst),
        "max_psi": round(worst, 4),
        "features": per_feature,
    }
