"""Credit-risk metrics: AUC/Gini/KS, Brier, calibration (ECE), PSI, threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    # quantile bins hold ~equal counts, so a plain mean is a good ECE approximation
    return float(np.mean(np.abs(frac - mean)))


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.03, 0.7, 68)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_stat(y, p),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p), "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def roc_points(y, p, max_points: int = 160) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return {"fpr": fpr[idx].round(5).tolist(), "tpr": tpr[idx].round(5).tolist()}


def calibration_points(y, p, bins: int = 10) -> dict:
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return {"mean_pred": mean.round(5).tolist(), "frac_pos": frac.round(5).tolist()}


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` vs the `expected` (reference) distribution."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
