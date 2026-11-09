"""Group fairness audit on the final decision (approve vs. not approve)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_report(df: pd.DataFrame, attr: str, approved_col: str = "approved", y_col: str = "y",
                 pd_col: str = "pd", min_n: int = 30) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(attr, observed=True):
        good, bad = d[y_col] == 0, d[y_col] == 1
        good_appr = d.loc[good, approved_col].mean() if good.any() else np.nan
        bad_appr = d.loc[bad, approved_col].mean() if bad.any() else np.nan
        auc = np.nan
        if len(d) >= min_n and good.any() and bad.any():
            auc = roc_auc_score(d[y_col], d[pd_col])
        rows.append({
            "group": str(g), "n": int(len(d)), "approval_rate": float(d[approved_col].mean()),
            "default_rate": float(d[y_col].mean()), "avg_pd": float(d[pd_col].mean()),
            "good_approval_rate": float(good_appr),            # equal-opportunity view
            "wrongly_rejected": float(1 - good_appr),          # false-positive rate
            "bad_rejected": float(1 - bad_appr),               # bad borrowers caught (TPR)
            "auc": float(auc), "small_sample": bool(len(d) < min_n),
        })
    rep = pd.DataFrame(rows)
    ref = rep["approval_rate"].max()
    rep["di_ratio"] = rep["approval_rate"] / ref if ref > 0 else np.nan
    return rep


def summarize(rep: pd.DataFrame) -> dict:
    ok = rep[~rep["small_sample"]] if (~rep["small_sample"]).any() else rep
    return {
        "disparate_impact": float(ok["di_ratio"].min()),
        "parity_gap": float(ok["approval_rate"].max() - ok["approval_rate"].min()),
        "equal_opportunity_gap": float(ok["good_approval_rate"].max() - ok["good_approval_rate"].min()),
        "passes_four_fifths": bool(ok["di_ratio"].min() >= 0.8),
    }
