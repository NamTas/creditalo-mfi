"""SHAP-based explanations. Contributions are in the 'risk direction': positive = raises PD."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Explainer:
    def __init__(self, model, kind: str):
        self.kind = kind
        if kind == "linear":
            self.scaler = model.named_steps["scale"]
            clf = model.named_steps["clf"]
            self.coef = clf.coef_[0]
            self.base = float(clf.intercept_[0])
        else:
            import shap  # imported lazily: heavy dependency

            self._exp = shap.TreeExplainer(model)
            ev = np.atleast_1d(self._exp.expected_value)
            self.base = float(ev[-1])

    def contributions(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.kind == "linear":
            return ((X - self.scaler.mean_) / self.scaler.scale_) * self.coef
        sv = self._exp.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        return sv


def aggregate_by_feature(contribs: np.ndarray, origin: list, feature_order: list) -> pd.DataFrame:
    """Sum the one-hot columns of a categorical feature back into one column."""
    df = pd.DataFrame(contribs, columns=origin)
    agg = df.T.groupby(level=0).sum().T
    return agg.reindex(columns=feature_order)
