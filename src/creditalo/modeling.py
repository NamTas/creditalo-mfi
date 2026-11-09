"""Candidate models, Platt calibration and the PD -> 0-100 score transformation."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (APPROVE_CUTOFF, BANDS, BASE_PD, BASE_SCORE, DECLINE_CUTOFF, PDO, RANDOM_STATE)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "LightGBM"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def build_models(seed: int = RANDOM_STATE) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=3000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=25, max_features="sqrt", n_jobs=-1, random_state=seed),
        "LightGBM": LGBMClassifier(
            n_estimators=350, learning_rate=0.04, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=seed, n_jobs=-1, verbose=-1),
    }


def explainer_kind(name: str) -> str:
    return "linear" if name == "Logistic Regression" else "tree"


class PlattCalibrator:
    """Sigmoid calibration of a model's raw probabilities (fit on a held-out validation set)."""

    def fit(self, p_raw, y) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000)
        self.lr_.fit(self._logit(p_raw), y)
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self.lr_.predict_proba(self._logit(p_raw))[:, 1]

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
        return np.log(p / (1 - p)).reshape(-1, 1)


# ------------------------------------------------------------------ score policy
def pd_to_score(pd_) -> np.ndarray:
    """Higher score = safer. Odds-doubling every PDO points, anchored at (BASE_PD, BASE_SCORE)."""
    p = np.clip(np.asarray(pd_, float), 1e-4, 1 - 1e-4)
    odds = (1 - p) / p
    base_odds = (1 - BASE_PD) / BASE_PD
    score = BASE_SCORE + (PDO / np.log(2)) * np.log(odds / base_odds)
    return np.clip(score, 0, 100)


def band_for(score: float) -> tuple[str, str, str]:
    for lo, letter, label, colour in BANDS:
        if score >= lo:
            return letter, label, colour
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def decision_for(score: float) -> str:
    if score >= APPROVE_CUTOFF:
        return "Approve"
    if score >= DECLINE_CUTOFF:
        return "Manual review"
    return "Decline"
