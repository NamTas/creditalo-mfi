"""Scorer: the one place that turns raw borrower fields into PD, 0-100 score, decision and
reason codes. Used by BOTH the FastAPI service and the Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, APPROVE_CUTOFF, DECLINE_CUTOFF
from .explain import Explainer, aggregate_by_feature
from .features import engineer
from .modeling import band_for, decision_for, explainer_kind, pd_to_score
from .schemas import Feature, Schema


def format_value(f: Feature, v, currency: str = "") -> str:
    try:
        if f.kind == "cat":
            return str(v)
        v = float(v)
        if f.fmt == "money":
            return f"{currency} {v:,.0f}".strip()
        if f.fmt == "pct":
            return f"{v * 100:.0f}%"
        if f.fmt == "int":
            return f"{v:.0f}"
        if f.fmt == "float1":
            return f"{v:.1f}"
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return str(v)


class Scorer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.schema = Schema.from_dict(bundle["schema"])
        self.encoder = bundle["encoder"]
        self.model_name = bundle["champion"]
        self.model = bundle["models"][self.model_name]
        self.calibrator = bundle["calibrators"][self.model_name]
        self.version = bundle["version"]
        self.explainer = Explainer(self.model, explainer_kind(self.model_name))
        self.feature_order = self.schema.model_features
        self.codes = {n: f"RC{i + 1:02d}" for i, n in enumerate(self.feature_order)}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Scorer":
        path = Path(path or ARTIFACT_DIR / "model_bundle.joblib")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python scripts/train.py` first.")
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ validation
    def validate(self, records: list[dict]) -> tuple[pd.DataFrame, list[list[str]]]:
        need = self.schema.input_names
        rows, warns = [], []
        for i, rec in enumerate(records):
            missing = [c for c in need if c not in rec or rec[c] is None]
            if missing:
                raise ValueError(f"Record {i}: missing required features {missing}")
            w = []
            for f in self.schema.inputs:
                v = rec[f.name]
                if f.kind == "cat":
                    if str(v) not in f.options:
                        w.append(f"{f.name}='{v}' is not a known category {f.options}")
                else:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        raise ValueError(f"Record {i}: '{f.name}' must be numeric, got {v!r}")
                    span = (f.max - f.min) or 1.0
                    if fv < f.min - 0.25 * span or fv > f.max + 0.25 * span:
                        w.append(f"{f.name}={fv:g} is far outside the training range [{f.min:g}, {f.max:g}]")
            rows.append({c: rec[c] for c in need})
            warns.append(w)
        return pd.DataFrame(rows), warns

    # ------------------------------------------------------------------ scoring
    def score_records(self, records: list[dict], top_k: int = 4) -> list[dict]:
        raw, warns = self.validate(records)
        eng = engineer(raw, self.schema.key)
        X = self.encoder.transform(eng).to_numpy(float)
        p_raw = self.model.predict_proba(X)[:, 1]
        pds = self.calibrator.predict(p_raw)
        scores = pd_to_score(pds)
        contribs = self.explainer.contributions(X)
        agg = aggregate_by_feature(contribs, self.encoder.origin_, self.feature_order)

        out = []
        for i in range(len(raw)):
            score = float(scores[i])
            letter, label, colour = band_for(score)
            impacts = agg.iloc[i]
            out.append({
                "pd": round(float(pds[i]), 5),
                "score": round(score, 1),
                "band": letter, "band_label": label, "band_colour": colour,
                "decision": decision_for(score),
                "reason_codes": self._reason_codes(impacts, eng.iloc[i], top_k),
                "feature_impacts": {k: round(float(v), 5) for k, v in impacts.items()},
                "base_value": round(self.explainer.base, 5),
                "warnings": warns[i],
                "model": {"name": self.model_name, "version": self.version},
            })
        return out

    def score_one(self, features: dict, top_k: int = 4) -> dict:
        return self.score_records([features], top_k=top_k)[0]

    def _reason_codes(self, impacts: pd.Series, row: pd.Series, k: int) -> dict:
        def item(name: str, val: float, risk: bool) -> dict:
            f = self.schema.feature(name)
            shown = format_value(f, row[name], self.schema.currency)
            phrase = "pushes default risk up" if risk else "works in the borrower's favour"
            return {"code": self.codes[name], "feature": name, "label": f.label, "value": shown,
                    "impact": round(float(val), 4), "text": f"{f.label} ({shown}) {phrase}"}

        ordered = impacts.sort_values(ascending=False)
        risk = [item(n, v, True) for n, v in ordered.items() if v > 1e-6][:k]
        protect = [item(n, v, False) for n, v in ordered[::-1].items() if v < -1e-6][:k]
        return {"risk": risk, "protective": protect}

    def model_info(self) -> dict:
        b = self.bundle
        return {
            "model": self.model_name, "version": self.version, "trained_at": b["trained_at"],
            "dataset": b["dataset"], "score_range": [0, 100],
            "policy": {"approve_at_or_above": APPROVE_CUTOFF, "decline_below": DECLINE_CUTOFF},
            "metrics": b["metrics"][self.model_name],
            "reason_code_dictionary": {c: self.schema.label(n) for n, c in self.codes.items()},
        }
