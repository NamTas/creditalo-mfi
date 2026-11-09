"""End-to-end training: data -> split -> 3 models -> calibration -> champion -> SHAP -> artifacts."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import __version__
from .config import (APPROVE_CUTOFF, ARTIFACT_DIR, BANDS, BASE_PD, BASE_SCORE, DECLINE_CUTOFF, PDO,
                     RANDOM_STATE)
from .data import load_dataset
from .explain import Explainer, aggregate_by_feature
from .features import TabularEncoder, engineer
from .metrics import (best_f1_threshold, calibration_points, evaluate, psi, roc_points)
from .modeling import (PlattCalibrator, band_for, build_models, decision_for, explainer_kind, pd_to_score,
                       slug)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def run_training(source: str = "synthetic", path: str | Path | None = None, n: int = 15000,
                 seed: int = RANDOM_STATE, out_dir: str | Path | None = None, max_rows: int = 60000,
                 shap_rows: int = 3000, verbose: bool = True) -> dict:
    out = Path(out_dir or ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    _log(verbose, f"[1/6] Loading data source '{source}' ...")
    raw_df, schema = load_dataset(source, path=path, n=n, seed=seed, max_rows=max_rows)
    df = engineer(raw_df, schema.key).reset_index(drop=True)
    y = df[schema.target].to_numpy(int)
    _log(verbose, f"      {len(df):,} borrowers | default rate {y.mean():.1%} | "
                  f"{len(schema.model_features)} model features")

    _log(verbose, "[2/6] Splitting 60/20/20 (train / validation / test, stratified) ...")
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    enc = TabularEncoder(schema.num_cols, schema.cat_cols).fit(df.iloc[tr])
    X = enc.transform(df).to_numpy(float)

    _log(verbose, "[3/6] Training + calibrating Logistic Regression, Random Forest, LightGBM ...")
    models, cals, metrics, curves, test_pd = {}, {}, {}, {}, {}
    for name, est in build_models(seed).items():
        est.fit(X[tr], y[tr])
        p_va_raw = est.predict_proba(X[va])[:, 1]
        p_te_raw = est.predict_proba(X[te])[:, 1]
        cal = PlattCalibrator().fit(p_va_raw, y[va])
        p_va, p_te = cal.predict(p_va_raw), cal.predict(p_te_raw)
        thr = best_f1_threshold(y[va], p_va)
        m = evaluate(y[te], p_te, thr)
        m["val_auc"] = float(roc_auc_score(y[va], p_va_raw))
        m["brier_uncalibrated"] = float(np.mean((p_te_raw - y[te]) ** 2))
        metrics[name] = m
        curves[name] = {"roc": roc_points(y[te], p_te), "calibration": calibration_points(y[te], p_te),
                        "calibration_raw": calibration_points(y[te], p_te_raw), "pd_col": f"pd_{slug(name)}"}
        models[name], cals[name], test_pd[name] = est, cal, p_te
        _log(verbose, f"      {name:<20} AUC {m['auc']:.3f} | KS {m['ks']:.3f} | Brier {m['brier']:.4f}")

    champion = max(metrics, key=lambda k: (round(metrics[k]["val_auc"], 4), -metrics[k]["brier"]))
    _log(verbose, f"[4/6] Champion model (best validation AUC): {champion}")

    # ---- test portfolio table -------------------------------------------------
    pd_champ = test_pd[champion]
    score = pd_to_score(pd_champ)
    test = df.iloc[te].copy()
    test["y"] = y[te]
    for name in models:
        test[curves[name]["pd_col"]] = test_pd[name]
    test["pd"] = pd_champ
    test["score"] = score.round(1)
    test["band"] = [band_for(s)[0] for s in score]
    test["decision"] = [decision_for(s) for s in score]
    test["approved"] = (score >= APPROVE_CUTOFF).astype(int)

    # ---- SHAP for the champion ------------------------------------------------
    _log(verbose, "[5/6] Computing SHAP explanations for the champion ...")
    explainer = Explainer(models[champion], explainer_kind(champion))
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(te), size=min(shap_rows, len(te)), replace=False))
    contribs = explainer.contributions(X[te][sample])
    agg = aggregate_by_feature(contribs, enc.origin_, schema.model_features)
    agg.insert(0, "applicant_id", test["applicant_id"].to_numpy()[sample])
    agg.to_csv(out / "shap_test.csv", index=False)
    test.to_csv(out / "test_predictions.csv", index=False)

    # ---- validation tables ----------------------------------------------------
    bands = (test.groupby("band").agg(borrowers=("y", "size"), default_rate=("y", "mean"), avg_pd=("pd", "mean"),
                                      avg_score=("score", "mean")).reset_index())
    bands["share"] = bands["borrowers"] / len(test)
    p_va_champ = cals[champion].predict(models[champion].predict_proba(X[va])[:, 1])
    psi_score = psi(pd_to_score(p_va_champ), score)
    num_inputs = [f.name for f in schema.inputs if f.kind == "num"]
    psi_features = {c: round(psi(df.iloc[tr][c].fillna(df.iloc[tr][c].median()),
                                 df.iloc[te][c].fillna(df.iloc[tr][c].median())), 4) for c in num_inputs}

    version = f"{schema.key}-{champion.lower().replace(' ', '')}-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset = {"source": source, "rows": int(len(df)), "default_rate": float(y.mean()),
               "train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
               "features": schema.model_features, "excluded_from_model": schema.audit_cols}

    _log(verbose, "[6/6] Saving artifacts ...")
    joblib.dump({"schema": schema.to_dict(), "encoder": enc, "models": models, "calibrators": cals,
                 "champion": champion, "version": version, "trained_at": trained_at, "dataset": dataset,
                 "metrics": metrics}, out / "model_bundle.joblib", compress=3)
    summary = {
        "version": version, "trained_at": trained_at, "package_version": __version__, "champion": champion,
        "dataset": dataset, "schema": schema.to_dict(), "models": {n: {**metrics[n], **curves[n]} for n in metrics},
        "bands": bands.to_dict(orient="records"),
        "psi": {"score_validation_vs_test": round(psi_score, 4), "features_train_vs_test": psi_features},
        "shap_base": explainer.base, "shap_rows": int(len(sample)),
        "policy": {"base_score": BASE_SCORE, "base_pd": BASE_PD, "pdo": PDO, "approve_cutoff": APPROVE_CUTOFF,
                   "decline_cutoff": DECLINE_CUTOFF, "bands": [list(b) for b in BANDS]},
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _log(verbose, f"\nDone. Champion = {champion} | test AUC {metrics[champion]['auc']:.3f} | "
                  f"PSI(score) {psi_score:.3f}\nArtifacts in: {out}")
    return summary
