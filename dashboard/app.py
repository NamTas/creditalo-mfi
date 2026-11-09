"""CreditAlo - interactive dashboard.   Run:  streamlit run dashboard/app.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works even without `pip install -e .`

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import tl_theme as T
from creditalo.config import APPROVE_CUTOFF, ARTIFACT_DIR, BAND_COLORS, BAND_ORDER, DECLINE_CUTOFF
from creditalo.fairness import group_report, summarize
from creditalo.schemas import Schema
from creditalo.scoring import Scorer

st.set_page_config(page_title="CreditAlo", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
T.register_plotly_template()
st.markdown(T.CSS, unsafe_allow_html=True)

DECISION_COLORS = {"Approve": T.GREEN, "Manual review": T.AMBER, "Decline": T.RED}
RISK_SCALE = [[0, T.TEAL], [0.5, T.AMBER], [1, T.PINK]]


# ============================================================================ helpers
def plot(fig, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def table(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def md(s):
    st.markdown(s, unsafe_allow_html=True)


# ============================================================================ artifacts
meta_file = ARTIFACT_DIR / "metrics.json"
if not meta_file.exists() or not (ARTIFACT_DIR / "model_bundle.joblib").exists():
    md(T.section("🛡️ CreditAlo", "No trained model found yet."))
    st.error("Train the models first, then refresh this page.")
    st.code("python scripts/train.py", language="bash")
    st.stop()

MTIME = meta_file.stat().st_mtime


@st.cache_data(show_spinner=False)
def load_artifacts(mtime):
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    shap_df = pd.read_csv(ARTIFACT_DIR / "shap_test.csv")
    return metrics, preds, shap_df


@st.cache_resource(show_spinner=False)
def load_scorer(mtime):
    return Scorer.load()


@st.cache_data(ttl=10, show_spinner=False)
def api_health(url):
    try:
        r = requests.get(url.rstrip("/") + "/health", timeout=0.7)
        return r.json() if r.ok else None
    except Exception:
        return None


M, PRED, SHAP = load_artifacts(MTIME)
SCORER = load_scorer(MTIME)
SCHEMA = Schema.from_dict(M["schema"])
CUR = SCHEMA.currency
CHAMP = M["champion"]
MODELS = list(M["models"].keys())
FEATS = SCHEMA.model_features


def money(v):
    return f"{CUR} {v:,.0f}"


# ============================================================================ sidebar
with st.sidebar:
    md(T.html('<div style="display:flex;gap:10px;align-items:center;margin-bottom:8px"><div class="tl-logo" '
              'style="width:42px;height:42px;font-size:22px;border-radius:12px">🛡️</div><div><b style="font-size:1.1rem">'
              'CreditAlo</b><br><span style="color:#8B97B5;font-size:.78rem">Explainable credit scoring</span></div></div>'))
    st.markdown("#### ⚙️ Scoring engine")
    mode = st.radio("Score borrowers via", ["In-process model", "FastAPI service"],
                    help="FastAPI mode sends every request to the running API (uvicorn creditalo.api:app).")
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    api_key = st.text_input("API key (optional)", type="password")

    st.markdown("#### 🎛️ Portfolio filters")
    mask = pd.Series(True, index=PRED.index)
    for col in SCHEMA.audit_cols:
        if col in PRED.columns and PRED[col].nunique() <= 8:
            opts = sorted(PRED[col].astype(str).unique())
            sel = st.multiselect(col.replace("_", " ").title(), opts, default=opts)
            mask &= PRED[col].astype(str).isin(sel)
    lo, hi = st.slider("Score range", 0, 100, (0, 100))
    mask &= PRED["score"].between(lo, hi)
    P = PRED[mask].copy()
    st.caption(f"{len(P):,} of {len(PRED):,} test borrowers selected")
    st.divider()
    st.caption(f"Model version `{M['version']}`  \nData: **{M['dataset']['source']}** · "
               f"{M['dataset']['rows']:,} borrowers")

# ============================================================================ hero
health = api_health(api_url)
api_chip = (f'<span class="tl-dot tl-on"></span>API online' if health and health.get("status") == "ok"
            else '<span class="tl-dot tl-off"></span>API offline')
champ_m = M["models"][CHAMP]
md(T.html(f"""
<div class="tl-hero"><div class="tl-row">
  <div class="tl-logo">🛡️</div>
  <div><div class="tl-title">CreditAlo</div>
  <div class="tl-tag">Explainable credit scoring for microfinance · probability of default → 0-100 score → reason codes</div></div>
  <div class="tl-chips">
    <span class="tl-chip">🏆 {CHAMP}</span>
    <span class="tl-chip">AUC {champ_m['auc']:.3f}</span>
    <span class="tl-chip">KS {champ_m['ks']:.2f}</span>
    <span class="tl-chip">{api_chip}</span>
  </div>
</div></div>"""))

tabs = st.tabs(["📊  Executive Overview", "🧮  Borrower Assessment", "🔍  Explainable AI",
                "🏁  Model Performance", "⚖️  Fairness & Validation"])


# ============================================================================ 1. OVERVIEW
def tab_overview():
    if P.empty:
        st.warning("No borrowers match the current filters.")
        return
    k = st.columns(5)
    k[0].markdown(T.kpi("Borrowers analysed", f"{len(P):,}", "held-out test portfolio", T.TEAL), unsafe_allow_html=True)
    k[1].markdown(T.kpi("Average PD", f"{P['pd'].mean():.1%}", "mean calibrated default probability", T.VIOLET), unsafe_allow_html=True)
    k[2].markdown(T.kpi("Observed default rate", f"{P['y'].mean():.1%}", "actual outcomes in this slice", T.PINK), unsafe_allow_html=True)
    k[3].markdown(T.kpi("Champion AUC", f"{champ_m['auc']:.3f}", f"{CHAMP} · Gini {champ_m['gini']:.2f}", T.AMBER), unsafe_allow_html=True)
    k[4].markdown(T.kpi("Auto-approval rate", f"{(P['decision'] == 'Approve').mean():.0%}",
                        f"score ≥ {APPROVE_CUTOFF:.0f}", T.GREEN), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.35])
    with c1:
        md(T.section("Risk distribution", "Borrowers by risk band (A = safest)"))
        cnt = P["band"].value_counts().reindex(BAND_ORDER).fillna(0)
        fig = go.Figure(go.Pie(labels=[f"Band {b}" for b in cnt.index], values=cnt.values, hole=0.66, sort=False,
                               marker=dict(colors=[BAND_COLORS[b] for b in cnt.index], line=dict(color="#070B17", width=3)),
                               textinfo="label+percent", textfont=dict(size=13)))
        fig.add_annotation(text=f"<b>{len(P):,}</b><br><span style='font-size:12px;color:#8B97B5'>borrowers</span>",
                           showarrow=False, font=dict(size=24))
        fig.update_layout(showlegend=False)
        plot(fig, 360)
    with c2:
        md(T.section("Score distribution", "Good vs. defaulted borrowers separate cleanly — that is model power"))
        fig = go.Figure()
        fig.add_histogram(x=P.loc[P["y"] == 0, "score"], name="Repaid", nbinsx=40, histnorm="percent", opacity=0.8,
                          marker_color=T.TEAL)
        fig.add_histogram(x=P.loc[P["y"] == 1, "score"], name="Defaulted", nbinsx=40, histnorm="percent", opacity=0.8,
                          marker_color=T.PINK)
        fig.add_vline(x=APPROVE_CUTOFF, line_dash="dash", line_color=T.GREEN, annotation_text="approve")
        fig.add_vline(x=DECLINE_CUTOFF, line_dash="dash", line_color=T.RED, annotation_text="decline")
        fig.update_layout(barmode="overlay", xaxis_title="Credit score (0-100)", yaxis_title="% of group")
        plot(fig, 360)

    c3, c4 = st.columns(2)
    with c3:
        md(T.section("Default rate by band", "Rank-ordering: risk must rise steadily from A to E"))
        g = P.groupby("band")["y"].agg(["mean", "size"]).reindex(BAND_ORDER).dropna()
        fig = go.Figure(go.Bar(x=[f"Band {b}" for b in g.index], y=g["mean"], marker_color=[BAND_COLORS[b] for b in g.index],
                               text=[f"{v:.1%}" for v in g["mean"]], textposition="outside",
                               customdata=g["size"], hovertemplate="%{x}<br>Default rate %{y:.1%}<br>n=%{customdata}<extra></extra>"))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Observed default rate")
        plot(fig, 340)
    with c4:
        md(T.section("Lift by score decile", "Decile 1 = riskiest 10% of borrowers"))
        d = P.copy()
        d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
        lift = d.groupby("decile")["y"].mean()
        fig = go.Figure(go.Bar(x=lift.index, y=lift.values, text=[f"{v:.0%}" for v in lift.values], textposition="outside",
                               marker=dict(color=lift.values, colorscale=[[0, T.TEAL], [0.5, T.AMBER], [1, T.PINK]])))
        fig.add_hline(y=P["y"].mean(), line_dash="dot", line_color="#fff", annotation_text="portfolio average")
        fig.update_layout(xaxis_title="Score decile (1 = riskiest)", yaxis_tickformat=".0%", yaxis_title="Default rate",
                          xaxis=dict(dtick=1))
        plot(fig, 340)

    cats = [c for c in SCHEMA.audit_cols + SCHEMA.cat_cols if c in P.columns and c != "age_group"
            and 2 <= P[c].nunique() <= 8]
    if len(cats) >= 2:
        md(T.section("Risk heat-map", f"Average PD by {cats[0].replace('_', ' ')} × {cats[1].replace('_', ' ')}"))
        piv = P.pivot_table(index=cats[0], columns=cats[1], values="pd", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.astype(str), y=piv.index.astype(str), colorscale=RISK_SCALE,
                                   texttemplate="%{z:.1%}", hovertemplate="%{y} · %{x}<br>Avg PD %{z:.1%}<extra></extra>",
                                   colorbar=dict(tickformat=".0%", title="PD")))
        plot(fig, 320)
    st.download_button("⬇️ Download scored portfolio (CSV)", P.to_csv(index=False).encode("utf-8"),
                       file_name="creditalo_scored_portfolio.csv", mime="text/csv")


# ============================================================================ 2. BORROWER ASSESSMENT
def is_int(f):
    return f.fmt == "int" or all(float(x).is_integer() for x in (f.min, f.max, f.step, f.default))


def cast(f, v):
    if f.kind == "cat":
        return v if v in f.options else f.default
    v = min(max(float(v), float(f.min)), float(f.max))
    return int(round(v)) if is_int(f) else round(v, 4)


def ncast(f, v):
    """Type-consistent number for Streamlit sliders (all ints or all floats), no clamping."""
    return int(round(float(v))) if is_int(f) else float(v)


def build_presets():
    presets = {"Form defaults": {f.name: f.default for f in SCHEMA.inputs}}
    ranked = PRED.sort_values("score").reset_index(drop=True)
    for label, q in [("Typical borrower", 0.5), ("Low-risk borrower", 0.92), ("High-risk borrower", 0.06)]:
        row = ranked.iloc[int(q * (len(ranked) - 1))]
        presets[label] = {f.name: row[f.name] for f in SCHEMA.inputs}
    return presets


PRESETS = build_presets()
for _f in SCHEMA.inputs:
    _k = f"in_{_f.name}"
    if _k not in st.session_state:
        st.session_state[_k] = cast(_f, _f.default)


def apply_preset():
    vals = PRESETS[st.session_state["preset"]]
    for f in SCHEMA.inputs:
        st.session_state[f"in_{f.name}"] = cast(f, vals[f.name])


def score_borrower(feats):
    t0 = time.perf_counter()
    if mode.startswith("FastAPI"):
        try:
            r = requests.post(api_url.rstrip("/") + "/score", json={"applicant_id": "dashboard", "features": feats},
                              headers={"X-API-Key": api_key} if api_key else {}, timeout=6)
            r.raise_for_status()
            res = r.json()
        except Exception as exc:
            return None, 0.0, f"API call failed: {exc}"
    else:
        res = SCORER.score_one(feats)
    return res, (time.perf_counter() - t0) * 1000, None


def gauge(score, colour):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=64, color="#fff")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#64748B", tickfont=dict(color="#8B97B5")),
                   bar=dict(color=colour, thickness=0.30), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, DECLINE_CUTOFF], color="rgba(239,68,68,.22)"),
                          dict(range=[DECLINE_CUTOFF, APPROVE_CUTOFF], color="rgba(255,176,32,.20)"),
                          dict(range=[APPROVE_CUTOFF, 65], color="rgba(132,204,22,.18)"),
                          dict(range=[65, 100], color="rgba(34,197,94,.25)")],
                   threshold=dict(line=dict(color="#fff", width=4), thickness=0.85, value=score))))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=0))
    return fig


def waterfall(impacts, base, n=9, title="Contribution to default risk (SHAP)"):
    s = pd.Series(impacts).sort_values(key=lambda x: x.abs(), ascending=False)
    top, rest = s.head(n), s.iloc[n:].sum()
    labels = [SCHEMA.label(i) for i in top.index] + (["All other factors"] if abs(rest) > 1e-9 else [])
    vals = top.tolist() + ([rest] if abs(rest) > 1e-9 else [])
    final = base + sum(vals)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        y=["Baseline (average borrower)"] + labels + ["This borrower"], x=[base] + vals + [0],
        text=[f"{base:+.2f}"] + [f"{v:+.2f}" for v in vals] + [f"{final:+.2f}"], textposition="outside",
        connector=dict(line=dict(color="rgba(255,255,255,.25)")),
        increasing=dict(marker=dict(color=T.PINK)), decreasing=dict(marker=dict(color=T.TEAL)),
        totals=dict(marker=dict(color=T.VIOLET))))
    fig.update_layout(title=title, xaxis_title="← lowers risk | raises risk →", yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=40, t=50, b=10))
    return fig


def reason_html(items, kind):
    if not items:
        return ""
    cls = "r" if kind == "risk" else "p"
    out = ""
    for it in items:
        out += (f'<div class="tl-rc {cls}"><b>{it["code"]} · {it["label"]}</b><br>'
                f'<span>{it["text"]} &nbsp;·&nbsp; impact {it["impact"]:+.2f}</span></div>')
    return T.html(out)


def tab_assessment():
    md(T.section("Borrower credit assessment", "Move any slider — the score, decision and reason codes update live."))
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.selectbox("Load a sample borrower", list(PRESETS), key="preset", on_change=apply_preset)
        cols = st.columns(2)
        for i, f in enumerate(SCHEMA.inputs):
            with cols[i % 2]:
                if f.kind == "cat":
                    st.selectbox(f.label, f.options, key=f"in_{f.name}", help=f.help or None)
                else:
                    st.slider(f.label, min_value=ncast(f, f.min), max_value=ncast(f, f.max), step=ncast(f, f.step),
                              key=f"in_{f.name}", help=f.help or None)
    feats = {f.name: st.session_state[f"in_{f.name}"] for f in SCHEMA.inputs}
    res, ms, err = score_borrower(feats)
    with right:
        if err:
            st.error(err)
            st.info("Start the API (`run_api.bat`) or switch the sidebar to “In-process model”.")
            return
        src = f"⚡ FastAPI · {ms:.0f} ms" if mode.startswith("FastAPI") else f"🧠 In-process · {ms:.0f} ms"
        md(T.html(f'<div style="text-align:right;color:#8B97B5;font-size:.8rem">{src} · {res["model"]["name"]}</div>'))
        plot(gauge(res["score"], res["band_colour"]), 270)
        c = st.columns(3)
        c[0].markdown(T.kpi("Default probability", f"{res['pd']:.1%}", "calibrated PD", T.VIOLET), unsafe_allow_html=True)
        c[1].markdown(T.kpi("Risk band", f"{res['band']}", res["band_label"], res["band_colour"]), unsafe_allow_html=True)
        c[2].markdown(T.kpi("Decision", res["decision"], f"policy: ≥{APPROVE_CUTOFF:.0f} approve",
                            DECISION_COLORS[res["decision"]]), unsafe_allow_html=True)
        for w in res.get("warnings", []):
            st.warning(w)

    md(T.section("Why this score?", "Reason codes ranked by SHAP impact"))
    a, b = st.columns(2)
    with a:
        st.markdown("**🔴 Factors raising risk**")
        md(reason_html(res["reason_codes"]["risk"], "risk") or "<span style='color:#8B97B5'>None — nothing raises risk.</span>")
    with b:
        st.markdown("**🟢 Factors protecting the borrower**")
        md(reason_html(res["reason_codes"]["protective"], "protective") or "<span style='color:#8B97B5'>None.</span>")
    plot(waterfall(res["feature_impacts"], res["base_value"]), 440)


# ============================================================================ 3. EXPLAINABLE AI
def value_rank(v):
    if pd.api.types.is_numeric_dtype(v):
        return v.rank(pct=True).to_numpy()
    codes = pd.factorize(v)[0]
    return codes / max(codes.max(), 1)


def tab_explain():
    S = SHAP[SHAP["applicant_id"].isin(P["applicant_id"])]
    if S.empty:
        st.warning("No explained borrowers in the current filter — widen the sidebar filters.")
        return
    imp = S[FEATS].abs().mean().sort_values(ascending=False)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        md(T.section("Global feature importance", f"Mean |SHAP| over {len(S):,} borrowers · {CHAMP}"))
        top = imp.head(12)[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[SCHEMA.label(i) for i in top.index], orientation="h",
                               marker=dict(color=top.values, colorscale=[[0, T.VIOLET], [1, T.TEAL]]),
                               text=[f"{v:.2f}" for v in top.values], textposition="outside"))
        fig.update_layout(xaxis_title="Average impact on model output")
        plot(fig, 470)
    with c2:
        md(T.section("SHAP summary", "Each dot is a borrower — colour shows the feature value (low → high)"))
        names = imp.head(10).index.tolist()
        M2 = S[["applicant_id"] + names].merge(P[["applicant_id"] + names], on="applicant_id", suffixes=("_shap", "_val"))
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for i, name in enumerate(names[::-1]):
            fig.add_trace(go.Scatter(
                x=M2[f"{name}_shap"], y=i + rng.normal(0, 0.09, len(M2)), mode="markers", showlegend=False,
                marker=dict(size=5, opacity=0.75, color=value_rank(M2[f"{name}_val"]),
                            colorscale=[[0, T.SKY], [0.5, T.VIOLET], [1, T.PINK]], cmin=0, cmax=1,
                            showscale=(i == 0), colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"])),
                hovertemplate=f"{SCHEMA.label(name)}<br>SHAP %{{x:.2f}}<extra></extra>"))
        fig.update_layout(yaxis=dict(tickvals=list(range(len(names))), ticktext=[SCHEMA.label(n) for n in names[::-1]]),
                          xaxis_title="← lowers risk | raises risk →")
        fig.add_vline(x=0, line_color="rgba(255,255,255,.3)")
        plot(fig, 470)

    md(T.section("Dependence plot", "How one feature moves the model's risk estimate"))
    feat = st.selectbox("Feature", imp.index.tolist(), format_func=SCHEMA.label, key="dep_feat")
    D = S[["applicant_id", feat]].merge(P[["applicant_id", feat, "score"]], on="applicant_id", suffixes=("_shap", "_val"))
    fig = go.Figure(go.Scatter(x=D[f"{feat}_val"], y=D[f"{feat}_shap"], mode="markers",
                               marker=dict(size=6, opacity=0.8, color=D["score"], colorscale=RISK_SCALE[::-1],
                                           showscale=True, colorbar=dict(title="Score"))))
    fig.add_hline(y=0, line_color="rgba(255,255,255,.3)")
    fig.update_layout(xaxis_title=SCHEMA.label(feat), yaxis_title="SHAP value (impact on risk)")
    plot(fig, 380)

    md(T.section("Single-borrower explanation", "Pick a borrower from the test set and see their reason codes"))
    pick = st.radio("Show", ["Riskiest borrower", "Safest borrower", "Random borrower"], horizontal=True)
    cand = P[P["applicant_id"].isin(S["applicant_id"])]
    if pick.startswith("Riskiest"):
        row = cand.loc[cand["pd"].idxmax()]
    elif pick.startswith("Safest"):
        row = cand.loc[cand["pd"].idxmin()]
    else:
        if st.button("🎲 Another random borrower") or "rand_seed" not in st.session_state:
            st.session_state["rand_seed"] = int(time.time()) % 100000
        row = cand.sample(1, random_state=st.session_state["rand_seed"]).iloc[0]
    res = SCORER.score_one({f.name: row[f.name] for f in SCHEMA.inputs})
    k = st.columns(4)
    k[0].markdown(T.kpi("Applicant", str(row["applicant_id"]), "test set", T.SKY), unsafe_allow_html=True)
    k[1].markdown(T.kpi("Score", f"{res['score']:.0f}", f"band {res['band']} · {res['band_label']}", res["band_colour"]), unsafe_allow_html=True)
    k[2].markdown(T.kpi("PD", f"{res['pd']:.1%}", res["decision"], DECISION_COLORS[res["decision"]]), unsafe_allow_html=True)
    k[3].markdown(T.kpi("Actual outcome", "Defaulted" if row["y"] == 1 else "Repaid", "ground truth",
                        T.PINK if row["y"] == 1 else T.TEAL), unsafe_allow_html=True)
    a, b = st.columns([1, 1.3])
    with a:
        md(reason_html(res["reason_codes"]["risk"], "risk"))
        md(reason_html(res["reason_codes"]["protective"], "protective"))
    with b:
        plot(waterfall(res["feature_impacts"], res["base_value"], title="SHAP waterfall"), 430)
    with st.expander("📖 Reason-code dictionary (what each code means)"):
        table(pd.DataFrame({"Code": list(SCORER.codes.values()), "Feature": [SCHEMA.label(n) for n in SCORER.codes]}))


# ============================================================================ 4. MODEL PERFORMANCE
def tab_models():
    md(T.section("Model leaderboard", "Three models, same data, same split — evaluated on a held-out test set"))
    cols = st.columns(len(MODELS))
    for col, name in zip(cols, MODELS):
        m = M["models"][name]
        crown = " 🏆" if name == CHAMP else ""
        col.markdown(T.kpi(name + crown, f"{m['auc']:.3f}",
                           f"AUC · KS {m['ks']:.2f} · Brier {m['brier']:.4f}", T.MODEL_COLORS.get(name, T.TEAL)),
                     unsafe_allow_html=True)

    rows = []
    for n in MODELS:
        m = M["models"][n]
        rows.append({"Model": n, "ROC-AUC": m["auc"], "Gini": m["gini"], "KS": m["ks"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "Brier": m["brier"], "ECE": m["ece"],
                     "Brier (uncalibrated)": m["brier_uncalibrated"], "F1 threshold": m["threshold"]})
    lb = pd.DataFrame(rows)
    table(lb.style.format({c: "{:.3f}" for c in lb.columns if c != "Model"})
          .apply(lambda r: ["background-color: rgba(0,229,199,.18)" if r["Model"] == CHAMP else "" for _ in r], axis=1))

    c1, c2 = st.columns(2)
    with c1:
        md(T.section("Discrimination", "ROC-AUC · Gini · KS · F1 side by side"))
        fig = go.Figure()
        for n in MODELS:
            m = M["models"][n]
            fig.add_bar(name=n, x=["ROC-AUC", "Gini", "KS", "Precision", "Recall", "F1"],
                        y=[m["auc"], m["gini"], m["ks"], m["precision"], m["recall"], m["f1"]],
                        marker_color=T.MODEL_COLORS.get(n))
        fig.update_layout(barmode="group", yaxis_range=[0, 1])
        plot(fig, 380)
    with c2:
        md(T.section("ROC curves", "Closer to the top-left corner is better"))
        fig = go.Figure()
        for n in MODELS:
            r = M["models"][n]["roc"]
            fig.add_scatter(x=r["fpr"], y=r["tpr"], name=f"{n} (AUC {M['models'][n]['auc']:.3f})",
                            line=dict(width=3, color=T.MODEL_COLORS.get(n)))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dot", color="#64748B"))
        fig.update_layout(xaxis_title="False-positive rate", yaxis_title="True-positive rate")
        plot(fig, 380)

    c3, c4 = st.columns(2)
    with c3:
        md(T.section("Calibration", "Do predicted PDs match reality? (diagonal = perfect)"))
        which = st.radio("Probabilities", ["Calibrated (used in production)", "Raw model output"], horizontal=True)
        key = "calibration" if which.startswith("Calibrated") else "calibration_raw"
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dot", color="#64748B"))
        top = 0.05
        for n in MODELS:
            c = M["models"][n][key]
            top = max(top, max(c["mean_pred"] + c["frac_pos"]))
            fig.add_scatter(x=c["mean_pred"], y=c["frac_pos"], mode="lines+markers", name=n,
                            line=dict(width=3, color=T.MODEL_COLORS.get(n)))
        fig.update_layout(xaxis_title="Mean predicted PD", yaxis_title="Observed default rate",
                          xaxis_range=[0, top * 1.1], yaxis_range=[0, top * 1.1])
        plot(fig, 380)
    with c4:
        md(T.section("Brier score", "Lower is better — calibration should help or at least not hurt"))
        fig = go.Figure()
        fig.add_bar(name="Uncalibrated", x=MODELS, y=[M["models"][n]["brier_uncalibrated"] for n in MODELS], marker_color="#64748B")
        fig.add_bar(name="Calibrated", x=MODELS, y=[M["models"][n]["brier"] for n in MODELS], marker_color=T.TEAL)
        lo_b = min(M["models"][n]["brier"] for n in MODELS) * 0.9
        fig.update_layout(barmode="group", yaxis_range=[lo_b, max(M["models"][n]["brier_uncalibrated"] for n in MODELS) * 1.03],
                          yaxis_title="Brier score (zoomed axis)")
        plot(fig, 380)

    md(T.section("Confusion matrix explorer", "Move the PD threshold to trade off bad-borrower capture vs. wrongly rejected good borrowers"))
    s1, s2 = st.columns([1, 2])
    with s1:
        sel = st.selectbox("Model", MODELS, index=MODELS.index(CHAMP))
        thr = st.slider("Flag as 'high risk' when PD ≥", 0.02, 0.60, float(round(M["models"][sel]["threshold"], 2)), 0.01)
    pdv, yv = PRED[M["models"][sel]["pd_col"]].to_numpy(), PRED["y"].to_numpy()
    flag = pdv >= thr
    tp, fp, fn, tn = int((flag & (yv == 1)).sum()), int((flag & (yv == 0)).sum()), int((~flag & (yv == 1)).sum()), int((~flag & (yv == 0)).sum())
    with s2:
        z = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted: OK", "Predicted: high risk"], y=["Actual: repaid", "Actual: defaulted"],
                                   text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}", textfont=dict(size=22),
                                   colorscale=[[0, "#0F1730"], [1, T.VIOLET]], showscale=False))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        plot(fig, 260)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    q = st.columns(4)
    q[0].markdown(T.kpi("Precision", f"{prec:.1%}", "flagged who truly default", T.TEAL), unsafe_allow_html=True)
    q[1].markdown(T.kpi("Recall", f"{rec:.1%}", "defaulters caught", T.PINK), unsafe_allow_html=True)
    q[2].markdown(T.kpi("F1", f"{f1:.2f}", "balance of the two", T.VIOLET), unsafe_allow_html=True)
    q[3].markdown(T.kpi("Good borrowers rejected", f"{fp / max(fp + tn, 1):.1%}", "false-positive rate", T.AMBER), unsafe_allow_html=True)


# ============================================================================ 5. FAIRNESS & VALIDATION
def tab_fairness():
    attrs = [c for c in SCHEMA.audit_cols if c in PRED.columns]
    md(T.section("Fairness audit", "Sensitive attributes are never model inputs — they are only used here to audit outcomes"))
    if not attrs:
        st.info("This dataset has no audit attributes.")
        return
    attr = st.selectbox("Audit attribute", attrs, format_func=lambda a: a.replace("_", " ").title())
    rep = group_report(PRED, attr)
    sm = summarize(rep)
    k = st.columns(4)
    k[0].markdown(T.kpi("Disparate-impact ratio", f"{sm['disparate_impact']:.2f}", "lowest group ÷ highest · need ≥ 0.80",
                        T.GREEN if sm["passes_four_fifths"] else T.RED), unsafe_allow_html=True)
    k[1].markdown(T.kpi("Approval-rate gap", f"{sm['parity_gap']:.1%}", "demographic-parity difference", T.VIOLET), unsafe_allow_html=True)
    k[2].markdown(T.kpi("Equal-opportunity gap", f"{sm['equal_opportunity_gap']:.1%}", "gap in approval of GOOD borrowers", T.AMBER), unsafe_allow_html=True)
    k[3].markdown(T.kpi("Four-fifths rule", "PASS" if sm["passes_four_fifths"] else "REVIEW", f"audited on {attr.replace('_', ' ')}",
                        T.GREEN if sm["passes_four_fifths"] else T.RED), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        md(T.section("Approval rate by group", "Dashed line = 80% of the best-treated group"))
        fig = go.Figure(go.Bar(x=rep["group"], y=rep["approval_rate"], text=[f"{v:.0%}" for v in rep["approval_rate"]],
                               textposition="outside", marker=dict(color=rep["approval_rate"], colorscale=[[0, T.PINK], [1, T.TEAL]])))
        fig.add_hline(y=0.8 * rep["approval_rate"].max(), line_dash="dash", line_color=T.AMBER)
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.08], xaxis_title=attr.replace("_", " ").title())
        plot(fig, 340)
    with c2:
        md(T.section("Error rates by group", "Wrongly rejected good borrowers vs. bad borrowers caught"))
        fig = go.Figure()
        fig.add_bar(name="Good borrowers wrongly rejected (FPR)", x=rep["group"], y=rep["wrongly_rejected"], marker_color=T.AMBER)
        fig.add_bar(name="Bad borrowers rejected (TPR)", x=rep["group"], y=rep["bad_rejected"], marker_color=T.VIOLET)
        fig.update_layout(barmode="group", yaxis_tickformat=".0%")
        plot(fig, 340)

    show = rep.rename(columns={"group": "Group", "n": "Borrowers", "approval_rate": "Approval", "default_rate": "Default rate",
                               "avg_pd": "Avg PD", "wrongly_rejected": "FPR", "bad_rejected": "TPR", "auc": "AUC",
                               "di_ratio": "DI ratio"})[["Group", "Borrowers", "Approval", "Default rate", "Avg PD", "FPR", "TPR", "AUC", "DI ratio"]]
    table(show.style.format({c: "{:.1%}" for c in ["Approval", "Default rate", "Avg PD", "FPR", "TPR"]} | {"AUC": "{:.3f}", "DI ratio": "{:.2f}"}))

    md(T.section("Model validation scorecard", "Automated checks a model-risk team would ask for"))
    audit_in_model = [a for a in SCHEMA.audit_cols if a in FEATS]
    bands = pd.DataFrame(M["bands"]).set_index("band").reindex(BAND_ORDER).dropna(subset=["default_rate"])
    monotone = bool(bands["default_rate"].is_monotonic_increasing)
    psi_s = M["psi"]["score_validation_vs_test"]
    g1, g2 = st.columns(2)
    with g1:
        md(T.check(not audit_in_model, "Sensitive attributes excluded from the model",
                   "Not used as inputs: " + ", ".join(SCHEMA.audit_cols) if not audit_in_model else f"Found in model: {audit_in_model}"))
        md(T.check(monotone, "Rank-ordering holds across bands A→E", "Default rate: " + " → ".join(f"{v:.1%}" for v in bands["default_rate"])))
        md(T.check(champ_m["ece"] < 0.02, "Probabilities are well calibrated", f"Expected calibration error {champ_m['ece']:.3f} (target < 0.02)"))
    with g2:
        md(T.check(psi_s < 0.10, "Score distribution is stable (PSI)", f"PSI validation vs test = {psi_s:.3f} (< 0.10 stable, > 0.25 drift)"))
        md(T.check(champ_m["ks"] >= 0.30, "Strong separation of good and bad", f"KS {champ_m['ks']:.2f} (≥ 0.30 is considered strong)"))
        md(T.check(sm["passes_four_fifths"], f"Four-fifths rule on {attr.replace('_', ' ')}", f"Disparate-impact ratio {sm['disparate_impact']:.2f}"))

    md(T.section("Feature stability (PSI, train vs test)", "Green < 0.10 · amber 0.10–0.25 · red > 0.25"))
    ps = pd.Series(M["psi"]["features_train_vs_test"]).sort_values()
    fig = go.Figure(go.Bar(x=ps.values, y=[SCHEMA.label(i) for i in ps.index], orientation="h",
                           marker_color=[T.GREEN if v < 0.1 else T.AMBER if v < 0.25 else T.RED for v in ps.values]))
    fig.add_vline(x=0.1, line_dash="dash", line_color=T.AMBER)
    fig.add_vline(x=0.25, line_dash="dash", line_color=T.RED)
    fig.update_layout(xaxis_title="PSI")
    plot(fig, max(300, 26 * len(ps) + 80))


with tabs[0]:
    tab_overview()
with tabs[1]:
    tab_assessment()
with tabs[2]:
    tab_explain()
with tabs[3]:
    tab_models()
with tabs[4]:
    tab_fairness()

md(T.html('<div style="text-align:center;color:#5E6A8A;font-size:.78rem;margin-top:30px">CreditAlo · '
          'scikit-learn · LightGBM · SHAP · FastAPI · Streamlit — decision support, not a substitute for credit-officer judgement.</div>'))
