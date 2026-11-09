# CV story & application pack — Softzency

## Which 2 projects to lead with (for this job)

1. **CreditAlo** (this repo) — flagship. Covers PD modelling, LR/RF/LightGBM, SHAP, fairness, FastAPI, Docker.
2. **credit-risk-prediction-ml** — your earlier Python PD project (keep it if it contains a PD model + metrics; add AUC,
   top-3 features and a one-line result to its README). If it is thin, use **LoanIQ Credit Policy Simulator** instead —
   it shows you understand credit-policy decisions (cut-offs, approval trade-offs), the business side of scoring.
   *Optional 3rd line:* **TrustGate ETL** for the “collect, harmonise, validate datasets” bullet.

## CV bullets (fill the numbers from your own `train.bat` output)

* Built **CreditAlo**, an explainable microfinance credit-scoring system: compared Logistic Regression, Random
  Forest and LightGBM (test ROC-AUC **0.xx**, KS **0.xx**), Platt-calibrated PDs (Brier **0.0xx**), mapped PD to a
  transparent 0–100 score with five-band policy.
* Implemented **SHAP-based reason codes** returned with every decision; audited fairness (disparate-impact ratio,
  equal-opportunity gap) with sensitive attributes excluded from the model.
* Shipped a **FastAPI** scoring service (single + batch, API-key auth, validation, Docker) and a **Streamlit** risk
  dashboard (5 modules) powered by the same scoring class.
* Validated pipeline on **real public credit-default data** (Give Me Some Credit, UCI Taiwan) with PSI/KS/rank-ordering checks.

## 30-second pitch

“Microfinance lenders need to know *why* a borrower is risky, not just a number. I built a pipeline that predicts
calibrated default probability, turns it into a 0–100 score, and returns plain-English reason codes via a FastAPI
service. I compared three models, checked fairness across gender and region, and monitored stability with PSI.”

## Likely interview questions

* Why calibrate? → trees give scores, lending needs probabilities (Brier/ECE shown in the dashboard).
* Why exclude gender/region? → disparate treatment; audited instead through outcomes (four-fifths rule).
* Why might the simple Logistic Regression win? → small tabular data + additive risk; interpretability is a feature.
* How would you monitor in production? → PSI on score/features, KS & default rate by band on matured cohorts, recalibration.
* What would you change with real MFI data? → out-of-time split, reject-inference caveat (you only see repayment for approved loans), cost-based cut-offs.

## Application message (email / form)

> Subject: Application — Data Scientist (Explainable Credit Scoring) — Namira Tasnim
>
> Dear Softzency team, I am applying for the data-scientist consulting role. Relevant work: **CreditAlo**
> (github.com/NamTas/creditalo) — calibrated PD models (LR/RF/LightGBM), SHAP reason codes, fairness audit, FastAPI
> + Docker, Streamlit dashboard; and **<second project link>**. Availability: **<hours/week, start date>**.
> Expected consulting rate: **<your figure — decide it before sending>**. CV attached. Thank you, Namira.

*Tip on the rate:* decide it yourself (hourly or fixed per milestone) after checking what junior data-science consultants
charge in Bangladesh/remote; state the currency and whether it is hourly or per-milestone.

## GitHub checklist

`git init` → `git add .` → `git commit -m "CreditAlo v1"` → create repo `creditalo` on GitHub → push.
Add 3 screenshots (Borrower Assessment, SHAP waterfall, Fairness tab) to the README, and pin the repo.
