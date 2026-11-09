# 🛡️ CreditAlo — Explainable Microfinance Credit Scoring (Alo = light in Bangla)

> Predict a borrower's **probability of default (PD)**, convert it to a transparent **0–100 credit score**, and
> explain every decision with **SHAP reason codes** — served through a **FastAPI** scoring service and an
> interactive **Streamlit** risk dashboard.

Built for the *Softzency — Explainable Credit Scoring for Microfinance* data-scientist role.

| Requirement in the job post | Where it lives in this repo |
|---|---|
| Python, scikit-learn, statistical modelling | `src/creditalo/` (pipeline, calibration, PSI, KS, Gini) |
| Credit-risk modelling & PD estimation | `modeling.py`, `train.py` — calibrated PD + score scaling |
| Logistic Regression, Random Forest, LightGBM, SHAP | `modeling.py`, `explain.py` — 3 models compared, SHAP on champion |
| Explainability, fairness, robust validation | `fairness.py`, `metrics.py`, dashboard tabs 3–5 |
| 0–100 score with interpretable reason codes | `scoring.py` (`RC01…` codes with plain-English text) |
| Production-ready scoring API (FastAPI) | `api.py` — `/score`, `/score/batch`, `/health`, API-key auth |
| Git, Docker | `Dockerfile`, `docker-compose.yml`, tests |
| Collect, harmonise, analyse datasets | `data.py` — synthetic MFI + 3 real-data loaders |

## Architecture

```
 data.py ──► features.py ──► train.py ──► artifacts/ (model_bundle, metrics.json, SHAP, predictions)
 (synthetic | real CSV)     (ratios +      │ LogReg / RF / LightGBM
                             encoder)      │ Platt calibration · champion pick · SHAP · PSI
                                           ▼
                        scoring.py  (PD → score → band → decision → reason codes)
                         ▲                                   ▲
        FastAPI  api.py ─┘                                   └─ Streamlit dashboard/app.py
```

## Quick start (Windows)

1. Install **Python 3.11** from python.org — tick **“Add Python to PATH”**.
2. Unzip this folder, e.g. to `C:\Projects\creditalo`, open it in **VS Code** (or PowerShell).
3. Double-click **`setup.bat`** (creates `.venv`, installs everything — ~3–5 min).
4. Double-click **`train.bat`** (≈1 min). It prints AUC/KS/Brier for the three models and picks a champion.
5. Double-click **`run_all.bat`** → API on <http://127.0.0.1:8000/docs>, dashboard on <http://localhost:8501>.

Prefer the terminal?

```powershell
python -m venv .venv
.venv\Scripts\activate            # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
pip install -e .
python scripts/train.py
uvicorn creditalo.api:app --port 8000        # terminal 1
streamlit run dashboard/app.py               # terminal 2
```

## Try the API

```powershell
curl.exe -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d "@docs/sample_request.json"
```

Response (shortened):

```json
{ "applicant_id": "DEMO-001", "pd": 0.035, "score": 70.3, "band": "B", "decision": "Approve",
  "reason_codes": { "risk": [{"code": "RC07", "text": "Requested loan amount (BDT 50,000) pushes default risk up"}],
                    "protective": [{"code": "RC10", "text": "Late payments (history) (0) works in the borrower's favour"}] } }
```

Set `CREDITALO_API_KEY=secret` before starting the API to require an `X-API-Key` header.

## Using real data

```powershell
python scripts/train.py --data gmsc   --path data/raw/cs-training.csv      # Kaggle: Give Me Some Credit
python scripts/train.py --data taiwan --path data/raw/UCI_Credit_Card.csv  # UCI: credit-card default
python scripts/train.py --data mfi_csv --path data/raw/mfi_loans.csv       # YOUR / a partner MFI's loan book
```
Full guide + honest CV wording: [`docs/REAL_DATA_GUIDE.md`](docs/REAL_DATA_GUIDE.md).
After retraining, restart the API and refresh the dashboard.

## Score policy

`score = 65 + (10 / ln 2) · ln( odds / odds₀ )`, clipped to 0–100, where odds = (1−PD)/PD and odds₀ is the odds at
PD = 5 %. Every time the good:bad odds double, the score rises 10 points.

| Band | Score | Decision |
|---|---|---|
| A / B | 65–100 | Approve |
| C | 50–65 | Approve |
| D | 35–50 | Manual review |
| E | < 35 | Decline |

Cut-offs live in `src/creditalo/config.py` — change them to match an institution's risk appetite.

## Design decisions worth explaining in an interview

* **Calibrated PDs.** Raw tree/forest scores are not probabilities; each model is Platt-calibrated on a separate
  validation set, and the Brier score before/after is reported.
* **Champion selection on validation AUC, reporting on a untouched test set** (60/20/20 stratified).
* **Sensitive attributes are not model inputs** (gender, region). They are only used in the fairness audit
  (disparate-impact ratio, equal-opportunity gap, FPR/TPR by group).
* **One scoring class (`Scorer`)** powers both the API and the dashboard, so the demo is the production path.
* **Reason codes** come from SHAP, summed back from one-hot columns to business features, and are returned with
  every score (adverse-action style).
* **Monitoring hooks:** PSI on the score and on each input feature.

## Tests & Docker

```powershell
test.bat                       # or: pytest -q
docker compose up --build      # API :8000, dashboard :8501
```

## Project layout

```
src/creditalo/   config · schemas · data · features · modeling · metrics · explain · fairness · scoring · train · api
dashboard/       app.py (5-tab Streamlit app) · tl_theme.py (CSS + Plotly theme)
scripts/train.py CLI          tests/   unit + API contract tests          docs/   real-data guide, CV story
```

## Limitations (say these before an interviewer asks)

* The default dataset is **synthetic** (shaped like group-lending portfolios) — metrics demonstrate the pipeline,
  not real-world performance. Use `--data gmsc/taiwan` for real defaults or `--data mfi_csv` for a real loan book.
* No time-based (out-of-time) split on synthetic data; with real MFI data, split by disbursement date.
* Decision thresholds are illustrative and should be set with the institution's loss/profit economics.
