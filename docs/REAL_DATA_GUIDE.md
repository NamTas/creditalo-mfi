# Using real data — and telling an honest CV story

**The hard truth:** labelled microfinance default data (who repaid, who didn't) is not publicly available — MFIs
treat it as confidential. So use a three-level strategy, and describe each level truthfully.

## Level 1 — Real public default data (proves the modelling on real defaults) · 15 minutes

| Dataset | Where | File to save | Command |
|---|---|---|---|
| **Give Me Some Credit** (150k real borrowers, 7 % default) | Kaggle competition “Give Me Some Credit” | `data/raw/cs-training.csv` | `python scripts/train.py --data gmsc` |
| **Default of Credit Card Clients** (Taiwan, 30k, 22 % default; has sex/marriage for fairness) | UCI ML Repository (also on Kaggle as `UCI_Credit_Card.csv`) | `data/raw/UCI_Credit_Card.csv` | `python scripts/train.py --data taiwan` |

Kaggle needs a free account (Kaggle → competition/dataset page → Download → unzip into `data/raw/`).
The dashboard adapts automatically: labels, sliders, SHAP, fairness (Taiwan data audits gender, marital status, age group).

## Level 2 — MFI-realistic synthetic data (shows domain understanding)

The default generator mimics Bangladesh-style group lending: BDT incomes, solidarity-group repayment rate,
bKash/Nagad-style mobile activity, utility-bill discipline, loan cycles, flat-rate instalments.
To make it *credible*, calibrate it to published numbers:

* Set `target_default_rate` in `generate_synthetic_mfi()` to a published portfolio-at-risk / write-off figure for
  Bangladeshi MFIs (look in Microcredit Regulatory Authority statistics, PKSF and large-MFI annual reports).
* Kaggle's **Kiva loans** (real MFI loans) can inspire realistic loan sizes, terms and sectors — as far as I know it
  has **no default label**, so never use it as the target.
* Write the assumptions in the README (already done in “Limitations”).

## Level 3 — A real MFI loan book (the real case) · what to ask a partner for

Save as `data/raw/mfi_loans.csv`, then run `python scripts/train.py --data mfi_csv --path data/raw/mfi_loans.csv`.

| Column | Type | Meaning |
|---|---|---|
| `default` | 0/1 | **Target.** 1 = 90+ days overdue / written off within the loan term (agree the definition in writing) |
| `age`, `dependents`, `years_in_business` | number | borrower profile |
| `business_type` | text | one of: Retail shop, Tailoring, Livestock, Agriculture, Food stall, Handicraft (or edit `BUSINESS_TYPES` in `schemas.py`) |
| `monthly_income`, `monthly_expenses`, `savings_balance` | BDT | household finances |
| `loan_amount`, `loan_term_months`, `prior_loans`, `prior_late_payments` | number | loan & history |
| `group_repayment_rate`, `utility_on_time_rate` | 0–1 | peer-group and alternative data |
| `mobile_txn_per_month` | number | mobile-wallet activity |
| `gender`, `region` *(optional)* | text | **audit only** — never used by the model |

Rules for real data: **remove names, NID, phone numbers**; get written permission; keep data on your PC / in a private
repo (`data/raw/` is git-ignored); split by disbursement date for an out-of-time test if you have dates.
Different columns? Edit `MICROFINANCE` in `src/creditalo/schemas.py` and the `engineer()` function — everything else
(UI sliders, API schema, reason codes) is generated from the schema.

## CV wording — honest and strong

✅ *“Built an explainable credit-scoring system for microfinance: calibrated PD models (Logistic Regression, Random
Forest, LightGBM), SHAP reason codes, fairness audit and a FastAPI scoring service; validated on public real-world
credit-default datasets (Give Me Some Credit, UCI Taiwan) and MFI-style synthetic data.”*

❌ Do **not** write “deployed at an MFI” or quote the synthetic AUC as real-world performance. Interviewers will ask
“which data?” — the honest 3-level answer above impresses more than a bluff.
