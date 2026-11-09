"""Data sources: a microfinance-shaped synthetic generator + loaders for real public datasets
and for a real MFI export (CSV with the microfinance column names)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR
from .features import age_group, engineer
from .schemas import GMSC, MICROFINANCE, TAIWAN, Schema

SOURCES = ("synthetic", "mfi_csv", "gmsc", "taiwan")


# ------------------------------------------------------------------ synthetic MFI data
def generate_synthetic_mfi(n: int = 15000, seed: int = 42, target_default_rate: float = 0.12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["Rural", "Semi-urban", "Urban"], n, p=[0.55, 0.25, 0.20])
    gender = rng.choice(["Female", "Male"], n, p=[0.72, 0.28])
    age = np.clip(rng.normal(35, 9, n), 20, 65).round()
    business = rng.choice(MICROFINANCE.feature("business_type").options, n, p=[0.25, 0.14, 0.15, 0.18, 0.16, 0.12])

    income_mult = pd.Series(region).map({"Rural": 0.82, "Semi-urban": 1.0, "Urban": 1.25}).to_numpy()
    income = np.clip(rng.lognormal(np.log(20000), 0.45, n) * income_mult, 6000, 140000).round(-2)
    expense_ratio = np.clip(rng.beta(6, 4, n) * 0.95 + 0.05, 0.3, 0.95)
    expenses = (income * expense_ratio).round(-2)
    dependents = np.clip(rng.poisson(2.6, n), 0, 8)
    years = np.clip(rng.gamma(2.0, 2.5, n), 0, 30).round(1)
    loan = np.clip(rng.lognormal(np.log(45000), 0.55, n) * (0.6 + income / 50000 * 0.4), 10000, 300000).round(-3)
    term = rng.choice([6, 12, 18, 24, 36], n, p=[0.15, 0.40, 0.20, 0.18, 0.07])
    prior_loans = np.clip(rng.poisson(2.0, n), 0, 12)
    late = np.clip(rng.poisson(0.45 + 0.1 * (dependents > 4), n), 0, 10)
    savings = np.clip(rng.lognormal(np.log(7000), 1.0, n) * (0.5 + (1 - expense_ratio)), 0, 200000).round(-2)
    group_rate = np.clip(rng.beta(12, 1.6, n), 0.3, 1.0).round(2)
    mob_mult = np.where(region == "Urban", 1.4, np.where(region == "Rural", 0.8, 1.0))
    mobile = np.clip(rng.poisson(16, n) * mob_mult, 0, 120).round()
    utility = np.clip(rng.beta(8, 2, n), 0.2, 1.0).round(2)

    df = pd.DataFrame({
        "applicant_id": [f"MFI-{i:06d}" for i in range(1, n + 1)],
        "age": age, "monthly_income": income, "monthly_expenses": expenses, "dependents": dependents,
        "years_in_business": years, "business_type": business, "loan_amount": loan,
        "loan_term_months": term, "prior_loans": prior_loans, "prior_late_payments": late,
        "savings_balance": savings, "group_repayment_rate": group_rate,
        "mobile_txn_per_month": mobile, "utility_on_time_rate": utility,
        "gender": gender, "region": region,
    })
    e = engineer(df, "microfinance")

    biz_effect = e["business_type"].map({"Agriculture": 0.30, "Livestock": 0.15, "Retail shop": -0.15,
                                         "Tailoring": -0.05, "Food stall": 0.10, "Handicraft": 0.0}).to_numpy()
    reg_effect = e["region"].map({"Rural": 0.10, "Semi-urban": 0.0, "Urban": -0.05}).to_numpy()
    z = (
        3.0 * (e["installment_burden"] - 0.2)
        + 0.9 * (e["installment_to_surplus"].clip(0, 3) - 0.55)
        + 0.5 * (e["prior_late_payments"] - 0.45)
        - 4.0 * (e["group_repayment_rate"] - 0.9)
        - 0.5 * (np.log1p(e["savings_to_loan"] * 10) - 0.9)
        - 0.04 * (e["years_in_business"] - 5)
        - 0.012 * (e["mobile_txn_per_month"] - 16)
        - 1.8 * (e["utility_on_time_rate"] - 0.8)
        + 0.10 * (e["dependents"] - 2.6)
        - 0.08 * (np.minimum(e["prior_loans"], 6) - 2)
        - 0.015 * (e["age"] - 35)
        + 0.25 * (e["loan_term_months"] >= 24)
        + 0.7 * ((e["installment_burden"] > 0.3) & (e["prior_late_payments"] >= 1))
        + 0.5 * (e["savings_to_loan"] < 0.03)
        + 0.8 * ((e["utility_on_time_rate"] < 0.6) & (e["mobile_txn_per_month"] < 8))
        - 0.5 * ((e["years_in_business"] > 8) & (e["group_repayment_rate"] > 0.95))
        + 6.0 * np.maximum(e["installment_burden"] - 0.35, 0) ** 2
        + biz_effect + reg_effect
        + rng.normal(0, 0.6, n)
    ).to_numpy()

    lo, hi = -15.0, 15.0           # solve intercept so the mean PD equals the target default rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_default_rate:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    df["default"] = (rng.random(n) < p).astype(int)
    df["age_group"] = age_group(df["age"])
    return df


# ------------------------------------------------------------------ real-data loaders
def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


def load_mfi_csv(path: Path | None = None) -> tuple[pd.DataFrame, Schema]:
    """A real MFI export. Needs the microfinance input columns + 'default' (0/1).
    Optional audit columns: gender, region."""
    path = _need(Path(path or RAW_DIR / "mfi_loans.csv"),
                 "Export your loan book to this file - see docs/REAL_DATA_GUIDE.md for the column list.")
    df = pd.read_csv(path)
    missing = [c for c in MICROFINANCE.input_names + ["default"] if c not in df.columns]
    if missing:
        raise ValueError(f"mfi_csv is missing columns: {missing}")
    if "applicant_id" not in df.columns:
        df.insert(0, "applicant_id", [f"MFI-{i:06d}" for i in range(1, len(df) + 1)])
    df["age_group"] = age_group(df["age"])
    audit = [c for c in MICROFINANCE.audit_cols if c in df.columns]
    schema = Schema(**{**MICROFINANCE.__dict__, "audit_cols": audit})
    return df, schema


def load_gmsc(path: Path | None = None, max_rows: int = 60000, seed: int = 42) -> tuple[pd.DataFrame, Schema]:
    path = _need(Path(path or RAW_DIR / "cs-training.csv"),
                 "Download 'Give Me Some Credit' from Kaggle and save cs-training.csv here.")
    raw = pd.read_csv(path)
    raw = raw.drop(columns=[c for c in raw.columns if c.lower().startswith("unnamed")], errors="ignore")
    df = raw.rename(columns={
        "SeriousDlqin2yrs": "default",
        "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
        "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
        "DebtRatio": "debt_ratio", "MonthlyIncome": "monthly_income",
        "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
        "NumberOfTimes90DaysLate": "late_90_days",
        "NumberRealEstateLoansOrLines": "real_estate_loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
        "NumberOfDependents": "dependents",
    })
    df = df[df["age"] >= 18].copy()
    for c in ("past_due_30_59", "past_due_60_89", "late_90_days"):
        df[c] = df[c].where(df[c] < 20, 20)          # 96/98 are data-entry codes
    df["revolving_utilization"] = df["revolving_utilization"].clip(upper=1.5)
    df["debt_ratio"] = df["debt_ratio"].clip(upper=5.0)
    df["monthly_income"] = df["monthly_income"].clip(upper=50000)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    df.insert(0, "applicant_id", [f"APP-{i:06d}" for i in range(1, len(df) + 1)])
    df["age_group"] = age_group(df["age"])
    return df, GMSC


def load_taiwan(path: Path | None = None, max_rows: int = 60000, seed: int = 42) -> tuple[pd.DataFrame, Schema]:
    path = _need(Path(path or RAW_DIR / "UCI_Credit_Card.csv"),
                 "Download 'Default of Credit Card Clients' (UCI / Kaggle) and save UCI_Credit_Card.csv here.")
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    target = next(c for c in raw.columns if c.lower().replace(".", " ").replace("_", " ") in
                  ("default payment next month", "default"))
    pay_cols = [c for c in ["PAY_0", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in raw.columns]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    edu = raw["EDUCATION"].map({1: "Graduate school", 2: "University", 3: "High school"}).fillna("Other")
    df = pd.DataFrame({
        "limit_bal": raw["LIMIT_BAL"], "age": raw["AGE"], "education": edu,
        "pay_recent": raw[pay_cols[0]].clip(-2, 8),
        "max_delay": raw[pay_cols].max(axis=1).clip(-2, 8),
        "months_delayed": (raw[pay_cols] > 0).sum(axis=1),
        "avg_bill_amt": raw[bill_cols].mean(axis=1), "avg_pay_amt": raw[amt_cols].mean(axis=1),
        "gender": raw["SEX"].map({1: "Male", 2: "Female"}).fillna("Unknown"),
        "marital": raw["MARRIAGE"].map({1: "Married", 2: "Single"}).fillna("Other"),
        "default": raw[target].astype(int),
    })
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    df.insert(0, "applicant_id", [f"APP-{i:06d}" for i in range(1, len(df) + 1)])
    df["age_group"] = age_group(df["age"])
    return df, TAIWAN


def load_dataset(source: str = "synthetic", path: str | Path | None = None, n: int = 15000,
                 seed: int = 42, max_rows: int = 60000) -> tuple[pd.DataFrame, Schema]:
    if source == "synthetic":
        return generate_synthetic_mfi(n=n, seed=seed), MICROFINANCE
    if source == "mfi_csv":
        return load_mfi_csv(path)
    if source == "gmsc":
        return load_gmsc(path, max_rows, seed)
    if source == "taiwan":
        return load_taiwan(path, max_rows, seed)
    raise ValueError(f"Unknown data source '{source}'. Choose from {SOURCES}")
