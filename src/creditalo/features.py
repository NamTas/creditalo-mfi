"""Feature engineering + a small, picklable encoder (numeric impute + one-hot)."""
from __future__ import annotations

import re

import pandas as pd

from .schemas import FLAT_RATE


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 24, 34, 44, 200], labels=["<25", "25-34", "35-44", "45+"]).astype(str)


def engineer(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Add the derived ratios used by the model. Works on raw API input as well."""
    out = df.copy()
    if key == "microfinance":
        term = pd.to_numeric(out["loan_term_months"], errors="coerce").clip(lower=1)
        loan = pd.to_numeric(out["loan_amount"], errors="coerce").clip(lower=1)
        income = pd.to_numeric(out["monthly_income"], errors="coerce").clip(lower=1)
        expenses = pd.to_numeric(out["monthly_expenses"], errors="coerce")
        savings = pd.to_numeric(out["savings_balance"], errors="coerce")
        instalment = loan * (1 + FLAT_RATE * term / 12.0) / term
        surplus = income - expenses
        out["installment_burden"] = (instalment / income).clip(0, 3)
        out["installment_to_surplus"] = (instalment / surplus.clip(lower=500)).clip(0, 5)
        out["surplus_ratio"] = (surplus / income).clip(-1, 1)
        out["savings_to_loan"] = (savings / loan).clip(0, 10)
    elif key == "gmsc":
        out["total_late"] = (
            pd.to_numeric(out["past_due_30_59"], errors="coerce").fillna(0)
            + pd.to_numeric(out["past_due_60_89"], errors="coerce").fillna(0)
            + pd.to_numeric(out["late_90_days"], errors="coerce").fillna(0)
        )
    elif key == "taiwan":
        bill = pd.to_numeric(out["avg_bill_amt"], errors="coerce")
        pay = pd.to_numeric(out["avg_pay_amt"], errors="coerce")
        limit = pd.to_numeric(out["limit_bal"], errors="coerce").clip(lower=1)
        out["utilization"] = (bill / limit).clip(-0.5, 2)
        out["pay_ratio"] = (pay / bill.clip(lower=1000)).clip(0, 3)
    return out


def _safe(text) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")


class TabularEncoder:
    """Median-impute numerics, one-hot categoricals. Keeps a map encoded-col -> original feature
    so SHAP values can be summed back to human-readable features."""

    def __init__(self, num_cols, cat_cols):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self.medians_ = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in self.num_cols}
        self.categories_ = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.cat_cols}
        self.columns_, self.origin_ = [], []
        for c in self.num_cols:
            self.columns_.append(c)
            self.origin_.append(c)
        for c in self.cat_cols:
            for cat in self.categories_[c]:
                self.columns_.append(f"{c}__{_safe(cat)}")
                self.origin_.append(c)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.num_cols:
            cols[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).to_numpy(float)
        for c in self.cat_cols:
            values = df[c].astype(str).to_numpy()
            for cat in self.categories_[c]:
                cols[f"{c}__{_safe(cat)}"] = (values == cat).astype(float)
        return pd.DataFrame(cols, index=df.index)[self.columns_]
