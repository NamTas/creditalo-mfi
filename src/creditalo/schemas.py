"""Dataset schemas. A schema tells the whole system which columns exist, how to label
them, what UI ranges to use and which columns are only used for fairness auditing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FLAT_RATE = 0.15  # flat annual interest used to estimate the monthly instalment


@dataclass
class Feature:
    name: str
    label: str
    kind: str = "num"           # "num" | "cat"
    min: float = 0.0
    max: float = 1.0
    default: Any = 0.0
    step: float = 1.0
    fmt: str = "float2"         # money | pct | int | float1 | float2 | text
    options: list = field(default_factory=list)
    help: str = ""


@dataclass
class Schema:
    key: str
    title: str
    target: str
    currency: str
    inputs: list
    engineered: list
    audit_cols: list
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        d = dict(d)
        d["inputs"] = [Feature(**f) for f in d["inputs"]]
        d["engineered"] = [Feature(**f) for f in d["engineered"]]
        return cls(**d)

    @property
    def all_features(self) -> list:
        return list(self.inputs) + list(self.engineered)

    @property
    def model_features(self) -> list:
        return [f.name for f in self.all_features]

    @property
    def input_names(self) -> list:
        return [f.name for f in self.inputs]

    @property
    def num_cols(self) -> list:
        return [f.name for f in self.all_features if f.kind == "num"]

    @property
    def cat_cols(self) -> list:
        return [f.name for f in self.all_features if f.kind == "cat"]

    def feature(self, name: str) -> Feature:
        for f in self.all_features:
            if f.name == name:
                return f
        raise KeyError(name)

    def label(self, name: str) -> str:
        try:
            return self.feature(name).label
        except KeyError:
            return name.replace("_", " ").title()


BUSINESS_TYPES = ["Retail shop", "Tailoring", "Livestock", "Agriculture", "Food stall", "Handicraft"]

MICROFINANCE = Schema(
    key="microfinance",
    title="Microfinance (Bangladesh-style group lending)",
    target="default",
    currency="BDT",
    inputs=[
        Feature("age", "Age", "num", 18, 70, 34, 1, "int"),
        Feature("monthly_income", "Monthly household income", "num", 5000, 150000, 22000, 500, "money"),
        Feature("monthly_expenses", "Monthly household expenses", "num", 3000, 120000, 14000, 500, "money"),
        Feature("dependents", "Dependents", "num", 0, 8, 3, 1, "int"),
        Feature("years_in_business", "Years in business", "num", 0, 30, 4, 0.5, "float1"),
        Feature("business_type", "Business type", "cat", options=BUSINESS_TYPES, default="Retail shop", fmt="text"),
        Feature("loan_amount", "Requested loan amount", "num", 10000, 300000, 50000, 1000, "money"),
        Feature("loan_term_months", "Loan term (months)", "num", 6, 36, 12, 1, "int"),
        Feature("prior_loans", "Previous loan cycles", "num", 0, 12, 2, 1, "int"),
        Feature("prior_late_payments", "Late payments (history)", "num", 0, 10, 0, 1, "int"),
        Feature("savings_balance", "Savings balance", "num", 0, 200000, 8000, 500, "money"),
        Feature("group_repayment_rate", "Peer-group repayment rate", "num", 0.3, 1.0, 0.92, 0.01, "pct",
                help="Share of instalments paid on time by the borrower's solidarity group"),
        Feature("mobile_txn_per_month", "Mobile-wallet transactions / month", "num", 0, 120, 18, 1, "int",
                help="Alternative data, e.g. bKash / Nagad activity"),
        Feature("utility_on_time_rate", "Utility bills paid on time", "num", 0.2, 1.0, 0.85, 0.01, "pct",
                help="Alternative data: electricity / mobile-bill payment discipline"),
    ],
    engineered=[
        Feature("installment_burden", "Instalment as % of income", "num", fmt="pct"),
        Feature("installment_to_surplus", "Instalment vs. monthly surplus", "num", fmt="float2"),
        Feature("surplus_ratio", "Monthly surplus ratio", "num", fmt="pct"),
        Feature("savings_to_loan", "Savings / loan amount", "num", fmt="float2"),
    ],
    audit_cols=["gender", "region", "age_group"],
    notes="Synthetic data shaped like MFI portfolios (group lending, alternative data). "
          "Replace with a real MFI export using --data mfi_csv.",
)

GMSC = Schema(
    key="gmsc",
    title="Give Me Some Credit (Kaggle, real consumer credit)",
    target="default",
    currency="$",
    inputs=[
        Feature("revolving_utilization", "Revolving credit utilisation", "num", 0, 1.5, 0.3, 0.01, "pct"),
        Feature("age", "Age", "num", 18, 95, 45, 1, "int"),
        Feature("past_due_30_59", "Times 30-59 days past due", "num", 0, 10, 0, 1, "int"),
        Feature("debt_ratio", "Debt ratio", "num", 0, 5, 0.35, 0.01, "float2"),
        Feature("monthly_income", "Monthly income", "num", 0, 50000, 5400, 100, "money"),
        Feature("open_credit_lines", "Open credit lines & loans", "num", 0, 30, 8, 1, "int"),
        Feature("late_90_days", "Times 90+ days late", "num", 0, 10, 0, 1, "int"),
        Feature("real_estate_loans", "Real-estate loans", "num", 0, 10, 1, 1, "int"),
        Feature("past_due_60_89", "Times 60-89 days past due", "num", 0, 10, 0, 1, "int"),
        Feature("dependents", "Dependents", "num", 0, 8, 0, 1, "int"),
    ],
    engineered=[Feature("total_late", "Total late events", "num", fmt="int")],
    audit_cols=["age_group"],
    notes="Public Kaggle dataset (cs-training.csv).",
)

TAIWAN = Schema(
    key="taiwan",
    title="UCI Taiwan credit-card default (real)",
    target="default",
    currency="NT$",
    inputs=[
        Feature("limit_bal", "Credit limit", "num", 10000, 1000000, 140000, 10000, "money"),
        Feature("age", "Age", "num", 21, 75, 34, 1, "int"),
        Feature("education", "Education", "cat", options=["Graduate school", "University", "High school", "Other"],
                default="University", fmt="text"),
        Feature("pay_recent", "Latest repayment status", "num", -2, 8, 0, 1, "int",
                help="-2/-1 = paid duly, 1+ = months of delay"),
        Feature("max_delay", "Worst repayment status (6m)", "num", -2, 8, 0, 1, "int"),
        Feature("months_delayed", "Months delayed (of 6)", "num", 0, 6, 0, 1, "int"),
        Feature("avg_bill_amt", "Average monthly bill", "num", -50000, 600000, 45000, 1000, "money"),
        Feature("avg_pay_amt", "Average monthly payment", "num", 0, 300000, 5000, 500, "money"),
    ],
    engineered=[
        Feature("utilization", "Credit utilisation", "num", fmt="pct"),
        Feature("pay_ratio", "Payment / bill ratio", "num", fmt="float2"),
    ],
    audit_cols=["gender", "marital", "age_group"],
    notes="UCI 'Default of Credit Card Clients' (Kaggle file UCI_Credit_Card.csv).",
)

SCHEMAS = {s.key: s for s in (MICROFINANCE, GMSC, TAIWAN)}
