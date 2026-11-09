"""CLI:  python scripts/train.py [--data synthetic|mfi_csv|gmsc|taiwan] [--path FILE] [--n 15000]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditalo.data import SOURCES
from creditalo.train import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train CreditAlo models")
    ap.add_argument("--data", default="synthetic", choices=SOURCES)
    ap.add_argument("--path", default=None, help="CSV path (defaults to data/raw/<standard name>)")
    ap.add_argument("--n", type=int, default=15000, help="rows for synthetic data")
    ap.add_argument("--max-rows", type=int, default=60000, help="cap rows for large real datasets")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_training(source=a.data, path=a.path, n=a.n, seed=a.seed, max_rows=a.max_rows)
