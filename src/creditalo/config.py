"""Central configuration: paths and the score policy (PD -> 0-100 score -> decision)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CREDITALO_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = Path(os.environ.get("CREDITALO_ARTIFACTS", ROOT / "artifacts"))

RANDOM_STATE = 42

# ---- Score policy (scorecard-style log-odds scaling, re-based to 0-100) ----------
# A borrower with PD = BASE_PD gets BASE_SCORE; every time the good:bad odds double,
# the score rises by PDO points. Higher score = safer borrower.
BASE_SCORE = 65.0
BASE_PD = 0.05
PDO = 10.0

APPROVE_CUTOFF = 50.0   # score >= 50  -> Approve
DECLINE_CUTOFF = 35.0   # score <  35  -> Decline, otherwise Manual review

# (minimum score, band letter, label, colour)
BANDS = [
    (80.0, "A", "Excellent", "#22C55E"),
    (65.0, "B", "Good", "#84CC16"),
    (50.0, "C", "Fair", "#FFB020"),
    (35.0, "D", "Weak", "#FF7A45"),
    (0.0, "E", "Very high risk", "#EF4444"),
]
BAND_ORDER = [b[1] for b in BANDS]
BAND_COLORS = {b[1]: b[3] for b in BANDS}
