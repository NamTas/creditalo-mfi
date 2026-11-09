import numpy as np
import pytest

from creditalo.data import generate_synthetic_mfi
from creditalo.metrics import psi
from creditalo.modeling import band_for, decision_for, pd_to_score


def test_score_is_monotonic_and_bounded():
    pds = np.linspace(0.001, 0.95, 200)
    s = pd_to_score(pds)
    assert np.all(np.diff(s) <= 1e-9)           # higher PD -> lower score
    assert s.min() >= 0 and s.max() <= 100


def test_policy_anchor():
    assert abs(float(pd_to_score(0.05)) - 65.0) < 1e-6


def test_band_and_decision():
    assert band_for(90)[0] == "A" and band_for(10)[0] == "E"
    assert decision_for(70) == "Approve" and decision_for(40) == "Manual review" and decision_for(5) == "Decline"


def test_synthetic_default_rate():
    df = generate_synthetic_mfi(4000, seed=1)
    assert 0.09 < df["default"].mean() < 0.15
    assert df["applicant_id"].is_unique


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6
