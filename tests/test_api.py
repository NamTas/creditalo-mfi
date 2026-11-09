import importlib

import pytest
from fastapi.testclient import TestClient

from creditalo.schemas import MICROFINANCE
from creditalo.train import run_training


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    out = tmp_path_factory.mktemp("artifacts")
    os.environ["CREDITALO_ARTIFACTS"] = str(out)
    run_training(source="synthetic", n=3000, out_dir=out, shap_rows=300, verbose=False)
    import creditalo.config as cfg, creditalo.scoring as sc, creditalo.api as api
    importlib.reload(cfg); importlib.reload(sc); importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


PAYLOAD = {"applicant_id": "T-1", "features": {f.name: f.default for f in MICROFINANCE.inputs}}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_score_contract(client):
    r = client.post("/score", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 100 and 0 <= body["pd"] <= 1
    assert body["decision"] in {"Approve", "Manual review", "Decline"}
    assert len(body["reason_codes"]["risk"]) <= 4


def test_missing_feature_is_422(client):
    r = client.post("/score", json={"features": {"age": 30}})
    assert r.status_code == 422


def test_batch(client):
    r = client.post("/score/batch", json={"borrowers": [PAYLOAD, PAYLOAD]})
    assert r.status_code == 200 and r.json()["count"] == 2
