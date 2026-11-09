"""CreditAlo scoring service (FastAPI).

Run:  uvicorn creditalo.api:app --port 8000      Docs: http://127.0.0.1:8000/docs
Optional auth: set env CREDITALO_API_KEY and send header  X-API-Key: <key>
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__
from .schemas import MICROFINANCE
from .scoring import Scorer

log = logging.getLogger("creditalo.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

STATE: dict[str, Any] = {}

EXAMPLE = {f.name: f.default for f in MICROFINANCE.inputs}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["scorer"] = Scorer.load()
        log.info("Model loaded: %s (%s)", STATE["scorer"].model_name, STATE["scorer"].version)
    except FileNotFoundError as exc:  # service still starts so /health explains what is wrong
        STATE["scorer"] = None
        log.error("%s", exc)
    yield


app = FastAPI(
    title="CreditAlo Scoring API",
    version=__version__,
    description="Explainable credit scoring for microfinance: probability of default, a 0-100 score, "
                "a lending decision and SHAP-based reason codes for every borrower.",
    lifespan=lifespan,
)


def get_scorer() -> Scorer:
    scorer = STATE.get("scorer")
    if scorer is None:
        raise HTTPException(503, "Model not loaded. Run `python scripts/train.py` and restart the API.")
    return scorer


def check_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("CREDITALO_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key header.")


# ----------------------------------------------------------------------- schemas
class ScoreRequest(BaseModel):
    applicant_id: Optional[str] = Field(default=None, description="Your own reference, echoed back")
    features: dict[str, Any] = Field(..., description="Borrower fields; see GET /schema")
    model_config = {"json_schema_extra": {"example": {"applicant_id": "DEMO-001", "features": EXAMPLE}}}


class BatchRequest(BaseModel):
    borrowers: list[ScoreRequest] = Field(..., min_length=1, max_length=500)


class ReasonCode(BaseModel):
    code: str
    feature: str
    label: str
    value: str
    impact: float
    text: str


class ReasonCodes(BaseModel):
    risk: list[ReasonCode]
    protective: list[ReasonCode]


class ScoreResponse(BaseModel):
    applicant_id: Optional[str] = None
    pd: float = Field(..., description="Calibrated probability of default")
    score: float = Field(..., description="0-100, higher = safer")
    band: str
    band_label: str
    decision: str
    reason_codes: ReasonCodes
    feature_impacts: dict[str, float]
    base_value: float
    warnings: list[str] = []
    model: dict[str, str]


# ----------------------------------------------------------------------- routes
@app.get("/health", tags=["ops"])
def health():
    s = STATE.get("scorer")
    return {"status": "ok" if s else "model_missing", "model": s.model_name if s else None,
            "version": s.version if s else None, "api_version": __version__}


@app.get("/model-info", tags=["ops"], dependencies=[Depends(check_key)])
def model_info(scorer: Scorer = Depends(get_scorer)):
    return scorer.model_info()


@app.get("/schema", tags=["ops"], dependencies=[Depends(check_key)])
def schema(scorer: Scorer = Depends(get_scorer)):
    return {"required_features": [
        {"name": f.name, "label": f.label, "type": "category" if f.kind == "cat" else "number",
         "min": None if f.kind == "cat" else f.min, "max": None if f.kind == "cat" else f.max,
         "options": f.options or None, "example": f.default}
        for f in scorer.schema.inputs]}


@app.post("/score", response_model=ScoreResponse, tags=["scoring"], dependencies=[Depends(check_key)])
def score(req: ScoreRequest, top_k: int = Query(4, ge=1, le=8), scorer: Scorer = Depends(get_scorer)):
    t0 = time.perf_counter()
    try:
        result = scorer.score_one(req.features, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    log.info("scored id=%s score=%.1f decision=%s (%.1f ms)", req.applicant_id, result["score"],
             result["decision"], (time.perf_counter() - t0) * 1000)
    return {"applicant_id": req.applicant_id, **result}


@app.post("/score/batch", tags=["scoring"], dependencies=[Depends(check_key)])
def score_batch(req: BatchRequest, top_k: int = Query(3, ge=1, le=8), scorer: Scorer = Depends(get_scorer)):
    try:
        results = scorer.score_records([b.features for b in req.borrowers], top_k=top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"count": len(results),
            "results": [{"applicant_id": b.applicant_id, **r} for b, r in zip(req.borrowers, results)]}
