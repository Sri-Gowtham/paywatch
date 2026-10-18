"""Shadow-mode scoring service for UPI traffic.

    PAYWATCH_UPI_MODEL_DIR=models/upi PAYWATCH_API_KEYS=key1,key2 PAYWATCH_LOG_SALT=... PAYWATCH_SHADOW_LOG=logs/decisions.jsonl \
    PAYWATCH_STATE_PATH=state/upi_state.json.gz uvicorn --factory src.upi.main:create_upi_app --workers 1

Shadow mode: every response is advisory (`"shadow": true`); this service never blocks anything. One worker only: the state lives in process
memory and is snapshotted to disk periodically and on shutdown. Refuses to start without API keys unless PAYWATCH_ALLOW_NO_AUTH=1
(tests and local experiments only).
"""

import hmac
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .predictor import UpiPredictor
from .schema import UpiTransaction
from .shadow_log import DecisionLog


class ScoreRequest(UpiTransaction):
    explain: bool = True
    commit: bool = True
    top_k: int = 5


class FeedbackRequest(BaseModel):
    transaction_id: str
    is_fraud: bool


def create_upi_app(model_dir: Optional[str] = None, api_keys: Optional[list] = None, allow_no_auth: Optional[bool] = None,
                   log_path: Optional[str] = None, log_salt: Optional[str] = None, state_path: Optional[str] = None,
                   autosave_seconds: Optional[float] = None, max_body_bytes: int = 65536, log_features: Optional[bool] = None) -> FastAPI:
    env = os.environ.get
    model_dir = model_dir or env("PAYWATCH_UPI_MODEL_DIR", "models/upi")
    keys = [k.strip() for k in (api_keys if api_keys is not None else env("PAYWATCH_API_KEYS", "").split(",")) if k and k.strip()]
    allow_no_auth = allow_no_auth if allow_no_auth is not None else env("PAYWATCH_ALLOW_NO_AUTH") == "1"
    if not keys and not allow_no_auth:
        raise RuntimeError("refusing to start without authentication: set PAYWATCH_API_KEYS (or PAYWATCH_ALLOW_NO_AUTH=1 for local tests)")
    log_path = log_path or env("PAYWATCH_SHADOW_LOG")
    log_salt = log_salt or env("PAYWATCH_LOG_SALT")
    state_path = state_path or env("PAYWATCH_STATE_PATH")
    autosave_seconds = float(autosave_seconds if autosave_seconds is not None else env("PAYWATCH_AUTOSAVE_SECONDS", "300"))

    log_features = log_features if log_features is not None else env("PAYWATCH_LOG_FEATURES") == "1"   # off by default: see shadow_log.py
    log = DecisionLog(log_path, log_salt) if log_path else None
    predictor = UpiPredictor(model_dir, log=log, state_path=state_path, log_features=log_features)

    @asynccontextmanager
    async def lifespan(app):
        stop = threading.Event()

        def loop():
            while not stop.wait(autosave_seconds):
                try:
                    predictor.save_state(state_path)
                except Exception:                      # a failed autosave must never take the service down
                    pass

        if state_path and autosave_seconds > 0:
            threading.Thread(target=loop, daemon=True).start()
        yield
        stop.set()
        if state_path:
            predictor.save_state(state_path)

    app = FastAPI(title="PayWatch UPI (shadow mode)", version="1.0.0", lifespan=lifespan,
                  description="Advisory fraud scoring for UPI traffic. Never blocks a payment.")
    app.state.predictor = predictor

    def require_key(x_api_key: Optional[str] = Header(None)):
        if allow_no_auth:
            return
        if not x_api_key or not any(hmac.compare_digest(x_api_key.encode(), k.encode()) for k in keys):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if int(request.headers.get("content-length") or 0) > max_body_bytes:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # report where and why, never the submitted value: a rejected request may contain a raw identifier
        return JSONResponse({"detail": [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]}, status_code=422)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": predictor.model_version, "shadow": True}

    @app.get("/model-info", dependencies=[Depends(require_key)])
    def model_info():
        return {"model_version": predictor.model_version, "n_features": len(predictor.spec["features"]), "n_models": len(predictor.boosters),
                "tier_thresholds_calibrated": predictor.thresholds, "trained_on": predictor.spec.get("trained_on"),
                "criteria_met_at_training": predictor.spec.get("criteria_met"), "shadow": True,
                "logging": log is not None, "logging_features": bool(log and log_features), "payers_tracked": len(predictor.fz.behavior)}

    @app.post("/score", dependencies=[Depends(require_key)])
    def score(req: ScoreRequest):
        txn = req.model_dump(exclude={"explain", "commit", "top_k"})
        return predictor.predict(txn, commit=req.commit, explain=req.explain, top_k=max(1, min(req.top_k, 15)))

    @app.post("/feedback", dependencies=[Depends(require_key)])
    def feedback(req: FeedbackRequest):
        if not predictor.feedback(req.transaction_id, req.is_fraud):
            raise HTTPException(status_code=404, detail="unknown or already-labelled transaction_id")
        return {"recorded": True}

    @app.post("/state/save", dependencies=[Depends(require_key)])
    def save_state_now():
        if not state_path:
            raise HTTPException(status_code=409, detail="no PAYWATCH_STATE_PATH configured")
        return {"saved_bytes": predictor.save_state(state_path)}

    return app
