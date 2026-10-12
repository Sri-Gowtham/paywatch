"""PayWatch scoring API.

Run:  uvicorn --factory src.api.main:create_app --workers 1
State (user behaviour, entity fraud history) lives in process memory, so use ONE worker.
Models/assets directory: env PAYWATCH_MODELS_DIR (default models/compact).
"""

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .predictor import PredictionError, Predictor
from .landing import landing_html
from .schemas import EXAMPLE_REQUEST, FeedbackRequest, FeedbackResponse, PredictRequest, PredictResponse


def create_app(models_dir: str = None) -> FastAPI:
    app = FastAPI(title="PayWatch", version="1.0.0",
                  description="UPI-style transaction fraud scoring with calibrated tiers and SHAP reasons")
    predictor = Predictor(models_dir or os.environ.get("PAYWATCH_MODELS_DIR", "models/compact"))
    warm = os.environ.get("PAYWATCH_WARM_STATE")
    if warm and os.path.exists(warm):
        predictor.load_state(warm)
    app.state.predictor = predictor

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing(request: Request):
        return landing_html(predictor.model_version, str(request.base_url))

    @app.get("/example")
    def example():
        """A complete synthetic request body to feed into POST /predict."""
        return EXAMPLE_REQUEST

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": predictor.model_version}

    @app.get("/model-info")
    def model_info():
        return {
            "model_version": predictor.model_version,
            "n_features": len(predictor.features),
            "n_models": len(predictor.boosters),
            "required_fields": predictor.required_fields,
            "tier_thresholds_calibrated": predictor.thresholds,
            "users_tracked": len(predictor.behavior),
            "pending_labels": len(predictor.history.pending),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            return predictor.predict(req.transaction_id, req.fields, explain=req.explain,
                                     commit=req.commit, top_k=req.top_k)
        except PredictionError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        ok = predictor.feedback(req.transaction_id, req.is_fraud)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown or already-labelled transaction_id")
        return FeedbackResponse(recorded=True, detail="entity fraud history updated")

    return app
