"""routes.py
-----------------
FastAPI routes for LoanGuard.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ..src.predict import predict_single, predict_batch
from ..src.risk_scoring import assess_risk
from ..src.explainability import explain_instance
from ..api.schemas import (
    ApplicantRecord,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
import joblib
import pathlib

router = APIRouter()

# Load model metadata once
MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
_model_meta = joblib.load(MODEL_DIR / "model_meta.pkl")

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy"}

@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "model_name": model_meta["model_name"],
        "validation_roc_auc": _model_meta["validation_roc_auc"],
        "best_params": _model_meta["best_params"],
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(record: ApplicantRecord):
    # Convert Pydantic model to dict
    rec_dict = record.dict()
    try:
        pred = predict_single(rec_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    risk = assess_risk(pred["probability"])
    shap_exp = explain_instance(rec_dict)
    return {
        "probability": pred["probability"],
        "prediction": pred["prediction"],
        "risk_score": risk["risk_score"],
        "risk_category": risk["risk_category"],
        "shap_explanation": shap_exp,
        "disclaimer": risk["disclaimer"],
    }

@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(records: List[ApplicantRecord]):
    preds: List[PredictionResponse] = []
    errors: List[Dict[str, Any]] = []
    # Build DataFrame for successful records
    good_records = []
    indices = []
    for i, r in enumerate(records):
        try:
            good_records.append(r.dict())
            indices.append(i)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    if good_records:
        df = __import__("pandas").DataFrame(good_records)
        batch_res = predict_batch(df)
        for idx, prob, pred in zip(indices, batch_res["probability"], batch_res["prediction"]):
            risk = assess_risk(prob)
            preds.append(
                PredictionResponse(
                    probability=prob,
                    prediction=int(pred),
                    risk_score=risk["risk_score"],
                    risk_category=risk["risk_category"],
                    shap_explanation=None,  # Skip SHAP for batch for performance
                    disclaimer=risk["disclaimer"],
                )
            )
    return {"predictions": preds, "errors": errors}
