"""schemas.py
-----------------
Pydantic models for FastAPI request and response payloads.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

class ApplicantRecord(BaseModel):
    """All features required for a single prediction. The field names must match the training columns (excluding the target)."""
    # Dynamically allow any extra fields but we will validate existence later
    class Config:
        extra = "allow"

class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default (0‑1)")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0 = non‑default, 1 = default)")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0‑100")
    risk_category: str = Field(..., description="Risk category: Low, Medium, High, Very High")
    shap_explanation: Optional[Dict[str, Any]] = None
    disclaimer: str = "This risk estimate is for decision‑support only and does not constitute a lending decision."

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    errors: List[Dict[str, Any]] = []

class ModelInfoResponse(BaseModel):
    model_name: str
    validation_roc_auc: float
    best_params: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str = "healthy"
"
