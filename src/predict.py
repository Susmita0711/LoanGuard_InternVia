"""predict.py
-----------------
Utility to load the trained model and make predictions on new data.
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

# Load artifacts once at import time for efficiency
_preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
_model = joblib.load(MODEL_DIR / "best_model.pkl")

def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    """Predict default risk for a single applicant record.

    Parameters
    ----------
    record: dict
        Mapping of feature names to values (must match training columns).

    Returns
    -------
    dict
        Contains ``probability`` (float) and ``prediction`` (0/1).
    """
    # Convert to DataFrame
    df = pd.DataFrame([record])
    # Apply preprocessing
    X_processed = _preprocessor.transform(df)
    prob = _model.predict_proba(X_processed)[0, 1]
    pred = int(prob >= 0.5)
    return {"probability": float(prob), "prediction": pred}

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict on a batch DataFrame and return a DataFrame with results.
    The input DataFrame should contain the same columns as the training data
    (excluding the target column).
    """
    X_processed = _preprocessor.transform(df)
    probas = _model.predict_proba(X_processed)[:, 1]
    preds = (probas >= 0.5).astype(int)
    result = df.copy()
    result["probability"] = probas
    result["prediction"] = preds
    return result

