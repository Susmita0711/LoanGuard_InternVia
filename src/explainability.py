"""explainability.py
-----------------
Utility functions for SHAP explainability.
"""
import pathlib
import joblib
import pandas as pd
import shap
import numpy as np
from typing import List, Dict, Any

MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
_preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
_model = joblib.load(MODEL_DIR / "best_model.pkl")

def shap_explainer() -> shap.Explainer:
    """Create a SHAP Explainer for the fitted model.
    Uses TreeExplainer for tree‑based models, KernelExplainer otherwise.
    """
    try:
        # XGBoost, LightGBM, RandomForest are tree based
        explainer = shap.Explainer(_model)
    except Exception:
        # Fallback to KernelExplainer with a background sample
        background = shap.sample(pd.DataFrame(_preprocessor.transform(pd.DataFrame())) , 100)
        explainer = shap.KernelExplainer(_model.predict_proba, background)
    return explainer

def global_feature_importance() -> pd.DataFrame:
    """Return mean absolute SHAP values for each feature.
    """
    explainer = shap_explainer()
    # Use a random sample of the training data for speed
    X_sample = shap.sample(pd.DataFrame(_preprocessor.transform(pd.DataFrame())) , 200)
    shap_values = explainer(X_sample)
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = _preprocessor.get_feature_names_out()
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values(by="importance", ascending=False)
    return df

def explain_instance(record: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SHAP values for a single applicant record.
    Returns a dictionary with ``shap_values`` (list of (feature, value)) and ``base_value``.
    """
    df = pd.DataFrame([record])
    X_processed = _preprocessor.transform(df)
    explainer = shap_explainer()
    shap_vals = explainer(X_processed)
    # For binary classification, shap_vals.values has shape (1, n_features, n_classes)
    # We take the values for class 1 (default)
    vals = shap_vals.values[0, :, 1] if shap_vals.values.ndim == 3 else shap_vals.values[0]
    base = shap_vals.base_values[0, 1] if shap_vals.base_values.ndim == 2 else shap_vals.base_values[0]
    feature_names = _preprocessor.get_feature_names_out()
    contributions = [{"feature": f, "shap_value": float(v)} for f, v in zip(feature_names, vals)]
    # Sort by absolute contribution magnitude
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return {"base_value": float(base), "contributions": contributions}

