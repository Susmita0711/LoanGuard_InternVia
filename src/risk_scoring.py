"""risk_scoring.py
-----------------
Functions to translate model probabilities into a risk score (0‑100) and categorical risk level.
"""
import os
import json
from typing import Dict, Any

# Default thresholds – can be overridden by environment variables or a config file.
DEFAULT_THRESHOLDS = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

def load_thresholds() -> Dict[str, float]:
    """Load risk thresholds from environment variable ``RISK_THRESHOLDS`` if set.

    The variable should be a JSON string like ``{"low":0.2,"medium":0.5,"high":0.8}``.
    """
    env = os.getenv("RISK_THRESHOLDS")
    if env:
        try:
            data = json.loads(env)
            # Ensure all required keys exist
            for key in ["low", "medium", "high"]:
                if key not in data:
                    data[key] = DEFAULT_THRESHOLDS[key]
            return data
        except Exception:
            pass
    return DEFAULT_THRESHOLDS

def probability_to_score(prob: float) -> int:
    """Convert a probability (0‑1) to a 0‑100 integer risk score.
    Simple linear scaling is used.
    """
    return int(round(prob * 100))

def categorize_score(score: int, thresholds: Dict[str, float] = None) -> str:
    """Return a risk category string based on the score.
    Parameters
    ----------
    score: int
        0‑100 risk score.
    thresholds: dict, optional
        Mapping with keys ``low``, ``medium``, ``high`` expressed as probabilities.
    """
    if thresholds is None:
        thresholds = load_thresholds()
    # Convert thresholds to 0‑100 scale for comparison
    low_thr = int(thresholds["low"] * 100)
    med_thr = int(thresholds["medium"] * 100)
    high_thr = int(thresholds["high"] * 100)
    if score < low_thr:
        return "Low"
    if score < med_thr:
        return "Medium"
    if score < high_thr:
        return "High"
    return "Very High"

def assess_risk(probability: float) -> Dict[str, Any]:
    """Return a full risk assessment dictionary.
    Includes raw probability, scaled score, category, and a disclaimer.
    """
    score = probability_to_score(probability)
    category = categorize_score(score)
    return {
        "probability": probability,
        "risk_score": score,
        "risk_category": category,
        "disclaimer": "This risk estimate is for decision‑support only and does not constitute a lending decision.",
    }

if __name__ == "__main__":
    # Simple demo when run directly
    demo_prob = 0.37
    print(assess_risk(demo_prob))

