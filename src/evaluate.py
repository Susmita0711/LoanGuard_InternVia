"""evaluate.py
-----------------
Evaluate the best model on the untouched test set and save metrics.
"""
import pathlib
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

from preprocessing import load_data, clean_data, split_data, build_preprocessor

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

def evaluate():
    # Load data
    df_raw = load_data()
    df = clean_data(df_raw)
    _, _, test_df = split_data(df)
    X_test = test_df.drop(columns=["default.payment.next.month"])
    y_test = test_df["default.payment.next.month"]

    # Load artifacts
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    model = joblib.load(MODEL_DIR / "best_model.pkl")

    X_test_proc = preprocessor.transform(X_test)
    probas = model.predict_proba(X_test_proc)[:, 1]
    preds = (probas >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probas),
        "pr_auc": average_precision_score(y_test, probas),
        "specificity": confusion_matrix(y_test, preds)[0,0] / (confusion_matrix(y_test, preds)[0,0] + confusion_matrix(y_test, preds)[0,1]),
        "brier_score": brier_score_loss(y_test, probas),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    # Save metrics
    joblib.dump(metrics, MODEL_DIR / "evaluation_metrics.pkl")
    print("Evaluation metrics saved to models/evaluation_metrics.pkl")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
        else:
            print(f"confusion_matrix: {v}")

if __name__ == "__main__":
    evaluate()

