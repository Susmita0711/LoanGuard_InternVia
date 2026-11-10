"""train.py
-----------------
Train multiple models and select the best based on ROC‑AUC on a validation set.
"""
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

from preprocessing import load_data, clean_data, split_data, build_preprocessor

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def train_and_select():
    # Load and prepare data
    df_raw = load_data()
    df = clean_data(df_raw)
    train_df, val_df, test_df = split_data(df)
    X_train = train_df.drop(columns=["default.payment.next.month"])
    y_train = train_df["default.payment.next.month"]
    X_val = val_df.drop(columns=["default.payment.next.month"])
    y_val = val_df["default.payment.next.month"]

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(df)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "C": uniform(0.01, 10),
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 15),
                "min_samples_split": randint(2, 10),
            },
        ),
        "xgboost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 12),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
            },
        ),
        "lightgbm": (
            LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE),
            {
                "n_estimators": randint(50, 300),
                "max_depth": randint(3, 12),
                "learning_rate": uniform(0.01, 0.3),
                "num_leaves": randint(20, 150),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
            },
        ),
    }

    best_score = -np.inf
    best_name = None
    best_model = None
    best_params = None

    for name, (model, param_dist) in models.items():
        print(f"Training {name} ...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train_proc, y_train)
        val_pred = search.predict_proba(X_val_proc)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        print(f"{name} validation ROC‑AUC: {auc:.4f}")
        if auc > best_score:
            best_score = auc
            best_name = name
            best_model = search.best_estimator_
            best_params = search.best_params_

    # Save artifacts
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    # Save meta info
    meta = {
        "model_name": best_name,
        "validation_roc_auc": best_score,
        "best_params": best_params,
    }
    joblib.dump(meta, MODEL_DIR / "model_meta.pkl")
    print(f"Best model: {best_name} with ROC‑AUC {best_score:.4f}")

if __name__ == "__main__":
    train_and_select()

