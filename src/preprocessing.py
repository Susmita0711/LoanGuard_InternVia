"""preprocessing.py
-----------------
Data preprocessing utilities for the LoanGuard project.
"""
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "default_of_credit_card_clients.xls"

TARGET_COLUMN = "default.payment.next.month"

def load_data() -> pd.DataFrame:
    """Load the dataset from the Excel file.
    Returns
    -------
    pd.DataFrame
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}. Run download_data first.")
    df = pd.read_excel(DATA_FILE, header=1)  # The UCI file has a header row at line 0
    # Rename target column for convenience
    df = df.rename(columns={"default payment next month": TARGET_COLUMN})
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove ID column, handle missing values, and drop duplicates."""
    # The original dataset includes an ID column "ID"
    df = df.drop(columns=[col for col in ["ID", "Unnamed: 0"] if col in df.columns], errors="ignore")
    # Drop duplicate rows
    df = df.drop_duplicates()
    # No explicit missing values in this dataset, but guard against them
    df = df.replace({"?": np.nan})
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """Stratified train/validation/test split.
    Returns train_df, val_df, test_df.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    # From the temporary set, split validation and test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, val_df, test_df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that scales numeric features and one‑hot encodes categoricals.
    The UCI credit default dataset is mostly numeric; however, we treat "SEX", "EDUCATION",
    "MARRIAGE" as categorical for demonstration.
    """
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COLUMN]]
    numeric_transformer = Pipeline(steps=[
        ("imputer",  __import__("sklearn.impute").impute.SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer",  __import__("sklearn.impute").impute.SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor

if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    train_df, val_df, test_df = split_data(df_clean)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

