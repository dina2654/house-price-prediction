# ==============================================================================
# DATA PREPROCESSING & TRANSFORMATIONS MODULE
# isolates all data loading, missing value imputation, feature scaling, and categorical encoding logic
# ==============================================================================
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Loads housing dataset from an Excel file and drops non-predictive ID column."""
    df = pd.read_excel(filepath)
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    return df


def split_features_and_target(df: pd.DataFrame, target_col: str = "SalePrice"):
    """Separates feature matrix X and target vector y."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y


def get_train_test_data(
    filepath: str, test_size: float = 0.2, random_state: int = 1234
):
    """Pipeline helper function to load, separate, and split training/test datasets."""
    df = load_data(filepath)
    X, y = split_features_and_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Constructs ColumnTransformer pipeline for numeric scaling and categorical encoding."""
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns

    # Pipeline for numeric features: Median Imputation + Standard Scaling
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Pipeline for categorical features: Mode Imputation + One-Hot Encoding
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor