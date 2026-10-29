# ==============================================================================
# MODEL TRAINING, EVALUATION & FEATURE IMPORTANCE MODULE (src/train_model.py)
# ==============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Ensure python can locate local modules inside src/
sys.path.append(str(Path(__file__).resolve().parent))

from data_preprocessing import build_preprocessor, get_train_test_data


def create_model_pipelines(preprocessor):
    """Constructs Baseline Linear Regression and Hyperparameter-Tuned RidgeCV pipelines[cite: 7, 9]."""
    # Grid of alpha values for regularisation tuning
    alphas = np.logspace(-2, 3, 50)

    linear_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    ridge_cv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RidgeCV(alphas=alphas, cv=5)),
    ])

    return linear_pipeline, ridge_cv_pipeline


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates model performance metrics on the test set[cite: 7, 11]."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"=== {model_name} Performance ===")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R²:   {r2:.4f}\n")

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def extract_feature_importance(pipeline, X_train):
    """Extracts top positive and negative feature driver coefficients."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns

    cat_onehot_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_cols)
    )

    feature_names = np.concatenate([num_cols, cat_onehot_names])

    coefs = pd.Series(model.coef_, index=feature_names)
    top_pos = coefs.sort_values(ascending=False).head(5)
    top_neg = coefs.sort_values(ascending=True).head(5)

    print("Top 5 Positive Price Drivers:")
    print(top_pos)
    print("\nTop 5 Negative Price Drivers:")
    print(top_neg)


# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    # 1. Load data via data_preprocessing module[cite: 7]
    X_train, X_test, y_train, y_test = get_train_test_data(
        "house-price-prediction/data/HousePricesDataset.xlsx"
    )

    # 2. Build preprocessor and model pipelines
    preprocessor = build_preprocessor(X_train)
    linear_model, ridge_cv_model = create_model_pipelines(preprocessor)

    # 3. Fit models
    linear_model.fit(X_train, y_train)
    ridge_cv_model.fit(X_train, y_train)

    # 4. Evaluate models[cite: 7, 15]
    evaluate_model(
        linear_model, X_test, y_test, model_name="Linear Regression (Baseline)"
    )

    optimal_alpha = ridge_cv_model.named_steps["model"].alpha_
    evaluate_model(
        ridge_cv_model,
        X_test,
        y_test,
        model_name=f"Ridge Regression (Optimal Alpha={optimal_alpha:.2f})",
    )

    # 5. Extract Feature Importance
    extract_feature_importance(ridge_cv_model, X_train)