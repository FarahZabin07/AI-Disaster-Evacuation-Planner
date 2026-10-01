"""
src/ml/train_model.py
Phases 6, 7, and 8:
- Prepares features via ColumnTransformer (StandardScaler + OneHotEncoder).
- Implements GroupShuffleSplit on 'event_id' to prevent earthquake data leakage.
- Trains an interpretable Logistic Regression model with balanced class weights.
- Evaluates model using Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
- Serializes the trained pipeline into models/road_risk_logistic_model.joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

DATASET_PATH = "data/processed/earthquake_road_dataset.csv"
MODEL_OUTPUT_PATH = "models/road_risk_logistic_model.joblib"


def train_logistic_regression():
    print(f"[INFO] Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Feature definitions
    numeric_features = [
        "magnitude",
        "depth_km",
        "epicentral_dist_km",
        "length_m",
        "lanes_clean",
    ]
    categorical_features = ["highway_clean", "oneway_clean"]
    all_features = numeric_features + categorical_features
    target = "is_blocked"

    X = df[all_features]
    y = df[target].values
    groups = df["event_id"].values

    print(f"[INFO] Dataset shape: {X.shape[0]:,} rows with {X.shape[1]} features.")
    print(f"[INFO] Number of unique earthquake groups: {len(np.unique(groups))}")

    # Group-based split by earthquake event
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    print(f"[INFO] Training set size: {len(X_train):,} rows ({len(train_groups)} earthquakes).")
    print(f"[INFO] Testing set size:  {len(X_test):,} rows ({len(test_groups)} earthquakes).")
    assert len(train_groups.intersection(test_groups)) == 0, "DATA LEAK DETECTED: Earthquakes overlap!"

    # Feature transformation pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Model definition
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    print("[INFO] Fitting Logistic Regression pipeline...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("      LOGISTIC REGRESSION EVALUATION METRICS       ")
    print("=" * 55)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"ROC-AUC Score:     {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"   [TN: {cm[0,0]:,}\tFP: {cm[0,1]:,}]")
    print(f"   [FN: {cm[1,0]:,}\tTP: {cm[1,1]:,}]")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Passable (0)", "Blocked (1)"]))
    print("=" * 55)

    # Inspect coefficients
    feature_names = numeric_features.copy()
    cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_encoded_names = feature_names + encoded_cat_names
    coefficients = model.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame(
        {"Feature": all_encoded_names, "Coefficient": coefficients}
    ).sort_values("Coefficient", ascending=False)

    print("\nModel Interpretability (Learned Coefficients):")
    print(coef_df.to_string(index=False))

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\n[SUCCESS] Model successfully saved to '{MODEL_OUTPUT_PATH}'.")


if __name__ == "__main__":
    train_logistic_regression()