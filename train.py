"""Train and compare churn classifiers.

Pipeline per model: ColumnTransformer (one-hot categoricals, scale numerics)
-> classifier. Compares logistic regression, random forest, and histogram
gradient boosting on accuracy / precision / recall / F1 / ROC-AUC, saves the
best model (by ROC-AUC) and a metrics JSON.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
CATEGORICAL = ["contract", "payment_method", "internet_service"]
NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges", "senior_citizen",
    "partner", "dependents", "phone_service", "multiple_lines",
    "online_security", "tech_support", "streaming_tv",
    "support_calls_6m", "late_payments_12m",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small, defensible feature engineering on top of the raw columns."""
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure_months"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"],
    )
    df["charges_per_tenure"] = df["monthly_charges"] / df["tenure_months"].clip(lower=1)
    df["high_support"] = (df["support_calls_6m"] >= 3).astype(int)
    df["has_late_payment"] = (df["late_payments_12m"] > 0).astype(int)
    add_ons = ["online_security", "tech_support", "streaming_tv",
               "multiple_lines", "phone_service"]
    df["addon_count"] = df[add_ons].sum(axis=1)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC + ["charges_per_tenure", "addon_count",
                                             "high_support", "has_late_payment"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL + ["tenure_group"]),
    ])


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    # Random forest: solid baseline, handles the mixed feature types well.
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    # HistGradientBoosting: sklearn-native boosting, usually the strongest
    # here without pulling in XGBoost/LightGBM as extra dependencies.
    "gradient_boosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, random_state=42
    ),
}


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    df = add_features(pd.read_csv("data/customers.csv"))
    X = df.drop(columns=[TARGET, "customer_id"])
    y = df[TARGET]

    # Stratified split so the churn rate is preserved in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, fitted = [], {}
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        print(f"\n== {name} ==")
        print(classification_report(y_test, pipe.predict(X_test),
                                    target_names=["stayed", "churned"]))

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best['model']} ({best['roc_auc']})")

    Path("models").mkdir(exist_ok=True)
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump({"model": fitted[best["model"]],
                     "name": best["model"],
                     "metrics": best}, f)
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved models/churn_model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
