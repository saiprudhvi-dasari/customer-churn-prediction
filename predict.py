"""Inference example: load the trained churn model and score a few customers.

Usage: python predict.py
"""
import pickle

import pandas as pd

from train import add_features  # reuse the same feature engineering


def main():
    with open("models/churn_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, name = bundle["model"], bundle["name"]
    print(f"Loaded model: {name}\n")

    # Three made-up customers: a risky one, a safe one, and one in between.
    examples = pd.DataFrame([
        {
            "customer_id": "NEW-001", "tenure_months": 3, "monthly_charges": 105.5,
            "total_charges": 310.0, "contract": "Month-to-month",
            "payment_method": "Electronic check", "internet_service": "Fiber optic",
            "senior_citizen": 0, "partner": 0, "dependents": 0, "phone_service": 1,
            "multiple_lines": 1, "online_security": 0, "tech_support": 0,
            "streaming_tv": 1, "support_calls_6m": 5, "late_payments_12m": 2,
        },
        {
            "customer_id": "NEW-002", "tenure_months": 60, "monthly_charges": 55.0,
            "total_charges": 3300.0, "contract": "Two year",
            "payment_method": "Bank transfer", "internet_service": "DSL",
            "senior_citizen": 0, "partner": 1, "dependents": 1, "phone_service": 1,
            "multiple_lines": 0, "online_security": 1, "tech_support": 1,
            "streaming_tv": 0, "support_calls_6m": 0, "late_payments_12m": 0,
        },
        {
            "customer_id": "NEW-003", "tenure_months": 18, "monthly_charges": 82.0,
            "total_charges": 1476.0, "contract": "One year",
            "payment_method": "Credit card", "internet_service": "Fiber optic",
            "senior_citizen": 1, "partner": 0, "dependents": 0, "phone_service": 1,
            "multiple_lines": 0, "online_security": 0, "tech_support": 1,
            "streaming_tv": 1, "support_calls_6m": 2, "late_payments_12m": 1,
        },
    ])

    X = add_features(examples).drop(columns=["customer_id"])
    proba = model.predict_proba(X)[:, 1]
    for cid, p in zip(examples["customer_id"], proba):
        flag = "HIGH RISK" if p >= 0.5 else "low risk"
        print(f"{cid}: churn probability {p:.1%}  ->  {flag}")


if __name__ == "__main__":
    main()
