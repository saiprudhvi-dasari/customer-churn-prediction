"""Generate a realistic synthetic telecom customer dataset for churn modeling.

The churn signal is built from a weighted mix of features (contract type,
tenure, support calls, late payments, ...) passed through a logistic
function, plus noise. Correlated enough to learn from, noisy enough
to be honest.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000


def main():
    tenure = RNG.integers(1, 73, N)
    monthly = np.round(RNG.uniform(20, 120, N), 2)

    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"], N, p=[0.5, 0.25, 0.25]
    )
    payment = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        N, p=[0.35, 0.2, 0.25, 0.2],
    )
    internet = RNG.choice(["Fiber optic", "DSL", "No"], N, p=[0.45, 0.35, 0.2])

    senior = RNG.choice([0, 1], N, p=[0.85, 0.15])
    partner = RNG.choice([0, 1], N, p=[0.5, 0.5])
    dependents = RNG.choice([0, 1], N, p=[0.7, 0.3])
    phone = RNG.choice([0, 1], N, p=[0.9, 0.1])
    multi_lines = np.where(phone == 1, RNG.choice([0, 1], N, p=[0.5, 0.5]), 0)
    online_security = RNG.choice([0, 1], N, p=[0.6, 0.4])
    tech_support = RNG.choice([0, 1], N, p=[0.65, 0.35])
    streaming = RNG.choice([0, 1], N, p=[0.55, 0.45])

    support_calls = RNG.poisson(1.2, N).clip(0, 10)
    late_payments = RNG.poisson(0.6, N).clip(0, 8)

    # Churn log-odds: month-to-month, fiber (pricier), support friction,
    # and short tenure push it up; long tenure and add-ons pull it down.
    logit = (
        -2.4
        + 1.6 * (contract == "Month-to-month")
        + 0.5 * (contract == "One year")
        - 0.035 * tenure
        + 0.45 * support_calls
        + 0.55 * late_payments
        + 0.6 * (internet == "Fiber optic")
        - 0.5 * tech_support
        - 0.3 * online_security
        + 0.01 * (monthly - 70)
        + 0.4 * (payment == "Electronic check")
        + RNG.normal(0, 0.5, N)  # noise: real life is messy
    )
    churn = (RNG.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(N)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure + RNG.normal(0, 50, N), 2),
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "phone_service": phone,
        "multiple_lines": multi_lines,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming,
        "support_calls_6m": support_calls,
        "late_payments_12m": late_payments,
        "churn": churn,
    })

    out = "data/customers.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df):,} rows, churn rate {df.churn.mean():.1%}")


if __name__ == "__main__":
    main()
