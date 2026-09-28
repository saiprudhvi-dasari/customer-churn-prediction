# Customer Churn Prediction

Binary classification on telecom customer data — will this customer churn or not.

## What it is

A churn model trained on 5,000 synthetic telecom customers. It compares three
classifiers (logistic regression, random forest, gradient boosting), reports
standard metrics, and saves the best one for scoring new customers.

## Why I built it

Churn prediction is one of those problems that comes up in almost every
subscription business I've worked around, and it's a good excuse to show the
full loop: data generation, feature engineering, model comparison, evaluation,
and inference — not just a notebook with an accuracy score. I generated the
data myself because I didn't want this to depend on downloading some external
dataset that might move or break.

## How to run

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/customers.csv
python train.py           # trains 3 models, saves the best to models/
python predict.py         # scores 3 example customers
```

## What's inside

- `generate_data.py` — builds the dataset. Churn is driven by a weighted mix
  of contract type, tenure, support calls, late payments, and a few other
  signals, run through a logistic function with noise added. Correlated enough
  to learn from, noisy enough to be realistic.
- `train.py` — feature engineering (tenure buckets, charges-per-tenure,
  add-on counts, support friction flags), a `ColumnTransformer` pipeline per
  model, stratified 80/20 split, and a metrics comparison. Saves the best
  model by ROC-AUC plus a `metrics.json`.
- `predict.py` — loads the saved model and scores three hand-made example
  customers so you can see the output shape immediately.

Typical results: logistic regression comes out on top at ~0.75 ROC-AUC, with
random forest and gradient boosting a couple of points behind. That's not a
mistake — the synthetic churn signal is linear by construction, so the linear
model has a natural advantage here. On real data the ranking is usually
reversed. Recall on the churn class is modest (~0.2–0.25 at the default 0.5
threshold), which is exactly why threshold tuning is first on the improvement
list below. Full numbers are in `models/metrics.json`.

## What I'd improve

- **Class imbalance handling.** Churn sits around 25-30% here, so plain
  training works fine, but on a real dataset with 5% churn I'd reach for
  class weights or threshold tuning instead of the default 0.5 cutoff.
- **Calibration.** The probabilities from gradient boosting aren't calibrated;
  if this fed an actual retention campaign, I'd add Platt scaling or isotonic
  regression and check a calibration curve.
- **Feature importance / SHAP.** Right now there's no explanation of *why* a
  customer is flagged. For a stakeholder-facing version, SHAP values per
  prediction would be the first thing I'd add.
- **The data is synthetic.** It captures the right structure, but real churn
  has seasonality, promo effects, and competitor moves that this doesn't.
  I wouldn't claim production readiness off this alone.
