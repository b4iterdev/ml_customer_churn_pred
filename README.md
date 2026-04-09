# Telco Customer Churn Prediction (Undergraduate ML Project)

This repository contains a complete undergraduate-scale Machine Learning project for predicting telecom customer churn.

## Project structure

- `data/raw/`: original downloaded dataset
- `data/processed/`: cleaned and transformed data
- `notebooks/`: EDA and experiment notebooks
- `src/churn/`: reusable training/evaluation code
- `scripts/`: setup and automation scripts
- `app/`: simple web app for inference
- `reports/`: report assets and generated tables/figures
- `models/`: trained model artifacts

## Quick start

1. Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download dataset:

```bash
python3 scripts/download_dataset.py
```

3. Run the web app (after model training is added):

```bash
streamlit run app/app.py
```

4. Train and compare 3 ML models, then save the best one:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_models.py
```

Generated artifacts:

- `reports/model_comparison.csv`
- `reports/model_comparison.md`
- `reports/best_model_summary.json`
- `models/best_model.joblib`

## Dataset

Primary source: Kaggle dataset `blastchar/telco-customer-churn`.

The download script supports:

- Kaggle API download (if credentials are configured), or
- Public fallback URL for this same dataset.
