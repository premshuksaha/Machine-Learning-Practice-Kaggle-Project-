# Machine Learning Practice Kaggle Project

This repository contains a supervised machine learning workflow for predicting purchase value from anonymized web analytics session data.

The project is organized around a single notebook where data loading, preprocessing, model training, tuning, and submission generation are developed end to end.

## Project Objective

Predict `purchaseValue` for each row in the test set using behavioral and device/session features.

## Repository Structure

```text
.
├── notebook.ipynb
├── Readme.md
└── data/
	├── train_data.csv
	├── test_data.csv
	└── sample_submission.csv
```

## Data Description

The dataset appears to be a cleaned, flattened analytics export with mixed feature types:

- Categorical fields: browser, traffic source, campaign/channel details, geo fields
- Numeric fields: page views, hits, session metadata
- Boolean-like fields: mobile flags and click/ad indicators
- Target column: `purchaseValue` (train only)

### File Summary

- `data/train_data.csv`: training data including target `purchaseValue`
- `data/test_data.csv`: test data without target
- `data/sample_submission.csv`: required prediction format (`ID`, `purchaseValue`)

## Current Modeling Workflow

Based on the notebook contents, the workflow includes:

1. Data loading and basic cleaning
2. Feature preprocessing for mixed data types
3. Baseline regression with `RandomForestRegressor`
4. Hyperparameter tuning with `XGBRegressor` and `RandomizedSearchCV`
5. Prediction generation for test data
6. Submission file creation in Kaggle-compatible format

## How To Run

### 1. Open the notebook

Use `notebook.ipynb` in Jupyter or VS Code Notebook mode.

### 2. Install dependencies

Use your preferred environment manager, then install the core packages:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### 3. Execute cells in order

Run all notebook cells from top to bottom so preprocessing and model artifacts are built in the correct order.

### 4. Export predictions

Ensure the output CSV matches this schema:

```text
ID,purchaseValue
0,0.0
1,0.0
...
```

## Suggested Improvements

- Add a reproducible train/validation split strategy and fixed random seeds everywhere
- Introduce feature importance and error analysis sections
- Compare additional regressors (LightGBM, CatBoost, ElasticNet)
- Track experiments and metrics in a small results table
- Convert notebook logic into reusable scripts for training/inference

## Notes

- Keep data files in the existing `data/` directory so notebook paths remain valid.
- If new columns are added in future data versions, update preprocessing logic before training.
