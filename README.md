# Ames Housing Regression Benchmark

This project starts with the official Kaggle **House Prices: Advanced Regression
Techniques** dataset. The first milestone evaluates Ridge, Random Forest,
XGBoost, and LightGBM using the same leakage-safe preprocessing and validation
split.

## Setup

```bash
.venv/bin/pip install -r requirements.txt
```

XGBoost and LightGBM require the macOS OpenMP runtime:

```bash
brew install libomp
```

Authenticate with Kaggle before downloading the competition files:

```python
import kagglehub

kagglehub.login()
```

## Run the baseline experiment

With Kaggle authentication configured:

```bash
.venv/bin/python baseline.py
```

Or download the official files manually and provide the directory or
`train.csv` directly:

```bash
.venv/bin/python baseline.py --train-csv /path/to/competition/files
```

The script removes `Id`, separates `SalePrice`, splits the data, and fits
imputation and encoding steps only on the training partition. Models train on
`log1p(SalePrice)`; the report includes raw RMSE, log-RMSE, MAE, R², and
training time. Results are saved to `baseline_results.csv`.

## Student Performance benchmark

The second benchmark uses the official Kaggle Student Performance dataset:
`larsen0966/student-performance-data-set`. The loader supports the original
`student-mat.csv` and `student-por.csv` files, which use semicolon delimiters.

Inspect the Portuguese records after Kaggle authentication:

```bash
.venv/bin/python student_data.py --language portuguese
```

Inspect a manually downloaded dataset without Kaggle authentication:

```bash
.venv/bin/python student_data.py \
  --data-path /path/to/student-performance-data-set \
  --language portuguese
```

The loader supports both research conditions:

- regression with `G1` and `G2` included when predicting `G3`
- regression with `G1` and `G2` removed
- binary classification using `G3 >= 10` as the pass label

Run Layer 1 algorithm selection after downloading the Student dataset:

```bash
.venv/bin/python student_selection.py \
  --data-path /path/to/student-performance-data-set \
  --task regression \
  --language math
```

For classification:

```bash
.venv/bin/python student_selection.py \
  --data-path /path/to/student-performance-data-set \
  --task classification \
  --language math \
  --without-prior-grades
```

The command evaluates Linear/Logistic Regression, Random Forest, SVM/SVR,
XGBoost, and LightGBM with cross-validation. It prints the selected algorithm,
the measured reason for selection, candidate metrics, and the Layer 2
optimization handoff. Results are saved to
`student_algorithm_selection_results.csv`.

## Stage 4: hyperparameter optimization

Optimize the Layer 1-selected Student regression algorithm with Optuna/TPE:

```bash
.venv/bin/python student_optimization.py --trials 30
```

For XGBoost explicitly:

```bash
.venv/bin/python student_optimization.py \
  --algorithm XGBoost \
  --trials 30
```

The XGBoost search includes `n_estimators` (100-1000), `max_depth` (3-12),
`learning_rate` (0.01-0.30), `subsample` (0.5-1.0), `colsample_bytree`
(0.5-1.0), `min_child_weight` (1-10), `gamma` (0-5), `reg_alpha` (0-10),
and `reg_lambda` (0-10). The output reports RMSE, MAE, R², number of trials,
optimization time, and best objective value.

## Telco Customer Churn benchmark

The Telco classification benchmark uses the local
`WA_Fn-UseC_-Telco-Customer-Churn.csv` file. The loader removes the identifier,
converts `TotalCharges` to numeric values, drops records where that conversion
produces a missing value, and maps `Churn` from `No`/`Yes` to `0`/`1`.

Run it with:

```bash
.venv/bin/python churn_data.py \
  /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The command prints the cleaned shape, churn counts, dataset characteristics,
and feature names.
