from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, download_competition, load_and_split


@dataclass(frozen=True)
class ModelResult:
    model: str
    rmse: float
    log_rmse: float
    mae: float
    r2: float
    training_time_seconds: float


def build_candidates(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Return baseline estimators before preprocessing is attached."""
    candidates: dict[str, RegressorMixin] = {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for the XGBoost baseline. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for the LightGBM baseline. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    candidates["XGBoost"] = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    candidates["LightGBM"] = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    return candidates


def evaluate_baselines(
    train_csv: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and evaluate all baseline models on one fixed validation split."""
    X_train, X_valid, y_train, y_valid = load_and_split(
        train_csv,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(X_train)
    results: list[ModelResult] = []

    for name, estimator in build_candidates(random_state).items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )
        started_at = time.perf_counter()
        model.fit(X_train, np.log1p(y_train))
        predictions = np.maximum(0.0, np.expm1(model.predict(X_valid)))
        elapsed = time.perf_counter() - started_at
        results.append(
            ModelResult(
                model=name,
                rmse=float(np.sqrt(mean_squared_error(y_valid, predictions))),
                log_rmse=float(
                    np.sqrt(
                        mean_squared_error(
                            np.log1p(y_valid), np.log1p(predictions)
                        )
                    )
                ),
                mae=float(mean_absolute_error(y_valid, predictions)),
                r2=float(r2_score(y_valid, predictions)),
                training_time_seconds=elapsed,
            )
        )

    return (
        pd.DataFrame([result.__dict__ for result in results])
        .sort_values("log_rmse")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Ames Housing baseline regressors."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        help="Path to train.csv or a directory containing train.csv. "
        "Downloads the official Kaggle files when omitted.",
    )
    args = parser.parse_args()

    dataset_path = args.train_csv or download_competition()
    results = evaluate_baselines(dataset_path)
    print(results.to_string(index=False))
    results.to_csv("baseline_results.csv", index=False)
    print("\nSaved baseline results to baseline_results.csv")


if __name__ == "__main__":
    main()
