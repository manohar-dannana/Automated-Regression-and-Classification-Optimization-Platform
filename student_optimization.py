from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_data import (
    download_student_dataset,
    load_student_data,
    make_regression_data,
)
from student_selection import select_algorithm

Task = Literal["regression"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns
    numerical = X.select_dtypes(include=[np.number]).columns
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
        ]
    )


def suggest_parameters(
    trial: optuna.Trial,
    algorithm: str,
    random_state: int,
) -> dict[str, Any]:
    """Return an optimization search space for the selected regressor."""
    if algorithm == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
        }
    if algorithm == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "objective": "regression",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
    if algorithm == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 1.0]
            ),
            "random_state": random_state,
            "n_jobs": -1,
        }
    if algorithm == "SVR":
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "kernel": "rbf",
        }
    raise ValueError(f"Optimization is not implemented for {algorithm!r}")


def make_estimator(algorithm: str, parameters: dict[str, Any]) -> Any:
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(**parameters)
    if algorithm == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**parameters)
    if algorithm == "Random Forest":
        return RandomForestRegressor(**parameters)
    if algorithm == "SVR":
        from sklearn.svm import SVR

        return SVR(**parameters)
    raise ValueError(f"Optimization is not implemented for {algorithm!r}")


def optimize_selected_algorithm(
    data_path: str | Path,
    algorithm: str,
    language: Literal["math", "portuguese"] = "portuguese",
    include_prior_grades: bool = True,
    n_trials: int = 30,
    folds: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Optimize one selected regressor and evaluate it on a held-out split."""
    data = load_student_data(data_path, language)
    X, y = make_regression_data(data, include_prior_grades)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial, algorithm, random_state)
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", make_estimator(algorithm, parameters)),
            ]
        )
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        return float(-scores.mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    started_at = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    optimization_time = time.perf_counter() - started_at

    best_model = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("model", make_estimator(algorithm, study.best_params)),
        ]
    )
    fit_started_at = time.perf_counter()
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    total_time = time.perf_counter() - fit_started_at + optimization_time
    metrics: dict[str, float | int] = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "number_of_trials": len(study.trials),
        "optimization_time_seconds": float(optimization_time),
        "total_time_seconds": float(total_time),
        "best_objective_value": float(study.best_value),
    }
    return study.best_params, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: optimize the selected Student regressor with TPE."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Student dataset directory or CSV. Downloads when omitted.",
    )
    parser.add_argument("--language", choices=("math", "portuguese"), default="portuguese")
    parser.add_argument("--without-prior-grades", action="store_true")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--algorithm", help="Skip Layer 1 and optimize this algorithm.")
    args = parser.parse_args()

    data_path = args.data_path or download_student_dataset()
    if args.algorithm:
        selected = args.algorithm
    else:
        selected, _, explanation = select_algorithm(
            data_path,
            task="regression",
            language=args.language,
            include_prior_grades=not args.without_prior_grades,
            folds=args.folds,
        )
        print(f"Layer 1 selected algorithm: {selected}")
        print(f"Selection reason: {explanation}")

    best_params, metrics = optimize_selected_algorithm(
        data_path,
        selected,
        language=args.language,
        include_prior_grades=not args.without_prior_grades,
        n_trials=args.trials,
        folds=args.folds,
    )
    print("\nSTAGE 4: HYPERPARAMETER OPTIMIZATION")
    print(f"Algorithm: {selected}")
    print("\nBest configuration:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print("\nEvaluation:")
    for name, value in metrics.items():
        if isinstance(value, int):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value:.4f}")
    pd.DataFrame([{"algorithm": selected, **best_params, **metrics}]).to_csv(
        "student_optimization_results.csv", index=False
    )
    print("\nSaved results to student_optimization_results.csv")


if __name__ == "__main__":
    main()
