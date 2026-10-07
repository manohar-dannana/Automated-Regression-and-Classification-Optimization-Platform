from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from student_data import (
    download_student_dataset,
    find_student_file,
    load_student_data,
    make_classification_data,
    make_regression_data,
)

Task = Literal["regression", "classification"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create leakage-safe preprocessing for mixed student data."""
    categorical = X.select_dtypes(include=["object", "category"]).columns
    numerical = X.select_dtypes(include=[np.number]).columns
    return ColumnTransformer(
        transformers=[
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


def candidate_models(task: Task, random_state: int) -> dict[str, object]:
    """Return the candidate algorithms evaluated by Layer 1."""
    if task == "regression":
        models: dict[str, object] = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=250, random_state=random_state, n_jobs=-1
            ),
            "SVR": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=250, random_state=random_state, n_jobs=-1
            ),
            "SVM": SVC(
                C=10.0,
                probability=True,
                random_state=random_state,
            ),
        }

    try:
        if task == "regression":
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                objective="binary:logistic",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        if task == "regression":
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(
                objective="regression",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            from lightgbm import LGBMClassifier

            models["LightGBM"] = LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return models


def explain_selection(
    selected: str,
    task: Task,
    X: pd.DataFrame,
    scores: pd.DataFrame,
) -> str:
    """Explain the selection using measured scores and dataset characteristics."""
    row = scores.loc[scores["algorithm"] == selected].iloc[0]
    reasons = [
        f"it achieved the best mean cross-validation {row['primary_metric_name']} "
        f"({row['primary_metric']:.4f}) among the candidates"
    ]
    categorical_count = len(X.select_dtypes(exclude="number").columns)
    if categorical_count:
        reasons.append(
            f"the dataset contains {categorical_count} categorical feature(s), "
            "which the preprocessing pipeline encodes consistently for every model"
        )
    if selected in {"Random Forest", "XGBoost", "LightGBM"}:
        reasons.append(
            "tree-based models can represent nonlinear relationships and feature "
            "interactions without manual interaction terms"
        )
    elif selected in {"SVM", "SVR"}:
        reasons.append(
            "the kernel model can capture nonlinear relationships after scaling"
        )
    else:
        reasons.append(
            "the linear model provides a strong, interpretable baseline for comparison"
        )
    return "; ".join(reasons) + "."


def select_algorithm(
    data_path: str | Path,
    task: Task = "regression",
    language: Literal["math", "portuguese"] = "math",
    include_prior_grades: bool = True,
    pass_mark: int = 10,
    folds: int = 5,
    random_state: int = 42,
) -> tuple[str, pd.DataFrame, str]:
    """Evaluate candidates with cross-validation and select the best algorithm."""
    data = load_student_data(data_path, language)
    if task == "regression":
        X, y = make_regression_data(data, include_prior_grades)
        scoring = {
            "primary": "neg_root_mean_squared_error",
            "secondary": "r2",
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        }
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        primary_name = "RMSE"
        higher_is_better = False
    else:
        X, y = make_classification_data(
            data, pass_mark, include_prior_grades
        )
        scoring = {
            "primary": "f1",
            "secondary": "accuracy",
            "roc_auc": "roc_auc",
        }
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
        primary_name = "F1"
        higher_is_better = True

    rows = []
    for name, estimator in candidate_models(task, random_state).items():
        model = Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                ("model", estimator),
            ]
        )
        result = cross_validate(
            model,
            X,
            y,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        primary_values = result["test_primary"]
        primary = float(primary_values.mean())
        if task == "regression":
            primary = -primary
            secondary = float(result["test_secondary"].mean())
            extra = float(-result["test_mae"].mean())
        else:
            secondary = float(result["test_secondary"].mean())
            extra = float(result["test_roc_auc"].mean())
        rows.append(
            {
                "algorithm": name,
                "primary_metric_name": primary_name,
                "primary_metric": primary,
                "secondary_metric": secondary,
                "extra_metric": extra,
                "cv_time_seconds": float(result["fit_time"].sum()),
            }
        )

    scores = pd.DataFrame(rows)
    scores = scores.sort_values(
        "primary_metric", ascending=not higher_is_better
    ).reset_index(drop=True)
    selected = str(scores.iloc[0]["algorithm"])
    explanation = explain_selection(selected, task, X, scores)
    return selected, scores, explanation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 1: select the best Student Performance algorithm."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Directory containing student-mat.csv/student-por.csv, or a CSV path. "
        "Downloads the official dataset when omitted.",
    )
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        default="regression",
    )
    parser.add_argument(
        "--language", choices=("math", "portuguese"), default="portuguese"
    )
    parser.add_argument(
        "--without-prior-grades",
        action="store_true",
        help="Exclude G1 and G2 from the features.",
    )
    parser.add_argument("--pass-mark", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data_path = args.data_path or download_student_dataset()
    selected, scores, explanation = select_algorithm(
        data_path,
        task=args.task,
        language=args.language,
        include_prior_grades=not args.without_prior_grades,
        pass_mark=args.pass_mark,
        folds=args.folds,
    )
    file_path = find_student_file(data_path, args.language)
    print("LAYER 1: ALGORITHM SELECTION")
    print(f"Dataset: {file_path}")
    print(f"Task: {args.task}")
    print(f"Prior grades included: {not args.without_prior_grades}")
    print("\nCandidate model cross-validation results:")
    print(scores.to_string(index=False))
    print(f"\nSelected algorithm: {selected}")
    print(f"Why it was selected: {explanation}")
    print("\nLAYER 2: HYPERPARAMETER OPTIMIZATION")
    print(f"Next step: optimize the hyperparameters of {selected}.")
    print("Compare TPE/Optuna, Random Search, and Genetic Algorithm.")
    scores.to_csv("student_algorithm_selection_results.csv", index=False)
    print("\nSaved results to student_algorithm_selection_results.csv")


if __name__ == "__main__":
    main()
