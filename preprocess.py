from pathlib import Path
from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_train_csv(dataset_path: str | Path) -> Path:
    """Return the competition's train.csv from a downloaded directory or file."""
    path = Path(dataset_path)
    if path.is_file():
        if path.name != "train.csv":
            raise ValueError(f"Expected train.csv, got: {path.name}")
        return path

    train_csv = path / "train.csv"
    if train_csv.is_file():
        return train_csv

    matches = list(path.rglob("train.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find train.csv under: {path}")
    raise RuntimeError(f"Found multiple train.csv files under: {path}")


def download_competition() -> Path:
    """Download the official Kaggle competition files and return their directory."""
    return Path(
        kagglehub.competition_download(
            "house-prices-advanced-regression-techniques"
        )
    )


def load_train_data(train_csv: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load train.csv and separate predictors from the SalePrice target."""
    data = pd.read_csv(find_train_csv(train_csv))
    required_columns = {"Id", "SalePrice"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return data.drop(columns=["Id", "SalePrice"]), data["SalePrice"]


def load_and_split(
    train_csv: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train.csv, remove Id, separate SalePrice, and split the data."""
    X, y = load_train_data(train_csv)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build missing-value, categorical-encoding, and numerical-scaling steps."""
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numerical", numerical_pipeline, numerical_columns),
        ],
        remainder="drop",
    )


def prepare_data(
    train_csv: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """Run the complete preprocessing workflow and return transformed partitions."""
    X_train, X_valid, y_train, y_valid = load_and_split(
        train_csv,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    return (
        X_train_transformed,
        X_valid_transformed,
        y_train,
        y_valid,
        preprocessor,
    )


if __name__ == "__main__":
    downloaded_path = download_competition()
    X_train, X_valid, y_train, y_valid, preprocessor = prepare_data(
        downloaded_path
    )

    print(f"Training features:   {X_train.shape}")
    print(f"Validation features: {X_valid.shape}")
    print(f"Training targets:    {y_train.shape}")
    print(f"Validation targets:  {y_valid.shape}")
    print(f"Encoded feature count: {len(preprocessor.get_feature_names_out())}")
