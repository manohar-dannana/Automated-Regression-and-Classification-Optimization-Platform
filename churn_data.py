from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"


def find_churn_csv(data_path: str | Path) -> Path:
    """Find the Telco churn CSV from a file or directory."""
    path = Path(data_path)
    if path.is_file():
        if path.name != DATASET_FILENAME:
            raise ValueError(
                f"Expected {DATASET_FILENAME}, got {path.name}"
            )
        return path

    matches = list(path.rglob(DATASET_FILENAME))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {DATASET_FILENAME} under: {path}"
        )
    raise RuntimeError(f"Found multiple {DATASET_FILENAME} files under: {path}")


def find_default_churn_csv() -> Path:
    """Find the dataset in common project locations when no path is supplied."""
    project_root = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / DATASET_FILENAME,
        Path.cwd() / "data" / DATASET_FILENAME,
        project_root / DATASET_FILENAME,
        project_root / "data" / DATASET_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {DATASET_FILENAME}. Place it in the project root or "
        f"data/ directory, or pass its full path as an argument."
    )


def load_churn_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load Telco churn records and return cleaned features and target."""
    data = pd.read_csv(find_churn_csv(data_path))
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"])

    target = data.pop(TARGET_COLUMN).map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No' values")

    features = data.drop(columns=["customerID"], errors="ignore")
    return features, target.astype("int8")


def describe_churn_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Return dataset characteristics for the platform analyzer."""
    return pd.DataFrame(
        {
            "rows": [len(features)],
            "features": [features.shape[1]],
            "numeric_features": [
                len(features.select_dtypes(include="number").columns)
            ],
            "categorical_features": [
                len(features.select_dtypes(exclude="number").columns)
            ],
            "missing_values": [int(features.isna().sum().sum())],
            "churn_rate": [float(target.mean())],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the Telco Customer Churn dataset."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        help=f"Path to {DATASET_FILENAME} or its containing directory. "
        "If omitted, the project root and data/ directory are searched.",
    )
    args = parser.parse_args()

    data_path = args.data_path or find_default_churn_csv()
    features, target = load_churn_data(data_path)
    print(f"Dataset: {find_churn_csv(data_path)}")
    print(f"Rows: {len(features)}")
    print(f"Columns after cleaning: {features.shape[1]}")
    print(f"Churn counts: {target.value_counts().sort_index().to_dict()}")
    print("\nDataset characteristics:")
    print(describe_churn_data(features, target).to_string(index=False))
    print("\nColumns:")
    print(", ".join(features.columns))


if __name__ == "__main__":
    main()
