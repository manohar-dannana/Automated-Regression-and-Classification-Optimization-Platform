from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import kagglehub
import pandas as pd
from kagglehub.exceptions import UnauthenticatedError

STUDENT_DATASET = "larsen0966/student-performance-data-set"
LANGUAGES = ("math", "portuguese")


def download_student_dataset() -> Path:
    """Download the official Kaggle Student Performance dataset."""
    try:
        return Path(kagglehub.dataset_download(STUDENT_DATASET))
    except UnauthenticatedError as exc:
        raise RuntimeError(
            "Kaggle authentication is required. Run script.py and complete "
            "the login, or pass --data-path /path/to/the/dataset."
        ) from exc


def find_student_file(data_path: str | Path, language: str) -> Path:
    """Find a language-specific UCI student CSV in a file or dataset directory."""
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")

    path = Path(data_path)
    if path.is_file():
        return path

    expected_name = {
        "math": "student-mat.csv",
        "portuguese": "student-por.csv",
    }[language]
    matches = list(path.rglob(expected_name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available_files = sorted(
            candidate.name for candidate in path.rglob("student-*.csv")
        )
        available = ", ".join(available_files) or "none"
        raise FileNotFoundError(
            f"Could not find {expected_name} under dataset path: {path}. "
            f"Available student files: {available}. "
            "Use --language portuguese if student-por.csv is available."
        )
    raise RuntimeError(f"Found multiple {expected_name} files under: {path}")


def load_student_data(
    data_path: str | Path,
    language: Literal["math", "portuguese"] = "math",
) -> pd.DataFrame:
    """Load the original semicolon-delimited UCI student records."""
    file_path = find_student_file(data_path, language)
    # The original UCI files are semicolon-delimited, while some Kaggle
    # mirrors export the same records as comma-delimited CSV.
    data = pd.read_csv(file_path, sep=None, engine="python")
    required_columns = {"G1", "G2", "G3"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required grade columns: "
            f"{sorted(missing_columns)}"
        )
    return data


def make_regression_data(
    data: pd.DataFrame,
    include_prior_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create regression features and the G3 target."""
    X = data.drop(columns=["G3"]).copy()
    if not include_prior_grades:
        X = X.drop(columns=["G1", "G2"])
    return X, data["G3"]


def make_classification_data(
    data: pd.DataFrame,
    pass_mark: int = 10,
    include_prior_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create binary pass/fail labels from the final grade G3."""
    X, _ = make_regression_data(data, include_prior_grades)
    return X, (data["G3"] >= pass_mark).astype("int8")


def describe_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return concise dataset characteristics for the platform analyzer."""
    numeric_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.select_dtypes(exclude="number").columns
    return pd.DataFrame(
        {
            "rows": [len(data)],
            "features": [data.shape[1] - 1],
            "numeric_features": [len(numeric_columns) - 1],
            "categorical_features": [len(categorical_columns)],
            "missing_values": [int(data.isna().sum().sum())],
            "target": ["G3"],
            "target_mean": [float(data["G3"].mean())],
            "target_std": [float(data["G3"].std())],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the official Kaggle Student Performance dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Dataset directory or CSV file. Downloads from Kaggle when omitted.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        default="portuguese",
        help="Student record language to inspect.",
    )
    args = parser.parse_args()

    dataset_path = args.data_path or download_student_dataset()
    data = load_student_data(dataset_path, args.language)
    X_with, _ = make_regression_data(data, include_prior_grades=True)
    X_without, _ = make_regression_data(data, include_prior_grades=False)

    print(f"Dataset: {find_student_file(dataset_path, args.language)}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {data.shape[1]}")
    print(f"Regression features with G1/G2: {X_with.shape[1]}")
    print(f"Regression features without G1/G2: {X_without.shape[1]}")
    print("\nDataset characteristics:")
    print(describe_student_data(data).to_string(index=False))
    print("\nColumns:")
    print(", ".join(data.columns))


if __name__ == "__main__":
    main()
