"""Load Task 1 metadata and provided feature files.

This script combines the metadata CSV files with the provided feature CSV files.
It can be imported by training scripts, or run directly to save merged feature
tables under data/processed/.
"""

import argparse
from pathlib import Path

import pandas as pd


# Paths are based on the project root, so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

# Short names used by command line arguments.
FEATURE_FILES = {
    "color": "color_histogram.csv",
    "hog": "hog_pca.csv",
    "additional": "additional_features.csv",
    "engineered": "engineered_features.csv",
    "deep_resnet50": "deep_resnet50_features.csv",
}


def read_csv_checked(file_path):
    """Read a CSV file and give a clear error if it is missing."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return pd.read_csv(file_path)


def choose_feature_sets(feature_names):
    """Convert 'all' into the full list of provided feature sets."""
    if feature_names == ["all"]:
        return list(FEATURE_FILES.keys())

    for name in feature_names:
        if name not in FEATURE_FILES:
            valid_names = ", ".join(FEATURE_FILES.keys())
            raise ValueError(f"Unknown feature set '{name}'. Choose from: {valid_names}")

    return feature_names


def load_feature_table(task_dir, feature_names):
    """Load and merge the selected feature CSV files using image_id."""
    selected_features = choose_feature_sets(feature_names)
    merged_features = None

    for feature_name in selected_features:
        file_name = FEATURE_FILES[feature_name]
        feature_table = read_csv_checked(task_dir / file_name)

        # Every feature table should have exactly one row for each image.
        if "image_id" not in feature_table.columns:
            raise ValueError(f"{file_name} does not contain image_id")
        if feature_table["image_id"].duplicated().any():
            raise ValueError(f"{file_name} contains duplicate image_id values")

        if merged_features is None:
            merged_features = feature_table
        else:
            merged_features = merged_features.merge(feature_table, on="image_id", how="inner")

    return merged_features


def load_task_data(task="task1", feature_names=None):
    """Load train/test metadata and merge them with selected features.

    Returns a dictionary with X_train, y_train, X_test, metadata, and class names.
    This simple format is enough for the baseline models we will train later.
    """
    if feature_names is None:
        feature_names = ["all"]

    task_dir = DATA_ROOT / "raw" / task

    train_metadata = read_csv_checked(task_dir / "train_metadata.csv")
    test_metadata = read_csv_checked(task_dir / "test_metadata.csv")
    feature_table = load_feature_table(task_dir, feature_names)

    # Join labels/paths with numeric features.
    train_data = train_metadata.merge(feature_table, on="image_id", how="left")
    test_data = test_metadata.merge(feature_table, on="image_id", how="left")

    # Missing values usually mean the image_id values did not match properly.
    if train_data.isna().any().any():
        raise ValueError("Training data contains missing values after merging features")
    if test_data.isna().any().any():
        raise ValueError("Test data contains missing values after merging features")

    metadata_columns = ["image_id", "image_path", "class_id", "class_name"]
    input_columns = [col for col in train_data.columns if col not in metadata_columns]

    class_names = (
        train_metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
        .set_index("class_id")["class_name"]
        .to_dict()
    )

    return {
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "X_train": train_data[input_columns],
        "y_train": train_data["class_id"],
        "X_test": test_data[input_columns],
        "feature_names": input_columns,
        "class_names": class_names,
    }


def save_processed_tables(task_data, task="task1"):
    """Save merged train and test feature tables for quick checking."""
    output_dir = DATA_ROOT / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = task_data["X_train"].copy()
    train_output.insert(0, "class_id", task_data["y_train"].values)
    train_output.insert(0, "image_id", task_data["train_metadata"]["image_id"].values)

    test_output = task_data["X_test"].copy()
    test_output.insert(0, "image_id", task_data["test_metadata"]["image_id"].values)

    train_output.to_csv(output_dir / f"{task}_train_features.csv", index=False)
    test_output.to_csv(output_dir / f"{task}_test_features.csv", index=False)


def parse_args():
    """Read command line options."""
    parser = argparse.ArgumentParser(description="Load and merge provided feature CSV files.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        help="Feature sets to use: all, color, hog, additional, engineered, deep_resnet50.",
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save merged CSV files under data/processed.",
    )
    return parser.parse_args()


def main():
    """Run the data loading script from the command line."""
    args = parse_args()
    task_data = load_task_data(task=args.task, feature_names=args.features)

    print(f"Loaded {args.task}")
    print(f"Train: {task_data['X_train'].shape[0]} rows, {task_data['X_train'].shape[1]} features")
    print(f"Test:  {task_data['X_test'].shape[0]} rows, {task_data['X_test'].shape[1]} features")
    print(f"Classes: {task_data['class_names']}")

    if args.save_processed:
        save_processed_tables(task_data, task=args.task)
        print(f"Saved merged tables to {DATA_ROOT / 'processed'}")


if __name__ == "__main__":
    main()
