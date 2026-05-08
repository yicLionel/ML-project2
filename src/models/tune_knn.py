"""Tune k-nearest neighbours with cross-validation.

KNN is sensitive to the choice of k and feature scaling. This script tests a
small grid of k values and distance weighting options, then saves the results.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


# Add the project root to Python's search path.
# This makes the script work both with "python -m" and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_task_data


OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def get_feature_label(feature_names):
    """Create a short name for the selected feature set."""
    return "_".join(feature_names)


def build_preprocessing(preprocess_name):
    """Create preprocessing steps before KNN."""
    if preprocess_name == "standard":
        return [("scaler", StandardScaler())]
    if preprocess_name == "normalizer":
        return [("normalizer", Normalizer())]
    if preprocess_name == "standard_normalizer":
        return [("scaler", StandardScaler()), ("normalizer", Normalizer())]

    raise ValueError(f"Unknown preprocessing option: {preprocess_name}")


def build_knn(k_value, weights, preprocess_name):
    """Create one KNN pipeline for cross-validation."""
    steps = build_preprocessing(preprocess_name)
    steps.append(
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=k_value,
                weights=weights,
            ),
        )
    )
    return Pipeline(steps)


def cross_validate_model(model, X, y, cv):
    """Return mean validation accuracy and macro F1 across folds."""
    accuracies = []
    macro_f1_scores = []

    for train_index, valid_index in cv.split(X, y):
        # StratifiedKFold gives integer row positions, so use iloc.
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        accuracies.append(accuracy_score(y_valid, predictions))
        macro_f1_scores.append(f1_score(y_valid, predictions, average="macro"))

    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "mean_macro_f1": sum(macro_f1_scores) / len(macro_f1_scores),
    }


def tune_knn(task, feature_names, folds, random_state):
    """Run cross-validation for KNN parameter combinations."""
    task_data = load_task_data(task=task, feature_names=feature_names)

    # Odd k values reduce ties in multi-class voting.
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    weight_options = ["uniform", "distance"]
    preprocess_options = ["standard", "normalizer", "standard_normalizer"]

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []

    for preprocess_name in preprocess_options:
        for weights in weight_options:
            for k_value in k_values:
                print(
                    f"Tuning KNN, preprocess={preprocess_name}, "
                    f"weights={weights}, k={k_value}",
                    flush=True,
                )
                model = build_knn(k_value, weights, preprocess_name)
                scores = cross_validate_model(
                    model,
                    task_data["X_train"],
                    task_data["y_train"],
                    cv,
                )

                results.append(
                    {
                        "task": task,
                        "model": "knn",
                        "features": "+".join(feature_names),
                        "preprocess": preprocess_name,
                        "weights": weights,
                        "k": k_value,
                        "cv_folds": folds,
                        "mean_accuracy": scores["mean_accuracy"],
                        "mean_macro_f1": scores["mean_macro_f1"],
                    }
                )

    return pd.DataFrame(results)


def save_results(results, task, feature_names):
    """Save KNN tuning results as a CSV file."""
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{feature_label}_knn_cv_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Tune KNN with cross-validation.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["color", "hog", "additional", "deep_efficientnet_v2_m"],
        help="Feature sets to use.",
    )
    parser.add_argument("--folds", type=int, default=9, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Run KNN cross-validation and print the best settings."""
    args = parse_args()

    results = tune_knn(
        task=args.task,
        feature_names=args.features,
        folds=args.folds,
        random_state=args.seed,
    )

    results = results.sort_values("mean_accuracy", ascending=False)
    print("\nBest KNN settings:")
    print(results.head(10).to_string(index=False))

    output_path = save_results(results, args.task, args.features)
    print(f"\nSaved KNN CV results to {output_path}")


if __name__ == "__main__":
    main()
