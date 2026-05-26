"""Tune Random Forest with cross-validation.

Random Forest can overfit when trees are too deep, especially for Task 2 where
the training set is small. This script tests tree depth together with a few
related settings, then saves the cross-validation results.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


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


def format_depth(max_depth):
    """Convert max_depth into a readable value for printing and saving."""
    if max_depth is None:
        return "None"
    return str(max_depth)


def build_random_forest(max_depth, min_samples_leaf, max_features, n_estimators, random_state):
    """Create one Random Forest model for cross-validation."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


def cross_validate_model(model, X, y, cv):
    """Return mean validation accuracy and macro F1 across folds."""
    accuracies = []
    macro_f1_scores = []

    for train_index, valid_index in cv.split(X, y):
        # StratifiedKFold returns row positions, so iloc keeps the split correct.
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


def tune_random_forest(task, feature_names, folds, n_estimators, random_state):
    """Run cross-validation for Random Forest parameter combinations."""
    task_data = load_task_data(task=task, feature_names=feature_names)

    # max_depth is the main parameter we want to study. The other two settings
    # control tree complexity and feature randomness, so they affect overfitting.
    depth_values = [3, 5, 8, 10, 15, 20, None]
    min_leaf_values = [1, 2, 4, 8]
    max_feature_values = ["sqrt", "log2", 0.3]

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []

    for max_depth in depth_values:
        for min_samples_leaf in min_leaf_values:
            for max_features in max_feature_values:
                print(
                    "Tuning Random Forest, "
                    f"max_depth={format_depth(max_depth)}, "
                    f"min_samples_leaf={min_samples_leaf}, "
                    f"max_features={max_features}",
                    flush=True,
                )

                model = build_random_forest(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
                scores = cross_validate_model(
                    model,
                    task_data["X_train"],
                    task_data["y_train"],
                    cv,
                )

                results.append(
                    {
                        "task": task,
                        "model": "random_forest",
                        "features": "+".join(feature_names),
                        "max_depth": format_depth(max_depth),
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "n_estimators": n_estimators,
                        "cv_folds": folds,
                        "mean_accuracy": scores["mean_accuracy"],
                        "mean_macro_f1": scores["mean_macro_f1"],
                    }
                )

    return pd.DataFrame(results)


def save_results(results, task, feature_names):
    """Save Random Forest tuning results as a CSV file."""
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{feature_label}_random_forest_cv_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Tune Random Forest with cross-validation.")
    parser.add_argument("--task", default="task2", help="Task folder under data/raw.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["hog", "additional", "deep_efficientnet_v2_l"],
        help="Feature sets to use.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees per forest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Run Random Forest cross-validation and print the best settings."""
    args = parse_args()

    results = tune_random_forest(
        task=args.task,
        feature_names=args.features,
        folds=args.folds,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )

    results = results.sort_values("mean_accuracy", ascending=False)
    print("\nBest Random Forest settings:")
    print(results.head(10).to_string(index=False))

    output_path = save_results(results, args.task, args.features)
    print(f"\nSaved Random Forest CV results to {output_path}")


if __name__ == "__main__":
    main()
