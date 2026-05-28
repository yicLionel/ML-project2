"""Tune Logistic Regression and Linear SVM with cross-validation.

This script focuses on the two strongest linear models. It tests several C
values and preprocessing choices, then saves the cross-validation results.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC


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
    """Create the preprocessing steps before the classifier."""
    # StandardScaler performed best in the current EfficientNet experiments, but
    # the other options are kept for comparison.
    if preprocess_name == "standard":
        return [("scaler", StandardScaler())]
    if preprocess_name == "normalizer":
        return [("normalizer", Normalizer())]
    if preprocess_name == "standard_normalizer":
        return [("scaler", StandardScaler()), ("normalizer", Normalizer())]

    raise ValueError(f"Unknown preprocessing option: {preprocess_name}")


def build_model(model_name, c_value, preprocess_name, random_state):
    """Create one model pipeline for cross-validation."""
    steps = build_preprocessing(preprocess_name)

    if model_name == "logistic_regression":
        # liblinear is used with one-vs-rest because this is a 10-class problem.
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=c_value,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )
        )
    elif model_name == "linear_svm":
        # dual=False is usually faster for our train size and feature dimension.
        classifier = LinearSVC(
            C=c_value,
            dual=False,
            max_iter=5000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("model", classifier))
    return Pipeline(steps)


def cross_validate_model(model, X, y, cv):
    """Return mean validation accuracy and macro F1 across folds."""
    accuracies = []
    macro_f1_scores = []

    for train_index, valid_index in cv.split(X, y):
        # Use iloc because StratifiedKFold returns integer row positions.
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        # Fit only on the training fold, then evaluate on the held-out fold.
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        accuracies.append(accuracy_score(y_valid, predictions))
        macro_f1_scores.append(f1_score(y_valid, predictions, average="macro"))

    return {
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "mean_macro_f1": sum(macro_f1_scores) / len(macro_f1_scores),
    }


def tune_models(task, feature_names, folds, random_state):
    """Run cross-validation for all model and parameter combinations."""
    task_data = load_task_data(task=task, feature_names=feature_names)

    # The grid is centred around the values that worked well after adding deep
    # CNN features. Smaller C means stronger regularisation.
    c_values = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    preprocess_options = ["standard", "normalizer", "standard_normalizer"]
    model_names = ["logistic_regression", "linear_svm"]

    # Stratified folds preserve class balance in each validation fold.
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []

    for model_name in model_names:
        for preprocess_name in preprocess_options:
            for c_value in c_values:
                print(
                    f"Tuning {model_name}, preprocess={preprocess_name}, C={c_value}",
                    flush=True,
                )
                model = build_model(model_name, c_value, preprocess_name, random_state)
                scores = cross_validate_model(
                    model,
                    task_data["X_train"],
                    task_data["y_train"],
                    cv,
                )

                results.append(
                    {
                        "task": task,
                        "model": model_name,
                        "features": "+".join(feature_names),
                        "preprocess": preprocess_name,
                        "C": c_value,
                        "cv_folds": folds,
                        "mean_accuracy": scores["mean_accuracy"],
                        "mean_macro_f1": scores["mean_macro_f1"],
                    }
                )

    return pd.DataFrame(results)


def save_results(results, task, feature_names):
    """Save tuning results as a CSV file."""
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{feature_label}_linear_cv_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Tune linear models with cross-validation.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["color", "hog", "additional", "deep_efficientnet_v2_l"],
        help="Feature sets to use.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Run cross-validation tuning and print the best settings."""
    args = parse_args()

    results = tune_models(
        task=args.task,
        feature_names=args.features,
        folds=args.folds,
        random_state=args.seed,
    )

    results = results.sort_values("mean_accuracy", ascending=False)
    print("\nBest settings:")
    print(results.head(10).to_string(index=False))

    output_path = save_results(results, args.task, args.features)
    print(f"\nSaved CV results to {output_path}")


if __name__ == "__main__":
    main()
