"""Compare strong candidate models with cross-validation.

This script is used after a promising feature set has been found. It compares a
small group of likely final models and records both mean and standard deviation
so we can judge accuracy and stability together.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Add the project root to Python's search path.
# This makes the script work both with "python -m" and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_task_data


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_NAMES = ["logistic_regression", "svm_rbf", "random_forest", "extra_trees"]


def get_feature_label(feature_names):
    """Create a short name for the selected feature set."""
    return "_".join(feature_names)


def format_depth(max_depth):
    """Convert max_depth into a readable value for printing and saving."""
    if max_depth is None:
        return "None"
    return str(max_depth)


def build_candidate_configs(model_names, n_estimators):
    """Create parameter configurations for the selected candidate models."""
    configs = []

    if "logistic_regression" in model_names:
        for c_value in [0.001, 0.003, 0.01, 0.03, 0.1]:
            configs.append(
                {
                    "model": "logistic_regression",
                    "C": c_value,
                    "gamma": None,
                    "max_depth": None,
                    "min_samples_leaf": None,
                    "max_features": None,
                    "n_estimators": None,
                }
            )

    if "svm_rbf" in model_names:
        for c_value in [1, 3, 10, 30, 100]:
            for gamma in ["scale", 0.0003, 0.001, 0.003, 0.01]:
                configs.append(
                    {
                        "model": "svm_rbf",
                        "C": c_value,
                        "gamma": gamma,
                        "max_depth": None,
                        "min_samples_leaf": None,
                        "max_features": None,
                        "n_estimators": None,
                    }
                )

    tree_depths = [8, 10, 15, 20, None]
    min_leaf_values = [1, 2, 4]
    max_feature_values = ["sqrt", "log2", 0.3]

    for tree_model in ["random_forest", "extra_trees"]:
        if tree_model not in model_names:
            continue

        for max_depth in tree_depths:
            for min_samples_leaf in min_leaf_values:
                for max_features in max_feature_values:
                    configs.append(
                        {
                            "model": tree_model,
                            "C": None,
                            "gamma": None,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "n_estimators": n_estimators,
                        }
                    )

    return configs


def build_model(config, random_state):
    """Create one classifier from a candidate configuration."""
    model_name = config["model"]

    if model_name == "logistic_regression":
        # Scale inputs because linear models depend on feature magnitudes.
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=config["C"],
                            solver="liblinear",
                            max_iter=2000,
                            random_state=random_state,
                        )
                    ),
                ),
            ]
        )

    if model_name == "svm_rbf":
        # RBF SVM is sensitive to feature scales, so use the same standard
        # scaling as the linear models.
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=config["C"],
                        gamma=config["gamma"],
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def cross_validate_model(model, X, y, cv):
    """Return fold-level and summary validation scores."""
    accuracies = []
    macro_f1_scores = []

    for train_index, valid_index in cv.split(X, y):
        # StratifiedKFold returns integer row positions, so use iloc.
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
        "std_accuracy": pd.Series(accuracies).std(ddof=0),
        "mean_macro_f1": sum(macro_f1_scores) / len(macro_f1_scores),
        "std_macro_f1": pd.Series(macro_f1_scores).std(ddof=0),
    }


def tune_candidate_models(task, feature_names, model_names, folds, n_estimators, random_state):
    """Run cross-validation for all candidate models and parameter settings."""
    task_data = load_task_data(task=task, feature_names=feature_names)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    configs = build_candidate_configs(model_names, n_estimators)
    results = []

    for config in configs:
        print(
            "Tuning "
            f"{config['model']}, C={config['C']}, gamma={config['gamma']}, "
            f"max_depth={format_depth(config['max_depth'])}, "
            f"min_samples_leaf={config['min_samples_leaf']}, "
            f"max_features={config['max_features']}",
            flush=True,
        )

        model = build_model(config, random_state)
        scores = cross_validate_model(model, task_data["X_train"], task_data["y_train"], cv)

        results.append(
            {
                "task": task,
                "model": config["model"],
                "features": "+".join(feature_names),
                "C": config["C"],
                "gamma": config["gamma"],
                "max_depth": format_depth(config["max_depth"]),
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"],
                "n_estimators": config["n_estimators"],
                "cv_folds": folds,
                "mean_accuracy": scores["mean_accuracy"],
                "std_accuracy": scores["std_accuracy"],
                "mean_macro_f1": scores["mean_macro_f1"],
                "std_macro_f1": scores["std_macro_f1"],
            }
        )

    return pd.DataFrame(results)


def save_results(results, task, feature_names):
    """Save candidate model tuning results as a CSV file."""
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{feature_label}_candidate_cv_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Tune candidate models with cross-validation.")
    parser.add_argument("--task", default="task2", help="Task folder under data/raw.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["hog", "additional", "deep_efficientnet_v2_l"],
        help="Feature sets to use.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Candidate models to tune.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees per forest.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Run candidate model cross-validation and print the best settings."""
    args = parse_args()

    results = tune_candidate_models(
        task=args.task,
        feature_names=args.features,
        model_names=args.models,
        folds=args.folds,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )

    results = results.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
    print("\nBest candidate settings:")
    print(results.head(15).to_string(index=False))

    output_path = save_results(results, args.task, args.features)
    print(f"\nSaved candidate CV results to {output_path}")


if __name__ == "__main__":
    main()
