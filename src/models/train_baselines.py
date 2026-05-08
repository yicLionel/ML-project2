"""Train simple baseline models for Task 1.

The goal of this script is to get reliable first results, not to find the best
possible model yet. These results will be useful as a starting point for the
report and later model improvements.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


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


def get_deep_feature_names(feature_names):
    """Return deep feature names from the selected feature list."""
    return [name for name in feature_names if name.startswith("deep_")]


def print_experiment_summary(task, feature_names, task_data):
    """Print the main settings before model training starts."""
    deep_features = get_deep_feature_names(feature_names)
    deep_feature_text = ", ".join(deep_features) if deep_features else "none"

    # This makes VSCode runs easier to check because command line arguments are
    # not always visible in the terminal output.
    print("Experiment setup")
    print(f"Task: {task}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Deep feature extractor: {deep_feature_text}")
    print(
        f"Training data: {task_data['X_train'].shape[0]} rows, "
        f"{task_data['X_train'].shape[1]} features"
    )
    print(f"Test data: {task_data['X_test'].shape[0]} rows")
    print()


def build_models(random_state):
    """Create the baseline models used in the first experiment."""
    # Models based on distances, margins, or feature weights need scaling
    # because their results depend on the size of the numeric feature values.
    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    # liblinear is fast for this feature size, so we wrap it in
                    # one-vs-rest to handle the 10-class task.
                    OneVsRestClassifier(
                        LogisticRegression(
                            solver="liblinear",
                            max_iter=2000,
                            C=0.03,
                            random_state=random_state,
                        )
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    # dual=False is faster when the number of samples is larger
                    # than the number of features after scaling.
                    LinearSVC(
                        C=0.001,
                        dual=False,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                # RBF SVM is kept as a non-linear comparison model.
                ("model", SVC(kernel="rbf", C=25, gamma="scale")),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "gaussian_nb": Pipeline(
            [
                ("scaler", StandardScaler()),
                # Naive Bayes is a simple probabilistic baseline.
                ("model", GaussianNB()),
            ]
        ),
        # Tree-based models do not require scaling because they split features
        # by thresholds rather than using distances or dot products.
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
        ),
    }
    return models


def evaluate_model(model, X_train, X_valid, y_train, y_valid):
    """Train one model and return validation metrics."""
    # Fit the model only on the training split, then evaluate on validation.
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    # Macro F1 treats all classes equally, which is useful for multi-class data.
    return {
        "accuracy": accuracy_score(y_valid, predictions),
        "macro_f1": f1_score(y_valid, predictions, average="macro"),
    }


def run_baselines(task, feature_names, test_size, random_state):
    """Load data, split train/validation data, and evaluate all baselines."""
    task_data = load_task_data(task=task, feature_names=feature_names)
    print_experiment_summary(task, feature_names, task_data)

    # Use a stratified split so each class keeps a similar proportion in the
    # training and validation sets.
    X_train, X_valid, y_train, y_valid = train_test_split(
        task_data["X_train"],
        task_data["y_train"],
        test_size=test_size,
        random_state=random_state,
        stratify=task_data["y_train"],
    )

    results = []
    models = build_models(random_state)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        scores = evaluate_model(model, X_train, X_valid, y_train, y_valid)

        # Store results in a list first, then convert it to a DataFrame later.
        results.append(
            {
                "task": task,
                "model": model_name,
                "features": "+".join(feature_names),
                "valid_accuracy": scores["accuracy"],
                "valid_macro_f1": scores["macro_f1"],
            }
        )

        print(
            f"{model_name}: "
            f"accuracy={scores['accuracy']:.4f}, "
            f"macro_f1={scores['macro_f1']:.4f}"
        )

    return pd.DataFrame(results)


def save_results(results, task, feature_names):
    """Save validation results as a CSV file."""
    # Keep experiment outputs separate from source code and raw data.
    output_dir = OUTPUT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{feature_label}_baseline_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Train Task 1 baseline models.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    # Default to the current strongest feature combination. Use --features all
    # only for broad comparison runs because it includes every registered file.
    parser.add_argument(
        "--features",
        nargs="+",
        default=["color", "hog", "additional", "deep_efficientnet_v2_m"],
        help="Feature sets to use: all, color, hog, additional, deep_resnet50, deep_efficientnet_v2_s, deep_efficientnet_v2_m.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Run baseline training from the command line."""
    args = parse_args()

    # Run the full baseline experiment with the selected feature sets.
    results = run_baselines(
        task=args.task,
        feature_names=args.features,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("\nValidation results:")
    print(results.sort_values("valid_accuracy", ascending=False).to_string(index=False))

    # Save the table so it can be used later in the report.
    output_path = save_results(results, args.task, args.features)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
