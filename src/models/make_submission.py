"""Create a Kaggle submission file for Task 1.

Kaggle requires exactly two columns:
image_id,class_id
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


# Add the project root to Python's search path.
# This helps the script run from both the terminal and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_task_data
from src.models.train_baselines import build_models


OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def get_feature_label(feature_names):
    """Create a short name for the selected feature set."""
    return "_".join(feature_names)


def train_model_for_submission(task, model_name, feature_names, random_state):
    """Train one selected model on the full training set."""
    # Reuse the same data loading function as the validation script, so the
    # submission is made from the same feature representation.
    task_data = load_task_data(task=task, feature_names=feature_names)

    # Build all baseline models first, then select the one requested by name.
    # This keeps model settings consistent with train_baselines.py.
    models = build_models(random_state)

    if model_name not in models:
        valid_names = ", ".join(models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {valid_names}")

    model = models[model_name]

    # For Kaggle, train on all labelled training data before predicting test data.
    model.fit(task_data["X_train"], task_data["y_train"])

    return model, task_data


def save_submission(task_data, predictions, task, model_name, feature_names):
    """Save predictions in the exact two-column Kaggle format."""
    # Kaggle submission files are kept separate from validation result files.
    output_dir = OUTPUT_ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The competition page requires exactly these two columns:
    # image_id and class_id. Do not include class_name here.
    submission = pd.DataFrame(
        {
            "image_id": task_data["test_metadata"]["image_id"],
            "class_id": predictions,
        }
    )

    feature_label = get_feature_label(feature_names)
    output_path = output_dir / f"{task}_{model_name}_{feature_label}_submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def save_model(model, task, model_name, feature_names):
    """Save the trained model so we know which model made the submission."""
    # Saving the model is useful for checking later which trained model produced
    # a particular Kaggle file.
    output_dir = OUTPUT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_label = get_feature_label(feature_names)
    model_path = output_dir / f"{task}_{model_name}_{feature_label}.joblib"
    joblib.dump(model, model_path)
    return model_path


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Create a Task 1 Kaggle submission file.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    # The default is Logistic Regression because it currently has the best
    # Kaggle score among our submitted Task 1 models.
    parser.add_argument(
        "--model",
        default="logistic_regression",
        help="Model name from train_baselines.py, such as logistic_regression or svm_rbf.",
    )
    # By default, use all provided CSV features.
    parser.add_argument(
        "--features",
        nargs="+",
        default=["all"],
        help="Feature sets to use: all, color, hog, additional, engineered, deep_resnet18.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    """Train the selected model and write the submission CSV."""
    args = parse_args()

    # Train the chosen model on the full labelled training set.
    model, task_data = train_model_for_submission(
        task=args.task,
        model_name=args.model,
        feature_names=args.features,
        random_state=args.seed,
    )

    # Predict class ids for the unlabelled test images.
    predictions = model.predict(task_data["X_test"])

    # Save both the Kaggle file and the fitted model for reproducibility.
    submission_path = save_submission(task_data, predictions, args.task, args.model, args.features)
    model_path = save_model(model, args.task, args.model, args.features)

    print(f"Saved submission to {submission_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
