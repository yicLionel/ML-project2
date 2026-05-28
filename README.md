# COMP30027 Project 2

Coarse-to-fine-grained image classification project for COMP30027 Machine Learning.

This repository contains Task 1 coarse-grained animal classification and Task 2 fine-grained bird species classification experiments.

## Project Layout

```text
data/
  raw/
    task1/              # Original Task 1 files and generated feature CSVs
    task2/              # Original Task 2 files and generated feature CSVs
  processed/            # Optional merged feature tables for checking
src/
  data/                 # Data loading and feature merging
  features/             # Feature extraction scripts
  models/               # Training, tuning, and submission scripts
outputs/
  results/              # Validation CSVs used in the report
  figures/              # Report figures and figure source CSVs
  submissions/          # Final Kaggle-format submission CSVs
  models/               # Optional fitted models created by make_submission.py
report/                 # Report draft and report assets
```

## How to Run

Create and activate a clean Python virtual environment first. The commands
below create the environment in `.venv/`, which is ignored by git.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After installation, check that the main dependencies are available:

```bash
python -c "import torch, torchvision, sklearn; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

When you return to the project later, reactivate the same environment before running any scripts:

```bash
source .venv/bin/activate
```

The EfficientNet-V2-L and ResNet-50 feature CSV files are already included in the submitted zip under `data/raw/task1/` and `data/raw/task2/`. Therefore, the final models can be trained directly from the provided feature files. If these CSV files are missing in the evaluation environment, they can be regenerated from the raw images by running the feature extraction commands below. The local Torch cache for downloaded pretrained weights is not included in the submitted zip archive.

```bash
python src/features/extract_deep_features.py --task task1 --model efficientnet_v2_l
python src/features/extract_deep_features.py --task task2 --model efficientnet_v2_l
python src/features/extract_deep_features.py --task task1 --model resnet50 --batch-size 64
python src/features/extract_deep_features.py --task task2 --model resnet50 --batch-size 32
```

Run the final Task 1 model and validation:

```bash
python src/models/train_baselines.py --task task1 --features color hog additional deep_efficientnet_v2_l
```

Run the final Task 2 model and validation:

```bash
python src/models/train_baselines.py --task task2 --features hog additional deep_efficientnet_v2_l
```

## Model Selection and Hyperparameter Tuning

The final training and submission scripts already use the selected model settings. This section is only included to show how the hyperparameters were chosen during development. It can be skipped if the goal is only to reproduce the final validation results and Kaggle submissions.

The scripts in `src/models/` were used in two stages. First,
`train_baselines.py` compares the main classifier families on a held-out
validation split and writes a validation CSV to `outputs/results/`. Then the tuning scripts use stratified cross-validation to test selected hyperparameter grids.

Linear models:

```bash
python src/models/tune_linear_models.py --task task1 --features color hog additional deep_efficientnet_v2_l
```

This compares Logistic Regression and Linear SVM across several `C` values and
preprocessing choices. It writes:

```text
outputs/results/<task>_<features>_linear_cv_results.csv
```

Random Forest:

```bash
python src/models/tune_random_forest.py --task task2 --features hog additional deep_efficientnet_v2_l
```

This tunes `max_depth`, `min_samples_leaf`, `max_features`, and the number of
trees. It writes:

```text
outputs/results/<task>_<features>_random_forest_cv_results.csv
```

KNN:

```bash
python src/models/tune_knn.py --task task1 --features color hog additional deep_efficientnet_v2_l
```

This tests different values of `k`, distance weighting, and preprocessing
choices. It writes:

```text
outputs/results/<task>_<features>_knn_cv_results.csv
```

Candidate model comparison:

```bash
python src/models/tune_candidate_models.py --task task2 --features hog additional deep_efficientnet_v2_l
```

This compares likely final candidates, including Logistic Regression, RBF SVM,
Random Forest, and Extra Trees, and records mean and standard deviation across
folds. It writes:

```text
outputs/results/<task>_<features>_candidate_cv_results.csv
```

Reproduce the report figure:

```bash
python src/evaluation/plot_task2_confusion_matrix.py
```

This creates:

```text
outputs/figures/task2_random_forest_confusion_matrix.png
outputs/figures/task2_random_forest_confusion_matrix.csv
```

The report tables are based on the retained validation CSV files under
`outputs/results/`.

## Final Designed Models

The final designed models used for the report are:

| Task | Final model | Features | Validation result file | Kaggle prediction file |
| --- | --- | --- | --- | --- |
| Task 1 | Logistic Regression | Color + HOG + additional + EfficientNet-V2-L | `outputs/results/task1_color_hog_additional_deep_efficientnet_v2_l_baseline_results.csv` | `outputs/submissions/task1_logistic_regression_color_hog_additional_deep_efficientnet_v2_l_submission.csv` |
| Task 2 | Random Forest | HOG + additional + EfficientNet-V2-L | `outputs/results/task2_hog_additional_deep_efficientnet_v2_l_baseline_results.csv` | `outputs/submissions/task2_random_forest_hog_additional_deep_efficientnet_v2_l_submission.csv` |

The final Kaggle-format prediction files submitted to Kaggle are stored in
`outputs/submissions/`. They can be reproduced with:

```bash
python src/models/make_submission.py --task task1 --model logistic_regression --features color hog additional deep_efficientnet_v2_l
python src/models/make_submission.py --task task2 --model random_forest --features hog additional deep_efficientnet_v2_l
```

## Notes

- The pretrained CNN is used only as a generic ImageNet feature extractor. It is not trained or fine-tuned on CIFAR-10, CUB-200-2011, or the project data.
