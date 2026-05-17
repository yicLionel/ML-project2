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
  results/              # Validation and cross-validation result CSVs
  submissions/          # Kaggle submission CSV files
  models/               # Saved fitted models
report/                 # Report draft and report assets
```

## Main Workflow

The normal workflow for each task is:

```text
prepare raw data
  -> extract deep features
  -> tune linear models with cross-validation
  -> tune Random Forest when it is needed as a comparison model
  -> run baseline validation
  -> create Kaggle submission
```

Task 2 first-pass commands:

```bash
.venv/bin/python src/features/extract_deep_features.py --task task2 --model efficientnet_v2_m --batch-size 8
.venv/bin/python src/data/load_data.py --task task2 --features hog additional deep_efficientnet_v2_m
.venv/bin/python src/models/train_baselines.py --task task2 --features hog additional deep_efficientnet_v2_m
.venv/bin/python src/models/tune_linear_models.py --task task2 --features hog additional deep_efficientnet_v2_m --folds 5
.venv/bin/python src/models/tune_random_forest.py --task task2 --features hog additional deep_efficientnet_v2_m --folds 5
.venv/bin/python src/models/make_submission.py --task task2 --model logistic_regression --features hog additional deep_efficientnet_v2_m
```


## Notes

- `src/data/load_data.py` is normally called by the model scripts. It does not need to be run directly.
- `src/features/extract_image_features.py` contains earlier hand-crafted feature engineering experiments.
- The pretrained CNN is used only as a generic ImageNet feature extractor. It is not trained or fine-tuned on CIFAR-10, CUB-200-2011, or the project data.
