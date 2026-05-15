# COMP30027 Project 2

Coarse-to-fine-grained image classification project for COMP30027 Machine Learning.

This repository currently focuses on Task 1: coarse-grained animal classification.

## Project Layout

```text
data/
  raw/
    task1/              # Original Task 1 files and generated feature CSVs
    task2/              # Reserved for Task 2 files
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

The normal Task 1 workflow is:

```text
prepare raw data
  -> extract deep features
  -> tune linear models with cross-validation
  -> run baseline validation
  -> create Kaggle submission
```


## Notes

- `src/data/load_data.py` is normally called by the model scripts. It does not need to be run directly.
- `src/features/extract_image_features.py` contains earlier hand-crafted feature engineering experiments.
- The pretrained CNN is used only as a generic ImageNet feature extractor. It is not trained or fine-tuned on CIFAR-10, CUB-200-2011, or the project data.
