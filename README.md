# COMP30027 Project 2

Coarse-to-fine-grained image classification project for COMP30027 Machine Learning.

## Project Layout

```text
data/
  raw/
    task1/              # Original Task 1 animal classification files
    task2/              # Original Task 2 bird classification files
  processed/            # Cleaned or merged feature tables
notebooks/              # Exploration and one-off experiments
src/
  data/                 # Data loading and dataset utilities
  features/             # Feature engineering and preprocessing
  models/               # Training and prediction scripts
  evaluation/           # Metrics, validation, confusion matrices
outputs/
  submissions/          # Kaggle submission CSV files
  models/               # Saved trained models
  figures/              # Plots for analysis and report
report/                 # Report draft, figures, references
```

## Suggested Workflow

1. Place the provided Task 1 files in `data/raw/task1/`.
2. Place the provided Task 2 files in `data/raw/task2/`.
3. Build baseline models using the provided CSV features.
4. Save validation results, confusion matrices, and Kaggle submissions under `outputs/`.
5. Keep report-ready figures in `outputs/figures/` or `report/`.

## Baseline

Run a first Task 1 baseline from the provided CSV features:

```bash
python3 -m src.models.baseline --task task1
```

This writes validation scores, per-model classification reports, confusion matrices,
the best fitted model, and a submission CSV under `outputs/`.
