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
  models/               # Saved fitted models
report/                 # Report draft and report assets
```

## How to Run

Install the dependencies first:

```bash
pip install -r requirements.txt
```

If `deep_efficientnet_v2_l_features.csv` is not already present, extract the EfficientNet-V2-L features first. 

```bash
python src/features/extract_deep_features.py --task task1 --model efficientnet_v2_l
python src/features/extract_deep_features.py --task task2 --model efficientnet_v2_l
```

Run the final Task 1 model and validation:

```bash
python src/models/train_baselines.py --task task1 --features color hog additional deep_efficientnet_v2_l
```

Run the final Task 2 model and validation:

```bash
python src/models/train_baselines.py --task task2 --features hog additional deep_efficientnet_v2_l
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

- `src/data/load_data.py` is normally called by the model scripts. It does not need to be run directly.
- If the EfficientNet-V2-L feature CSVs are missing, generate them with `src/features/extract_deep_features.py` before training.
- The pretrained CNN is used only as a generic ImageNet feature extractor. It is not trained or fine-tuned on CIFAR-10, CUB-200-2011, or the project data.
