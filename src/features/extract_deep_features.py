"""Extract ImageNet pretrained deep features for Task 1.

This script uses a ResNet18 model pretrained on ImageNet as a generic feature
extractor. It does not train or fine-tune the CNN on the project data. The saved
features can then be used by our normal classifiers such as Logistic Regression
or SVM.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


# Add the project root to Python's search path.
# This makes the script work both from the terminal and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Store downloaded ImageNet weights inside this project folder. This avoids
# writing to the user's home cache directory.
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

from src.data.load_data import DATA_ROOT, read_csv_checked


class ImageMetadataDataset(Dataset):
    """Dataset that loads images listed in a metadata CSV table."""

    def __init__(self, task_dir, metadata, transform):
        self.task_dir = task_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = self.task_dir / row["image_path"]

        # Convert to RGB because ImageNet models expect three colour channels.
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return row["image_id"], image


def choose_device():
    """Use Apple GPU if available, otherwise use CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet18_feature_extractor(device):
    """Load ImageNet pretrained ResNet18 without its final classifier."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Remove the final classification layer. The model output becomes a
    # 512-dimensional image representation instead of 1000 ImageNet classes.
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    return model, weights.transforms()


def extract_features(model, data_loader, device):
    """Run images through the CNN and collect feature vectors."""
    rows = []

    with torch.no_grad():
        for batch_index, (image_ids, images) in enumerate(data_loader):
            images = images.to(device)
            features = model(images).cpu().numpy()

            for image_id, feature_vector in zip(image_ids, features):
                row = {"image_id": image_id}
                for feature_index, value in enumerate(feature_vector):
                    row[f"deep_resnet18_{feature_index}"] = value
                rows.append(row)

            # Print progress occasionally because feature extraction can take a
            # few minutes on CPU.
            if (batch_index + 1) % 20 == 0:
                processed = (batch_index + 1) * data_loader.batch_size
                print(f"Processed about {processed} images...")

    return pd.DataFrame(rows)


def build_deep_feature_table(task, batch_size):
    """Extract ResNet18 features for all train and test images."""
    task_dir = DATA_ROOT / "raw" / task
    train_metadata = read_csv_checked(task_dir / "train_metadata.csv")
    test_metadata = read_csv_checked(task_dir / "test_metadata.csv")
    all_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)

    device = choose_device()
    print(f"Using device: {device}")

    model, transform = build_resnet18_feature_extractor(device)
    dataset = ImageMetadataDataset(task_dir, all_metadata, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return extract_features(model, data_loader, device)


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Extract ImageNet ResNet18 features.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per batch.")
    return parser.parse_args()


def main():
    """Extract deep features and save them as a CSV file."""
    args = parse_args()
    task_dir = DATA_ROOT / "raw" / args.task
    output_path = task_dir / "deep_resnet18_features.csv"

    features = build_deep_feature_table(args.task, args.batch_size)
    features.to_csv(output_path, index=False)

    print(f"Saved {features.shape[0]} rows and {features.shape[1] - 1} features")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
