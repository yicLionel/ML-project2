"""Extract ImageNet pretrained deep features for Task 1.

This script uses an ImageNet pretrained CNN as a generic feature extractor. It
does not train or fine-tune the CNN on the project data. The saved features can
then be used by our normal classifiers such as Logistic Regression or SVM.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import efficientnet_v2_m, efficientnet_v2_s, resnet50


# Add the project root to Python's search path.
# This makes the script work both from the terminal and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Store downloaded ImageNet weights inside this project folder. This avoids
# writing to the user's home cache directory.
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

from src.data.load_data import DATA_ROOT, read_csv_checked


MODEL_CONFIGS = {
    # Each entry contains the torchvision constructor, pretrained weights, CSV
    # column prefix, and output filename for one feature extractor.
    "resnet50": {
        "builder": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "feature_prefix": "deep_resnet50",
        "output_file": "deep_resnet50_features.csv",
    },
    "efficientnet_v2_s": {
        "builder": efficientnet_v2_s,
        "weights": EfficientNet_V2_S_Weights.DEFAULT,
        "feature_prefix": "deep_efficientnet_v2_s",
        "output_file": "deep_efficientnet_v2_s_features.csv",
    },
    "efficientnet_v2_m": {
        "builder": efficientnet_v2_m,
        "weights": EfficientNet_V2_M_Weights.DEFAULT,
        "feature_prefix": "deep_efficientnet_v2_m",
        "output_file": "deep_efficientnet_v2_m_features.csv",
    },
}


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

        # Use the preprocessing attached to the pretrained weights, so the
        # input format matches how the ImageNet model was trained.
        image = self.transform(image)

        return row["image_id"], image


def choose_device():
    """Use Apple GPU if available, otherwise use CPU."""
    # MPS is available on many Apple Silicon machines and speeds up extraction.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_extractor(model_name, device):
    """Load an ImageNet pretrained CNN without its final classifier."""
    if model_name not in MODEL_CONFIGS:
        valid_names = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown deep model '{model_name}'. Choose from: {valid_names}")

    config = MODEL_CONFIGS[model_name]
    weights = config["weights"]
    model = config["builder"](weights=weights)

    # Remove the final classification layer. This gives us image features
    # instead of ImageNet class scores.
    if model_name == "resnet50":
        model.fc = torch.nn.Identity()
    elif model_name in ["efficientnet_v2_s", "efficientnet_v2_m"]:
        model.classifier = torch.nn.Identity()

    # Evaluation mode disables training behaviour such as dropout updates.
    model.eval()
    model.to(device)

    return model, weights.transforms()


def extract_features(model, data_loader, device, feature_prefix):
    """Run images through the CNN and collect feature vectors."""
    rows = []

    with torch.no_grad():
        for batch_index, (image_ids, images) in enumerate(data_loader):
            # Only forward passes are needed because the CNN is not fine-tuned.
            images = images.to(device)
            features = model(images).cpu().numpy()

            for image_id, feature_vector in zip(image_ids, features):
                row = {"image_id": image_id}
                for feature_index, value in enumerate(feature_vector):
                    row[f"{feature_prefix}_{feature_index}"] = value
                rows.append(row)

            # Print progress occasionally because feature extraction can take a
            # few minutes, especially for larger models.
            if (batch_index + 1) % 20 == 0:
                processed = (batch_index + 1) * data_loader.batch_size
                print(f"Processed about {processed} images...")

    return pd.DataFrame(rows)


def build_deep_feature_table(task, model_name, batch_size):
    """Extract deep features for all train and test images."""
    task_dir = DATA_ROOT / "raw" / task
    train_metadata = read_csv_checked(task_dir / "train_metadata.csv")
    test_metadata = read_csv_checked(task_dir / "test_metadata.csv")

    # Extract features for both train and test images so they can be merged
    # through the same load_data.py path later.
    all_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Using model: {model_name}")

    model, transform = build_feature_extractor(model_name, device)
    dataset = ImageMetadataDataset(task_dir, all_metadata, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    feature_prefix = MODEL_CONFIGS[model_name]["feature_prefix"]
    return extract_features(model, data_loader, device, feature_prefix)


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Extract ImageNet pretrained CNN features.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    parser.add_argument(
        "--model",
        default="efficientnet_v2_m",
        choices=list(MODEL_CONFIGS.keys()),
        help="ImageNet pretrained model to use.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Images per batch.")
    return parser.parse_args()


def main():
    """Extract deep features and save them as a CSV file."""
    args = parse_args()
    task_dir = DATA_ROOT / "raw" / args.task
    output_path = task_dir / MODEL_CONFIGS[args.model]["output_file"]

    features = build_deep_feature_table(args.task, args.model, args.batch_size)
    features.to_csv(output_path, index=False)

    print(f"Saved {features.shape[0]} rows and {features.shape[1] - 1} features")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
