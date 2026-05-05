"""Extract extra image features for Task 1.

The provided CSV features are useful, but they may miss some colour and spatial
information from the raw images. This script creates another feature file named
engineered_features.csv that can be merged with the original provided features.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# Add the project root to Python's search path.
# This makes the script work both from the terminal and the IDE Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import DATA_ROOT, read_csv_checked


def normalised_histogram(values, bins, value_range):
    """Create a histogram and scale it so the bin counts sum to 1."""
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def channel_skew(values):
    """Calculate a simple skewness value for one image channel."""
    mean = values.mean()
    std = values.std()

    # If the channel is almost constant, skewness is not meaningful.
    if std < 1e-8:
        return 0.0

    return np.mean(((values - mean) / std) ** 3)


def extract_colour_features(rgb_image, hsv_image):
    """Extract global colour histograms and colour moments."""
    features = {}

    # HSV separates colour type from brightness, which can help with animals
    # that have distinctive colours.
    hsv_bins = {"h": 16, "s": 8, "v": 8}
    for channel_index, channel_name in enumerate(["h", "s", "v"]):
        hist = normalised_histogram(
            hsv_image[:, :, channel_index],
            bins=hsv_bins[channel_name],
            value_range=(0, 1),
        )
        for bin_index, value in enumerate(hist):
            features[f"hsv_{channel_name}_{bin_index}"] = value

    # Mean, standard deviation, and skewness describe the overall colour
    # distribution without using too many features.
    for image_name, image_array, channels in [
        ("rgb", rgb_image, ["r", "g", "b"]),
        ("hsv", hsv_image, ["h", "s", "v"]),
    ]:
        for channel_index, channel_name in enumerate(channels):
            values = image_array[:, :, channel_index].reshape(-1)
            features[f"{image_name}_{channel_name}_mean"] = values.mean()
            features[f"{image_name}_{channel_name}_std"] = values.std()
            features[f"{image_name}_{channel_name}_skew"] = channel_skew(values)

    return features


def extract_spatial_colour_features(rgb_image, grid_size=4):
    """Extract RGB mean and standard deviation from a small image grid."""
    features = {}
    height, width, _ = rgb_image.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Spatial colour features keep rough location information, such as whether
    # the object or background colour appears near the centre or corners.
    for row in range(grid_size):
        for col in range(grid_size):
            row_start = row * cell_height
            col_start = col * cell_width
            row_end = height if row == grid_size - 1 else row_start + cell_height
            col_end = width if col == grid_size - 1 else col_start + cell_width

            cell = rgb_image[row_start:row_end, col_start:col_end, :]
            for channel_index, channel_name in enumerate(["r", "g", "b"]):
                values = cell[:, :, channel_index].reshape(-1)
                features[f"grid_{row}_{col}_{channel_name}_mean"] = values.mean()
                features[f"grid_{row}_{col}_{channel_name}_std"] = values.std()

    return features


def extract_gray_features(gray_image):
    """Extract texture, edge, and low-resolution shape features."""
    features = {}

    # Basic grayscale statistics describe brightness and contrast.
    features["gray_mean"] = gray_image.mean()
    features["gray_std"] = gray_image.std()
    features["gray_min"] = gray_image.min()
    features["gray_max"] = gray_image.max()
    features["gray_p25"] = np.percentile(gray_image, 25)
    features["gray_p50"] = np.percentile(gray_image, 50)
    features["gray_p75"] = np.percentile(gray_image, 75)

    gray_hist = normalised_histogram(gray_image, bins=16, value_range=(0, 1))
    entropy = -np.sum(gray_hist * np.log2(gray_hist + 1e-12))
    features["gray_entropy"] = entropy

    # Use simple image gradients as a lightweight edge descriptor.
    grad_y, grad_x = np.gradient(gray_image)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = (np.arctan2(grad_y, grad_x) + np.pi) % np.pi

    features["edge_density"] = np.mean(magnitude > magnitude.mean())
    features["edge_mean"] = magnitude.mean()
    features["edge_std"] = magnitude.std()

    edge_hist, _ = np.histogram(
        angle,
        bins=8,
        range=(0, np.pi),
        weights=magnitude,
    )
    if edge_hist.sum() > 0:
        edge_hist = edge_hist / edge_hist.sum()

    for bin_index, value in enumerate(edge_hist):
        features[f"edge_angle_{bin_index}"] = value

    return features


def extract_thumbnail_features(image_path, size=16):
    """Flatten a small grayscale thumbnail to keep rough shape information."""
    image = Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    thumbnail = np.asarray(image, dtype=float).reshape(-1) / 255.0

    return {f"thumb_{index}": value for index, value in enumerate(thumbnail)}


def extract_features_for_image(image_path):
    """Extract all engineered features for one image."""
    rgb_pil = Image.open(image_path).convert("RGB")
    hsv_pil = rgb_pil.convert("HSV")

    rgb_image = np.asarray(rgb_pil, dtype=float) / 255.0
    hsv_image = np.asarray(hsv_pil, dtype=float) / 255.0
    gray_image = np.asarray(rgb_pil.convert("L"), dtype=float) / 255.0

    features = {}
    features.update(extract_colour_features(rgb_image, hsv_image))
    features.update(extract_spatial_colour_features(rgb_image))
    features.update(extract_gray_features(gray_image))
    features.update(extract_thumbnail_features(image_path))

    return features


def extract_features_for_metadata(task_dir, metadata):
    """Extract features for every image listed in a metadata table."""
    rows = []

    for row_index, row in metadata.iterrows():
        image_path = task_dir / row["image_path"]
        image_features = extract_features_for_image(image_path)
        image_features["image_id"] = row["image_id"]
        rows.append(image_features)

        # Print progress occasionally so long runs do not look frozen.
        if (row_index + 1) % 500 == 0:
            print(f"Processed {row_index + 1} images...")

    return pd.DataFrame(rows)


def build_engineered_features(task):
    """Create engineered features for both train and test images."""
    task_dir = DATA_ROOT / "raw" / task
    train_metadata = read_csv_checked(task_dir / "train_metadata.csv")
    test_metadata = read_csv_checked(task_dir / "test_metadata.csv")
    all_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)

    features = extract_features_for_metadata(task_dir, all_metadata)

    # Keep image_id as the first column to match the provided feature files.
    columns = ["image_id"] + [col for col in features.columns if col != "image_id"]
    return features[columns]


def parse_args():
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description="Extract extra image features for Task 1.")
    parser.add_argument("--task", default="task1", help="Task folder under data/raw.")
    return parser.parse_args()


def main():
    """Run feature extraction and save engineered_features.csv."""
    args = parse_args()
    task_dir = DATA_ROOT / "raw" / args.task
    output_path = task_dir / "engineered_features.csv"

    features = build_engineered_features(args.task)
    features.to_csv(output_path, index=False)

    print(f"Saved {features.shape[0]} rows and {features.shape[1] - 1} features")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
