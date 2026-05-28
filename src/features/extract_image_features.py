"""Extract engineered image features for the project tasks.

The provided CSV features are useful, but they may miss some colour, texture,
and spatial layout information from the raw images. This script creates an
optional feature file named engineered_features.csv for extra experiments.
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


def extract_hsv_histogram_features(hsv_image):
    """Extract global HSV colour histograms."""
    features = {}

    # HSV separates colour type from brightness. This is useful for Task 2
    # birds because species often differ by head, wing, or body colour.
    hsv_bins = {"h": 24, "s": 12, "v": 12}
    for channel_index, channel_name in enumerate(["h", "s", "v"]):
        hist = normalised_histogram(
            hsv_image[:, :, channel_index],
            bins=hsv_bins[channel_name],
            value_range=(0, 1),
        )
        for bin_index, value in enumerate(hist):
            features[f"hsv_{channel_name}_{bin_index}"] = value

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


def extract_lbp_texture_features(gray_image):
    """Extract a Local Binary Pattern texture histogram."""
    features = {}

    # LBP compares each pixel with its 8 neighbours. It can capture simple
    # feather texture, spots, and streaks without training another model.
    center = gray_image[1:-1, 1:-1]
    lbp_codes = np.zeros(center.shape, dtype=np.uint8)

    neighbours = [
        gray_image[:-2, :-2],
        gray_image[:-2, 1:-1],
        gray_image[:-2, 2:],
        gray_image[1:-1, 2:],
        gray_image[2:, 2:],
        gray_image[2:, 1:-1],
        gray_image[2:, :-2],
        gray_image[1:-1, :-2],
    ]

    for bit_index, neighbour in enumerate(neighbours):
        lbp_codes += ((neighbour >= center) << bit_index).astype(np.uint8)

    lbp_hist = normalised_histogram(lbp_codes.reshape(-1), bins=256, value_range=(0, 256))
    for bin_index, value in enumerate(lbp_hist):
        features[f"lbp_{bin_index}"] = value

    return features


def extract_features_for_image(image_path):
    """Extract all engineered features for one image."""
    rgb_pil = Image.open(image_path).convert("RGB")
    hsv_pil = rgb_pil.convert("HSV")

    rgb_image = np.asarray(rgb_pil, dtype=float) / 255.0
    hsv_image = np.asarray(hsv_pil, dtype=float) / 255.0
    gray_image = np.asarray(rgb_pil.convert("L"), dtype=float) / 255.0

    features = {}
    features.update(extract_hsv_histogram_features(hsv_image))
    features.update(extract_spatial_colour_features(rgb_image))
    features.update(extract_lbp_texture_features(gray_image))

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
    parser = argparse.ArgumentParser(description="Extract engineered image features.")
    parser.add_argument("--task", default="task2", help="Task folder under data/raw.")
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
