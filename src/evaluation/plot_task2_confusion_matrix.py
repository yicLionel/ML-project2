"""Create a Task 2 validation confusion matrix figure.

The figure uses the same validation setup as the baseline result table:
HOG + additional features + EfficientNet-V2-L, stratified 80/20 split, seed 42.
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_task_data
from src.models.train_baselines import build_models


FEATURES = ["hog", "additional", "deep_efficientnet_v2_l"]
MODEL_NAME = "random_forest"
SEED = 42
TEST_SIZE = 0.2
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
PNG_PATH = OUTPUT_DIR / "task2_random_forest_confusion_matrix.png"
CSV_PATH = OUTPUT_DIR / "task2_random_forest_confusion_matrix.csv"


def get_font(size):
    """Load a readable system font, falling back to Pillow's default font."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def short_label(label):
    """Make class labels compact enough for the report figure."""
    return label.replace("_", " ").replace("American ", "Am. ").replace("Red winged", "Red-winged")


def draw_matrix(matrix, labels, output_path):
    """Draw a labelled confusion matrix as a PNG without matplotlib."""
    n_classes = len(labels)
    cell = 45
    left = 145
    top = 45
    right = 25
    bottom = 145
    width = left + n_classes * cell + right
    height = top + n_classes * cell + bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_font = get_font(11)
    number_font = get_font(13)

    max_value = max(1, int(matrix.max()))
    draw.text((left + n_classes * cell / 2 - 42, top + n_classes * cell + 118), "Predicted label", fill="black", font=label_font)
    draw.text((8, top - 24), "True label", fill="black", font=label_font)

    for i, true_label in enumerate(labels):
        y = top + i * cell
        draw.text((8, y + 17), short_label(true_label), fill="black", font=label_font)

    for j, pred_label in enumerate(labels):
        x = left + j * cell
        label = short_label(pred_label)
        temp = Image.new("RGBA", (112, 20), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp)
        temp_draw.text((0, 0), label, fill="black", font=label_font)
        temp = temp.rotate(45, expand=True)
        image.paste(temp, (x - 7, top + n_classes * cell + 5), temp)

    for i in range(n_classes):
        for j in range(n_classes):
            value = int(matrix[i, j])
            intensity = int(255 - 185 * (value / max_value))
            fill = (intensity, intensity + 15 if intensity < 240 else 255, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=(210, 210, 210))
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=number_font)
            tx = x0 + (cell - (bbox[2] - bbox[0])) / 2
            ty = y0 + (cell - (bbox[3] - bbox[1])) / 2
            draw.text((tx, ty), text, fill="black", font=number_font)

    image.save(output_path)


def main():
    """Train the selected model on the validation split and save the matrix."""
    task_data = load_task_data(task="task2", feature_names=FEATURES)
    X_train, X_valid, y_train, y_valid = train_test_split(
        task_data["X_train"],
        task_data["y_train"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=task_data["y_train"],
    )

    model = build_models(SEED)[MODEL_NAME]
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    class_ids = sorted(task_data["class_names"])
    labels = [task_data["class_names"][class_id] for class_id in class_ids]
    matrix = confusion_matrix(y_valid, predictions, labels=class_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(CSV_PATH)
    draw_matrix(matrix, labels, PNG_PATH)
    print(f"Saved confusion matrix PNG to {PNG_PATH}")
    print(f"Saved confusion matrix CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
