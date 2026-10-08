"""
DecodeLabs Project 4
Path 1: Optical Character Recognition (OCR)

Pipeline:
1. Load image
2. Convert to grayscale
3. Gaussian blur
4. Adaptive threshold
5. Optional deskew
6. Tesseract OCR
7. Apply the mandatory 80% confidence gate
8. Create visual confirmation and text output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output


CONFIDENCE_THRESHOLD = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DecodeLabs Project 4 - OCR with OpenCV preprocessing"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated results."
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional path to the Tesseract executable."
    )
    return parser.parse_args()


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert the input to a clean black/white OCR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur reduces small noise before thresholding.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles uneven lighting/shadows better than
    # one global threshold value.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return binary


def deskew_image(binary: np.ndarray) -> np.ndarray:
    """Correct small text rotation so Tesseract receives a level baseline."""
    coords = np.column_stack(np.where(binary < 255))

    if len(coords) < 20:
        return binary

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Avoid rotating an already level image.
    if abs(angle) < 0.5:
        return binary

    height, width = binary.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def run_ocr(processed: np.ndarray) -> pd.DataFrame:
    """Run Tesseract and return word-level OCR data."""
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=Output.DATAFRAME,
    )

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce")

    data = data[(data["text"] != "") & data["conf"].notna()]
    return data


def accepted_detections(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only OCR words meeting the mandatory 80% gate."""
    return data[data["conf"] >= CONFIDENCE_THRESHOLD].copy()


def create_annotated_image(
    original: np.ndarray,
    accepted: pd.DataFrame,
) -> np.ndarray:
    """Draw accepted OCR words and confidence on the original image."""
    annotated = original.copy()

    for _, row in accepted.iterrows():
        x, y = int(row["left"]), int(row["top"])
        w, h = int(row["width"]), int(row["height"])
        text = str(row["text"])
        conf = float(row["conf"])

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 180, 0),
            2,
        )
        label = f"{text} ({conf:.0f}%)"

        # Keep label above the box when possible.
        label_y = max(20, y - 8)
        cv2.putText(
            annotated,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def build_text(accepted: pd.DataFrame) -> str:
    """Reconstruct accepted OCR text in reading order."""
    if accepted.empty:
        return ""

    # Tesseract gives line/block information; sort in reading order.
    ordered = accepted.sort_values(
        ["block_num", "par_num", "line_num", "left"]
    )

    lines = []
    current_key = None
    current_words = []

    for _, row in ordered.iterrows():
        key = (
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
        )

        if current_key is not None and key != current_key:
            if current_words:
                lines.append(" ".join(current_words))
            current_words = []

        current_key = key
        current_words.append(str(row["text"]))

    if current_words:
        lines.append(" ".join(current_words))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    # Pre-processing: grayscale -> Gaussian blur -> adaptive threshold.
    processed = preprocess_image(image)

    # Correct small rotation before OCR.
    processed = deskew_image(processed)

    data = run_ocr(processed)
    accepted = accepted_detections(data)

    # Save preprocessing output for technical validation.
    cv2.imwrite(str(output_dir / "processed.png"), processed)

    # Save visual confirmation.
    annotated = create_annotated_image(image, accepted)
    cv2.imwrite(str(output_dir / "annotated.png"), annotated)

    # Save final accepted OCR text.
    recognized_text = build_text(accepted)
    (output_dir / "recognized.txt").write_text(
        recognized_text + ("\n" if recognized_text else ""),
        encoding="utf-8",
    )

    # Save raw word-level confidence data for benchmarking/audit.
    data.to_csv(output_dir / "ocr_data.csv", index=False)

    print("=" * 60)
    print("DecodeLabs Project 4 - OCR")
    print("=" * 60)
    print(f"Input: {image_path}")
    print(f"Detected OCR words: {len(data)}")
    print(f"Accepted (>= {CONFIDENCE_THRESHOLD:.0f}%): {len(accepted)}")
    print(f"Rejected (< {CONFIDENCE_THRESHOLD:.0f}%): {len(data) - len(accepted)}")
    print()
    print("Final accepted text:")
    print(recognized_text or "[No text passed the 80% confidence gate]")
    print()
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
