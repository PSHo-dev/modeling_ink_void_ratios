import csv
import os
import re
import sys

import cv2
import numpy as np

# Proportional split points within each image's actual brightness range.
# These are fractions of the range [darkest_pixel, brightest_pixel].
#
#   dark   → brightness <  dark/medium boundary
#   medium → brightness >= dark/medium boundary and < medium/light boundary
#   light  → brightness >= medium/light boundary
#
# Change these to shift the boundaries — values must be between 0 and 1,
# and DARK_MEDIUM_SPLIT must be less than MEDIUM_LIGHT_SPLIT.

DARK_MEDIUM_SPLIT = 0.333
MEDIUM_LIGHT_SPLIT = 0.750

#### visually, it looks like the below representation                                                   ####
#### low----------------dark_medium_split--------------------medium_light_split--------------------high ####

# Output folder for mask images
MASK_OUTPUT_DIR = "masks"

# Colors for each category (BGR format for OpenCV)
COLOR_LIGHT = (255, 0, 0)  # Blue
COLOR_MEDIUM = (0, 0, 255)  # Red
COLOR_DARK = (0, 255, 0)  # Green


def compute_thresholds(gray):
    """
    Derives absolute brightness thresholds using percentiles instead of
    min/max, so outlier pixels don't skew the boundaries.
    """
    # How far into the brightness distribution to anchor the range endpoints.
    # 2/98 means the bottom 2% and top 2% of pixels are treated as outliers.
    LOW_PERCENTILE = 5
    HIGH_PERCENTILE = 95
    low = np.percentile(gray, LOW_PERCENTILE)
    high = np.percentile(gray, HIGH_PERCENTILE)
    brightness_range = high - low

    # Edge case: nearly uniform image
    if brightness_range == 0:
        return low, low

    dark_threshold = low + DARK_MEDIUM_SPLIT * brightness_range
    light_threshold = low + MEDIUM_LIGHT_SPLIT * brightness_range
    return dark_threshold, light_threshold


def save_mask_image(dark_mask, medium_mask, light_mask, image_path, output_dir):
    """
    Creates a color-coded mask image:
      - Blue  → light
      - Red   → medium
      - Green  → dark
    """

    os.makedirs(output_dir, exist_ok=True)

    h, w = dark_mask.shape
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)

    mask_img[dark_mask] = COLOR_DARK
    mask_img[medium_mask] = COLOR_MEDIUM
    mask_img[light_mask] = COLOR_LIGHT

    base = os.path.basename(image_path)
    stem, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{stem}_mask{ext}")

    success = cv2.imwrite(out_path, mask_img)
    if success:
        print(f"  Mask saved → {out_path}")
    else:
        print(f"  [WARNING] Could not save mask to '{out_path}'.")
        return None

    return out_path


def categorize_pixels(image_path, mask_output_dir=MASK_OUTPUT_DIR):
    """
    Reads an image, converts to grayscale, derives per-image thresholds,
    and categorizes each pixel into dark / medium / light.

    Also exports a color-coded mask image to mask_output_dir.

    Returns (label, image_name, light_pct, medium_pct, dark_pct) or None on failure.
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARNING] Could not read '{image_path}'. Skipping.")
        return None
    # Apply contrast stretching using NORM_MINMAX
    # alpha is the lower boundary (0) and beta is the upper boundary (255)
    stretched_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    gray = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2HSV)

    lower_sat = np.array([25, 60, 0])
    upper_sat = np.array([179, 255, 255])

    dark_thresh, light_thresh = compute_thresholds(gray)

    # Categorize pixels
    color_mask = cv2.inRange(hsv, lower_sat, upper_sat) > 0
    dark_mask = (gray < dark_thresh) | color_mask
    light_mask = (gray >= light_thresh) & ~dark_mask
    medium_mask = ~dark_mask & ~light_mask

    dark_pixels = int(np.sum(dark_mask))
    light_pixels = int(np.sum(light_mask))
    medium_pixels = int(np.sum(medium_mask))

    total_pixels = gray.size

    light_pct = round(light_pixels / total_pixels, 3)
    medium_pct = round(medium_pixels / total_pixels, 3)
    dark_pct = round(dark_pixels / total_pixels, 3)

    image_name = os.path.basename(image_path)
    season = re.split(r"[ _]", image_name, maxsplit=1)[0]
    label = re.split(r"[ ]", image_name, maxsplit=1)[0]

    print(
        f"  Brightness range: {int(gray.min())}-{int(gray.max())}  |  "
        f"Thresholds → dark/medium: {dark_thresh:.1f}, medium/light: {light_thresh:.1f}"
    )
    print(
        f"  Light: {round(light_pct * 100.0, 2)}%  |  "
        f"Medium: {round(medium_pct * 100.0, 2)}%  |  "
        f"Dark: {round(dark_pct * 100.0, 2)}%"
    )

    # --- Save visualization ---
    save_mask_image(dark_mask, medium_mask, light_mask, image_path, mask_output_dir)

    return season, label, image_name, light_pct, medium_pct, dark_pct


def process_images(image_paths, output_csv="pixel_results.csv", mask_output_dir=MASK_OUTPUT_DIR):
    """
    Processes a list of image paths and writes results to a CSV file.
    Mask images are saved to mask_output_dir.
    """
    results = []
    for path in image_paths:
        print(f"Processing: {path}")
        result = categorize_pixels(path, mask_output_dir=mask_output_dir)
        if result:
            results.append(result)

    if not results:
        print("No valid images were processed.")
        return

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["season", "label", "image_name", "light_%", "medium_%", "dark_%"])
        writer.writerows(results)

    print(f"\nResults saved to : {output_csv}")
    print(f"Mask images saved to : {mask_output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(paths):
    """
    Accepts a mix of file paths and directory paths.
    Directories are scanned (non-recursively) for supported image files.
    Returns a flat list of image file paths.
    """
    image_files = []
    for path in paths:
        if os.path.isdir(path):
            found = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ]
            if not found:
                print(f"  [WARNING] No supported images found in directory '{path}'.")
            image_files.extend(found)
        elif os.path.isfile(path):
            image_files.append(path)
        else:
            print(f"  [WARNING] '{path}' is not a valid file or directory. Skipping.")
    return image_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pixel_darkness_masks.py <image_or_dir> [image_or_dir2] ...")
        print("Examples:")
        print("  python pixel_darkness_masks.py ./images/")
        print("  python pixel_darkness_masks.py photo.jpg ./scans/ archive.png")
        sys.exit(1)

    image_files = collect_images(sys.argv[1:])

    if not image_files:
        print("No valid images found. Exiting.")
        sys.exit(1)

    process_images(image_files, output_csv="pixel_results.csv", mask_output_dir=MASK_OUTPUT_DIR)
