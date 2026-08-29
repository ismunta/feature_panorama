"""
image_utils.py - Image loading, validation, and preprocessing.

Provides functions for:
- Automatic image discovery from a directory
- Image validation (existence, readability, dimensions)
- Grayscale conversion
- Aspect-ratio-aware resizing
- Full preprocessing pipeline
- Dataset summary reporting
"""
import os
import cv2
import numpy as np
import time
from typing import Optional


def discover_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")) -> list[str]:
    """Discover all supported image files in a directory, sorted by name.

    Args:
        directory: Path to the image directory.
        extensions: Tuple of valid file extensions.

    Returns:
        Sorted list of full file paths.
    """
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(extensions):
            files.append(os.path.join(directory, f))
    return files


def validate_image(path: str) -> dict:
    """Validate a single image file.

    Checks:
    - File exists
    - File is readable by OpenCV
    - Reports width, height, channels

    Returns:
        Dictionary with validation results.
    """
    result = {
        "path": path,
        "filename": os.path.basename(path),
        "exists": os.path.isfile(path),
        "readable": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "error": None,
    }

    if not result["exists"]:
        result["error"] = "File does not exist"
        return result

    try:
        img = cv2.imread(path)
        if img is None:
            result["error"] = "OpenCV cannot read the file (corrupt or unsupported format)"
            return result
        result["readable"] = True
        result["height"], result["width"] = img.shape[:2]
        result["channels"] = img.shape[2] if len(img.shape) == 3 else 1
    except Exception as e:
        result["error"] = str(e)

    return result


def load_image(path: str, flag: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Load an image from disk.

    Args:
        path: File path.
        flag: OpenCV load flag (default: BGR color).

    Returns:
        Image array or None on failure.
    """
    img = cv2.imread(path, flag)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale. Returns as-is if already single-channel."""
    if len(img.shape) == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    """Resize image so that width equals max_width, preserving aspect ratio.

    If the image is already narrower than max_width, it is returned unchanged.
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    ratio = max_width / w
    new_h = int(h * ratio)
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(
    path: str,
    max_width: int = 800,
) -> dict:
    """Load and preprocess a single image.

    Returns a dictionary with:
    - 'color': original colour image (possibly resized)
    - 'gray': grayscale version
    - 'path': file path
    - 'original_size': (width, height) before resize
    - 'final_size': (width, height) after resize
    - 'load_time': seconds taken to load and preprocess
    """
    t0 = time.perf_counter()

    color = load_image(path)
    if color is None:
        return {"path": path, "error": "Failed to load image"}

    original_size = (color.shape[1], color.shape[0])

    color = resize_to_width(color, max_width)
    gray = to_grayscale(color)

    final_size = (color.shape[1], color.shape[0])
    elapsed = time.perf_counter() - t0

    return {
        "path": path,
        "filename": os.path.basename(path),
        "color": color,
        "gray": gray,
        "original_size": original_size,
        "final_size": final_size,
        "load_time": elapsed,
    }


def print_dataset_summary(images: list[dict]) -> str:
    """Print and return a formatted dataset summary.

    Args:
        images: List of validation result dicts or preprocess dicts.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 50)
    lines.append(f"Number of images: {len(images)}")
    lines.append("")

    for i, img in enumerate(images, 1):
        lines.append(f"Image {i}:")
        lines.append(f"  Name:      {img.get('filename', img.get('name', 'N/A'))}")
        lines.append(f"  Width:     {img.get('width', img.get('final_size', (0, 0))[0])}")
        lines.append(f"  Height:    {img.get('height', img.get('final_size', (0, 0))[1])}")
        lines.append(f"  Channels:  {img.get('channels', 'N/A')}")
        if "error" in img and img["error"]:
            lines.append(f"  ERROR:     {img['error']}")
        lines.append("")

    lines.append("=" * 50)
    summary = "\n".join(lines)
    print(summary)
    return summary
