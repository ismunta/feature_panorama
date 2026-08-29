"""
visualization.py - Visualization utilities for input images and results.

Provides functions for:
- Displaying all input images in a single figure
- Drawing keypoints on images
- Drawing matches between image pairs
"""
import os
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_input_images(images: list[dict], save_path: str, dpi: int = 150):
    """Display all loaded input images side by side and save to file.

    Args:
        images: List of dicts with 'color' and 'filename' keys.
        save_path: Where to save the figure.
        dpi: Resolution.
    """
    n = len(images)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img_dict in zip(axes, images):
        bgr = img_dict["color"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(img_dict.get("filename", "Image"), fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved input visualization -> {save_path}")


def draw_keypoints_on_image(
    image: np.ndarray,
    keypoints: list,
    max_keypoints: int = 200,
) -> np.ndarray:
    """Draw detected keypoints on an image for visualization.

    Args:
        image: BGR colour image.
        keypoints: List of cv2.KeyPoint.
        max_keypoints: Limit drawn keypoints for readability.

    Returns:
        Image with keypoints drawn.
    """
    vis = image.copy()
    kp_to_draw = keypoints[:max_keypoints] if len(keypoints) > max_keypoints else keypoints
    return cv2.drawKeypoints(
        vis, kp_to_draw, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )
