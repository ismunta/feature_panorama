"""
stitching.py - Multi-image warping and blending for panorama construction.

Provides functions for:
- Computing panorama canvas from transformed corners
- Warping images to a common coordinate system using homographies
- Distance-transform-based weighted blending
- Cropping black borders
"""
import os
import cv2
import numpy as np


def warp_images_to_common_canvas(
    images: list[dict],
    homographies: dict[int, np.ndarray],
) -> dict:
    """Warp all images into a common coordinate system and compute canvas.

    Convention:
        homographies[i] maps image i -> reference image coordinate system.
        For the reference image itself, use np.eye(3).

    Args:
        images: List of preprocessed image dicts with 'color' key.
        homographies: Dict mapping image index to 3x3 homography matrix.

    Returns:
        Dictionary with warped images, masks, and canvas dimensions.
    """
    result = {
        "warped_images": [],
        "masks": [],
        "canvas_width": 0,
        "canvas_height": 0,
        "offset_x": 0,
        "offset_y": 0,
        "error": None,
    }

    if not images or not homographies:
        result["error"] = "No images or homographies provided"
        return result

    # Step 1: Transform all image corners to find canvas bounds
    all_corners = []
    for i, img in enumerate(images):
        if i not in homographies:
            continue
        h, w = img["color"].shape[:2]
        corners = np.array([
            [[0, 0]],
            [[w, 0]],
            [[w, h]],
            [[0, h]],
        ], dtype=np.float32)

        H = homographies[i]
        transformed = cv2.perspectiveTransform(corners, H)
        all_corners.append(transformed)

    if not all_corners:
        result["error"] = "No corners could be transformed"
        return result

    all_corners = np.concatenate(all_corners, axis=0)

    # Step 2: Compute canvas bounding box
    min_x = int(np.floor(all_corners[:, 0, 0].min()))
    min_y = int(np.floor(all_corners[:, 0, 1].min()))
    max_x = int(np.ceil(all_corners[:, 0, 0].max()))
    max_y = int(np.ceil(all_corners[:, 0, 1].max()))

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    # Step 3: Translation matrix to shift negative coordinates to positive
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1],
    ], dtype=np.float64)

    result["canvas_width"] = canvas_width
    result["canvas_height"] = canvas_height
    result["offset_x"] = min_x
    result["offset_y"] = min_y

    # Step 4: Warp each image
    warped_images = []
    masks = []

    for i, img in enumerate(images):
        if i not in homographies:
            continue

        H_composite = T @ homographies[i]

        warped = cv2.warpPerspective(
            img["color"], H_composite,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Create mask: 1 where warped image has valid pixels
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray_warped > 0).astype(np.uint8)

        # Dilate mask slightly to avoid edge artifacts
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        warped_images.append(warped)
        masks.append(mask)

    result["warped_images"] = warped_images
    result["masks"] = masks
    return result


def blend_warped_images(
    warped_images: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict:
    """Blend multiple warped images using distance-transform feathering.

    In overlap regions, each pixel is weighted by its distance to the
    nearest valid-image boundary, producing smooth transitions.

    Args:
        warped_images: List of warped BGR images (all same canvas size).
        masks: List of binary masks (1 = valid pixel).

    Returns:
        Dictionary with blended panorama and metadata.
    """
    result = {
        "panorama": None,
        "error": None,
    }

    if not warped_images:
        result["error"] = "No warped images to blend"
        return result

    n = len(warped_images)
    h, w = warped_images[0].shape[:2]

    # Compute distance-transform weights for each image
    weights = []
    for mask in masks:
        # Distance transform: distance from each pixel to nearest zero pixel
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        weights.append(dist.astype(np.float32))

    # Normalize weights so they sum to 1 at each pixel
    weight_sum = sum(weights)
    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-6)

    # Weighted blend
    panorama = np.zeros((h, w, 3), dtype=np.float32)

    for warped, w_dist in zip(warped_images, weights):
        norm_weight = w_dist / weight_sum
        # Expand to 3 channels
        norm_weight_3ch = np.stack([norm_weight] * 3, axis=-1)
        panorama += warped.astype(np.float32) * norm_weight_3ch

    panorama = np.clip(panorama, 0, 255).astype(np.uint8)

    # Crop black borders
    panorama = crop_black_borders(panorama)

    result["panorama"] = panorama
    return result


def crop_black_borders(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Crop black borders from a panorama.

    Finds the bounding box of non-black pixels and crops to that region.

    Args:
        image: BGR panorama image.
        threshold: Pixel value below which is considered black.

    Returns:
        Cropped image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Find bounding box of all non-black content
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + cw)
        y_max = max(y_max, y + ch)

    # Add small padding
    pad = 5
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(image.shape[1], x_max + pad)
    y_max = min(image.shape[0], y_max + pad)

    return image[y_min:y_max, x_min:x_max]
