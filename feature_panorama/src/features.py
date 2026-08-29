"""
features.py - SIFT and ORB feature detection and description.

Provides functions for:
- Creating SIFT and ORB detectors
- Detecting keypoints and computing descriptors
- Measuring extraction time
- Returning structured results for each method
"""
import cv2
import numpy as np
import time


def create_sift_detector(nfeatures: int = 0) -> cv2.SIFT:
    """Create a SIFT detector.

    Args:
        nfeatures: Max features to detect (0 = unlimited).

    Returns:
        cv2.SIFT instance.
    """
    return cv2.SIFT_create(nfeatures=nfeatures)


def create_orb_detector(nfeatures: int = 5000) -> cv2.ORB:
    """Create an ORB detector.

    Args:
        nfeatures: Max features to detect.

    Returns:
        cv2.ORB instance.
    """
    return cv2.ORB_create(nfeatures=nfeatures)


def detect_and_describe(
    detector,
    gray: np.ndarray,
    method_name: str = "unknown",
) -> dict:
    """Detect keypoints and compute descriptors for a single image.

    Args:
        detector: OpenCV feature detector (SIFT or ORB).
        gray: Grayscale input image.
        method_name: Label for results.

    Returns:
        Dictionary with keypoints, descriptors, shape, timing, etc.
    """
    t0 = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    elapsed = time.perf_counter() - t0

    n_kp = len(keypoints)
    desc_shape = descriptors.shape if descriptors is not None else None

    return {
        "method": method_name,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "num_keypoints": n_kp,
        "descriptor_shape": desc_shape,
        "descriptor_rows": desc_shape[0] if desc_shape is not None else 0,
        "descriptor_cols": desc_shape[1] if desc_shape is not None else 0,
        "extraction_time": elapsed,
    }
