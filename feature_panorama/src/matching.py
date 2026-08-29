"""
matching.py - Descriptor matching with Lowe's ratio test.

Provides functions for:
- BFMatcher creation (L2 for SIFT, Hamming for ORB)
- KNN matching
- Lowe's ratio test filtering
- Structured matching results
"""
import cv2
import numpy as np
import time


def create_matcher(method: str) -> cv2.BFMatcher:
    """Create a BFMatcher appropriate for the descriptor type.

    Args:
        method: "sift" or "orb".

    Returns:
        cv2.BFMatcher instance.

    Raises:
        ValueError: If method is not recognized.
    """
    if method.lower() == "sift":
        return cv2.BFMatcher(cv2.NORM_L2)
    elif method.lower() == "orb":
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sift' or 'orb'.")


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str,
    ratio_threshold: float = 0.75,
) -> dict:
    """Match descriptors between two images using KNN and Lowe's ratio test.

    Pipeline:
        descriptors -> KNN(k=2) -> ratio test -> good matches

    Args:
        desc1: Descriptors from image 1 (NxD float32 or binary).
        desc2: Descriptors from image 2.
        method: "sift" or "orb" (determines distance norm).
        ratio_threshold: Lowe's ratio test threshold.

    Returns:
        Dictionary with raw knn matches, good matches, and timing.
    """
    if desc1 is None or desc2 is None:
        return {
            "raw_knn_matches": 0,
            "initial_matches": 0,
            "matches": [],
            "matching_time": 0.0,
            "error": "One or both descriptor sets are None",
        }

    if len(desc1) < 2 or len(desc2) < 2:
        return {
            "raw_knn_matches": 0,
            "initial_matches": 0,
            "matches": [],
            "matching_time": 0.0,
            "error": "Insufficient descriptors for KNN(k=2)",
        }

    matcher = create_matcher(method)

    t0 = time.perf_counter()
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    elapsed = time.perf_counter() - t0

    # Apply Lowe's ratio test
    good_matches = []
    for m_pair in knn_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return {
        "raw_knn_matches": len(knn_matches),
        "initial_matches": len(good_matches),
        "matches": good_matches,
        "matching_time": elapsed,
        "error": None,
    }
