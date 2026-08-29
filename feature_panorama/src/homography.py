"""
homography.py - RANSAC-based homography estimation and reprojection error.

Provides functions for:
- Extracting corresponding point arrays from feature matches
- Estimating homography using cv2.findHomography with RANSAC
- Computing reprojection error statistics (mean, median, max)
- Saving/loading homography matrices
- Visualizing inlier matches after RANSAC
"""
import os
import cv2
import numpy as np
import time


def extract_corresponding_points(
    keypoints1: list,
    keypoints2: list,
    matches: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ordered point arrays from feature matches.

    Each match m maps keypoints1[m.queryIdx] -> keypoints2[m.trainIdx].
    We return two arrays of shape (N, 1, 2) suitable for cv2.findHomography.

    Args:
        keypoints1: Keypoints from image 1 (source).
        keypoints2: Keypoints from image 2 (destination).
        matches: List of cv2.DMatch objects.

    Returns:
        (src_pts, dst_pts) each of shape (N, 1, 2) as float32.
    """
    src_pts = []
    dst_pts = []
    for m in matches:
        src_pts.append(keypoints1[m.queryIdx].pt)
        dst_pts.append(keypoints2[m.trainIdx].pt)

    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    return src_pts, dst_pts


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    reproj_threshold: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
) -> dict:
    """Estimate homography using cv2.findHomography with RANSAC.

    Convention:
        H maps source points to destination points: dst = H * src
        i.e., image1 (source) -> image2 (destination)

    Args:
        src_pts: Source points, shape (N, 1, 2).
        dst_pts: Destination points, shape (N, 1, 2).
        reproj_threshold: RANSAC reprojection threshold in pixels.
        max_iters: Maximum RANSAC iterations.
        confidence: RANSAC confidence level.

    Returns:
        Dictionary with homography matrix, inlier mask, and metrics.
    """
    result = {
        "H": None,
        "success": False,
        "initial_matches": len(src_pts),
        "ransac_inliers": 0,
        "ransac_outliers": 0,
        "inlier_ratio": 0.0,
        "mean_reprojection_error": float("inf"),
        "median_reprojection_error": float("inf"),
        "max_reprojection_error": float("inf"),
        "error": None,
    }

    if len(src_pts) < 4:
        result["error"] = f"Need >= 4 correspondences, got {len(src_pts)}"
        return result

    t0 = time.perf_counter()

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC,
        reproj_threshold, None, max_iters, confidence,
    )

    elapsed = time.perf_counter() - t0
    result["computation_time"] = elapsed

    if H is None:
        result["error"] = "cv2.findHomography returned None"
        return result

    if mask is None:
        result["error"] = "RANSAC mask is None"
        return result

    # Count inliers
    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(np.sum(inlier_mask))
    n_outliers = len(inlier_mask) - n_inliers

    result["H"] = H
    result["success"] = True
    result["ransac_inliers"] = n_inliers
    result["ransac_outliers"] = n_outliers
    result["inlier_ratio"] = n_inliers / len(src_pts) if len(src_pts) > 0 else 0.0
    result["inlier_mask"] = inlier_mask

    # Compute reprojection error for inliers only
    errors = compute_reprojection_error(H, src_pts[inlier_mask], dst_pts[inlier_mask])
    result["mean_reprojection_error"] = float(np.mean(errors))
    result["median_reprojection_error"] = float(np.median(errors))
    result["max_reprojection_error"] = float(np.max(errors))

    return result


def compute_reprojection_error(
    H: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean reprojection error for each point pair.

    Args:
        H: 3x3 homography matrix.
        src_pts: Source points (N, 1, 2).
        dst_pts: Destination points (N, 1, 2).

    Returns:
        Array of per-point errors, shape (N,).
    """
    if len(src_pts) == 0:
        return np.array([])

    # Project source points using homography
    projected = cv2.perspectiveTransform(src_pts, H)

    # Euclidean distance between projected and destination
    diff = projected - dst_pts
    errors = np.sqrt(np.sum(diff ** 2, axis=2)).ravel()
    return errors


def save_homography(H: np.ndarray, path: str):
    """Save a 3x3 homography matrix as a tab-separated text file.

    Args:
        H: 3x3 homography matrix.
        path: Output file path.
    """
    np.savetxt(path, H, fmt="%.8f", delimiter="\t")


def load_homography(path: str) -> np.ndarray:
    """Load a homography matrix from a text file.

    Args:
        path: Input file path.

    Returns:
        3x3 numpy array.
    """
    return np.loadtxt(path, dtype=np.float64)


def draw_inlier_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    keypoints1: list,
    keypoints2: list,
    matches: list,
    inlier_mask: np.ndarray,
    max_draw: int = 200,
) -> np.ndarray:
    """Draw only the inlier matches after RANSAC.

    Args:
        image1: BGR image 1.
        image2: BGR image 2.
        keypoints1: Keypoints from image 1.
        keypoints2: Keypoints from image 2.
        matches: Original list of matches (before RANSAC filtering).
        inlier_mask: Boolean mask of length len(matches).
        max_draw: Maximum number of inlier matches to draw.

    Returns:
        Image with inlier correspondences drawn.
    """
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
    inlier_matches = inlier_matches[:max_draw]

    return cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
    )
