"""
config.py - Central configuration for the panorama project.

All tunable parameters are defined here so that no magic numbers
appear in the rest of the codebase.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
TRANSFORMED_DIR = os.path.join(DATA_DIR, "transformed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

KEYPOINTS_DIR = os.path.join(RESULTS_DIR, "keypoints")
MATCHES_DIR = os.path.join(RESULTS_DIR, "matches")
RANSAC_DIR = os.path.join(RESULTS_DIR, "ransac")
HOMOGRAPHY_DIR = os.path.join(RESULTS_DIR, "homography")
PANORAMAS_DIR = os.path.join(RESULTS_DIR, "panoramas")
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────
MAX_IMAGE_WIDTH = 800          # Resize large images to this width
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
MIN_IMAGES_REQUIRED = 3        # Need at least this many overlapping images

# ──────────────────────────────────────────────
# Feature detection
# ──────────────────────────────────────────────
SIFT_NFEATURES = 0             # 0 = no limit
ORB_NFEATURES = 5000           # ORB cap on features

# ──────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────
RATIO_TEST_THRESHOLD = 0.75    # Lowe's ratio test threshold

# ──────────────────────────────────────────────
# RANSAC / Homography
# ──────────────────────────────────────────────
RANSAC_REPROJECTION_THRESHOLD = 5.0  # pixels
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.999

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
VIS_DPI = 150
