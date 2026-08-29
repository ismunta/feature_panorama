# Feature-Based Image Matching and Automatic Panorama Construction

University-level computer vision project implementing the classical pipeline:

**Feature Detection → Description → Matching → RANSAC → Homography → Alignment → Panorama**

## Comparison

| Method | Descriptor Type | Distance Metric |
|--------|----------------|-----------------|
| SIFT   | Float (128-D)  | L2 / Euclidean  |
| ORB    | Binary (256-bit) | Hamming       |

## Project Structure

```
feature_panorama/
├── data/input/          # Input overlapping images
├── data/transformed/    # Warping output (Phase 5)
├── results/             # All outputs
│   ├── keypoints/       # Keypoint visualizations
│   ├── matches/         # Match visualizations
│   ├── ransac/          # RANSAC inlier visualizations
│   ├── homography/      # Homography matrices
│   ├── panoramas/       # Final stitched panoramas
│   ├── experiments/     # Experimental CSVs
│   └── plots/           # Comparison plots
├── src/                 # Source modules
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Input Requirements

Place at least 3 overlapping JPEG/PNG images of the same scene in `data/input/`.
