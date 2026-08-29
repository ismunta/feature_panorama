"""
main.py - Entry point for Feature-Based Panorama Construction.

Phase 1: Image loading, validation, preprocessing, dataset report
Phase 2: SIFT & ORB feature detection and descriptor extraction
Phase 3: Descriptor matching with Lowe's ratio test
Phase 4: RANSAC, homography estimation, reprojection error
Phase 5: Multi-image homography composition and warping
Phase 6: Image blending and final panorama
"""
import os
import sys
import csv
import cv2
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.image_utils import (
    discover_images, validate_image, preprocess_image,
    print_dataset_summary,
)
from src.visualization import visualize_input_images, draw_keypoints_on_image
from src.features import create_sift_detector, create_orb_detector, detect_and_describe
from src.matching import match_descriptors
from src.homography import (
    extract_corresponding_points, estimate_homography,
    save_homography, draw_inlier_matches,
)
from src.stitching import warp_images_to_common_canvas, blend_warped_images


def ensure_directories():
    """Create all output directories if they don't exist."""
    for d in [
        config.KEYPOINTS_DIR, config.MATCHES_DIR, config.RANSAC_DIR,
        config.HOMOGRAPHY_DIR, config.PANORAMAS_DIR, config.EXPERIMENTS_DIR,
        config.PLOTS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def phase1_discover_and_preprocess():
    """Phase 1: Discover, validate, and preprocess input images.

    Returns:
        List of preprocessed image dicts, or exits if insufficient images.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Image Discovery, Validation, and Preprocessing")
    print("=" * 60)

    # 1. Discover images
    image_paths = discover_images(config.INPUT_DIR, config.SUPPORTED_EXTENSIONS)
    print(f"\nDiscovered {len(image_paths)} image(s) in {config.INPUT_DIR}")

    # 2. Validate
    validations = []
    for p in image_paths:
        v = validate_image(p)
        validations.append(v)

    valid = [v for v in validations if v["readable"]]
    print(f"Valid images: {len(valid)}/{len(validations)}")

    for v in validations:
        status = "OK" if v["readable"] else f"FAIL: {v['error']}"
        print(f"  {v['filename']}: {status}")

    # 3. Check minimum requirement
    if len(valid) < config.MIN_IMAGES_REQUIRED:
        print(f"\nERROR: At least {config.MIN_IMAGES_REQUIRED} overlapping images required.")
        print("Please add images to data/input/ and re-run.")
        sys.exit(1)

    # 4. Preprocess
    print(f"\nPreprocessing (max_width={config.MAX_IMAGE_WIDTH})...")
    images = []
    for v in valid:
        res = preprocess_image(v["path"], max_width=config.MAX_IMAGE_WIDTH)
        if "error" in res:
            print(f"  WARNING: {v['filename']}: {res['error']}")
            continue
        images.append(res)
        print(f"  {res['filename']}: {res['original_size'][0]}x{res['original_size'][1]}"
              f" -> {res['final_size'][0]}x{res['final_size'][1]}")

    # 5. Dataset summary
    summary_data = []
    for img in images:
        summary_data.append({
            "filename": img["filename"],
            "width": img["final_size"][0],
            "height": img["final_size"][1],
            "channels": img["color"].shape[2] if len(img["color"].shape) == 3 else 1,
        })
    print_dataset_summary(summary_data)

    # 6. Visualization
    vis_path = os.path.join(config.RESULTS_DIR, "phase1_input_images.png")
    visualize_input_images(images, vis_path, dpi=config.VIS_DPI)

    print("\nPhase 1 complete.")
    return images


def phase2_feature_detection(images: list[dict]) -> dict:
    """Phase 2: Detect keypoints and extract descriptors using SIFT and ORB.

    Args:
        images: List of preprocessed image dicts.

    Returns:
        Dictionary with results keyed by (method, image_index).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: SIFT and ORB Feature Detection")
    print("=" * 60)

    sift = create_sift_detector(nfeatures=config.SIFT_NFEATURES)
    orb = create_orb_detector(nfeatures=config.ORB_NFEATURES)

    results = {}
    csv_rows = []

    for i, img in enumerate(images):
        gray = img["gray"]
        fname = img["filename"]

        for method_name, detector in [("sift", sift), ("orb", orb)]:
            print(f"\n  {method_name.upper()} on {fname}...")
            res = detect_and_describe(detector, gray, method_name=method_name)
            key = (method_name, i)
            results[key] = res

            print(f"    Keypoints:     {res['num_keypoints']}")
            print(f"    Descriptor:    {res['descriptor_shape']}")
            print(f"    Extract time:  {res['extraction_time']:.4f}s")

            csv_rows.append({
                "method": method_name,
                "image": fname,
                "number_of_keypoints": res["num_keypoints"],
                "descriptor_rows": res["descriptor_rows"],
                "descriptor_cols": res["descriptor_cols"],
                "feature_extraction_time": f"{res['extraction_time']:.6f}",
            })

            # Save keypoint visualization
            kp_vis = draw_keypoints_on_image(img["color"], res["keypoints"])
            vis_name = f"{method_name}_{os.path.splitext(fname)[0]}.jpg"
            vis_path = os.path.join(config.KEYPOINTS_DIR, vis_name)
            cv2.imwrite(vis_path, kp_vis)
            print(f"    Saved:         {vis_path}")

    # Save CSV
    csv_path = os.path.join(config.EXPERIMENTS_DIR, "phase2_features.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "image", "number_of_keypoints",
            "descriptor_rows", "descriptor_cols", "feature_extraction_time"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Metrics saved to {csv_path}")

    print("\nPhase 2 complete.")
    return results


def phase3_descriptor_matching(images: list[dict], feature_results: dict) -> dict:
    """Phase 3: Match descriptors between adjacent overlapping images.

    Args:
        images: List of preprocessed image dicts.
        feature_results: Dict from phase2.

    Returns:
        Dictionary with matching results keyed by (method, pair_index).
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Descriptor Matching with Lowe's Ratio Test")
    print("=" * 60)

    n = len(images)
    if n < 2:
        print("  Not enough images for matching.")
        return {}

    results = {}
    csv_rows = []

    for method in ["sift", "orb"]:
        print(f"\n  --- {method.upper()} matching ---")

        for pair_idx in range(n - 1):
            i1, i2 = pair_idx, pair_idx + 1
            fname1 = images[i1]["filename"]
            fname2 = images[i2]["filename"]
            pair_label = f"{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}"

            desc1 = feature_results[(method, i1)]["descriptors"]
            desc2 = feature_results[(method, i2)]["descriptors"]

            print(f"\n  Pair: {fname1} <-> {fname2}")

            match_res = match_descriptors(
                desc1, desc2,
                method=method,
                ratio_threshold=config.RATIO_TEST_THRESHOLD,
            )

            if match_res["error"]:
                print(f"    WARNING: {match_res['error']}")
            else:
                print(f"    Raw KNN matches:    {match_res['raw_knn_matches']}")
                print(f"    After ratio test:   {match_res['initial_matches']}")
                print(f"    Matching time:      {match_res['matching_time']:.4f}s")

            key = (method, pair_idx)
            results[key] = match_res

            csv_rows.append({
                "method": method,
                "image_pair": pair_label,
                "raw_knn_matches": match_res["raw_knn_matches"],
                "initial_matches": match_res["initial_matches"],
                "matching_time": f"{match_res['matching_time']:.6f}",
            })

            # Save match visualization
            kp1 = feature_results[(method, i1)]["keypoints"]
            kp2 = feature_results[(method, i2)]["keypoints"]
            matches = match_res["matches"]

            vis = cv2.drawMatches(
                images[i1]["color"], kp1,
                images[i2]["color"], kp2,
                matches[:200], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            vis_name = f"{method}_{pair_label}_before_ransac.jpg"
            vis_path = os.path.join(config.MATCHES_DIR, vis_name)
            cv2.imwrite(vis_path, vis)
            print(f"    Saved:             {vis_path}")

    # Save CSV
    csv_path = os.path.join(config.EXPERIMENTS_DIR, "phase3_matches.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "image_pair", "raw_knn_matches",
            "initial_matches", "matching_time"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Metrics saved to {csv_path}")

    print("\nPhase 3 complete.")
    return results


def phase4_ransac_homography(
    images: list[dict],
    feature_results: dict,
    match_results: dict,
) -> dict:
    """Phase 4: RANSAC-based homography estimation and reprojection error.

    Args:
        images: List of preprocessed image dicts.
        feature_results: Dict from phase2.
        match_results: Dict from phase3.

    Returns:
        Dictionary with homography results keyed by (method, pair_index).
    """
    print("\n" + "=" * 60)
    print("PHASE 4: RANSAC, Homography Estimation, and Reprojection Error")
    print("=" * 60)

    n = len(images)
    results = {}
    csv_rows = []

    for method in ["sift", "orb"]:
        print(f"\n  --- {method.upper()} RANSAC + Homography ---")

        for pair_idx in range(n - 1):
            i1, i2 = pair_idx, pair_idx + 1
            fname1 = images[i1]["filename"]
            fname2 = images[i2]["filename"]
            pair_label = f"{os.path.splitext(fname1)[0]}_{os.path.splitext(fname2)[0]}"

            print(f"\n  Pair: {fname1} -> {fname2}")

            # Get matches
            match_key = (method, pair_idx)
            match_res = match_results.get(match_key)
            if match_res is None or match_res["error"]:
                print(f"    Skipping: no valid matches")
                continue

            matches = match_res["matches"]
            initial_matches = match_res["initial_matches"]

            if initial_matches < 4:
                print(f"    Too few matches ({initial_matches}) for homography")
                results[(method, pair_idx)] = {
                    "success": False,
                    "error": f"Only {initial_matches} matches (need >= 4)",
                    "initial_matches": initial_matches,
                    "ransac_inliers": 0,
                    "inlier_ratio": 0.0,
                }
                continue

            # Extract corresponding points
            kp1 = feature_results[(method, i1)]["keypoints"]
            kp2 = feature_results[(method, i2)]["keypoints"]
            src_pts, dst_pts = extract_corresponding_points(kp1, kp2, matches)

            # Estimate homography with RANSAC
            h_res = estimate_homography(
                src_pts, dst_pts,
                reproj_threshold=config.RANSAC_REPROJECTION_THRESHOLD,
                max_iters=config.RANSAC_MAX_ITERS,
                confidence=config.RANSAC_CONFIDENCE,
            )

            if not h_res["success"]:
                print(f"    FAILED: {h_res['error']}")
                results[(method, pair_idx)] = h_res
                continue

            # Print results
            print(f"    Initial matches:   {h_res['initial_matches']}")
            print(f"    RANSAC inliers:    {h_res['ransac_inliers']}")
            print(f"    RANSAC outliers:   {h_res['ransac_outliers']}")
            print(f"    Inlier ratio:      {h_res['inlier_ratio']:.4f}")
            print(f"    Mean reproj err:   {h_res['mean_reprojection_error']:.2f} px")
            print(f"    Median reproj err: {h_res['median_reprojection_error']:.2f} px")
            print(f"    Max reproj err:    {h_res['max_reprojection_error']:.2f} px")
            print(f"    Time:              {h_res['computation_time']:.4f}s")

            results[(method, pair_idx)] = h_res

            # Save homography matrix
            h_path = os.path.join(config.HOMOGRAPHY_DIR, f"{method}_H_{pair_label}.txt")
            save_homography(h_res["H"], h_path)
            print(f"    Saved H:           {h_path}")

            # Save after-RANSAC visualization
            vis = draw_inlier_matches(
                images[i1]["color"], images[i2]["color"],
                kp1, kp2, matches,
                h_res["inlier_mask"],
                max_draw=200,
            )
            vis_name = f"{method}_{pair_label}_after_ransac.jpg"
            vis_path = os.path.join(config.RANSAC_DIR, vis_name)
            cv2.imwrite(vis_path, vis)
            print(f"    Saved RANSAC vis:  {vis_path}")

            csv_rows.append({
                "method": method,
                "image_pair": pair_label,
                "initial_matches": h_res["initial_matches"],
                "ransac_inliers": h_res["ransac_inliers"],
                "ransac_outliers": h_res["ransac_outliers"],
                "inlier_ratio": f"{h_res['inlier_ratio']:.6f}",
                "mean_reprojection_error": f"{h_res['mean_reprojection_error']:.4f}",
                "median_reprojection_error": f"{h_res['median_reprojection_error']:.4f}",
                "max_reprojection_error": f"{h_res['max_reprojection_error']:.4f}",
                "computation_time": f"{h_res['computation_time']:.6f}",
            })

    # Save CSV
    csv_path = os.path.join(config.EXPERIMENTS_DIR, "phase4_ransac.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "image_pair", "initial_matches",
            "ransac_inliers", "ransac_outliers", "inlier_ratio",
            "mean_reprojection_error", "median_reprojection_error",
            "max_reprojection_error", "computation_time",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Metrics saved to {csv_path}")

    print("\nPhase 4 complete.")
    return results


def phase5_warp_images(
    images: list[dict],
    feature_results: dict,
    match_results: dict,
    homography_results: dict,
) -> dict:
    """Phase 5: Multi-image homography composition and warping.

    Uses the middle image as reference:
        image1 -> image2 <- image3

    Args:
        images: List of preprocessed image dicts.
        feature_results: Dict from phase2.
        match_results: Dict from phase3.
        homography_results: Dict from phase4.

    Returns:
        Dictionary with warped images, masks, and canvas info.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Multi-Image Homography Composition and Warping")
    print("=" * 60)

    n = len(images)
    if n < 2:
        print("  Not enough images for warping.")
        return {}

    warped_results = {}

    for method in ["sift", "orb"]:
        print(f"\n  --- {method.upper()} warping ---")

        # Check homography results for this method
        # For 3 images: H_01 maps image0->image1, H_21 maps image2->image1
        # We use image1 (middle) as reference
        ref_idx = 1  # Middle image is reference

        # Collect homographies: maps image_i -> reference
        homographies = {}

        # Identity for reference image
        homographies[ref_idx] = np.eye(3, dtype=np.float64)

        # H_01 maps image0 -> image1
        pair_01 = (method, 0)  # pair_idx=0 means images[0]->images[1]
        if pair_01 in homography_results and homography_results[pair_01]["success"]:
            homographies[0] = homography_results[pair_01]["H"]
            print(f"    H(0->1): OK")
        else:
            print(f"    H(0->1): FAILED - skipping")
            continue

        # H_21 maps image2 -> image1
        pair_12 = (method, 1)  # pair_idx=1 means images[1]->images[2]
        if pair_12 in homography_results and homography_results[pair_12]["success"]:
            # H_12 maps image1->image2, so H_21 = inv(H_12)
            H_12 = homography_results[pair_12]["H"]
            H_21 = np.linalg.inv(H_12)
            homographies[2] = H_21
            print(f"    H(2->1): OK (inverted from H_12)")
        else:
            print(f"    H(2->1): FAILED - skipping")
            continue

        # Warp all images to common canvas
        print(f"    Warping {n} images to common canvas...")
        warp_res = warp_images_to_common_canvas(images, homographies)

        if warp_res["error"]:
            print(f"    Warping FAILED: {warp_res['error']}")
            continue

        print(f"    Canvas size: {warp_res['canvas_width']}x{warp_res['canvas_height']}")
        print(f"    Warped images: {len(warp_res['warped_images'])}")

        warped_results[method] = warp_res

        # Save intermediate warped images
        inter_dir = os.path.join(config.PANORAMAS_DIR, "intermediate")
        os.makedirs(inter_dir, exist_ok=True)

        for i, warped in enumerate(warp_res["warped_images"]):
            fname = f"{method}_warped_image{i}.jpg"
            path = os.path.join(inter_dir, fname)
            cv2.imwrite(path, warped)
            print(f"    Saved: {path}")

        # Save masks
        for i, mask in enumerate(warp_res["masks"]):
            fname = f"{method}_mask_image{i}.jpg"
            path = os.path.join(inter_dir, fname)
            cv2.imwrite(path, mask * 255)
            print(f"    Saved mask: {path}")

    print("\nPhase 5 complete.")
    return warped_results


def phase6_blend_panorama(
    images: list[dict],
    warped_results: dict,
) -> dict:
    """Phase 6: Image blending and final panorama construction.

    Args:
        images: List of preprocessed image dicts.
        warped_results: Dict from phase5.

    Returns:
        Dictionary with final panorama paths and metadata.
    """
    print("\n" + "=" * 60)
    print("PHASE 6: Image Blending and Final Panorama")
    print("=" * 60)

    final_results = {}

    for method in ["sift", "orb"]:
        if method not in warped_results:
            print(f"\n  Skipping {method.upper()}: no warped images available")
            continue

        print(f"\n  --- {method.upper()} blending ---")
        warp_res = warped_results[method]

        # Blend the warped images
        blend_res = blend_warped_images(
            warp_res["warped_images"],
            warp_res["masks"],
        )

        if blend_res["error"]:
            print(f"    Blending FAILED: {blend_res['error']}")
            continue

        panorama = blend_res["panorama"]

        # Save final panorama
        pano_path = os.path.join(config.PANORAMAS_DIR, f"{method}_panorama.jpg")
        cv2.imwrite(pano_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"    Saved panorama: {pano_path}")

        # Print metadata
        h, w = panorama.shape[:2]
        total_pixels = h * w
        valid_pixels = int(np.sum(panorama.sum(axis=2) > 0))
        valid_pct = 100.0 * valid_pixels / total_pixels if total_pixels > 0 else 0

        print(f"    Dimensions:     {w}x{h}")
        print(f"    Valid pixels:   {valid_pct:.1f}%")
        print(f"    Input images:   {len(warp_res['warped_images'])}")

        final_results[method] = {
            "path": pano_path,
            "width": w,
            "height": h,
            "valid_pixel_percentage": valid_pct,
            "num_images": len(warp_res["warped_images"]),
        }

    # Save metadata CSV
    csv_path = os.path.join(config.EXPERIMENTS_DIR, "phase6_panorama.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "panorama_width", "panorama_height",
            "valid_pixel_percentage", "num_input_images",
        ])
        writer.writeheader()
        for method, res in final_results.items():
            writer.writerow({
                "method": method,
                "panorama_width": res["width"],
                "panorama_height": res["height"],
                "valid_pixel_percentage": f"{res['valid_pixel_percentage']:.2f}",
                "num_input_images": res["num_images"],
            })
    print(f"\n  Metadata saved to {csv_path}")

    print("\nPhase 6 complete.")
    return final_results


def main():
    """Run the full panorama pipeline (Phases 1-6)."""
    print("Feature-Based Panorama Construction")
    print("Phases 1-6: Full Pipeline")

    ensure_directories()

    # ── Phase 1 ──
    images = phase1_discover_and_preprocess()

    # ── Phase 2 ──
    feature_results = phase2_feature_detection(images)

    # ── Phase 3 ──
    match_results = phase3_descriptor_matching(images, feature_results)

    # ── Phase 4 ──
    homography_results = phase4_ransac_homography(
        images, feature_results, match_results
    )

    # ── Phase 5 ──
    warped_results = phase5_warp_images(
        images, feature_results, match_results, homography_results
    )

    # ── Phase 6 ──
    final_results = phase6_blend_panorama(images, warped_results)

    print("\n" + "=" * 60)
    print("ALL PHASES (1-6) COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("  - Keypoint visualizations:  results/keypoints/")
    print("  - Before-RANSAC matches:     results/matches/")
    print("  - After-RANSAC matches:      results/ransac/")
    print("  - Homography matrices:       results/homography/")
    print("  - Intermediate warps:        results/panoramas/intermediate/")
    print("  - Final panoramas:           results/panoramas/")
    print("  - All metrics:               results/experiments/")


if __name__ == "__main__":
    main()
