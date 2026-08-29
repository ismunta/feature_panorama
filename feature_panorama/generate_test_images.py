"""
generate_test_images.py - Create synthetic overlapping test images.
Run once to populate data/input/ with test images.
"""
import cv2
import numpy as np
import os

def generate_scene(width=1600, height=900):
    """Generate a synthetic scene with diverse features."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 2):
        t = y / (height // 2)
        img[y, :] = [int(200 - 60 * t), int(180 - 40 * t), int(100 + 80 * t)]

    # Ground
    img[height // 2:, :] = [80, 140, 60]

    # Buildings
    buildings = [(100, 200, 200, 450), (350, 150, 300, 450),
                 (650, 250, 180, 450), (900, 180, 220, 450),
                 (1150, 220, 200, 450)]
    for x, w, h_top, h_bot in buildings:
        color = (np.random.randint(100, 200), np.random.randint(80, 160), np.random.randint(60, 140))
        cv2.rectangle(img, (x, h_top), (x + w, h_bot), color, -1)
        # Windows
        for wy in range(h_top + 20, h_bot - 20, 40):
            for wx in range(x + 15, x + w - 15, 35):
                cv2.rectangle(img, (wx, wy), (wx + 20, wy + 25), (200, 220, 255), -1)

    # Road
    cv2.rectangle(img, (0, height - 120), (width, height - 40), (60, 60, 60), -1)
    for x in range(0, width, 80):
        cv2.rectangle(img, (x, height - 85), (x + 40, height - 75), (255, 255, 255), -1)

    # Trees
    for tx in [50, 300, 580, 850, 1100, 1400]:
        cv2.rectangle(img, (tx - 5, 350), (tx + 5, 450), (40, 30, 20), -1)
        cv2.circle(img, (tx, 320), 40, (20, 100, 30), -1)

    # Sun
    cv2.circle(img, (1400, 80), 50, (0, 200, 255), -1)

    # Text
    cv2.putText(img, "CITY VIEW", (600, 850), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)

    # Random feature points (dots for texture)
    rng = np.random.RandomState(42)
    for _ in range(800):
        x, y = rng.randint(0, width), rng.randint(0, height)
        r = rng.randint(1, 5)
        c = tuple(int(v) for v in rng.randint(50, 255, 3))
        cv2.circle(img, (x, y), r, c, -1)

    # Diagonal lines
    for i in range(0, width + height, 60):
        cv2.line(img, (i, 0), (max(0, i - height), height), (150, 150, 150), 1)

    return img


def create_overlapping_images(scene, output_dir):
    """Create 3 overlapping crops from the scene."""
    h, w = scene.shape[:2]
    crop_w = int(w * 0.5)   # each crop is 50% of scene width
    overlap = int(w * 0.15) # 15% additional overlap
    step = crop_w - overlap  # 35% step

    for i in range(3):
        x_start = i * step
        x_end = x_start + crop_w
        if x_end > w:
            x_start = w - crop_w
            x_end = w

        crop = scene[:, x_start:x_end].copy()

        # Simulate slight illumination/rotation differences
        if i == 1:
            crop = cv2.convertScaleAbs(crop, alpha=1.08, beta=8)
        if i == 2:
            M = cv2.getRotationMatrix2D((crop.shape[1] // 2, crop.shape[0] // 2), 1.5, 1.0)
            crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT)

        path = os.path.join(output_dir, f"image{i+1}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {path} ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "input")
    os.makedirs(out, exist_ok=True)
    print("Generating synthetic scene...")
    scene = generate_scene()
    print("Creating overlapping images...")
    create_overlapping_images(scene, out)
    print("Done.")
