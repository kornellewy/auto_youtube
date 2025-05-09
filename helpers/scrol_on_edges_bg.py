import cv2
import numpy as np
from pathlib import Path

# === Config ===
image_path = Path("/home/kornellewy/Desktop/CAPPELLA_SISTINA_Ceiling.jpg")
window_size = (200, 200)
scroll_speed = 2  # pixels per frame

# === Load image ===
img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

img_h, img_w, _ = img.shape
h, w = window_size


# === Rotation helper ===
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return rotated


# === Path logic ===
def generate_edge_positions(img_h, img_w, h, w, step):
    positions_with_angles = []

    # Top edge (left → right): 0°
    for x in range(0, img_w - w, step):
        positions_with_angles.append(((0, x), 180))

    # Right edge (top → bottom): 90°
    for y in range(0, img_h - h, step):
        positions_with_angles.append(((y, img_w - w), 270))

    # Bottom edge (right → left): 180°
    for x in range(img_w - w, 0, -step):
        positions_with_angles.append(((img_h - h, x), 0))

    # Left edge (bottom → top): 270°
    for y in range(img_h - h, 0, -step):
        positions_with_angles.append(((y, 0), 90))

    return positions_with_angles


positions_with_angles = generate_edge_positions(img_h, img_w, h, w, scroll_speed)

# === Display loop ===
cv2.namedWindow("Edge Scroll", cv2.WINDOW_AUTOSIZE)

idx = 0
while True:
    (y, x), angle = positions_with_angles[idx % len(positions_with_angles)]
    cropped = img[y : y + h, x : x + w]
    rotated = rotate_image(cropped, angle)

    cv2.imshow("Edge Scroll", rotated)

    key = cv2.waitKey(10)
    if key == 27:
        break

    idx += 1

cv2.destroyAllWindows()
