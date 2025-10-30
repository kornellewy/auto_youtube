import cv2
from pathlib import Path

# --- CONFIGURATION (edit these two lines only) ---
TARGET_DIR = (
    Path(__file__).parent.parent / "media" / "slides_backgound"
)  # <— change only this  # folder that holds your images
TARGET_RES = (1920, 1080)  # (width, height) you want
# -------------------------------------------------

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

for img_path in TARGET_DIR.rglob("*"):
    if img_path.suffix.lower() not in SUPPORTED:
        continue

    img = cv2.imread(str(img_path))
    if img is None:  # skip broken files
        continue

    h, w = img.shape[:2]
    # enlarge ➜ INTER_CUBIC (best quality for upscaling)
    # shrink  ➜ INTER_AREA   (best quality for downscaling)
    interp = (
        cv2.INTER_CUBIC if (TARGET_RES[0] > w or TARGET_RES[1] > h) else cv2.INTER_AREA
    )

    resized = cv2.resize(img, TARGET_RES, interpolation=interp)
    cv2.imwrite(str(img_path), resized)  # overwrite original
