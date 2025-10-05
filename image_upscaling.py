"""
upscale_dir.py – batch-upscale every image in a directory
keeps aspect ratio, uses the simpler RealESRGAN wrapper
6 GB VRAM friendly (half-precision + tiling)
"""

import torch
from PIL import Image
from pathlib import Path
from RealESRGAN import RealESRGAN

# -----------------------------------------------------------
# CONFIGURATION – change these two lines
SOURCE_DIR = (
    Path(__file__).parent / "media" / "images"
)  # folder that contains the images
DEST_DIR = (
    Path(__file__).parent / "media" / "images"
)  # folder to save the upscaled images
MODEL_SCALE = 2  # 2 or 4
# -----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model once
model = RealESRGAN(device, scale=MODEL_SCALE)
model.load_weights(
    f"/media/kornellewy/jan_dysk_3/auto_youtube/weights_upscaler/RealESRGAN_x{MODEL_SCALE}.pth",
    download=True,
)

DEST_DIR.mkdir(parents=True, exist_ok=True)

# process every image
for img_path in SOURCE_DIR.iterdir():
    if img_path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
    }:
        continue

    print(f"Upscaling {img_path.name} ...")

    img = Image.open(img_path).convert("RGB")
    if img.size[0] > 1000 or img.size[1] > 1000:
        print(f"Skipping {img_path.name} – too big")
        continue
    sr_img = model.predict(img)  # keeps aspect ratio automatically

    out_path = DEST_DIR / f"{img_path.stem}_x{MODEL_SCALE}{img_path.suffix}"
    sr_img.save(out_path)

print("Done – all images saved to", DEST_DIR)
