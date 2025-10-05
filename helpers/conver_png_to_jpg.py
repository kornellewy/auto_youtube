#!/usr/bin/env python3
"""
Convert every PIL-readable image in a folder to JPEG (quality 95).
No command-line args – just edit TARGET_DIR.
Original files are kept; JPEGs are saved alongside with .jpg extension.
"""
from pathlib import Path
from PIL import Image

TARGET_DIR = Path(
    "/media/kornellewy/jan_dysk_3/auto_youtube/media/raw/"
)  # <— change only this
QUALITY = 95

for file in TARGET_DIR.iterdir():
    if file.suffix.lower() in {".png"}:
        try:
            with Image.open(file) as im:
                rgb = im.convert("RGB")  # JPEG has no alpha
                out = file.with_suffix(".jpg")
                rgb.save(out, format="JPEG", quality=QUALITY, optimize=True)
                print(f"saved  {out.name}")
                file.unlink()  # remove original PNG
        except Exception as e:
            print(f"skip {file.name}  ({e})")

print("All done ✔")
