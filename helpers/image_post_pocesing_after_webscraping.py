"""
python, pathlib, cv2, pil,uuid, python program that load all images and movies from given directory,
for images it will chek if they corupted, hash them and remove dulpicats, use ocr like easyocr to extract text from images, and use this text and uuid to save it to jpg file
for movies it will extract frames, hash them and remove duplicates, use ocr like easyocr to extract text from frames, and use this text and uuid to save it to gif file, ignoring audio

"""

#!/usr/bin/env python3
"""
media_ocr_dedup.py  –  image & movie OCR + duplicate removal
usage: python media_ocr_dedup.py <input_dir> <output_dir>
"""

import argparse
import hashlib
import os
import uuid
from pathlib import Path
from typing import List, Tuple
import time
import subprocess
import shlex

import cv2
import easyocr
from PIL import Image, ImageFile
from tqdm import tqdm
import google.generativeai as genai
import ffmpeg
import numpy as np
from ffmpeg import FFmpeg


from dotenv import load_dotenv

load_dotenv()

ImageFile.LOAD_TRUNCATED_IMAGES = False  # raise on corrupt images


SUPPORTED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_VID_EXT = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".webm",
    ".webp",
}
FRAME_INTERVAL_SEC = 1  # extract 1 frame per second from videos
GIF_HEIGHT = 320  # px; keep aspect ratio.  Set -1 to disable scaling
GIF_FPS = 10


# --------------------------------------------------------------------------- #
#                               Helper Routines                               #
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_pil(im: Image.Image) -> str:
    """Fast SHA-256 for PIL images (lossless PNG in memory)."""
    return sha256_bytes(im.tobytes())


def sha256_cv2(im) -> str:
    """Fast SHA-256 for OpenCV ndarray."""
    return sha256_bytes(im.tobytes())


def sanitize_filename(text: str, max_len: int = 30) -> str:
    """Keep alphanumerics and spaces, truncate."""
    safe = "".join(c for c in text if c.isalnum() or c in " -_.").rstrip()
    return (safe[:max_len] or "no_text").strip()


# --------------------------------------------------------------------------- #
#                           EasyOCR  (GPU if available)                       #
# --------------------------------------------------------------------------- #
reader = easyocr.Reader(["en"], gpu=True)  # add more languages here


def ocr_text(im) -> str:
    """Return concatenated OCR text from PIL image or cv2 ndarray."""
    if isinstance(im, Image.Image):
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    results = reader.readtext(im, detail=0)
    return " ".join(results).strip()


# --------------------------------------------------------------------------- #
#                              Image Pipeline                                 #
# --------------------------------------------------------------------------- #
def process_images(root: Path, out_dir: Path, hashes: set):
    files = [p for p in root.glob("*") if p.suffix.lower() in SUPPORTED_IMG_EXT]
    for path in tqdm(files, desc="Images"):
        try:
            with Image.open(path) as img:
                img.verify()  # quick corruption check
            with Image.open(path) as img:
                img = img.convert("RGB")
                h = sha256_pil(img)
                if h in hashes:
                    continue
                hashes.add(h)
                text = ocr_text(img)
                uid = uuid.uuid4().hex
                name = sanitize_filename(text)
                out_name = f"{uid}_{name}.jpg"
                img.save(out_dir / out_name, quality=95)
                normal_description, _ = describe_image(path)
                (out_dir / f"{uid}_{name}.txt").write_text(
                    normal_description, encoding="utf-8"
                )
                time.sleep(10)  # avoid overwhelming the filesystem
        except Exception as e:
            tqdm.write(f"Skip corrupted image {path}: {e}")


def describe_image(image_path, model_name="gemini-2.5-flash-lite-preview-06-17"):
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(model_name)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Attempt to describe as a normal image first
        normal_prompt = "Provide a concise, high-level overview or background caption for this image."
        response_normal = model.generate_content(
            [normal_prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        normal_description = response_normal.text.strip()

        # Attempt to describe as a graph, focusing on specific details
        graph_prompt = (
            "Analyze this image as if it were a graph or chart. "
            "Identify the main topic, what is being measured or represented, how the data is presented (e.g., bar chart, line graph), "
            "and if possible, what are the approximate maximum and minimum values or trends. "
            "Conclude with a brief statement about what the graph signifies or means."
        )
        response_graph = model.generate_content(
            [graph_prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        graph_description = response_graph.text.strip()

        return normal_description, graph_description

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


# --------------------------------------------------------------------------- #
#                              Movie Pipeline                                 #
# --------------------------------------------------------------------------- #


def process_movies(root: Path, out_dir: Path, hashes: set):
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_VID_EXT]
    print("number of files :", len(files))
    for path in tqdm(files, desc="Movies"):
        frame_idx = 0
        first_image = None
        first_image_path = out_dir / "tmp.jpg"

        # ---------- 2. Whole-video GIF ----------
        name_stub = uuid.uuid4().hex

        gif_path = (out_dir / f"{name_stub}.gif").with_suffix(".gif")

        webp_to_gif(path, gif_path)

        # --- 3. google describe 1 image is engught
        # save 1 frame
        cap = cv2.VideoCapture(str(gif_path))
        if not cap.isOpened():
            tqdm.write(f"Cannot open video {gif_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(fps * FRAME_INTERVAL_SEC)))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx == 0 and ok:
                first_image = frame.copy()
                cv2.imwrite(str(first_image_path), first_image)

            frame_idx += 1
        cap.release()

        if first_image_path.exists():
            normal_description, _ = describe_image(first_image_path)
            (gif_path.with_suffix(".txt")).write_text(
                normal_description, encoding="utf-8"
            )

        if normal_description is None or not first_image_path.exists():
            gif_path.unlink(missing_ok=True)

        first_image_path.unlink(missing_ok=True)


def webp_to_gif(src: Path, dst: Path):
    with Image.open(src) as im:
        im.save(
            dst,
            format="GIF",
            save_all=True,
            loop=0,
            optimize=True,
            duration=im.info.get("duration", 100),
        )


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main():
    INPUT_DIR = Path(__file__).parent.parent / "media" / "raw"
    OUTPUT_DIR = Path(__file__).parent.parent / "media" / "test"

    if not INPUT_DIR.is_dir():
        raise SystemExit(f"{INPUT_DIR} is not a directory")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    # process_images(INPUT_DIR, OUTPUT_DIR, seen_hashes)
    process_movies(INPUT_DIR, OUTPUT_DIR, seen_hashes)
    print("Done.")


if __name__ == "__main__":
    main()
