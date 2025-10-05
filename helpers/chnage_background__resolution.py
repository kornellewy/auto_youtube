#!/usr/bin/env python3
"""
Hard-coded script: rescales every *.mp4 in the SAME folder to 1920×1080 (FHD)
using OpenCV (cv2) + pathlib.  No arguments – just edit SOURCE_DIR.
Original files are overwritten (set OUT_SUFFIX = ""  ==>  in-place).
"""
from pathlib import Path
import cv2

SOURCE_DIR = Path(r"/full/path/to/your/videos")  # <— change only this
OUT_SUFFIX = "_fhd"  # ""  =  overwrite originals

TARGET_W, TARGET_H = 1920, 1080
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # H.264 MP4

for mp4 in [
    Path(
        "/media/kornellewy/jan_dysk_3/auto_youtube/media/background_clips/113368-697718069_large.mp4"
    ),
    Path(
        "/media/kornellewy/jan_dysk_3/auto_youtube/media/background_clips/138082-768352300_large.mp4"
    ),
]:
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmp_path = mp4.with_name(mp4.stem + OUT_SUFFIX + ".mp4")
    writer = cv2.VideoWriter(str(tmp_path), FOURCC, fps, (TARGET_W, TARGET_H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    cap.release()
    writer.release()

    # optional: overwrite original
    if not OUT_SUFFIX:
        # mp4.unlink()
        tmp_path.rename(mp4)

print("All MP4 → FHD complete ✔")
