from __future__ import annotations
import numpy as np
import cv2
import moviepy as mp
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

FONT_NAME = "/usr/share/fonts/urw-base35/URWGothic-Demi.otf"
FONT_COLOUR = (0, 90, 0)
BG_COLOUR = (0, 0, 0)
FPS = 24
BACKGROUND_DATABASE_PATH = Path(__file__).parent / "sides_backgrounds"


def create_writing_title(
    text: str,
    size: tuple[int, int] = (1920, 1080),
    total_duration: float = 6.0,
    writing_ratio: float = 0.75,
) -> mp.VideoClip:
    w, h = size
    writing_time = total_duration * writing_ratio
    background_image_path = random.choice([p for p in BACKGROUND_DATABASE_PATH.glob("*.jpeg")])


    # ---------- 1.  FIND BEST FONT SIZE (once) ----------
    for size_candidate in range(200, 10, -10):
        try:
            font = ImageFont.truetype(FONT_NAME, size_candidate)
        except OSError:
            font = ImageFont.load_default()
        # img = Image.new("RGB", (w, h), BG_COLOUR)
        img = Image.open(background_image_path)
        draw = ImageDraw.Draw(img)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:4]
        if tw <= w * 0.9 and th <= h * 0.9:  # fits with 10 % margin
            best_font = font
            best_tw, best_th = tw, th
            break
    else:  # emergency fallback
        best_font = ImageFont.load_default()
        best_tw, best_th = draw.textbbox((0, 0), text, font=best_font)[2:4]

    x = (w - best_tw) // 2
    y = (h - best_th) // 2

    # ---------- 2.  ANIMATION USING THAT FONT ----------
    def make_frame(t):
        if t >= writing_time:
            partial = text
        else:
            idx = int(len(text) * t / writing_time)
            partial = text[: max(0, idx)]

        # img = Image.new("RGB", (w, h), BG_COLOUR)
        img = Image.open(background_image_path)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), partial, font=best_font, fill=FONT_COLOUR)
        return np.array(img)[:, :, ::-1]

    return mp.VideoClip(make_frame, duration=total_duration).with_fps(FPS)


# ---------------- demo ----------------
if __name__ == "__main__":
    clip = create_writing_title("Artistic Turtles Unite!")
    clip.write_videofile(str(Path(__file__).parent/ "writing_title.mp4") , preset="superfast")
