import moviepy as mp
import math
from PIL import Image
import numpy
import cv2
import numpy as np


def Zoom(clip, mode="in", position="center", speed=1):
    fps = clip.fps
    duration = clip.duration
    total_frames = int(duration * fps)

    def main(get_frame, t):
        frame = get_frame(t)
        h, w = frame.shape[:2]
        i = t * fps
        if mode == "out":
            i = total_frames - i
        zoom = 1 + (i * ((0.1 * speed) / total_frames))
        positions = {
            "center": [(w - (w * zoom)) / 2, (h - (h * zoom)) / 2],
            "left": [0, (h - (h * zoom)) / 2],
            "right": [(w - (w * zoom)), (h - (h * zoom)) / 2],
            "top": [(w - (w * zoom)) / 2, 0],
            "topleft": [0, 0],
            "topright": [(w - (w * zoom)), 0],
            "bottom": [(w - (w * zoom)) / 2, (h - (h * zoom))],
            "bottomleft": [0, (h - (h * zoom))],
            "bottomright": [(w - (w * zoom)), (h - (h * zoom))],
        }
        tx, ty = positions[position]
        M = np.array([[zoom, 0, tx], [0, zoom, ty]])
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame

    return clip.transform(main)


# img = "/media/kornellewy/jan_dysk_3/auto_youtube/media/images/0a0a87b3-a553-495d-9413-0dac60d234b0.jpg"  # using  the image link above

# clip = mp.ImageClip(img).with_fps(30).with_duration(5)
# clip = Zoom(clip, mode="in", position="center", speed=1.2)

# clip.write_videofile("zoomin2.mp4", preset="superfast")
