import cv2
import numpy as np
import random
from pathlib import Path
from time import sleep

# === Config ===
screen_width, screen_height = 1920, 1080
num_dots = 200
line_probability = 0.1  # chance to draw a line between two dots
dot_radius = 5
dot_flicker_range = (180, 255)  # range of brightness for flickering dots
fps = 30
duration_seconds = 60  # total video length in seconds
output_path = Path(
    "/media/kornellewy/jan_dysk_3/auto_youtube/media/clips/starfield_backgound.mp4"
)

# === Generate random dot positions ===
dots = [
    (random.randint(0, screen_width - 1), random.randint(0, screen_height - 1))
    for _ in range(num_dots)
]

# === Setup video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    str(output_path), fourcc, fps, (screen_width, screen_height)
)

# === Main loop ===
# cv2.namedWindow("Starfield", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Starfield", 960, 540)

frame_count = int(fps * duration_seconds)

for frame_idx in range(frame_count):
    old_frame = 0
    if frame_idx % 30 == 0 or frame_idx == 0:
        frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Draw dots with flicker
        for x, y in dots:
            brightness = random.randint(*dot_flicker_range)
            cv2.circle(
                frame, (x, y), dot_radius, (brightness, brightness, brightness), -1
            )

        # Randomly draw lines between some dots
        for _ in range(num_dots // 2):
            if random.random() < line_probability:
                pt1 = random.choice(dots)
                pt2 = random.choice(dots)
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        # add
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        old_frame = frame.copy()

    # cv2.imshow("Starfield", frame)
    video_writer.write(frame)

    key = cv2.waitKey(int(1000 / fps))
    if key == 27:
        break

video_writer.release()
# cv2.destroyAllWindows()
print(f"Video saved to: {output_path.resolve()}")
