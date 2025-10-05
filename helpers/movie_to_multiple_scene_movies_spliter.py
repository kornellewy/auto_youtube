#!/usr/bin/env python3
"""
split_scenes.py
Detect shot changes in a movie and save every scene as a separate mp4 file.

Prerequisites:
    pip install "scenedetect[opencv]"

Only two things to change: INPUT_MOVIE and OUTPUT_DIR (both are hard-coded).
"""

from pathlib import Path
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

# ------------------------------------------------------------------
# 1.  HARD-CODED PATHS – edit these two lines
# ------------------------------------------------------------------
INPUT_MOVIE = (
    Path(__file__).parent.parent
    / "media"
    / "raw"
    / "stock_ai_clips"
    / "Titan Forge - Warhammer 40,000 Adeptus Mechanicus Animation.mp4"
)  # <-- change me
OUTPUT_DIR = (
    Path(__file__).parent.parent / "media" / "raw" / "stock_ai_clips"
)  # <-- change me
# ------------------------------------------------------------------


def main() -> None:
    # Ensure the output folder exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open the video
    video = open_video(str(INPUT_MOVIE))

    # Create a scene manager and add a content detector
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))  # tweak if needed

    # Detect scenes
    scene_manager.detect_scenes(video)

    # Get the timecodes of each scene
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        print("No scenes detected.")
        return

    # Split the original file into one clip per scene
    split_video_ffmpeg(
        str(INPUT_MOVIE),
        scene_list,
        output_dir=str(OUTPUT_DIR),
        show_progress=True,
        # scene_prefix="scene_",
        arg_override="-c:v libx264 -c:v copy -c:a copy -f mp4",  # re-encode; change if you want copy
    )

    print(f"Done! {len(scene_list)} scenes saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
