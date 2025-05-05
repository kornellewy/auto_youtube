import re
import yaml
from pathlib import Path
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)


# Dummy TTS engines (placeholder)
def generate_tts(text, voice_id, engine):
    path = Path("voices") / f"{voice_id}_{hash(text)}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # Simulate generation
    return path


# Parse tag line
TAG_PATTERN = re.compile(r"\[(.*?)\]")


def parse_script(script_path):
    blocks = []
    with open(script_path, "r") as f:
        lines = f.readlines()

    current_tags = {
        "photo": None,
        "music": None,
        "movie": None,
        "voice": None,
        "anim": "none",
    }

    for line in lines:
        tags = TAG_PATTERN.findall(line)
        content = TAG_PATTERN.sub("", line).strip()

        for tag in tags:
            if ":" in tag:
                key, val = tag.lower().split(":", 1)
                current_tags[key.strip()] = val.strip()

        if content:
            blocks.append({"text": content, "tags": current_tags.copy()})

    return blocks


def apply_animation(image_clip, anim_type, duration):
    if anim_type == "fadein+zoom":
        return image_clip.fadein(1).resize(lambda t: 1 + 0.05 * t)
    elif anim_type == "fadein":
        return image_clip.fadein(1)
    else:
        return image_clip


def build_video_blocks(blocks, config):
    media_path = Path(config["media_library"]["image_dir"])
    clips = []

    for block in blocks:
        text = block["text"]
        tags = block["tags"]

        voice = tags["voice"] or config["defaults"]["voice"]
        engine = config["voices"][voice]["tts"]
        voice_file = generate_tts(text, voice, engine)
        audio_clip = AudioFileClip(str(voice_file))

        image_id = tags["photo"] or config["defaults"]["photo"]
        image_path = media_path / f"{image_id}.jpg"
        image_clip = ImageClip(str(image_path)).set_duration(audio_clip.duration)
        image_clip = apply_animation(image_clip, tags["anim"], audio_clip.duration)

        final = image_clip.set_audio(audio_clip)
        clips.append(final)

    return clips


def main():
    config_path = Path("config.yaml")
    config = yaml.safe_load(config_path.read_text())

    script_path = Path(config["script"]["file"])
    blocks = parse_script(script_path)

    video_clips = build_video_blocks(blocks, config)

    final_video = concatenate_videoclips(video_clips)
    output_path = Path(config["project"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_video.write_videofile(str(output_path), fps=24)


if __name__ == "__main__":
    main()
