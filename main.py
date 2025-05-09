import re
import yaml
from pathlib import Path
from PIL import Image
from moviepy import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    afx,
)
from moviepy.video.fx import FadeIn, Resize, FadeOut, SlideIn, SlideOut
from transformers import AutoProcessor, AutoModel
from scipy.io import wavfile
import noisereduce


processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")
# Parse tag line
TAG_PATTERN = re.compile(r"\[(.*?)\]")
TARGET_RES = (1920, 1080)


def noise_reduction(input_path, output_path, prop_decrease=1.0):
    rate, data = wavfile.read(input_path)
    reduced_noise = noisereduce.reduce_noise(
        y=data, sr=rate, prop_decrease=prop_decrease
    )
    wavfile.write(output_path, rate, reduced_noise)
    return True


# Dummy TTS engines (placeholder)
def generate_tts(text, voice_id, engine, script_name, script_sentence_id):
    # TODO: Impelement multi leangu with https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
    # TODO: Implement tags like laught and happy crying
    path = (
        Path(__file__).parent / "voices" / f"{script_name}___{script_sentence_id}.wav"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    if engine == "bark":
        inputs = processor(
            text=[text],
            return_tensors="pt",
            voice_preset=voice_id,
        )

        speech_values = model.generate(**inputs, do_sample=True)

        sampling_rate = model.config.codec_config.sampling_rate
        wavfile.write(
            str(path),
            rate=sampling_rate,
            data=speech_values.cpu().numpy().squeeze(),
        )

    return path


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


def apply_animation(image_clip, anim_type):
    fade_in = FadeIn(duration=2)
    fade_out = FadeOut(duration=2)
    slide_in = SlideIn(duration=2, side="right")
    slide_out = SlideOut(duration=2, side="left")
    w, h = image_clip.size
    # resize = Resize(h * 1.05, w * 1.05)
    if anim_type == "fadein+zoom":
        image_clip = fade_in.apply(image_clip)
        # image_clip = resize.apply(image_clip)
        image_clip = fade_out.apply(image_clip)
        return image_clip
    elif anim_type == "fadein":
        image_clip = fade_in.apply(image_clip)
        image_clip = fade_out.apply(image_clip)
        return image_clip
    elif anim_type == "slidein":
        image_clip = slide_in.apply(image_clip)
        image_clip = slide_out.apply(image_clip)
        return image_clip
    else:
        return image_clip


def build_video_blocks(blocks, config, script_name):
    media_path = Path(__file__).parent / "media" / "images"
    clips = []

    for block_idx, block in enumerate(blocks):
        text = block["text"]
        tags = block["tags"]

        voice = tags["voice"]
        engine = config["voices"][voice]["tts"]
        voice_id = config["voices"][voice]["engine_params"]["bark_id"]

        voice_file = generate_tts(text, voice_id, engine, script_name, block_idx)
        noise_reduction(str(voice_file), str(voice_file))
        audio_clip = AudioFileClip(str(voice_file))

        image_id = tags["photo"]
        image_path = media_path / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = media_path / f"{image_id}.png"
        image_clip = ImageClip(str(image_path))
        image_clip = image_clip.with_duration(audio_clip.duration)

        image_clip = apply_animation(image_clip, tags["anim"])

        final = image_clip.with_audio(audio_clip)
        final = final.with_duration(audio_clip.duration)
        clips.append(final)
        final.write_videofile(
            "./test.mp4", fps=30, audio_fps=44100, codec="libx264", audio_codec="aac"
        )

        return clips


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())

    script_path = Path(__file__).parent / "projects" / "scripts" / "ai_history_1.txt"
    blocks = parse_script(script_path)

    video_clips = build_video_blocks(blocks, config, script_path.stem)

    final_video = concatenate_videoclips(video_clips)
    output_path = Path(__file__) / "output" / f"{script_path.stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_video.write_videofile(str(output_path), fps=60)


if __name__ == "__main__":
    main()
