import re
import os
import yaml
from pathlib import Path
from PIL import Image
from moviepy import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    CompositeVideoClip,
    VideoFileClip,
    afx,
    vfx,
    CompositeAudioClip,
)
import captacity
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from moviepy.audio.fx import MultiplyVolume
from scipy.io import wavfile
from transformers import pipeline
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy import TextClip  # ensure TextClip import

# import whisper_timestamped as whisper
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
import whisper_timestamped as whisper

from helpers.moviepy_zoom_in_effect2 import Zoom
from helpers.create_writing_slide import create_writing_title

# Parse tag line
TAG_PATTERN = re.compile(r"\[(.*?)\]")

# TODO: remove add bc debug
BACKGROUND_RES = (1920, 1080)  # (1920, 1080)
TARGET_IMAGE_RES = (int(BACKGROUND_RES[0] * 0.7), int(BACKGROUND_RES[1] * 0.7))
TALKING_HEAD_RES = (int(BACKGROUND_RES[0] * 0.4), int(BACKGROUND_RES[1] * 0.4))
BACKGROUND_MOVIE_PATH = Path(
    Path(__file__).parent / "media" / "clips" / "starfield_backgound.mp4"
)
BACKGROUND_MOVIE_PATHS = [
    p for p in (Path(__file__).parent / "media" / "background_clips").glob("*.mp4")
]
STROCK_MOVIES_PATHS = [
    p for p in (Path(__file__).parent / "media" / "raw" / "stock_ai_clips").glob("*.mp4")
]

checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=32, temp=0.6, device=torch.device("cuda")
)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)


def biased_coin(idx: int, low: int, high: int) -> int:
    """Return 0 or 1 with probability sliding from 0→1 as idx moves low→high."""
    return int(random.random() < (idx - low) / (high - low))

def create_title_slide(
    text: str,
    size: tuple[int, int],
    save_path: Path,
    background_color: tuple[int, int, int] = (30, 30, 30),  # dark background
    text_color: tuple[int, int, int] = (255, 255, 255),  # white text
    font_size: int = 150,
) -> Path:
    width, height = size
    # Create empty image with background color using PIL
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)
    while True:
        # Load default font or fallback
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default(font_size)

        # Calculate text position for centering
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2
        if text_width > width or text_height > height:
            # If text is too large, reduce font size and try again
            font_size -= 10
        else:
            break

    # Draw the text
    draw.text((x, y), text, fill=text_color, font=font)

    # Convert PIL to OpenCV format and save with cv2
    image_np = np.array(image)[:, :, ::-1]  # RGB to BGR
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image_np)

    return save_path


def get_rms_volume(clip):
    """Return RMS value (0-1) of the clip's audio track, safely handling short audio."""
    if clip.audio is None:
        return 0.0

    audio = clip.audio
    # Ensure audio covers the full clip duration
    if audio.duration < clip.duration:
        audio = afx.audio_loop(audio, duration=clip.duration)

    # Now safe to call to_soundarray
    samples = audio.to_soundarray(fps=44100)
    rms = float(np.sqrt(np.mean(samples**2)))
    return rms


def match_volume(reference_clip, target_clip):
    # Loop audio if shorter than video
    ref_audio = reference_clip.audio
    tgt_audio = target_clip.audio
    if ref_audio and ref_audio.duration < reference_clip.duration:
        ref_audio = afx.audio_loop(ref_audio, duration=reference_clip.duration)
    if tgt_audio and tgt_audio.duration < target_clip.duration:
        tgt_audio = afx.audio_loop(tgt_audio, duration=target_clip.duration)

    src_rms = get_rms_volume(reference_clip)
    tgt_rms = get_rms_volume(target_clip)

    gain = 1.0 if tgt_rms == 0 else src_rms / tgt_rms
    target_clip = target_clip.with_audio(tgt_audio).with_effects([afx.volumex(gain)])
    return target_clip


def resize_keep_aspect_no_padding(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    # Compute scale to fit inside the target box
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def resize_keep_aspect_with_padding(img, target_size):
    target_w, target_h = target_size

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute top-left corner for centering
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste resized image onto black canvas
    result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return result


def generate_tts_long(text, voice_id, engine, script_name, script_sentence_id):

    path = (
        Path(__file__).parent / "voices" / f"{script_name}___{script_sentence_id}.wav"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating TTS for script_sentence_id {script_sentence_id} with voice {voice_id} using {engine} engine"
    )
    if path.exists():
        return path

    voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

    print("Generating audio...")

    pcms = []

    def _on_frame(frame):
        print("Step", len(pcms), end="\r")
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))

    # You could also generate multiple audios at once by extending the following lists.
    all_entries = [entries]
    all_condition_attributes = [condition_attributes]
    with tts_model.mimi.streaming(len(all_entries)):
        result = tts_model.generate(
            all_entries, all_condition_attributes, on_frame=_on_frame
        )

    print("Done generating.")
    audio = np.concatenate(pcms, axis=-1)

    wavfile.write(
        str(path),
        rate=tts_model.mimi.sample_rate,
        data=audio.astype(np.float32),
    )
    # del checkpoint_info
    # del tts_model
    # torch.cuda.empty_cache()
    return path


def parse_script(script_path):
    blocks = []
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    text = ""
    tags = {}
    if len(lines) == 1:
        blocks.append({"text": lines[0], "tags": tags.copy()})
    else:
        for line_idx, line in enumerate(lines):
            print("line_idx", line_idx)
            print("len(blocks)", len(blocks))
            if idx == 0:
                for tag in TAG_PATTERN.findall(line):
                    key, value = tag.split(": ")
                    key = key.lower()
                    tags[key.strip().lower()] = value.strip()
                idx = 1
                continue
            elif idx == 1:
                text = line
                idx = 2
                if line_idx == (len(lines) - 1):
                    blocks.append({"text": text, "tags": tags.copy()})
                continue
            elif idx == 2:
                blocks.append({"text": text, "tags": tags.copy()})
                text = ""
                tags = {}
                idx = 0
                continue
            else:
                raise ValueError("")

    return blocks


def get_transcript(audio_path: Path) -> dict:
    """Return full whisper result with word-level timestamps."""
    print("[Audio] Running global whisper …")
    audio = whisper.load_audio(str(audio_path))
    model = whisper.load_model("tiny", device="cuda")
    return whisper.transcribe(model, audio, language="en")


def build_video_blocks_from_article(blocks, config, script_name):
    meme_dir_path = Path(__file__).parent / "media" / "images"
    background = VideoFileClip(random.choice(BACKGROUND_MOVIE_PATHS), audio=False)
    audio_clips = []
    video_clips = []
    downloaded_article_dir_path = (
        Path(__file__).parent / config["downloaded_article_dir_path"]
    )
    thumbnail_picture = downloaded_article_dir_path / "thumbnail.png"
    if not thumbnail_picture.exists():
        thumbnail_picture = downloaded_article_dir_path / "thumbnail.jpg"
    elif not thumbnail_picture.exists():
        thumbnail_picture = downloaded_article_dir_path / "thumbnail.jpeg"
    else:
        raise (f"thumbnail_picture dont exists in dir {downloaded_article_dir_path}")

    # load images from paper
    all_id_to_image = list(downloaded_article_dir_path.rglob("*.png"))
    all_id_to_image += list(downloaded_article_dir_path.rglob("*.jpg"))
    all_id_to_image += list(downloaded_article_dir_path.rglob("*.jpeg"))
    all_id_to_image = {
        int(image_path.stem.replace("image_", "")): image_path
        for image_path in all_id_to_image
        if image_path.stem.startswith("image_")
    }
    # load tables from paper
    all_id_to_table_image = list(downloaded_article_dir_path.rglob("*.png"))
    all_id_to_table_image += list(downloaded_article_dir_path.rglob("*.jpg"))
    all_id_to_table_image += list(downloaded_article_dir_path.rglob("*.jpeg"))
    all_id_to_table_image = {
        int(image_path.stem.replace("table_", "")): image_path
        for image_path in all_id_to_table_image
        if image_path.stem.startswith("table_")
    }
    # load all infograci form paper
    all_id_to_infografic = list(downloaded_article_dir_path.rglob("*.png"))
    all_id_to_infografic += list(downloaded_article_dir_path.rglob("*.jpg"))
    all_id_to_infografic += list(downloaded_article_dir_path.rglob("*.jpeg"))
    all_id_to_infografic = {
        int(image_path.stem.replace("infografic_", "")): image_path
        for image_path in all_id_to_infografic
        if image_path.stem.startswith("infografic_")
    }

    print(f"len of blocks {len(blocks)}")
    # blocks = new_blocks_all
    
    for block_idx, block in enumerate(blocks):
        print("block_idx", block_idx)
        # print("text", block["text"])
        print("################################")
        final_clip = None
        text = block["text"]
        # tags = block["tags"]

        voice = "narrator"
        engine = config["voices"][voice]["tts"]
        voice_id = config["voices"][voice]["engine_params"]["bark_id"]

        voice_file = generate_tts_long(text, voice_id, engine, script_name, block_idx)
        audio_clip = AudioFileClip(str(voice_file))
        # transcript = get_transcript(str(voice_file))
        # if there is addnotaion to figure in text
        if "figure " in text.lower():
            print("figure in text")
            figure_number = text.lower().split("figure ")[1].split(" ")[0][0]
            if figure_number.isdigit():
                figure_number = int(figure_number)
                if figure_number in all_id_to_image.keys():
                    image_path = all_id_to_image[figure_number]
                    block["photo"] = image_path
                    thumbnail_picture = image_path

                    image = cv2.imread(str(thumbnail_picture))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
                    image_clip = ImageClip(image)

                    background_copy = background.copy()
                    background_copy = background_copy.with_effects(
                        [vfx.Loop(duration=audio_clip.duration)]
                    )
                    image_clip = image_clip.with_duration(audio_clip.duration)
                    background_copy.layer = 0
                    image_clip.layer = 1
                    image_clip = image_clip.with_position(
                        ("center", "center"), relative=True
                    )
                    final_clip = CompositeVideoClip([background_copy, image_clip])
                    final_clip = final_clip.with_audio(audio_clip)
        # if there is table in text
        elif "table " in text.lower():
            print("figure in table")
            table_number = text.lower().split("table ")[1].split(" ")[0][0]
            if table_number.isdigit():
                table_number = int(table_number)
                if table_number in all_id_to_table_image.keys():
                    image_path = all_id_to_table_image[table_number]
                    block["photo"] = image_path
                    thumbnail_picture = image_path
                    image = cv2.imread(str(thumbnail_picture))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
                    image_clip = ImageClip(image)

                    background_copy = background.copy()
                    background_copy = background_copy.with_effects(
                        [vfx.Loop(duration=audio_clip.duration)]
                    )
                    image_clip = image_clip.with_duration(audio_clip.duration)
                    background_copy.layer = 0
                    image_clip.layer = 1
                    image_clip = image_clip.with_position(
                        ("center", "center"), relative=True
                    )
                    final_clip = CompositeVideoClip([background_copy, image_clip])
                    final_clip = final_clip.with_audio(audio_clip)
        # infografic case

        elif "infographic " in text.lower() or "infographic_" in text.lower():
            print("infografic in table")
            if "infographic " in text.lower():
                infographic_number = text.lower().split("infographic ")[1].split(" ")[0][0]
            elif "infographic_" in text.lower():
                infographic_number = text.lower().split("infographic_")[1].split(" ")[0][0]
            if infographic_number.isdigit():
                infographic_number = int(infographic_number)
                if infographic_number in all_id_to_infografic.keys():
                    image_path = all_id_to_infografic[infographic_number]
                    block["photo"] = image_path
                    thumbnail_picture = image_path
                    image = cv2.imread(str(thumbnail_picture))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
                    image_clip = ImageClip(image)

                    background_copy = background.copy()
                    background_copy = background_copy.with_effects(
                        [vfx.Loop(duration=audio_clip.duration)]
                    )
                    image_clip = image_clip.with_duration(audio_clip.duration)
                    background_copy.layer = 0
                    image_clip.layer = 1
                    image_clip = image_clip.with_position(
                        ("center", "center"), relative=True
                    )
                    final_clip = CompositeVideoClip([background_copy, image_clip])
                    final_clip = final_clip.with_audio(audio_clip)
        # heading case
        elif text.startswith("***"):
            text_of_heading = text.replace("***", "").strip()
            final_clip = create_writing_title(text_of_heading, BACKGROUND_RES)
            background = VideoFileClip(
                random.choice(BACKGROUND_MOVIE_PATHS), audio=False
            )
        # add meme or filler
        # if something brak or missing add meme

        

        video_clips.append(final_clip)

    return video_clips


def main(
    config_path: Path = Path(__file__).parent / "config.yaml",
    script_path: Path = None,
    downloaded_article_dir_path: Path = None,
    soundtrack_path: Path = (
        Path(__file__).parent
        / "media"
        / "music"
        / "Mechanicus Soundtrack - Children of the Omnissiah (Extended - Seamless loop).mp3"
    ),
    outro_video_path: Path = None,
    output_video_path: Path = None,
):
    config = yaml.safe_load(config_path.read_text())
    if "script_path" in config.keys():
        script_path = Path(config["script_path"])

    if "downloaded_article_dir_path" in config.keys():
        pass
    else:
        config["downloaded_article_dir_path"] = downloaded_article_dir_path

    if soundtrack_path in config.keys():
        soundtrack_path = Path(config["soundtrack_path"])
    else:
        soundtrack_path = random.choice(
            list([p for p in (Path(__file__).parent / "media" / "music").glob("*.mp3")])
        )

    if "outro_video_path" in config.keys():
        outro_video_path = Path(config["outro_video_path"])

    blocks = parse_script(script_path)
    # blocks = blocks[:4]

    video_clips = build_video_blocks_from_article(blocks, config, script_path.stem)

    final_video = concatenate_videoclips(video_clips)
    # add overlay music
    music = AudioFileClip(str(soundtrack_path))
    music = music.with_effects(
        [afx.AudioLoop(duration=final_video.duration), MultiplyVolume(0.03)]
    )
    combined_audio = CompositeAudioClip([final_video.audio, music])
    final_video = final_video.with_audio(combined_audio)
    # add outro video
    if outro_video_path is not None:
        outro_video_path = Path(outro_video_path)
        outro_movie = VideoFileClip(str(outro_video_path))
        background = VideoFileClip(random.choice(BACKGROUND_MOVIE_PATHS), audio=False)
        background_copy = background.copy()
        background_copy = background_copy.with_effects(
            [vfx.Loop(duration=outro_movie.duration)]
        )
        background_copy.layer = 0
        outro_movie.layer = 1
        outro_movie = outro_movie.with_position(("center", "center"), relative=True)
        outro_movie = CompositeVideoClip([background_copy, outro_movie])
        try:
            outro_movie = match_volume(final_video, outro_movie)
        except Exception as e:
            print(f"Error matching volume: {e}")
            save_path = downloaded_article_dir_path / "sound_alert.jpg"
            thumbnail_picture = create_writing_title(
                "!!! SOUND ALERT !!!", BACKGROUND_RES
            )
            image = cv2.imread(str(thumbnail_picture))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
            alert_clip = ImageClip(image)
            alert_clip = alert_clip.with_duration(5)
            outro_movie = concatenate_videoclips([alert_clip, outro_movie])
        final_video = concatenate_videoclips([final_video, outro_movie])

    # save final video
    if output_video_path is None:
        output_video_path = Path(__file__).parent / "output" / f"{script_path.stem}.mp4"
        (Path(__file__).parent / "output").mkdir(parents=True, exist_ok=True)
    # final_video.write_videofile(str(output_video_path), fps=30)#, preset="superfast")
    

    captacity.add_captions(
        video_file=output_video_path,
        output_file=output_video_path,
    )
    print(f"save output_video_path in {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    
    config_path= Path(__file__).parent / "config.yaml"
    script_path = Path(__file__).parent / "scraped_articles" / "EvoTest_Evolutionary_Test-Time_Learning_for_Self-Improving_Agentic_Systems" / "yt_short_script_1.txt"
    downloaded_article_dir_path = Path(__file__).parent / "scraped_articles" / "EvoTest_Evolutionary_Test-Time_Learning_for_Self-Improving_Agentic_Systems"
    output_video_path = Path(__file__).parent / "scraped_articles" / "EvoTest_Evolutionary_Test-Time_Learning_for_Self-Improving_Agentic_Systems" / "yt_short_1.mp4"

    main(
        config_path= config_path,
        script_path=script_path,
        downloaded_article_dir_path=downloaded_article_dir_path,
        output_video_path=output_video_path
    )