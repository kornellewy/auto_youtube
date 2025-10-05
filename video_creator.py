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
    concatenate_audioclips,
    CompositeVideoClip,
    VideoFileClip,
    afx,
    CompositeAudioClip,
)
import torch
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from moviepy.video.fx import FadeIn, FadeOut, SlideIn, SlideOut
from moviepy.audio.fx import MultiplyVolume
from scipy.io import wavfile
from transformers import pipeline

# import whisper_timestamped as whisper
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from sentence_transformers import SentenceTransformer, util


from helpers.moviepy_zoom_in_effect import zoom_in_effect

# Parse tag line
TAG_PATTERN = re.compile(r"\[(.*?)\]")

# TODO: remove add bc debug
BACKGROUND_RES = (1920, 1080)  # (1920, 1080)
TARGET_IMAGE_RES = (int(BACKGROUND_RES[0] * 0.7), int(BACKGROUND_RES[1] * 0.7))
TALKING_HEAD_RES = (int(BACKGROUND_RES[0] * 0.4), int(BACKGROUND_RES[1] * 0.4))
BACKGROUND_MOVIE_PATH = Path(
    Path(__file__).parent / "media" / "clips" / "starfield_backgound.mp4"
)
voice_id_to_face_images_map = {
    "narrator": (
        Path(__file__).parent / "media" / "images" / "rockefeller.png",
        Path(__file__).parent / "media" / "images" / "rockefeller_month_open.png",
    ),
    "cyborg": (
        Path(__file__).parent / "media" / "images" / "rockefeller.png",
        Path(__file__).parent / "media" / "images" / "rockefeller_month_open.png",
    ),
    "philosopher": Path(__file__).parent / "media" / "clips" / "acent_talking_Cat.mp4",
    "skynet": Path(__file__).parent / "media" / "clips" / "cyborg_talking.mp4",
}
checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=32, temp=0.6, device=torch.device("cuda")
)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)


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


def apply_animation(image_clip, anim_type):
    fade_in = FadeIn(duration=1)
    fade_out = FadeOut(duration=1)
    slide_in = SlideIn(duration=1, side="right")
    slide_out = SlideOut(duration=1, side="left")
    w, h = image_clip.size
    # resize = Resize(h * 1.05, w * 1.05)
    if anim_type == "zoomin":
        image_clip = zoom_in_effect(image_clip, 0.08)
        return image_clip
    elif anim_type == "fadein":
        image_clip = fade_in.apply(image_clip)
        return image_clip
    elif anim_type == "fadeout":
        image_clip = fade_out.apply(image_clip)
        return image_clip
    elif anim_type == "fadein+fadeout":
        image_clip = fade_in.apply(image_clip)
        image_clip = fade_out.apply(image_clip)
        return image_clip
    elif anim_type == "slidein":
        image_clip = slide_in.apply(image_clip)
        image_clip = slide_out.apply(image_clip)
        return image_clip
    else:
        return image_clip


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


def detect_emotion(text):
    """
    Returns the top predicted emotion for the given text
    """
    results = emotion_classifier(text)[0]  # list of dicts with scores
    top_emotion = max(results, key=lambda x: x["score"])
    return top_emotion["label"].lower(), top_emotion["score"]


def load_media_clip(path, target_res, resize_keep_aspect_no_padding, duration=None):
    """
    Loads an image or GIF/video into a moviepy clip with optional resizing and duration.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".gif", ".mp4", ".mov", ".avi"]:
        clip = VideoFileClip(path)
        if duration:
            clip = clip.subclip(0, min(duration, clip.duration))
        clip = clip.resize(height=target_res[1])  # keep aspect ratio by height
    else:
        meme = cv2.imread(str(path))
        meme = cv2.cvtColor(meme, cv2.COLOR_BGR2RGB)
        meme = resize_keep_aspect_no_padding(meme, target_res)
        clip = ImageClip(meme).with_duration(duration)
    return clip


def build_meme_clip(
    block_idx,
    all_blocks_text_embeddings,
    meme_text_embeddings,
    meme_database,
    thumbnail_picture,
    blocks,
    TARGET_IMAGE_RES,
    background,
    audio_clip,
    resize_keep_aspect_no_padding,
):
    block_text = blocks[block_idx]["text"]

    # --- 1. Detect emotion ---
    emotion_label, emotion_score = detect_emotion(block_text)
    print(
        f"Detected emotion for block {block_idx}: {emotion_label} ({emotion_score:.2f})"
    )

    # --- 2. Compute similarity scores ---
    block_text_embedding = all_blocks_text_embeddings[block_idx]
    cos_scores = util.cos_sim(block_text_embedding, meme_text_embeddings)[0]

    # --- 3. Combine embedding score + emotion match ---
    combined_scores = []
    for i, meme in enumerate(meme_database):
        score = float(cos_scores[i])
        # Boost if meme text matches detected emotion
        if emotion_label in meme["text"].lower():
            score += 0.001
        combined_scores.append(score)
    best_idx = int(torch.tensor(combined_scores).argmax())
    best_meme = meme_database[best_idx]
    best_score = combined_scores[best_idx]

    # Fallback if score is too low, random
    if best_score > 0.05:
        meme_path = thumbnail_picture
        # chosen_meme = {"text": "fallback", "image_path": meme_path}
        chosen_meme = random.choice(meme_database)
        best_idx = meme_database.index(chosen_meme)
        best_meme = meme_database[best_idx]
        meme_path = best_meme["image_path"]
    else:
        meme_path = best_meme["image_path"]
        chosen_meme = best_meme

    # --- 4. Remove chosen meme from database ---
    remove_idx = meme_database.index(chosen_meme)
    del meme_database[remove_idx]
    mask = torch.ones(meme_text_embeddings.size(0), dtype=torch.bool)
    mask[remove_idx] = False
    meme_text_embeddings = meme_text_embeddings[mask]

    print(
        f"Best meme for block {block_idx}: {chosen_meme['text']} with score {best_score}"
    )

    # --- 5. Load meme (image or GIF) ---
    ext = os.path.splitext(meme_path)[1].lower()
    if ext in [".gif", ".mp4", ".mov"]:  # support GIF/video memes
        meme_clip = VideoFileClip(meme_path)
    else:
        meme = cv2.imread(str(meme_path))
        meme = cv2.cvtColor(meme, cv2.COLOR_BGR2RGB)
        meme = resize_keep_aspect_no_padding(meme, TARGET_IMAGE_RES)
        meme_clip = ImageClip(meme)

    # --- 6. Adjust duration ---
    # If text is long, add a second meme or extend duration
    text_length_factor = len(block_text) / 100  # arbitrary scale
    final_duration = audio_clip.duration
    meme_clip = meme_clip.with_duration(final_duration)
    background_copy = background.copy().with_duration(final_duration)

    # --- 7. Compose final clip ---
    background_copy.layer = 0
    meme_clip.layer = 1
    meme_clip = meme_clip.with_position(("center", "center"), relative=True)

    final_clip = CompositeVideoClip([background_copy, meme_clip])
    final_clip = final_clip.with_audio(audio_clip.with_duration(final_duration))

    return final_clip, meme_database, meme_text_embeddings


def build_video_blocks_from_article(blocks, config, script_name):
    meme_dir_path = Path(__file__).parent / "media" / "images"
    background = VideoFileClip(BACKGROUND_MOVIE_PATH, audio=False)
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
    # load all memes and text
    meme_database = []
    for image_path in meme_dir_path.glob("*.jpg"):
        text_path = image_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        meme_data = {
            "text_path": text_path,
            "image_path": image_path,
            "text": text,
        }
        meme_database.append(meme_data)
    for image_path in meme_dir_path.glob("*.gif"):
        text_path = image_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        meme_data = {
            "text_path": text_path,
            "image_path": image_path,
            "text": text,
        }
        meme_database.append(meme_data)

    # embeding memes
    model = SentenceTransformer("all-MiniLM-L6-v2")
    meme_text_embeddings = model.encode(
        [m["text"] for m in meme_database],
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    # split text into blocks bs multiple figure refrence in single pice of text
    new_blocks_all = []
    for block in blocks:
        text = block["text"]
        if "figure" in text.lower():
            # Count how many times "figure" appears
            figures_per_block = sum(
                1 for word in text.lower().split() if "figure" in word
            )

            if figures_per_block > 1:
                # Split text into sentences
                sentences = text.split(".")
                sentences = [s.strip() for s in sentences if s.strip()]  # remove empty

                current_block_sentences = []
                figure_count = 0

                for sentence in sentences:
                    current_block_sentences.append(sentence)
                    if "figure" in sentence.lower():
                        figure_count += 1
                        # When one 'figure' is included, make a block
                        new_text = ". ".join(current_block_sentences).strip() + "."
                        new_block = block.copy()
                        new_block["text"] = new_text
                        new_blocks_all.append(new_block)
                        current_block_sentences = []

                # Add any leftover sentences as a block (if not empty)
                if current_block_sentences:
                    new_text = ". ".join(current_block_sentences).strip() + "."
                    new_block = block.copy()
                    new_block["text"] = new_text
                    new_blocks_all.append(new_block)

            else:
                new_blocks_all.append(block)
        else:
            new_blocks_all.append(block)

    print(f"len of blocks {len(blocks)}")
    # embeding all blocks text
    all_texts = [block["text"] for block in new_blocks_all]
    all_blocks_text_embeddings = model.encode(
        all_texts, convert_to_tensor=True, show_progress_bar=True
    )
    del model
    blocks = new_blocks_all

    for block_idx, block in enumerate(blocks):
        print("block_idx", block_idx)
        # print("text", block["text"])
        print("################################")
        final_clip = None
        text = block["text"]
        tags = block["tags"]

        voice = tags["voice"]
        engine = config["voices"][voice]["tts"]
        voice_id = config["voices"][voice]["engine_params"]["bark_id"]

        voice_file = generate_tts_long(text, voice_id, engine, script_name, block_idx)
        audio_clip = AudioFileClip(str(voice_file))
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
                    image_clip = image_clip.with_duration(audio_clip.duration)
                    background_copy = background_copy.with_duration(audio_clip.duration)
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
                    image_clip = image_clip.with_duration(audio_clip.duration)
                    background_copy = background_copy.with_duration(audio_clip.duration)
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
            save_path = (
                downloaded_article_dir_path / f"{text_of_heading.replace(' ', '_')}.jpg"
            )
            thumbnail_picture = create_title_slide(
                text_of_heading, TARGET_IMAGE_RES, save_path
            )
            image = cv2.imread(str(thumbnail_picture))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
            image_clip = ImageClip(image)
            final_clip = image_clip.with_duration(5)
        # add meme
        elif block_idx > 0:
            final_clip, meme_database, meme_text_embeddings = build_meme_clip(
                block_idx,
                all_blocks_text_embeddings,
                meme_text_embeddings,
                meme_database,
                thumbnail_picture,
                blocks,
                TARGET_IMAGE_RES,
                background,
                audio_clip,
                resize_keep_aspect_no_padding,
            )
        # idx 0 and thumnaile
        else:
            image = cv2.imread(str(thumbnail_picture))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_keep_aspect_with_padding(image, BACKGROUND_RES)
            image_clip = ImageClip(image)
            image_clip = image_clip.with_duration(audio_clip.duration)
            final_clip = image_clip.with_audio(audio_clip)
            final_clip = final_clip.with_duration(audio_clip.duration)

        # if something brak or missing add meme
        if final_clip is None:
            final_clip, meme_database, meme_text_embeddings = build_meme_clip(
                block_idx,
                all_blocks_text_embeddings,
                meme_text_embeddings,
                meme_database,
                thumbnail_picture,
                blocks,
                TARGET_IMAGE_RES,
                background,
                audio_clip,
                resize_keep_aspect_no_padding,
            )

        """
        TODO: Add gifs and then must be addes some sort of meme if leanth of text and meme dont mech
        """

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

    if "outro_video_path" in config.keys():
        outro_video_path = Path(config["outro_video_path"])

    blocks = parse_script(script_path)
    # blocks = blocks[:7]

    video_clips = build_video_blocks_from_article(blocks, config, script_path.stem)

    final_video = concatenate_videoclips(video_clips)
    # add overlay music
    music = AudioFileClip(str(soundtrack_path))
    music = music.with_effects(
        [afx.AudioLoop(duration=final_video.duration), MultiplyVolume(0.05)]
    )
    combined_audio = CompositeAudioClip([final_video.audio, music])
    final_video = final_video.with_audio(combined_audio)
    # add outro video
    if outro_video_path is not None:
        outro_video_path = Path(outro_video_path)
        outro_movie = VideoFileClip(str(outro_video_path))
        background = VideoFileClip(BACKGROUND_MOVIE_PATH, audio=False)
        background_copy = background.copy()
        background_copy = background_copy.with_duration(outro_movie.duration)
        background_copy.layer = 0
        outro_movie.layer = 1
        outro_movie = outro_movie.with_position(("center", "center"), relative=True)
        outro_movie = CompositeVideoClip([background_copy, outro_movie])
        try:
            outro_movie = match_volume(final_video, outro_movie)
        except Exception as e:
            print(f"Error matching volume: {e}")
            save_path = downloaded_article_dir_path / "sound_alert.jpg"
            thumbnail_picture = create_title_slide(
                "!!! SOUND ALERT !!!", TARGET_IMAGE_RES, save_path
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
    final_video.write_videofile(str(output_video_path), fps=30)
    return output_video_path


if __name__ == "__main__":
    main()


# video_file = Path(__file__).parent.parent / "media" / "raw"/ "Apple’s New AI SHOCKS The Industry With 85X More Speed (Beating Everyone).mp4"
# output_dir = Path(__file__).parent / "analysis_output"
# output_dir.mkdir(exist_ok=True)
