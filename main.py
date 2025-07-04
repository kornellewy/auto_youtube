import re
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
import cv2
import numpy as np
from moviepy.video.fx import FadeIn, Resize, FadeOut, SlideIn, SlideOut
from transformers import AutoProcessor, AutoModel
from scipy.io import wavfile
import nltk
import json
import whisper_timestamped as whisper

# import noisereduce
import torch

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
    print(
        f"Generating TTS for script_sentence_id {script_sentence_id} with voice {voice_id} using {engine} engine"
    )
    if path.exists():
        return path

    if engine == "bark":
        inputs = processor(
            text=[text],
            return_tensors="pt",
            voice_preset=voice_id,
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        inputs["history_prompt"] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs["history_prompt"].items()
        }
        with torch.no_grad():
            speech_values = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=processor.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.9,
                history_prompt=inputs["history_prompt"],
            )

        sampling_rate = model.config.codec_config.sampling_rate
        speech_values = speech_values.cpu().numpy().squeeze().astype(np.float32)
        wavfile.write(
            str(path),
            rate=sampling_rate,
            data=speech_values,
        )

    return path


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

    # load model
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = AutoModel.from_pretrained("suno/bark")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine == "bark":
        sentences = nltk.sent_tokenize(text)
        chunks = [""]
        token_counter = 0
        for sentence in sentences:
            current_tokens = len(nltk.Text(sentence))
            if token_counter + current_tokens <= 250:
                token_counter = token_counter + current_tokens
                chunks[-1] = chunks[-1] + " " + sentence
            else:
                chunks.append(sentence)
                token_counter = current_tokens
        audio_arrays = []
        for prompt in chunks:
            inputs = processor(
                text=[prompt],
                return_tensors="pt",
                voice_preset=voice_id,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            inputs["history_prompt"] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs["history_prompt"].items()
            }
            with torch.no_grad():
                speech_values = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.8,
                    history_prompt=inputs["history_prompt"],
                )

            sampling_rate = model.config.codec_config.sampling_rate
            speech_values = speech_values.cpu().numpy().squeeze().astype(np.float32)
            audio_arrays.append(speech_values)
        combined_audio = np.concatenate(audio_arrays)
        wavfile.write(
            str(path),
            rate=sampling_rate,
            data=combined_audio,
        )

    del model
    del processor
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


def get_talking_head(voice_id, duration):
    results = voice_id_to_face_images_map[voice_id]
    if isinstance(results, tuple):
        mouth_closed_img_path, mouth_open_img_path = results
    else:
        if Path(results).suffix == ".mp4":
            video = VideoFileClip(str(results))
            video = video.with_duration(duration)
            return video

    mouth_closed_img = cv2.imread(str(mouth_closed_img_path), cv2.IMREAD_UNCHANGED)
    mouth_closed_img = cv2.cvtColor(mouth_closed_img, cv2.COLOR_BGR2RGB)
    mouth_closed_img = resize_keep_aspect_no_padding(mouth_closed_img, TALKING_HEAD_RES)
    mouth_open_img = cv2.imread(str(mouth_open_img_path), cv2.IMREAD_UNCHANGED)
    mouth_open_img = cv2.cvtColor(mouth_open_img, cv2.COLOR_BGR2RGB)
    mouth_open_img = resize_keep_aspect_no_padding(mouth_open_img, TALKING_HEAD_RES)

    closed = ImageClip(mouth_closed_img).with_duration(1 / 5)
    open_ = ImageClip(mouth_open_img).with_duration(1 / 5)

    # Repeat alternating clips
    one_cycle = [closed, open_]
    cycles_needed = int(duration * 30 / 2)
    all_clips = one_cycle * cycles_needed

    final = concatenate_videoclips(all_clips, method="compose")
    final = final.with_duration(duration)
    return final

    # final.write_videofile(str(output_path), fps=30, codec="libx264", preset="ultrafast")


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


def build_video_blocks(blocks, config, script_name):
    media_path = Path(__file__).parent / "media" / "images"
    clips = []
    background = VideoFileClip(BACKGROUND_MOVIE_PATH, audio=False)

    print(f"len of blocks {len(blocks)}")
    for block_idx, block in enumerate(blocks):
        print("block_idx", block_idx)
        print("text", block["text"])
        print("################################")
        text = block["text"]
        tags = block["tags"]

        voice = tags["voice"]
        engine = config["voices"][voice]["tts"]
        voice_id = config["voices"][voice]["engine_params"]["bark_id"]

        voice_file = generate_tts_long(text, voice_id, engine, script_name, block_idx)
        # noise_reduction(str(voice_file), str(voice_file))
        # continue
        audio_clip = AudioFileClip(str(voice_file))

        image_id = tags["photo"]
        image_path = media_path / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = media_path / f"{image_id}.png"
        if not image_path.exists():
            image_path = media_path / f"{image_id}.jpeg"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_keep_aspect_no_padding(image, TARGET_IMAGE_RES)
        image_clip = ImageClip(image)

        image_clip = image_clip.with_duration(audio_clip.duration)

        # image_clip = (image_clip, tags["anim"])
        # tutaj sceny
        # talking head to 1
        # konfrence room to 2
        # show_vide_scene to 3
        # talking_head = get_talking_head(voice, audio_clip.duration)

        final = image_clip.with_audio(audio_clip)
        final = final.with_duration(audio_clip.duration)
        background = background.with_duration(audio_clip.duration)
        background.layer = 0
        final.layer = 1
        # talking_head.layer = 2
        final = final.with_position(("center", "center"), relative=True)
        # talking_head = talking_head.with_position(("left", "bottom"), relative=True)
        composite = CompositeVideoClip([background, final])

        # composite.write_videofile(
        #     "./test.mp4", fps=30, audio_fps=44100, codec="libx264", audio_codec="aac"
        # )
        clips.append(composite)

        # return clips
    return clips


def get_transcript_from_audio(audio_file_path, transcript_path):
    if Path(transcript_path).exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        audio = whisper.load_audio(audio_file_path)
        model = whisper.load_model("tiny", device="cuda")
        result = whisper.transcribe(model, audio, language="en")
        # Save transcript to file
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        del model
        del audio
    return result


def build_video_blocks_from_article(blocks, config, script_name):
    media_path = Path(__file__).parent / "media" / "images"
    audio_clips = []
    video_clips = []
    downloaded_article_dir_path = (
        Path(__file__).parent / config["downloaded_article_dir_path"]
    )

    all_id_to_image = list(downloaded_article_dir_path.rglob("*.png"))
    all_id_to_image += list(downloaded_article_dir_path.rglob("*.jpg"))
    all_id_to_image += list(downloaded_article_dir_path.rglob("*.jpeg"))
    all_id_to_image = {
        int(image_path.stem.replace("image_", "")): image_path
        for image_path in all_id_to_image
        if image_path.stem.startswith("image_")
    }

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
    old_picture = "/media/kornellewy/jan_dysk_3/auto_youtube/scraped_articles/DoRA_Weight-Decomposed_Low-Rank_Adaptation/thumbnail.png"
    for block_idx, block in enumerate(blocks):
        print("block_idx", block_idx)
        # print("text", block["text"])
        print("################################")
        text = block["text"]
        tags = block["tags"]

        voice = tags["voice"]
        engine = config["voices"][voice]["tts"]
        voice_id = config["voices"][voice]["engine_params"]["bark_id"]

        voice_file = generate_tts_long(text, voice_id, engine, script_name, block_idx)
        audio_clip = AudioFileClip(str(voice_file))
        if "figure " in text.lower():
            figure_number = text.lower().split("figure ")[1].split(" ")[0][0]
            if figure_number.isdigit():
                figure_number = int(figure_number)
                image_path = all_id_to_image[figure_number]
                block["photo"] = image_path
                old_picture = image_path

        print("picture", old_picture)
        image = cv2.imread(old_picture)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_keep_aspect_with_padding(image, TARGET_IMAGE_RES)
        image_clip = ImageClip(image)

        image_clip = image_clip.with_duration(audio_clip.duration)
        final_clip = image_clip.with_audio(audio_clip)
        final_clip = final_clip.with_duration(audio_clip.duration)

        audio_clips.append(audio_clip)
        video_clips.append(final_clip)

    return video_clips

    # audio_clips = concatenate_audioclips(audio_clips)
    # audio_path = Path(__file__).parent / "voices" / f"{script_name}.mp3"
    # if not audio_path.exists():
    #     print(f"Saving audio to {audio_path}")
    #     audio_clips.write_audiofile(str(audio_path))

    # get transcripts
    # transcript_path = (
    #     Path(__file__).parent / "voices" / f"{script_name}_transcript.json"
    # )
    # transcript = get_transcript_from_audio(str(audio_path), transcript_path)
    # print(json.dumps(transcript, indent=2, ensure_ascii=False))
    # start_times = [0]
    # for segment in transcript["segments"]:
    #     if "figure" in segment['text'].lower():
    #         start_times.append(segment['start'])
    # TODO: in trascipt number of figures are in words, so its hard to conect them to given image from paper


def main():
    config_path = Path(__file__).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text())

    script_path = (
        Path(__file__).parent
        / "projects"
        / "scripts"
        / "03_07_2025_DoRA_Weight-Decomposed_Low-Rank_Adaptation.txt"
    )
    config["downloaded_article_dir_path"] = (
        "scraped_articles/DoRA_Weight-Decomposed_Low-Rank_Adaptation"
    )
    blocks = parse_script(script_path)
    # blocks = blocks[:3]

    video_clips = build_video_blocks_from_article(blocks, config, script_path.stem)

    # TODO: remove add bc debug
    # video_clips = video_clips[:3]

    final_video = concatenate_videoclips(video_clips)
    output_path = Path(__file__).parent / "output" / f"{script_path.stem}.mp4"
    (Path(__file__).parent / "output").mkdir(parents=True, exist_ok=True)
    final_video.write_videofile(str(output_path), fps=60)


if __name__ == "__main__":
    main()
