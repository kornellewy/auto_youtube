import json
import whisper_timestamped as whisper

audio = whisper.load_audio(
    "/media/kornellewy/jan_dysk_3/auto_youtube/voices/08_06_2025_haggingface_smolvam___0.wav"
)

model = whisper.load_model("tiny", device="cuda")

result = whisper.transcribe(model, audio, language="en")


print(json.dumps(result, indent=2, ensure_ascii=False))
