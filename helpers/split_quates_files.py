from pathlib import Path
import ffmpeg
from pydub import AudioSegment
from pydub.silence import split_on_silence


def split_quotes(
    audio_path: str | Path,
    out_folder: str | Path = "/media/kornellewy/jan_dysk_3/auto_youtube/helpers/quotes",
    min_silence_len: int = 2000,  # ms
    silence_thresh: int = -40,  # dBFS
    keep_silence: int = 100,
) -> list[Path]:
    """
    Split quotes file (mp3/m4a) into single-quote files.
    Returns list[Path] of new clips (numbered 0001.mp3, 0002.mp3 …).
    """
    src = Path(audio_path)
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Load audio (handles both mp3 and m4a automatically)
    audio = AudioSegment.from_file(src)

    # Split on silence
    chunks = split_on_silence(
        audio,
        # min_silence_len=min_silence_len,
        # silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    new_files = []
    for idx, chunk in enumerate(chunks, 1):
        out_file = out_folder / f"{idx:04d}.mp3"
        chunk.export(out_file, format="mp3", bitrate="192k")
        new_files.append(out_file)

    return new_files


# ------------------------------------------------------------------
# quick demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    quotes = split_quotes(
        "/media/kornellewy/jan_dysk_3/auto_youtube/media/raw/Archmagos quotes .m4a"
    )  # or .mp3
    print(f"Created {len(quotes)} quote files:")
    for q in quotes:
        print(" →", q)
