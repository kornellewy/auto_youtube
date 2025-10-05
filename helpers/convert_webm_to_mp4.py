import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"  # or full path
CRF = 23  # quality 18-28
PRESET = "fast"


def convert_and_replace(src: Path) -> None:
    dst = src.with_suffix(".mp4")
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-crf",
        str(CRF),
        "-preset",
        PRESET,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    src.unlink()  # remove original webm
    print(f"converted & deleted  {src.name}")


def main() -> None:
    here = Path("/media/kornellewy/jan_dysk_3/auto_youtube/media/raw/stock_ai_clips")
    webms = list(here.glob("*.webm"))
    if not webms:
        print("No .webm files here.")
        return
    for f in webms:
        convert_and_replace(f)
    print("All done ✔")


if __name__ == "__main__":
    main()
