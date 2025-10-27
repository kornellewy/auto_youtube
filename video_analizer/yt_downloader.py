# #!/usr/bin/env python3
"""
yt_dl.py – reusable YouTube downloader class
pip install yt-dlp
"""
import pathlib
import subprocess


class YouTubeDownloader:
    """Lightweight wrapper around yt-dlp."""

    def __init__(
        self,
        output_dir: str = str(pathlib.Path(__file__).parent),
        quality: str = "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        fname_template: str = "%(title)s.%(ext)s",
        playlist: bool = True,
        ytdlp_path: str = "yt-dlp",
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.quality = quality
        self.fname_template = fname_template
        self.playlist = playlist
        self.ytdlp_path = ytdlp_path
        self.output_dir.mkdir(exist_ok=True)

    def download(self, url: str) -> None:
        """Download a single video (or playlist if playlist=True)."""
        cmd = [
            self.ytdlp_path,
            "-f",
            self.quality,
            "-o",
            str(self.output_dir / self.fname_template),
            "--yes-playlist" if self.playlist else "--no-playlist",
            url,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    url = "https://www.youtube.com/playlist?list=PLhe77z_zTzRY-bmtKMydUF0eoWbGz_cCM"
    dl = YouTubeDownloader()
    dl.download(url)
