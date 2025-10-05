from pytube import YouTube
import os

# List of YouTube video URLs to download
video_urls = [
    "https://youtu.be/LnR0pN8o63g",  # replace with your URLs
    "https://youtu.be/WxmICiOXw2c",
    "https://youtu.be/v8xGWXuNSKo",
    "https://youtu.be/kE5O-jVc_uE",
]

# Directory to save videos
DOWNLOAD_DIR = "/media/kornellewy/jan_dysk_3/auto_youtube/media/clips"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

for url in video_urls:
    try:
        yt = YouTube(url)
        print(f"📥 Downloading: {yt.title}")

        # Choose highest resolution progressive stream (video + audio)
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
        stream.download(output_path=DOWNLOAD_DIR)

        print(f"✅ Done: {yt.title}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
