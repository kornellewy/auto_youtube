"""
youtube_uploader.py – Single-call wrapper around the original upload_video_final.py
"""
import httplib2
import random
import sys
import time
from pathlib import Path

from apiclient.discovery import build
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow

# ------------------------------------------------------------------
# Constants (unchanged from original)
# ------------------------------------------------------------------
httplib2.RETRIES = 1
MAX_RETRIES = 10
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError)
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")

# ------------------------------------------------------------------
class YoutubeUploader:
    """
    One-shot YouTube uploader with optional thumbnail & playlist addition.
    """
    def __init__(self,
                 client_secrets_file: str | Path = "client_secrets.json",
                 category: str = "22",
                 privacy_status: str = "private",
                 made_for_kids: bool = False,
                 publish_at: str | None = None,
                 playlist_id: str | None = None):
        """
        Default values you rarely change can be set once here.
        """
        self.client_secrets_file = Path(client_secrets_file)
        self.category = category
        self.privacy_status = privacy_status
        self.made_for_kids = made_for_kids
        self.publish_at = publish_at
        self.playlist_id = playlist_id

        self.youtube = self._get_authenticated_service()

    # --------------------------------------------------------------
    # Public entry point
    # --------------------------------------------------------------
    def upload(self,
               video_path: str | Path,
               thumbnail_path: str | Path | None = None,
               title: str = "Untitled Video",
               description: str = "",
               tags: list[str] | None = None,
               privacy_status: str | None = "private",
               publish_at: str | None = None,
               playlist_id: str | None = None) -> str:
        """
        Upload a video, optionally add a thumbnail and add it to a playlist.
        Returns the new video ID.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if thumbnail_path:
            thumbnail_path = Path(thumbnail_path)
            if not thumbnail_path.is_file():
                print(f"Warning: thumbnail not found: {thumbnail_path}")
                thumbnail_path = None

        privacy_status = privacy_status or self.privacy_status
        publish_at = publish_at or self.publish_at
        playlist_id = playlist_id or self.playlist_id

        video_id = self._upload_video(
            video_path=video_path,
            title=title,
            description=description,
            tags=tags or [],
            privacy_status=privacy_status,
            publish_at=publish_at
        )

        if thumbnail_path:
            self._add_thumbnail(video_id, thumbnail_path)

        if playlist_id and playlist_id != "PL_ID_GOES_HERE":
            self._add_video_to_playlist(video_id, playlist_id)

        return video_id

    # --------------------------------------------------------------
    # Internal helpers (mostly lifted from original script)
    # --------------------------------------------------------------
    def _get_authenticated_service(self):
        if not self.client_secrets_file.exists():
            sys.exit(f"Client secrets file not found: {self.client_secrets_file}")

        flow = flow_from_clientsecrets(
            str(self.client_secrets_file),
            scope=YOUTUBE_UPLOAD_SCOPE,
            message=f"Missing {self.client_secrets_file}"
        )
        token_path = Path(__file__).with_suffix(".oauth2.json")
        storage = Storage(str(token_path))
        credentials = storage.get()

        if credentials is None or credentials.invalid:
            credentials = run_flow(flow, storage, None)

        return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                     http=credentials.authorize(httplib2.Http()))

    def _upload_video(self, *, video_path, title, description, tags,
                      privacy_status, publish_at) -> str:
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": self.category,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": self.made_for_kids,
            },
        }
        if publish_at:
            body["status"]["publishAt"] = publish_at

        insert_request = self.youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
        )

        return self._resumable_upload(insert_request)

    def _resumable_upload(self, insert_request):
        response = None
        error = None
        retry = 0
        while response is None:
            try:
                print("Uploading file…")
                status, response = insert_request.next_chunk()
                if response and "id" in response:
                    print(f"Video id '{response['id']}' was successfully uploaded.")
                    return response["id"]
                else:
                    sys.exit(f"Upload failed with unexpected response: {response}")
            except HttpError as e:
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    error = f"Retriable HTTP error {e.resp.status}:\n{e.content}"
                else:
                    raise
            except RETRIABLE_EXCEPTIONS as e:
                error = f"Retriable error: {e}"

            if error:
                print(error)
                retry += 1
                if retry > MAX_RETRIES:
                    sys.exit("No longer attempting to retry.")
                sleep_seconds = random.random() * (2 ** retry)
                print(f"Sleeping {sleep_seconds:.2f}s and then retrying…")
                time.sleep(sleep_seconds)
                error = None

    def _add_thumbnail(self, video_id, thumbnail_path):
        print("Uploading thumbnail…")
        request = self.youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path))
        )
        request.execute()
        print("Thumbnail was successfully uploaded.")

    def _add_video_to_playlist(self, video_id, playlist_id):
        print(f"Adding video '{video_id}' to playlist '{playlist_id}'…")
        request = self.youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        )
        request.execute()
        print("Video was successfully added to the playlist.")

if __name__ == "__main__":
    yt_uploader = YoutubeUploader()
    yt_uploader.upload(
        video_path=Path("output/smol_vma_only_audio.mp4"),
        thumbnail_path=Path("/media/kornellewy/jan_dysk_3/auto_youtube/media/raw/Google_AI_Studio_2025-08-08T13_22_02.918Z.jpg"),
        title="My Awesome Video",
        description="Uploaded with pathlib-only script",
        tags="python,automation,youtube" ,
        publish_at="2025-10-27T12:00:00.0Z" 
    )