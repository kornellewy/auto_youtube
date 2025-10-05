"""
youtube_full_scheduler.py
Single-file drop-in that:
1. Authenticates once (shared helper)
2. Lists all scheduled videos (private + future publishAt)
3. Builds a calendar of open slots for given weekdays
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict
import httplib2

from apiclient.discovery import build
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow

# ------------------------------------------------------------------
# Constants shared with legacy uploader
# ------------------------------------------------------------------
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
CLIENT_SECRETS_FILE = Path("client_secrets.json")

# ------------------------------------------------------------------
# Shared authentication helper
# ------------------------------------------------------------------
def _get_authenticated_service(
    secrets_file: Path = CLIENT_SECRETS_FILE,
    service_name: str = YOUTUBE_API_SERVICE_NAME,
    version: str = YOUTUBE_API_VERSION,
    scope: str = YOUTUBE_UPLOAD_SCOPE,
) -> "googleapiclient.discovery.Resource":
    if not secrets_file.exists():
        raise FileNotFoundError(secrets_file)

    flow = flow_from_clientsecrets(str(secrets_file), scope=scope)
    storage_path = Path(__file__).with_suffix(".oauth2.json")
    storage = Storage(str(storage_path))
    credentials = storage.get()
    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, None)

    http = credentials.authorize(httplib2.Http())
    return build(service_name, version, http=http)


# ------------------------------------------------------------------
# 1) Scheduler: list videos already queued
# ------------------------------------------------------------------
class YoutubeScheduler:
    def __init__(self, secrets_file: Path = CLIENT_SECRETS_FILE):
        self._service = _get_authenticated_service(secrets_file)

    def list_scheduled_videos(self) -> List[Dict[str, str]]:
        """Return list[dict] with id/title/scheduled_time for private + future videos."""
        videos: List[Dict[str, str]] = []
        next_page = None
        while True:
            resp = (
                self._service.search()
                .list(
                    part="snippet",
                    forMine=True,
                    type="video",
                    maxResults=50,
                    order="date",
                    pageToken=next_page,
                )
                .execute()
            )

            items = resp.get("items", [])
            if not items:
                break

            video_ids = [item["id"]["videoId"] for item in items]
            status_resp = (
                self._service.videos()
                .list(part="status,snippet", id=",".join(video_ids))
                .execute()
            )

            for item in status_resp["items"]:
                status = item["status"]
                if status.get("privacyStatus") == "private" and status.get("publishAt"):
                    pub_at = datetime.fromisoformat(
                        status["publishAt"].replace("Z", "+00:00")
                    )
                    if pub_at > datetime.now(timezone.utc):
                        videos.append(
                            {
                                "id": item["id"],
                                "title": item["snippet"]["title"],
                                "scheduled_time": status["publishAt"],
                            }
                        )

            next_page = resp.get("nextPageToken")
            if not next_page:
                break

        # sort ascending by scheduled_time
        videos.sort(key=lambda v: v["scheduled_time"])
        return videos


# ------------------------------------------------------------------
# 2) Calendar generator
# ------------------------------------------------------------------
_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_weekdays(names: List[str]) -> List[int]:
    try:
        return [_WEEKDAY_MAP[n.strip().lower()] for n in names]
    except KeyError as bad:
        raise ValueError(f"Invalid weekday name: {bad}") from None


def _next_weekday(from_date: datetime, weekday: int) -> datetime:
    """Return the next occurrence of `weekday` on or after `from_date`."""
    days_ahead = weekday - from_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return from_date + timedelta(days=days_ahead)


def generate_calendar(
    target_weekdays: List[str],
    look_ahead_days: int = 30,
    tz: timezone = timezone.utc,
) -> List[datetime]:
    """
    target_weekdays : e.g. ["friday", "saturday"]
    look_ahead_days : how many open slots to return (default 30)
    tz              : timezone for all datetimes (defaults to UTC)

    Returns list[datetime] that are free for scheduling.
    """
    # Fetch already-booked dates
    booked = {
        datetime.fromisoformat(v["scheduled_time"].replace("Z", "+00:00"))
        .astimezone(tz)
        .date()
        for v in YoutubeScheduler().list_scheduled_videos()
    }

    allowed = _parse_weekdays(target_weekdays)
    today = datetime.now(tz).date()
    cursor = max(today, max(booked) if booked else today)

    slots: List[datetime] = []
    while len(slots) < look_ahead_days:
        for wd in allowed:
            candidate = _next_weekday(
                datetime.combine(cursor, datetime.min.time()), wd
            ).date()
            if candidate not in booked:
                slots.append(
                    datetime.combine(candidate, datetime.min.time()).replace(tzinfo=tz)
                )
                booked.add(candidate)
                if len(slots) >= look_ahead_days:
                    break
        cursor += timedelta(days=7)

    return sorted(slots)


# ------------------------------------------------------------------
# CLI quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    calendar = generate_calendar(["friday"], look_ahead_days=8)
    for dt in calendar:
        print(dt.isoformat())