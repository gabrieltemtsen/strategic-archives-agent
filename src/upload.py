"""
YouTube Uploader - Uploads videos via YouTube Data API v3
Handles OAuth2 auth, metadata, thumbnail, and scheduling
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import pytz

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]
TOKEN_PATH = ".youtube_token.pkl"
CLIENT_SECRETS_PATH = "client_secret.json"


class YouTubeUploader:
    def __init__(self, config: dict, channel: dict = None):
        """
        config: full app config
        channel: specific channel config (from config['channels']['key'])
        """
        self.config = config
        # Channel-level youtube config overrides global
        self.yt_config = (channel or {}).get("youtube", config.get("youtube", {}))
        self.channel = channel or {}
        self.service = None

    def _authenticate(self):
        """Handle OAuth2 authentication."""
        creds = None

        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, "rb") as f:
                creds = pickle.load(f)

        if creds and creds.valid:
            pass
        elif creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PATH, "wb") as f:
                pickle.dump(creds, f)
        else:
            if not os.path.exists(CLIENT_SECRETS_PATH):
                raise FileNotFoundError(
                    f"YouTube OAuth client secrets not found at '{CLIENT_SECRETS_PATH}'. "
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
            creds = flow.run_local_server(port=8080)
            with open(TOKEN_PATH, "wb") as f:
                pickle.dump(creds, f)

        self.service = build("youtube", "v3", credentials=creds)
        logger.info("YouTube API authenticated ✓")

    def _build_schedule_time(self, upload_time: str, tz_name: str) -> str:
        """Convert 'HH:MM' + timezone to RFC3339 UTC publish time (next occurrence)."""
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
        hour, minute = map(int, upload_time.split(":"))
        scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if scheduled <= now:
            from datetime import timedelta
            scheduled += timedelta(days=1)
        return scheduled.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list,
        thumbnail_path: Optional[str] = None,
        language: str = "en",
        schedule: bool = True,
        video_format: str = "long",
    ) -> dict:
        """
        Upload a video to YouTube.
        Returns dict with video_id and video_url.
        """
        if not self.service:
            self._authenticate()

        # Channel upload_time takes priority over app-level
        upload_time = self.channel.get("upload_time") \
                      or self.config.get("app", {}).get("upload_time", "18:00")
        tz_name = self.channel.get("_timezone") \
                  or self.config.get("app", {}).get("timezone", "Africa/Lagos")

        # Build tags
        default_tags = self.yt_config.get("tags", [])
        all_tags = list(set(default_tags + tags))[:500]  # YouTube limit

        # Shorts-specific adjustments
        is_short = video_format == "short"
        if is_short:
            if "#Shorts" not in title:
                title = f"{title} #Shorts"
            all_tags = list(set(all_tags + ["Shorts", "YouTubeShorts", "KidsShorts"]))[:500]

        # Build snippet
        snippet = {
            "title": title[:100],  # YouTube 100 char limit
            "description": f"{description}\n\n"
                           f"#KidsVideos #ChildrensContent #EducationalKids"
                           + (" #Shorts" if is_short else ""),
            "tags": all_tags,
            "categoryId": self.yt_config.get("category_id", "22"),
            "defaultLanguage": language,
        }

        # Status — shorts go public immediately, long-form can be scheduled
        if not is_short and schedule:
            publish_at = self._build_schedule_time(upload_time, tz_name)
            status = {
                "privacyStatus": "private",
                "publishAt": publish_at,
                "selfDeclaredMadeForKids": self.yt_config.get("made_for_kids", True),
            }
            logger.info(f"Video scheduled for: {publish_at}")
        else:
            publish_at = None
            status = {
                "privacyStatus": self.yt_config.get("privacy_status", "public"),
                "selfDeclaredMadeForKids": self.yt_config.get("made_for_kids", True),
            }

        body = {"snippet": snippet, "status": status}

        # Upload video
        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=50 * 1024 * 1024  # 50MB chunks
        )

        logger.info(f"Uploading '{title}' to YouTube...")
        request = self.service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status_obj, response = request.next_chunk()
            if status_obj:
                progress = int(status_obj.progress() * 100)
                logger.info(f"Upload progress: {progress}%")

        video_id = response["id"]
        video_url = f"https://youtu.be/{video_id}"
        logger.info(f"Upload complete: {video_url}")

        # Upload thumbnail if provided (YouTube limit: 2MB)
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                # Compress if over 2MB
                file_size = os.path.getsize(thumbnail_path)
                upload_path = thumbnail_path
                if file_size > 2 * 1024 * 1024:
                    from PIL import Image
                    img = Image.open(thumbnail_path).convert("RGB")
                    upload_path = thumbnail_path.replace(".png", "_thumb.jpg")
                    quality = 85
                    while quality > 20:
                        img.save(upload_path, "JPEG", quality=quality)
                        if os.path.getsize(upload_path) <= 2 * 1024 * 1024:
                            break
                        quality -= 10
                    logger.info(f"Thumbnail compressed: {file_size//1024}KB → {os.path.getsize(upload_path)//1024}KB")

                self.service.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(upload_path, mimetype="image/jpeg" if upload_path.endswith(".jpg") else "image/png")
                ).execute()
                logger.info("Thumbnail uploaded ✓")
            except Exception as e:
                logger.warning(f"Thumbnail upload failed: {e}")

        return {
            "video_id": video_id,
            "video_url": video_url,
            "title": title,
            "scheduled_for": publish_at if publish_at else "immediate",
        }
