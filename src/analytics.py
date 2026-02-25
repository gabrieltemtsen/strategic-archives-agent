"""
YouTube Analytics - Fetch video performance data
Reuses existing YouTube OAuth credentials
"""

import os
import logging
import pickle
from typing import Optional, List

logger = logging.getLogger(__name__)

TOKEN_PATH = ".youtube_token.pkl"


class YouTubeAnalytics:
    """Fetch video statistics from YouTube Data API."""

    def __init__(self):
        self.service = None

    def _authenticate(self):
        """Reuse existing YouTube OAuth token."""
        if not os.path.exists(TOKEN_PATH):
            logger.warning("YouTube token not found — analytics unavailable")
            return False

        try:
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request

            with open(TOKEN_PATH, "rb") as f:
                creds = pickle.load(f)

            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(TOKEN_PATH, "wb") as f:
                    pickle.dump(creds, f)

            self.service = build("youtube", "v3", credentials=creds)
            return True
        except Exception as e:
            logger.error(f"YouTube auth failed: {e}")
            return False

    def get_video_stats(self, video_ids: List[str]) -> List[dict]:
        """Get statistics for multiple videos."""
        if not self.service and not self._authenticate():
            return []

        if not video_ids:
            return []

        try:
            # YouTube API accepts comma-separated IDs (max 50)
            results = []
            for batch_start in range(0, len(video_ids), 50):
                batch = video_ids[batch_start:batch_start + 50]
                response = self.service.videos().list(
                    part="snippet,statistics,status",
                    id=",".join(batch)
                ).execute()

                for item in response.get("items", []):
                    stats = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    status = item.get("status", {})
                    results.append({
                        "video_id": item["id"],
                        "title": snippet.get("title", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                        "privacy": status.get("privacyStatus", ""),
                        "views": int(stats.get("viewCount", 0)),
                        "likes": int(stats.get("likeCount", 0)),
                        "comments": int(stats.get("commentCount", 0)),
                        "url": f"https://youtu.be/{item['id']}"
                    })

            return results
        except Exception as e:
            logger.error(f"Failed to fetch video stats: {e}")
            return []

    def get_channel_stats(self) -> Optional[dict]:
        """Get channel overview statistics."""
        if not self.service and not self._authenticate():
            return None

        try:
            response = self.service.channels().list(
                part="snippet,statistics",
                mine=True
            ).execute()

            if not response.get("items"):
                return None

            item = response["items"][0]
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            return {
                "channel_name": snippet.get("title", ""),
                "channel_id": item["id"],
                "subscribers": int(stats.get("subscriberCount", 0)),
                "total_views": int(stats.get("viewCount", 0)),
                "total_videos": int(stats.get("videoCount", 0)),
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            }
        except Exception as e:
            logger.error(f"Failed to fetch channel stats: {e}")
            return None
