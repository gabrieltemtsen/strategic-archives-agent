"""
Channel Loader
Reads all channel info from a single CHANNELS env var (JSON array).

Each channel only needs: key, name, youtube_id, niche, upload_time
The agent infers content style from the niche string via Gemini.

Example env var:
CHANNELS=[
  {
    "key": "kids_universe",
    "name": "KidsUniverseFirst",
    "youtube_id": "UCxxxxxxxxxxxxxxxxxx",
    "niche": "kids education - bedtime stories and fun facts for children aged 3-10",
    "upload_time": "18:00",
    "made_for_kids": true,
    "emoji": "🧒"
  },
  {
    "key": "strategic_archives",
    "name": "StrategicArchives",
    "youtube_id": "UCxxxxxxxxxxxxxxxxxx",
    "niche": "AI tools and workflows for everyday people",
    "upload_time": "17:00",
    "emoji": "📺"
  }
]
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_channels(config: dict) -> dict[str, dict]:
    """
    Load channels from CHANNELS env var (JSON array).
    Returns {key: enriched_channel_dict}
    Raises ValueError if CHANNELS is missing or invalid.
    """
    raw = os.getenv("CHANNELS", "").strip()

    if not raw:
        raise ValueError(
            "CHANNELS env var not set.\n"
            "Set it as a JSON array, e.g.:\n"
            'CHANNELS=[{"key":"kids_universe","name":"KidsUniverseFirst",'
            '"youtube_id":"UCxxx","niche":"kids education","upload_time":"18:00"}]'
        )

    try:
        channels_list = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"CHANNELS env var is not valid JSON: {e}")

    if not isinstance(channels_list, list) or not channels_list:
        raise ValueError("CHANNELS must be a non-empty JSON array.")

    defaults = config.get("defaults", {})
    active = {}

    for ch in channels_list:
        key = ch.get("key", "").strip()
        name = ch.get("name", key)
        niche = ch.get("niche", "").strip()
        youtube_id = ch.get("youtube_id", "").strip()

        if not key:
            logger.warning(f"Skipping channel with no 'key': {ch}")
            continue
        if not niche:
            logger.warning(f"Channel '{key}' has no niche — Gemini will use generic content")

        enriched = {
            # Core identity
            "key":         key,
            "name":        name,
            "niche":       niche,
            "emoji":       ch.get("emoji", "📺"),
            "upload_time": ch.get("upload_time", "18:00"),

            # YouTube
            "_youtube_channel_id": youtube_id,
            "_timezone": config.get("app", {}).get("timezone", "Africa/Lagos"),
            "youtube": {
                **defaults.get("youtube", {}),
                "made_for_kids": ch.get("made_for_kids", False),
                "category_id": ch.get("category_id", "22"),
                "default_language": ch.get("language", "en"),
                "tags": ch.get("tags", []),
            },

            # Inherit global defaults (channel can override)
            "tts":     {**defaults.get("tts", {}),     **ch.get("tts", {})},
            "visuals": {**defaults.get("visuals", {}), **ch.get("visuals", {})},
            "music":   {**defaults.get("music", {}),   **ch.get("music", {})},
            "content": {
                "video":     {**defaults.get("video", {}), **ch.get("video", {})},
                "languages": {
                    "primary":   ch.get("language", "en"),
                    "supported": ch.get("languages", ["en"]),
                },
            },
        }

        if not youtube_id:
            logger.warning(
                f"  ⚠️  Channel '{key}' has no youtube_id — "
                "uploads will use the default authenticated channel."
            )

        active[key] = enriched
        logger.info(
            f"  ✅ Channel loaded: [{key}] {enriched['emoji']} {name} "
            f"| niche: \"{niche[:50]}\" | upload: {enriched['upload_time']} WAT"
        )

    if not active:
        raise ValueError("No valid channels found in CHANNELS env var.")

    return active


def get_channel_menu(active_channels: dict) -> str:
    """Readable menu string for Telegram messages."""
    lines = []
    for ch in active_channels.values():
        lines.append(
            f"{ch['emoji']} <b>{ch['name']}</b>\n"
            f"   <i>{ch['niche'][:80]}</i>"
        )
    return "\n\n".join(lines)
