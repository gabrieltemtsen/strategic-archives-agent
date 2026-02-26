"""
Channel Loader
Reads active channels from ACTIVE_CHANNELS env var (comma-separated keys).
Resolves each channel's YouTube channel ID from YOUTUBE_CHANNEL_ID_<KEY>.

ENV example:
  ACTIVE_CHANNELS=kids_universe,ai_tools,ai_side_hustles,african_folklore
  YOUTUBE_CHANNEL_ID_KIDS_UNIVERSE=UCxxxxxxxxxxxxxxxxxx
  YOUTUBE_CHANNEL_ID_AI_TOOLS=UCxxxxxxxxxxxxxxxxxx
  YOUTUBE_CHANNEL_ID_AI_SIDE_HUSTLES=UCxxxxxxxxxxxxxxxxxx
  YOUTUBE_CHANNEL_ID_AFRICAN_FOLKLORE=UCxxxxxxxxxxxxxxxxxx
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_active_channels(config: dict) -> dict[str, dict]:
    """
    Load active channel configs from ACTIVE_CHANNELS env var.

    Returns: {channel_key: enriched_channel_config}
    Raises: ValueError if ACTIVE_CHANNELS contains unknown keys.
    """
    raw = os.getenv("ACTIVE_CHANNELS", "").strip()
    all_channels = config.get("channels", {})

    if not raw:
        # Fall back to default channel
        default = config.get("app", {}).get("default_channel", "kids_universe")
        logger.warning(
            f"ACTIVE_CHANNELS not set — defaulting to '{default}'. "
            f"Set ACTIVE_CHANNELS=kids_universe,ai_tools etc. in your env."
        )
        raw = default

    keys = [k.strip() for k in raw.split(",") if k.strip()]
    active = {}

    for key in keys:
        if key not in all_channels:
            logger.warning(
                f"Channel '{key}' in ACTIVE_CHANNELS not found in config.yaml — skipping. "
                f"Available: {list(all_channels.keys())}"
            )
            continue
        channel = _enrich(key, all_channels[key], config)
        active[key] = channel
        logger.info(
            f"  ✅ Loaded channel: [{key}] {channel['name']} "
            f"({channel['niche']}) @ {channel.get('upload_time', 'N/A')} WAT"
        )

    if not active:
        raise ValueError(
            "No valid channels loaded. Check ACTIVE_CHANNELS matches keys in config.yaml."
        )

    return active


def _enrich(key: str, channel: dict, config: dict) -> dict:
    """Add runtime-resolved fields to a channel config."""
    enriched = channel.copy()
    enriched["_key"] = key

    # Resolve YouTube channel ID from env
    env_var = f"YOUTUBE_CHANNEL_ID_{key.upper()}"
    yt_id = os.getenv(env_var, "").strip()
    if not yt_id:
        logger.warning(
            f"  ⚠️  {env_var} not set for channel '{key}'. "
            "Upload will use the default authenticated channel."
        )
    enriched["_youtube_channel_id"] = yt_id

    # Inherit global timezone
    enriched["_timezone"] = config.get("app", {}).get("timezone", "Africa/Lagos")

    return enriched


def get_channel(config: dict, key: str) -> Optional[dict]:
    """Get a single channel by key (enriched)."""
    all_channels = config.get("channels", {})
    if key not in all_channels:
        return None
    return _enrich(key, all_channels[key], config)


def list_channel_menu(active_channels: dict) -> str:
    """Format a readable menu of active channels for Telegram."""
    lines = []
    for key, ch in active_channels.items():
        emoji = ch.get("emoji", "📺")
        lines.append(f"{emoji} *{ch['name']}* — _{ch.get('niche', '').replace('_', ' ')}_")
    return "\n".join(lines)
