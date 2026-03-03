"""
Music Mixer
Downloads royalty-free background tracks per channel niche and mixes
them under the TTS voiceover at low volume using FFmpeg.
"""

import os
import logging
import subprocess
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Royalty-free tracks from Pixabay CDN (no login required, CC0 license)
MUSIC_TRACKS = {
    "kids_universe": [
        "https://cdn.pixabay.com/download/audio/2022/03/15/audio_1a609c8b64.mp3",   # happy upbeat
        "https://cdn.pixabay.com/download/audio/2021/11/13/audio_cb31e68d5e.mp3",   # playful
    ],
    "strategic_archives": [
        "https://cdn.pixabay.com/download/audio/2022/10/21/audio_c4d9cf4f33.mp3",   # epic orchestral
        "https://cdn.pixabay.com/download/audio/2021/08/04/audio_0625c1539c.mp3",   # dramatic
    ],
    "stories_tales": [
        "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3",   # dark ambient
        "https://cdn.pixabay.com/download/audio/2022/08/02/audio_884fe92c21.mp3",   # eerie
    ],
    "gabe_dev_codes": [
        "https://cdn.pixabay.com/download/audio/2022/11/22/audio_febc508520.mp3",   # techy electronic
        "https://cdn.pixabay.com/download/audio/2021/09/06/audio_bb6a982d39.mp3",   # futuristic
    ],
    "default": [
        "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",   # cinematic ambient
    ],
}


class MusicMixer:
    def __init__(self, channel_key: str, assets_dir: str = "./assets"):
        self.channel_key = channel_key
        self.music_dir = Path(assets_dir) / "music"
        self.music_dir.mkdir(parents=True, exist_ok=True)

    def _get_track(self) -> str | None:
        """Download and cache a random track for the channel."""
        urls = MUSIC_TRACKS.get(self.channel_key, MUSIC_TRACKS["default"])
        url = random.choice(urls)
        filename = url.split("/")[-1].split("?")[0]
        track_path = self.music_dir / f"{self.channel_key}_{filename}"

        if track_path.exists():
            logger.info(f"Using cached music: {track_path.name}")
            return str(track_path)

        logger.info(f"Downloading music track: {url[:60]}...")
        try:
            import requests
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                track_path.write_bytes(r.content)
                logger.info(f"  ✅ Track cached: {track_path} ({len(r.content)//1024}KB)")
                return str(track_path)
            else:
                logger.warning(f"  Music download failed ({r.status_code}) — no music")
                return None
        except Exception as e:
            logger.warning(f"  Music download error: {e} — no music")
            return None

    def mix(self, video_path: str, output_path: str, music_volume: float = 0.12) -> str:
        """
        Mix background music into video at low volume under existing audio.

        Args:
            video_path:    Input video (with TTS voiceover already)
            output_path:   Output path for video with music
            music_volume:  Music level (0.0–1.0), default 12%

        Returns:
            Path to video with music mixed in
        """
        track = self._get_track()
        if not track:
            logger.warning("No music track available — skipping music mix")
            return video_path

        logger.info(f"Mixing music at {int(music_volume*100)}% volume...")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", track,
            "-filter_complex",
            (
                f"[1:a]volume={music_volume},apad[music];"
                f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=3[aout]"
            ),
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            size = Path(output_path).stat().st_size // 1024
            logger.info(f"  ✅ Music mixed → {output_path} ({size}KB)")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Music mix failed: {e.stderr.decode()[:300]}")
            return video_path  # return original if mix fails
