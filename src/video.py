"""
Video Compiler - Combines images + audio + music + subtitles into final MP4
Uses FFmpeg and MoviePy
"""

import os
import logging
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VideoCompiler:
    def __init__(self, config: dict, output_dir: str = "./output", assets_dir: str = "./assets"):
        self.config = config
        self.video_config = config.get("content", {}).get("video", {})
        self.music_config = config.get("music", {})
        self.output_dir = Path(output_dir)
        self.assets_dir = Path(assets_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resolution = self.video_config.get("resolution", "1920x1080")
        self.width, self.height = map(int, self.resolution.split("x"))
        self.fps = self.video_config.get("fps", 24)

    def _get_background_music(self) -> Optional[str]:
        """Get a random background music file from assets."""
        music_dir = self.assets_dir / "music"
        if not music_dir.exists():
            return None
        music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
        return str(random.choice(music_files)) if music_files else None

    def compile(
        self,
        image_paths: List[str],
        audio_path: str,
        job_id: str,
        title: str = "",
        subtitle_text: Optional[str] = None,
    ) -> str:
        """
        Compile final video from images + audio.
        Returns path to output MP4 file.
        """
        from moviepy.editor import (
            ImageClip, AudioFileClip, CompositeAudioClip,
            AudioFileClip as BGMusic, concatenate_videoclips,
            afx
        )
        import numpy as np

        output_path = str(self.output_dir / f"{job_id}_final.mp4")
        logger.info(f"Compiling video: {len(image_paths)} scenes → {output_path}")

        # Load audio to get total duration
        narration = AudioFileClip(audio_path)
        total_duration = narration.duration
        logger.info(f"Audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

        # Calculate duration per image
        n_images = len(image_paths)
        time_per_image = total_duration / n_images
        logger.info(f"~{time_per_image:.1f}s per image across {n_images} images")

        # Build video clips from images
        clips = []
        transition = self.video_config.get("transition", "fade")

        for i, img_path in enumerate(image_paths):
            clip = ImageClip(img_path, duration=time_per_image)
            clip = clip.resize((self.width, self.height))

            if transition == "fade" and i > 0:
                clip = clip.crossfadein(0.5)

            clips.append(clip)

        # Concatenate all clips
        video = concatenate_videoclips(clips, method="compose", padding=-0.5 if transition == "fade" else 0)
        video = video.set_duration(total_duration)

        # Add background music
        bg_music_path = self._get_background_music()
        if bg_music_path:
            try:
                bg_music = BGMusic(bg_music_path)
                # Loop music if shorter than video
                if bg_music.duration < total_duration:
                    loops = int(total_duration / bg_music.duration) + 1
                    from moviepy.editor import concatenate_audioclips
                    bg_music = concatenate_audioclips([bg_music] * loops)
                bg_music = bg_music.subclip(0, total_duration)
                # Apply volume and fade
                music_vol = self.music_config.get("volume", 0.15)
                fade_in = self.music_config.get("fade_in", 3)
                fade_out = self.music_config.get("fade_out", 5)
                bg_music = (bg_music
                            .volumex(music_vol)
                            .audio_fadein(fade_in)
                            .audio_fadeout(fade_out))
                # Mix narration + music
                final_audio = CompositeAudioClip([narration, bg_music])
            except Exception as e:
                logger.warning(f"Background music failed, using narration only: {e}")
                final_audio = narration
        else:
            final_audio = narration

        video = video.set_audio(final_audio)

        # Write final video
        logger.info("Rendering final video (this may take a few minutes)...")
        video.write_videofile(
            output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            bitrate="4000k",
            preset="medium",
            threads=4,
            logger=None  # Suppress moviepy progress (use our own logger)
        )

        # Cleanup clips
        video.close()
        narration.close()
        logger.info(f"Video compiled successfully: {output_path}")
        return output_path

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata using ffprobe."""
        import subprocess, json
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ], capture_output=True, text=True)
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        return {
            "duration": float(fmt.get("duration", 0)),
            "size_mb": int(fmt.get("size", 0)) / (1024 * 1024),
            "format": fmt.get("format_name", ""),
        }
