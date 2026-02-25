"""
Video Compiler - Combines images + audio + music + subtitles into final MP4
Uses FFmpeg and MoviePy with Ken Burns zoom/pan effects
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

    def _apply_ken_burns(self, clip, effect_type: str, duration: float):
        """
        Apply Ken Burns zoom/pan effect to make still images feel alive.
        Creates gentle, slow movement over the duration of each scene.
        """
        import numpy as np

        w, h = clip.size

        # Scale factor: how much to zoom (1.0 = no zoom, 1.15 = 15% zoom)
        zoom_range = 0.12  # subtle 12% zoom

        if effect_type == "zoom_in":
            def make_frame(get_frame, t):
                progress = t / duration
                scale = 1.0 + (zoom_range * progress)
                frame = get_frame(t)
                from PIL import Image
                img = Image.fromarray(frame)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                # Center crop back to original size
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
                return np.array(img)
            return clip.transform(make_frame)

        elif effect_type == "zoom_out":
            def make_frame(get_frame, t):
                progress = t / duration
                scale = (1.0 + zoom_range) - (zoom_range * progress)
                frame = get_frame(t)
                from PIL import Image
                img = Image.fromarray(frame)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
                return np.array(img)
            return clip.transform(make_frame)

        elif effect_type == "pan_left":
            def make_frame(get_frame, t):
                progress = t / duration
                frame = get_frame(t)
                from PIL import Image
                img = Image.fromarray(frame)
                scale = 1.0 + zoom_range
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                # Pan from right to left
                max_offset = new_w - w
                left = int(max_offset * (1 - progress))
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
                return np.array(img)
            return clip.transform(make_frame)

        elif effect_type == "pan_right":
            def make_frame(get_frame, t):
                progress = t / duration
                frame = get_frame(t)
                from PIL import Image
                img = Image.fromarray(frame)
                scale = 1.0 + zoom_range
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                # Pan from left to right
                max_offset = new_w - w
                left = int(max_offset * progress)
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
                return np.array(img)
            return clip.transform(make_frame)

        return clip  # No effect

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
        from moviepy import (
            ImageClip, AudioFileClip, CompositeAudioClip,
            concatenate_videoclips, concatenate_audioclips,
            vfx, afx
        )

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

        # Ken Burns effect types — cycle through them for variety
        kb_effects = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

        # Build video clips from images
        clips = []
        transition = self.video_config.get("transition", "fade")
        crossfade_duration = 1.0

        for i, img_path in enumerate(image_paths):
            clip = ImageClip(img_path, duration=time_per_image)
            clip = clip.resized((self.width, self.height))

            # Apply Ken Burns effect — alternate effects for visual variety
            effect = kb_effects[i % len(kb_effects)]
            clip = self._apply_ken_burns(clip, effect, time_per_image)

            if transition == "fade" and i > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(crossfade_duration)])

            clips.append(clip)

        # Concatenate all clips
        padding = -crossfade_duration if transition == "fade" else 0
        video = concatenate_videoclips(clips, method="compose", padding=padding)
        video = video.with_duration(total_duration)

        # Add background music
        bg_music_path = self._get_background_music()
        if bg_music_path:
            try:
                bg_music = AudioFileClip(bg_music_path)
                # Loop music if shorter than video
                if bg_music.duration < total_duration:
                    loops = int(total_duration / bg_music.duration) + 1
                    bg_music = concatenate_audioclips([bg_music] * loops)
                bg_music = bg_music.subclipped(0, total_duration)
                # Apply volume and fade
                music_vol = self.music_config.get("volume", 0.15)
                fade_in = self.music_config.get("fade_in", 3)
                fade_out = self.music_config.get("fade_out", 5)
                bg_music = bg_music.with_effects([
                    afx.MultiplyVolume(music_vol),
                    afx.AudioFadeIn(fade_in),
                    afx.AudioFadeOut(fade_out),
                ])
                # Mix narration + music
                final_audio = CompositeAudioClip([narration, bg_music])
            except Exception as e:
                logger.warning(f"Background music failed, using narration only: {e}")
                final_audio = narration
        else:
            final_audio = narration

        video = video.with_audio(final_audio)

        # Write final video with higher quality settings
        logger.info("Rendering final video (this may take a few minutes)...")
        video.write_videofile(
            output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            bitrate="8000k",
            preset="slow",
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
