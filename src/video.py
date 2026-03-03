"""
Video Compiler
Stitches animated clips (from Higgsfield) + TTS audio into a final MP4.
Falls back to static image compilation when Higgsfield clips aren't available.
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VideoCompiler:
    def __init__(
        self,
        channel: dict,
        output_dir: str = "./output",
        assets_dir: str = "./assets",
        video_format: str = "long",
    ):
        self.channel = channel
        self.output_dir = Path(output_dir)
        self.assets_dir = Path(assets_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_format = video_format

        if video_format == "short":
            self.width, self.height = 1080, 1920
        else:
            self.width, self.height = 1920, 1080
        self.fps = 24

    # ── Primary: Clip-based compilation (Higgsfield output) ──────────────────

    def compile_from_clips(
        self,
        clip_paths: List[str],
        audio_path: str,
        job_id: str,
        title: str = "",
    ) -> str:
        """
        Stitch animated video clips + TTS audio into a final MP4.

        Args:
            clip_paths: List of .mp4 clip paths from Higgsfield
            audio_path: TTS narration audio (.mp3 or .wav)
            job_id:     Job identifier
            title:      Episode title (for logging)

        Returns:
            Path to the final compiled video
        """
        logger.info(f"Compiling {len(clip_paths)} clips → final video...")

        # 1. Write concat list
        concat_file = self.output_dir / f"{job_id}_concat.txt"
        with open(concat_file, "w") as f:
            for clip in clip_paths:
                f.write(f"file '{Path(clip).resolve()}'\n")

        # 2. Concatenate clips (mute original audio — we add TTS instead)
        raw_video = self.output_dir / f"{job_id}_raw.mp4"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-an",                          # drop original audio from clips
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                   f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:black",
            str(raw_video),
        ]
        self._run(concat_cmd, "Concat clips")

        # 3. Get total video duration
        video_duration = self._get_duration(str(raw_video))
        audio_duration = self._get_duration(audio_path)
        logger.info(f"Video: {video_duration:.1f}s | Audio: {audio_duration:.1f}s")

        # 4. Merge TTS audio onto video (loop video if audio is longer)
        final_path = self.output_dir / f"{job_id}_final.mp4"
        if audio_duration > video_duration:
            # Audio longer — loop last clip to fill
            logger.info("Audio longer than video — looping final clip to fill...")
            extended = self._extend_video(str(raw_video), audio_duration, job_id)
            merge_input = extended
        else:
            merge_input = str(raw_video)

        merge_cmd = [
            "ffmpeg", "-y",
            "-i", merge_input,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(final_path),
        ]
        self._run(merge_cmd, "Merge audio")

        # 5. Add cinematic colour grade (subtle LUT-style via ffmpeg)
        graded_path = self.output_dir / f"{job_id}_graded.mp4"
        grade_cmd = [
            "ffmpeg", "-y",
            "-i", str(final_path),
            "-vf", "eq=contrast=1.05:brightness=0.02:saturation=1.1,"
                   "unsharp=5:5:0.5:3:3:0.0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            str(graded_path),
        ]
        try:
            self._run(grade_cmd, "Colour grade")
            output = str(graded_path)
        except Exception:
            logger.warning("Colour grading failed — using ungraded video")
            output = str(final_path)

        # Cleanup temp files
        for f in [concat_file, raw_video]:
            try: Path(f).unlink(missing_ok=True)
            except: pass

        size_mb = Path(output).stat().st_size / (1024 * 1024)
        logger.info(f"✅ Video compiled: {output} ({size_mb:.1f}MB)")
        return output

    # ── Legacy: Image-based compilation (Ken Burns fallback) ─────────────────

    def compile(
        self,
        image_paths: List[str],
        audio_path: str,
        job_id: str,
        title: str = "",
    ) -> str:
        """
        Legacy fallback: compile static images with Ken Burns zoom effect.
        Used when Higgsfield animation is unavailable.
        """
        logger.info(f"[Legacy] Compiling {len(image_paths)} images with Ken Burns...")

        audio_dur = self._get_duration(audio_path)
        scene_dur = max(3.0, audio_dur / max(len(image_paths), 1))

        clip_paths = []
        for i, img in enumerate(image_paths):
            clip_path = self.output_dir / f"{job_id}_kbclip{i:02d}.mp4"
            direction = ["in", "out", "left", "right"][i % 4]
            zoom_filter = self._ken_burns_filter(direction, scene_dur)
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", img,
                "-vf", f"{zoom_filter},scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                       f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:black",
                "-t", str(scene_dur),
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                str(clip_path),
            ]
            self._run(cmd, f"Ken Burns clip {i}")
            clip_paths.append(str(clip_path))

        return self.compile_from_clips(clip_paths, audio_path, job_id, title)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ken_burns_filter(self, direction: str, duration: float) -> str:
        frames = int(duration * self.fps)
        if direction == "in":
            return f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps}"
        elif direction == "out":
            return f"zoompan=z='if(lte(zoom,1.0),1.3,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps}"
        elif direction == "left":
            return f"zoompan=z='1.2':x='if(gte(x,iw-iw/zoom),iw-iw/zoom,x+1)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps}"
        else:
            return f"zoompan=z='1.2':x='if(lte(x,0),0,x-1)':y='ih/2-(ih/zoom/2)':d={frames}:s={self.width}x{self.height}:fps={self.fps}"

    def _extend_video(self, video_path: str, target_duration: float, job_id: str) -> str:
        """Loop a video to reach target_duration."""
        out_path = self.output_dir / f"{job_id}_extended.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-t", str(target_duration),
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        self._run(cmd, "Extend video")
        return str(out_path)

    def _get_duration(self, path: str) -> float:
        """Get media duration in seconds via ffprobe."""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15,
            )
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    def get_video_info(self, path: str) -> dict:
        duration = self._get_duration(path)
        size_mb = Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0
        return {"duration": duration, "size_mb": size_mb}

    def _run(self, cmd: list, label: str):
        """Run an FFmpeg command, raising on failure."""
        logger.debug(f"  FFmpeg [{label}]: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"FFmpeg [{label}] failed:\n{result.stderr[-500:]}")
            raise RuntimeError(f"FFmpeg [{label}] failed: {result.stderr[-200:]}")
        logger.debug(f"  ✅ [{label}] done")
