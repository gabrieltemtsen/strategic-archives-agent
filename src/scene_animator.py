"""
Scene Animator — Higgsfield API client
Turns FLUX-generated scene images into cinematic video clips.

Models used:
  - kling-video/v2.1/pro/image-to-video  → character scenes (best quality)
  - higgsfield-ai/dop/standard           → establishing/action shots (faster)
"""

import os
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://platform.higgsfield.ai"

# Model routing by scene type
MODELS = {
    "character":    "kling-video/v2.1/pro/image-to-video",
    "establishing": "higgsfield-ai/dop/standard",
    "action":       "higgsfield-ai/dop/standard",
    "montage":      "higgsfield-ai/dop/standard",
    "default":      "higgsfield-ai/dop/standard",
}


class SceneAnimator:
    def __init__(self, output_dir: str = "./output"):
        key_id = os.getenv("HIGGSFIELD_KEY_ID", "")
        secret = os.getenv("HIGGSFIELD_SECRET", "")
        if not key_id or not secret:
            raise RuntimeError(
                "HIGGSFIELD_KEY_ID and HIGGSFIELD_SECRET must be set.\n"
                "Get them from: https://cloud.higgsfield.ai/settings/api-keys"
            )
        self.auth = f"Key {key_id}:{secret}"
        self.headers = {
            "Authorization": self.auth,
            "Content-Type": "application/json",
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _public_image_url(self, image_path: str) -> str:
        """
        Return a publicly accessible URL for an image.
        Uses the Railway deployment URL to serve files from /output.
        Falls back to imgbb upload if no Railway URL is configured.
        """
        base_url = os.getenv("RAILWAY_PUBLIC_URL", "").rstrip("/")
        if base_url:
            filename = Path(image_path).name
            return f"{base_url}/files/{filename}"

        # Fallback: upload to imgbb
        imgbb_key = os.getenv("IMGBB_API_KEY", "")
        if imgbb_key:
            return self._upload_imgbb(image_path, imgbb_key)

        raise RuntimeError(
            "No way to serve images publicly. Set RAILWAY_PUBLIC_URL "
            "(e.g. https://your-app.up.railway.app) or IMGBB_API_KEY."
        )

    def _upload_imgbb(self, image_path: str, api_key: str) -> str:
        """Upload image to imgbb and return public URL."""
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": api_key, "image": b64},
            timeout=30,
        )
        r.raise_for_status()
        url = r.json()["data"]["url"]
        logger.info(f"Uploaded to imgbb: {url}")
        return url

    def animate_scene(
        self,
        image_path: str,
        motion_prompt: str,
        scene_type: str = "default",
        duration: int = 5,
        job_id: str = "scene",
        scene_idx: int = 0,
    ) -> str:
        """
        Animate a static image into a video clip.

        Args:
            image_path:    Path to the source FLUX-generated image
            motion_prompt: Cinematic motion description
            scene_type:    "character" | "establishing" | "action" | "montage"
            duration:      Clip length in seconds (5 or 10)
            job_id:        Job identifier for output filename
            scene_idx:     Scene index for output filename

        Returns:
            Path to the downloaded .mp4 clip
        """
        model = MODELS.get(scene_type, MODELS["default"])
        image_url = self._public_image_url(image_path)

        logger.info(f"Animating scene {scene_idx} [{scene_type}] with {model.split('/')[0]}...")
        logger.debug(f"  Image: {image_url}")
        logger.debug(f"  Prompt: {motion_prompt[:80]}")

        r = requests.post(
            f"{BASE_URL}/{model}",
            headers=self.headers,
            json={"image_url": image_url, "prompt": motion_prompt, "duration": duration},
            timeout=30,
        )

        if r.status_code not in (200, 201):
            raise RuntimeError(f"Higgsfield submit failed ({r.status_code}): {r.text[:300]}")

        data = r.json()
        request_id = data.get("request_id")
        logger.info(f"  Queued → request_id: {request_id}")

        # Poll until done (max 5 min)
        for attempt in range(60):
            time.sleep(5)
            sr = requests.get(
                f"{BASE_URL}/requests/{request_id}/status",
                headers=self.headers,
                timeout=15,
            )
            d = sr.json()
            status = d.get("status")

            if attempt % 6 == 0:
                logger.info(f"  [{attempt+1}] {status}...")

            if status == "completed":
                video_url = d.get("video", {}).get("url") or d.get("output", {}).get("url", "")
                if not video_url:
                    # scan all string values for an mp4 URL
                    import re
                    matches = re.findall(r'https?://[^\s"\']+\.mp4[^\s"\']*', str(d))
                    video_url = matches[0] if matches else ""

                if not video_url:
                    raise RuntimeError(f"No video URL in response: {d}")

                # Download
                out_path = self.output_dir / f"{job_id}_scene{scene_idx:02d}.mp4"
                vr = requests.get(video_url, timeout=120)
                out_path.write_bytes(vr.content)
                logger.info(f"  ✅ Clip saved: {out_path} ({len(vr.content)//1024}KB)")
                return str(out_path)

            elif status in ("failed", "error"):
                raise RuntimeError(
                    f"Higgsfield render failed for scene {scene_idx}: {d.get('error', d)}"
                )

        raise RuntimeError(f"Timed out waiting for scene {scene_idx} ({request_id})")

    def animate_scenes(
        self,
        scenes: list,
        image_paths: list,
        job_id: str,
    ) -> list:
        """
        Animate a list of scenes in sequence.

        Args:
            scenes:      List of scene dicts from script_gen (with motion_prompt, type, duration)
            image_paths: Parallel list of image paths (one per scene)
            job_id:      Job identifier

        Returns:
            List of clip paths in order
        """
        clip_paths = []
        for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
            try:
                clip = self.animate_scene(
                    image_path=img_path,
                    motion_prompt=scene.get("motion_prompt", "slow cinematic camera push forward"),
                    scene_type=scene.get("type", "default"),
                    duration=scene.get("duration", 5),
                    job_id=job_id,
                    scene_idx=i,
                )
                clip_paths.append(clip)
            except Exception as e:
                logger.error(f"Scene {i} animation failed: {e} — using static fallback")
                # Fallback: convert static image to video with FFmpeg
                fallback = self._static_to_video(img_path, job_id, i, scene.get("duration", 5))
                clip_paths.append(fallback)

        return clip_paths

    def _static_to_video(self, image_path: str, job_id: str, idx: int, duration: int) -> str:
        """Fallback: convert static image to video with basic Ken Burns zoom."""
        import subprocess
        out_path = self.output_dir / f"{job_id}_scene{idx:02d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path,
            "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.001,1.3)':d={duration*25}:s=1920x1080",
            "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Static fallback clip: {out_path}")
        return str(out_path)
