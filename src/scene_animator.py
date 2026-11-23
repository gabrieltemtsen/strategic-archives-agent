"""
Scene Animator — Gemini Video (Veo / Omni Flash) + optional Higgsfield fallback.
Turns generated scene images into cinematic video clips.

Primary:
  - Gemini Video API (`gemini-omni-1.2-flash`, `gemini-omni-1.1-flash`) via GEMINI_API_KEY
Optional Fallback (if HIGGSFIELD_KEY_ID & HIGGSFIELD_SECRET are set):
  - kling-video/v2.1/pro/image-to-video  → character scenes
  - higgsfield-ai/dop/standard           → establishing/action shots
"""

import os
import time
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://platform.higgsfield.ai"


class HiggsFieldOutOfCreditsError(Exception):
    """Raised when Higgsfield returns 403 Not enough credits."""
    pass


class HiggsFieldAuthError(Exception):
    """Raised when Higgsfield returns 401 Unauthorized."""
    pass

# Model routing by scene type
MODELS = {
    "character":    "kling-video/v2.1/pro/image-to-video",
    "establishing": "higgsfield-ai/dop/standard",
    "action":       "higgsfield-ai/dop/standard",
    "montage":      "higgsfield-ai/dop/standard",
    "default":      "higgsfield-ai/dop/standard",
}


GEMINI_VIDEO_MODELS = (
    "gemini-omni-1.2-flash",
    "gemini-omni-1.1-flash",
)


class SceneAnimator:
    def __init__(self, output_dir: str = "./output", aspect_ratio: str = "16:9"):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.higgs_key_id = os.getenv("HIGGSFIELD_KEY_ID", "").strip()
        self.higgs_secret = os.getenv("HIGGSFIELD_SECRET", "").strip()
        self.aspect_ratio = aspect_ratio

        if not self.gemini_key and not (self.higgs_key_id and self.higgs_secret):
            raise RuntimeError(
                "GEMINI_API_KEY (or HIGGSFIELD_KEY_ID + HIGGSFIELD_SECRET) must be set."
            )

        self.auth = f"Key {self.higgs_key_id}:{self.higgs_secret}" if self.higgs_key_id else ""
        self.headers = {
            "Authorization": self.auth,
            "Content-Type": "application/json",
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gemini_exhausted = False
        self.credits_exhausted = False

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

    def _animate_scene_gemini(
        self,
        image_path: str,
        motion_prompt: str,
        job_id: str,
        scene_idx: int,
    ) -> Optional[str]:
        """Animate a scene image into an MP4 clip via Gemini Video API."""
        if self.gemini_exhausted or not self.gemini_key:
            return None

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_key)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = "image/jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "image/png"

        for model_name in GEMINI_VIDEO_MODELS:
            try:
                logger.info(f"Animating scene {scene_idx} with Gemini ({model_name}, {self.aspect_ratio})...")
                operation = client.models.generate_videos(
                    model=model_name,
                    prompt=motion_prompt,
                    image={
                        "image_bytes": image_bytes,
                        "mime_type": mime_type,
                    },
                    config=types.GenerateVideosConfig(
                        aspect_ratio=self.aspect_ratio,
                        number_of_videos=1,
                    ),
                )

                # Poll until done (max ~5 min)
                for attempt in range(30):
                    if getattr(operation, "done", False):
                        break
                    time.sleep(10)
                    operation = client.operations.get(operation)
                    if attempt % 3 == 0:
                        logger.info(f"  [Gemini video {attempt + 1}] rendering...")

                if not getattr(operation, "done", False):
                    raise RuntimeError(f"Gemini video timed out for scene {scene_idx}")

                resp = getattr(operation, "response", None)
                videos = getattr(resp, "generated_videos", None) or []
                if not videos:
                    raise RuntimeError(f"No generated_videos returned by {model_name}")

                out_path = self.output_dir / f"{job_id}_scene{scene_idx:02d}.mp4"
                generated_video = videos[0]
                client.files.download(file=generated_video.video)
                generated_video.video.save(str(out_path))
                logger.info(f"  ✅ Gemini clip saved: {out_path}")
                return str(out_path)

            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Gemini video ({model_name}) failed for scene {scene_idx}: {e}")
                if any(k in err_str for k in ("quota", "429", "resource_exhausted", "billing", "403", "permission")):
                    self.gemini_exhausted = True
                    logger.warning("Gemini video quota/billing limit hit — disabling Gemini video for remaining scenes")
                    return None
        return None

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

        if r.status_code == 403:
            msg = r.json().get("detail", r.text[:200])
            if "credit" in msg.lower() or "enough" in msg.lower():
                raise HiggsFieldOutOfCreditsError(
                    f"Higgsfield credits exhausted! "
                    f"Top up at https://cloud.higgsfield.ai — falling back to Ken Burns for all remaining scenes."
                )
            raise RuntimeError(f"Higgsfield forbidden ({r.status_code}): {msg}")
        elif r.status_code == 401:
            raise HiggsFieldAuthError(
                f"Higgsfield auth failed — check HIGGSFIELD_KEY_ID and HIGGSFIELD_SECRET env vars."
            )
        elif r.status_code not in (200, 201):
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

    def check_credits(self) -> bool:
        """
        Quick credits check before starting a batch of scenes.
        Returns True if credits are available, False if exhausted.
        Logs a warning but does NOT raise — caller decides what to do.
        """
        try:
            # Higgsfield doesn't have a credits endpoint — attempt a harmless GET
            # A 403 here means no credits; 405 means endpoint exists = auth OK
            r = requests.get(BASE_URL, headers=self.headers, timeout=10)
            # Any non-401 response means auth is valid
            if r.status_code == 401:
                logger.error("Higgsfield auth invalid — check HIGGSFIELD_KEY_ID / HIGGSFIELD_SECRET")
                return False
            return True
        except Exception as e:
            logger.warning(f"Higgsfield credit check failed: {e}")
            return True  # assume OK if check fails

    def animate_scenes(
        self,
        scenes: list,
        image_paths: list,
        job_id: str,
    ) -> list:
        """
        Animate a list of scenes in sequence.
        Tries Gemini Video API first (GEMINI_API_KEY), then optional Higgsfield fallback.
        Returns a list of clip paths (or None for any scene that failed/exhausted
        credits so VideoCompiler can fall back to aspect-ratio-safe Ken Burns).
        """
        clip_paths = []
        higgs_available = bool(
            self.higgs_key_id
            and self.higgs_secret
            and (os.getenv("RAILWAY_PUBLIC_URL") or os.getenv("IMGBB_API_KEY"))
        )
        higgs_exhausted = not higgs_available

        for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
            raw_dur = float(scene.get("duration", 5))
            clip_dur = 10 if raw_dur > 6.5 else 5
            scene_type = scene.get("scene_type") or scene.get("type") or "character"
            motion_prompt = (
                scene.get("motion_prompt")
                or "character speaking expressively with natural body movement, cinematic camera"
            )

            # 1. Primary: Gemini Video API (free/included with GEMINI_API_KEY)
            if self.gemini_key and not self.gemini_exhausted:
                clip = self._animate_scene_gemini(
                    image_path=img_path,
                    motion_prompt=motion_prompt,
                    job_id=job_id,
                    scene_idx=i,
                )
                if clip:
                    clip_paths.append(clip)
                    continue

            # 2. Optional Fallback: Higgsfield (only if configured & has credits)
            if higgs_exhausted:
                logger.info(f"  Scene {i}: Ken Burns fallback")
                clip_paths.append(None)
                continue

            try:
                clip = self.animate_scene(
                    image_path=img_path,
                    motion_prompt=motion_prompt,
                    scene_type=scene_type,
                    duration=clip_dur,
                    job_id=job_id,
                    scene_idx=i,
                )
                clip_paths.append(clip)

            except (HiggsFieldOutOfCreditsError, HiggsFieldAuthError) as e:
                logger.warning(f"⚠️ {e}")
                higgs_exhausted = True
                clip_paths.append(None)

            except Exception as e:
                logger.error(f"Scene {i} animation failed: {e} — Ken Burns fallback for this scene")
                clip_paths.append(None)

        self.credits_exhausted = self.gemini_exhausted and higgs_exhausted
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


def animate_scenes_if_enabled(
    channel: dict,
    scenes: list,
    scene_timings: list,
    image_paths: list,
    job_id: str,
    output_dir: str = "./output",
    aspect_ratio: str = "16:9",
):
    """
    Animate scene images into real moving video clips via Gemini Video API
    (or optional Higgsfield fallback) when enabled for the channel
    (e.g. stories_tales Nollywood mode).
    Returns list of clip paths (or None entries for fallback), or None if skipped.
    """
    ch_key = (channel.get("key") or "").lower()
    niche = (channel.get("niche") or "").lower()
    should_animate = (
        ch_key == "stories_tales"
        or channel.get("visuals", {}).get("animate", False)
        or any(k in niche for k in ("nollywood", "nigerian movie", "market drama"))
    )
    if not should_animate:
        return None

    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_higgs = bool(
        os.getenv("HIGGSFIELD_KEY_ID")
        and os.getenv("HIGGSFIELD_SECRET")
        and (os.getenv("RAILWAY_PUBLIC_URL") or os.getenv("IMGBB_API_KEY"))
    )
    if not has_gemini and not has_higgs:
        logger.warning("Neither GEMINI_API_KEY nor Higgsfield credentials are set — falling back to Ken Burns")
        return None

    anim_scenes = []
    for i, sc in enumerate(scenes):
        sc_copy = dict(sc)
        if scene_timings and i < len(scene_timings):
            sc_copy["duration"] = scene_timings[i].get("duration", 5)
        anim_scenes.append(sc_copy)

    try:
        animator = SceneAnimator(output_dir=output_dir, aspect_ratio=aspect_ratio)
        return animator.animate_scenes(anim_scenes, image_paths, job_id)
    except Exception as e:
        logger.warning(f"Scene animation skipped due to error ({e}) — falling back to Ken Burns")
        return None
