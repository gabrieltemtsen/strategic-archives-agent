"""
Visuals Generator - Consistent, cost-effective AI image generation for video scenes.
Provider chain:
  1. Hugging Face Inference API (if HUGGINGFACE_API_TOKEN set and credits available)
  2. Gemini Image Generation (uses existing GEMINI_API_KEY)
  3. Pollinations.ai (100% free, no API key required, seeded for visual consistency)
  4. Styled Gradient Card (offline PIL fallback)
"""

import os
import io
import time
import hashlib
import logging
import urllib.parse
import requests
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"

_hf_unavailable = False
_gemini_unavailable = False


class VisualsGenerator:
    def __init__(
        self,
        config: dict,
        output_dir: str = "./output",
        aspect_ratio: str = "16:9",
        style_guide: Optional[str] = None,
    ):
        self.config = config.get("visuals", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
        self.model = self.config.get("model", "black-forest-labs/FLUX.1-schnell")

        default_style = (
            "cinematic digital illustration, cohesive color grading, "
            "dramatic soft lighting, detailed composition, high quality, no text or watermarks"
        )
        self.style_prefix = (style_guide or self.config.get("style") or default_style).strip()
        self.negative_prompt = self.config.get(
            "negative_prompt",
            "blurry, low quality, watermark, text overlay, deformed, ugly, extra limbs",
        )
        self.max_workers = self.config.get("parallel_workers", 3)
        self.aspect_ratio = aspect_ratio
        if aspect_ratio == "9:16":
            self.img_width, self.img_height = 1080, 1920
            self.gen_width, self.gen_height = 768, 1344
        else:
            self.img_width, self.img_height = 1920, 1080
            self.gen_width, self.gen_height = 1280, 720

    def _build_prompt(self, scene_prompt: str) -> str:
        """Combine cohesive style guide with specific scene prompt."""
        return f"{self.style_prefix}. Scene: {scene_prompt}. Aspect ratio {self.aspect_ratio}."

    def _save_and_fit_image(self, raw_bytes: bytes, output_path: str) -> str:
        """Validate image bytes and fit/crop cleanly to exact target resolution."""
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        fitted = ImageOps.fit(
            img,
            (self.img_width, self.img_height),
            method=Image.Resampling.LANCZOS,
        )
        fitted.save(output_path, "PNG", optimize=True)
        return output_path

    def _generate_image_hf(self, prompt: str, output_path: str, retries: int = 2) -> Optional[str]:
        global _hf_unavailable
        if _hf_unavailable or not self.hf_token:
            raise requests.exceptions.RequestException("HF unavailable or no token")

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": self.gen_width, "height": self.gen_height},
        }
        url = HF_API_URL + self.model

        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code == 503:
                    time.sleep(15 * (attempt + 1))
                    continue
                if response.status_code in (402, 410, 429):
                    _hf_unavailable = True
                    logger.warning(f"HF API {response.status_code} — switching to free fallback.")
                    raise requests.exceptions.RequestException(f"HF {response.status_code}")
                response.raise_for_status()
                return self._save_and_fit_image(response.content, output_path)
            except requests.exceptions.RequestException:
                if _hf_unavailable or attempt == retries - 1:
                    raise
                time.sleep(3)
        return None

    def _generate_image_gemini(self, prompt: str, output_path: str) -> Optional[str]:
        """Generate image via Imagen 3, falling back to gemini-2.0-flash-exp IMAGE modality."""
        global _gemini_unavailable
        if _gemini_unavailable or not os.getenv("GEMINI_API_KEY"):
            return None

        # Map pixel dimensions to Imagen 3 supported aspect ratios
        if self.img_width > self.img_height:
            imagen_aspect = "16:9"
        elif self.img_width < self.img_height:
            imagen_aspect = "9:16"
        else:
            imagen_aspect = "1:1"

        # --- Attempt 1: Imagen 3 (best quality, purpose-built image gen) ---
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=imagen_aspect,
                    safety_filter_level="BLOCK_ONLY_HIGH",
                    person_generation="ALLOW_ALL",
                ),
            )
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Imagen 3 image saved: {output_path}")
                return output_path
            logger.warning("Imagen 3 returned no images")

        except Exception as e:
            err_str = str(e).lower()
            logger.warning(f"Imagen 3 failed: {e}")
            if any(x in err_str for x in ("quota", "429", "resource_exhausted")):
                # Quota hit — disable for this batch
                _gemini_unavailable = True
                logger.warning("Gemini quota exceeded, disabling for this batch")
                return None

        # --- Attempt 2: gemini-2.0-flash-exp with IMAGE modality ---
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=f"Generate a high-quality {self.aspect_ratio} visual illustration: {prompt}",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    return self._save_and_fit_image(part.inline_data.data, output_path)
            logger.warning("Gemini flash returned no image data")

        except Exception as e:
            logger.error(f"Gemini flash image gen also failed: {e}")
            err_str = str(e).lower()
            if any(x in err_str for x in ("404", "not found", "not supported", "invalid", "quota", "429")):
                _gemini_unavailable = True
                logger.warning("Gemini image model unavailable, switching to Pollinations")

        return None

    def _generate_image_pollinations(
        self, prompt: str, output_path: str, seed: int = 42, retries: int = 2
    ) -> Optional[str]:
        """100% free image generation via Pollinations.ai (no API key needed)."""
        encoded = urllib.parse.quote(prompt[:900], safe="")
        url = (
            f"{POLLINATIONS_URL}{encoded}"
            f"?width={self.gen_width}&height={self.gen_height}"
            f"&seed={seed}&nologo=true&model=flux"
        )
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=90)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                    return self._save_and_fit_image(resp.content, output_path)
                logger.warning(f"Pollinations status {resp.status_code} (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Pollinations error (attempt {attempt + 1}): {e}")
            time.sleep(3)
        return None

    def _generate_fallback_image(self, scene_prompt: str, output_path: str, index: int) -> str:
        """Offline fallback: clean dark-gradient visual card with wrapped scene caption."""
        import textwrap
        from PIL import Image, ImageDraw, ImageFont

        palettes = [
            ((18, 22, 42), (46, 58, 108)),
            ((24, 34, 46), (52, 92, 118)),
            ((34, 22, 48), (88, 52, 112)),
            ((20, 38, 34), (46, 98, 84)),
        ]
        c1, c2 = palettes[index % len(palettes)]
        img = Image.new("RGB", (self.img_width, self.img_height))
        draw = ImageDraw.Draw(img)
        for y in range(self.img_height):
            t = y / max(1, self.img_height - 1)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (self.img_width, y)], fill=(r, g, b))

        font = ImageFont.load_default()
        wrapped = textwrap.fill(scene_prompt[:140], width=42)
        draw.multiline_text(
            (self.img_width // 2, self.img_height // 2),
            wrapped,
            fill=(235, 240, 255),
            font=font,
            anchor="mm",
            align="center",
        )
        img.save(output_path, "PNG")
        return output_path

    def _job_seed(self, job_id: str, index: int) -> int:
        digest = hashlib.md5(f"{job_id}:{index}".encode()).hexdigest()
        return int(digest[:8], 16)

    def _generate_single_scene(self, args: tuple) -> tuple[int, Optional[str]]:
        i, scene_prompt, job_id, total = args
        output_path = str(self.output_dir / f"{job_id}_scene_{i:03d}.png")
        logger.info(f"Generating image {i + 1}/{total}: {scene_prompt[:60]}...")
        full_prompt = self._build_prompt(scene_prompt)
        seed = self._job_seed(job_id, i)

        # 1. Hugging Face (if token configured)
        if self.hf_token:
            try:
                path = self._generate_image_hf(full_prompt, output_path)
                if path:
                    return (i, path)
            except Exception:
                pass

        # 2. Gemini Image (existing GEMINI_API_KEY)
        path = self._generate_image_gemini(full_prompt, output_path)
        if path:
            return (i, path)

        # 3. Pollinations.ai (100% free, no API key)
        path = self._generate_image_pollinations(full_prompt, output_path, seed=seed)
        if path:
            return (i, path)

        # 4. Styled offline card fallback
        logger.warning(f"All image providers failed for scene {i + 1}, using styled card")
        return (i, self._generate_fallback_image(scene_prompt, output_path, i))

    def generate_scene_images(
        self,
        scene_prompts: List[str],
        job_id: str,
        use_fallback: bool = False,
    ) -> List[str]:
        global _hf_unavailable, _gemini_unavailable
        _hf_unavailable = False
        _gemini_unavailable = False

        total = len(scene_prompts)
        if use_fallback:
            return [
                self._generate_fallback_image(
                    p, str(self.output_dir / f"{job_id}_scene_{i:03d}.png"), i
                )
                for i, p in enumerate(scene_prompts)
            ]

        args_list = [(i, p, job_id, total) for i, p in enumerate(scene_prompts)]
        results = {}
        workers = min(self.max_workers, max(1, total))
        logger.info(f"Generating {total} images with {workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._generate_single_scene, a): a[0] for a in args_list}
            for future in as_completed(futures):
                idx, path = future.result()
                if path:
                    results[idx] = path

        image_paths = [results[i] for i in sorted(results.keys())]
        logger.info(f"Generated {len(image_paths)}/{total} images")
        return image_paths

    def generate_thumbnail(self, thumbnail_prompt: str, job_id: str) -> Optional[str]:
        output_path = str(self.output_dir / f"{job_id}_thumbnail.png")
        full_prompt = (
            f"{self.style_prefix}. {thumbnail_prompt}. "
            "Eye-catching YouTube thumbnail composition, dramatic contrast, vibrant focal point, no text."
        )
        seed = self._job_seed(job_id, 999)

        if self.hf_token:
            try:
                path = self._generate_image_hf(full_prompt, output_path)
                if path:
                    return path
            except Exception:
                pass

        path = self._generate_image_gemini(full_prompt, output_path)
        if path:
            return path

        path = self._generate_image_pollinations(full_prompt, output_path, seed=seed)
        if path:
            return path

        return self._generate_fallback_image(thumbnail_prompt, output_path, 0)
