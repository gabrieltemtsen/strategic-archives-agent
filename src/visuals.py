"""
Visuals Generator - AI image generation for video scenes
Primary: Hugging Face Inference API (FLUX.1-schnell)
Fallback: Gemini Imagen (uses existing GEMINI_API_KEY)
"""

import os
import io
import time
import logging
import requests
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"

# Track if providers are unavailable so we skip retries for the whole batch
_hf_unavailable = False
_gemini_unavailable = False


class VisualsGenerator:
    def __init__(self, config: dict, output_dir: str = "./output", aspect_ratio: str = "16:9"):
        self.config = config.get("visuals", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.model = self.config.get(
            "model", "black-forest-labs/FLUX.1-schnell"
        )
        self.style_prefix = self.config.get(
            "style",
            "high quality digital illustration, Pixar-style 3D render, "
            "vibrant colors, soft lighting, children's animation, "
            "professional studio quality, detailed, 4K"
        )
        self.negative_prompt = self.config.get(
            "negative_prompt",
            "scary, dark, violent, adult content, realistic, photographic, "
            "ugly, deformed, blurry, low quality"
        )
        self.max_workers = self.config.get("parallel_workers", 3)
        # Aspect ratio: "16:9" for landscape, "9:16" for shorts/vertical
        self.aspect_ratio = aspect_ratio
        if aspect_ratio == "9:16":
            self.img_width, self.img_height = 1080, 1920
        else:
            self.img_width, self.img_height = 1920, 1080

    def _build_prompt(self, scene_prompt: str) -> str:
        """Combine scene prompt with style prefix."""
        return f"{self.style_prefix}, {scene_prompt}"

    def _generate_image_hf(
        self, prompt: str, output_path: str, retries: int = 3
    ) -> Optional[str]:
        """Generate image via Hugging Face Inference API."""
        global _hf_unavailable

        # Skip HF entirely if we already know credits are depleted
        if _hf_unavailable:
            raise requests.exceptions.RequestException("HF credits depleted, skipping")

        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "width": self.img_width,
                "height": self.img_height,
            }
        }

        url = HF_API_URL + self.model
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code == 503:
                    wait_time = 20 * (attempt + 1)
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                if response.status_code in (402, 429):
                    # Credits depleted or rate limited — mark unavailable
                    _hf_unavailable = True
                    logger.warning(f"HF API {response.status_code}: credits depleted. Switching to Gemini Imagen.")
                    raise requests.exceptions.RequestException(f"HF {response.status_code}: credits depleted")
                if response.status_code == 410:
                    _hf_unavailable = True
                    logger.warning("HF model deprecated. Switching to Gemini Imagen.")
                    raise requests.exceptions.RequestException("HF model deprecated")
                if not response.ok:
                    logger.warning(f"HF API error {response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Image saved: {output_path}")
                return output_path
            except requests.exceptions.RequestException as e:
                logger.warning(f"Image generation attempt {attempt+1} failed: {e}")
                if _hf_unavailable:
                    raise  # Don't retry, go straight to fallback
                if attempt == retries - 1:
                    raise
                time.sleep(5)
        return None

    def _generate_image_gemini(
        self, prompt: str, output_path: str
    ) -> Optional[str]:
        """
        Generate image via Imagen 3 (imagen-3.0-generate-002).
        Falls back to gemini-2.0-flash-exp IMAGE modality if Imagen 3 fails.
        """
        global _gemini_unavailable
        if _gemini_unavailable:
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
            from PIL import Image

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=f"Generate an image: {prompt}",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img_data = part.inline_data.data
                    image = Image.open(io.BytesIO(img_data))
                    image = image.resize((self.img_width, self.img_height), Image.LANCZOS)
                    image.save(output_path, "PNG")
                    logger.info(f"Gemini flash image saved: {output_path}")
                    return output_path
            logger.warning("Gemini flash returned no image data")

        except Exception as e:
            logger.error(f"Gemini flash image gen also failed: {e}")
            err_str = str(e).lower()
            if any(x in err_str for x in ("404", "not found", "not supported", "invalid")):
                _gemini_unavailable = True
                logger.warning("Gemini image model unavailable, disabling for this batch")

        return None

    def _generate_fallback_image(
        self, scene_prompt: str, output_path: str, index: int
    ) -> str:
        """
        Last resort: create a simple solid-color image with PIL
        when all APIs are unavailable.
        """
        try:
            from PIL import Image, ImageDraw
            colors = [
                (255, 179, 102), (102, 179, 255), (179, 255, 102),
                (255, 102, 179), (179, 102, 255), (102, 255, 179),
                (255, 220, 100), (100, 220, 255), (220, 100, 255)
            ]
            bg_color = colors[index % len(colors)]
            img = Image.new("RGB", (self.img_width, self.img_height), bg_color)
            draw = ImageDraw.Draw(img)
            words = scene_prompt[:80]
            draw.text((50, 500), words, fill=(50, 50, 50))
            img.save(output_path, "PNG")
            return output_path
        except ImportError:
            logger.error("PIL not available for fallback image generation")
            raise

    def _generate_single_scene(self, args: tuple) -> Optional[str]:
        """Generate a single scene image (used by ThreadPoolExecutor)."""
        i, scene_prompt, job_id, total = args
        output_path = str(self.output_dir / f"{job_id}_scene_{i:03d}.png")
        logger.info(f"Generating image {i+1}/{total}: {scene_prompt[:60]}...")

        full_prompt = self._build_prompt(scene_prompt)

        # Try HF first, then Gemini Imagen, then PIL fallback
        try:
            path = self._generate_image_hf(full_prompt, output_path)
            if path:
                return (i, path)
        except Exception:
            pass

        # Gemini Imagen fallback
        try:
            path = self._generate_image_gemini(full_prompt, output_path)
            if path:
                return (i, path)
        except Exception as e:
            logger.warning(f"Gemini Imagen also failed for image {i+1}: {e}")

        # Last resort: colored rectangle
        try:
            path = self._generate_fallback_image(scene_prompt, output_path, i)
            return (i, path)
        except Exception as fe:
            logger.error(f"All image providers failed for image {i+1}: {fe}")
            return (i, None)

    def generate_scene_images(
        self,
        scene_prompts: List[str],
        job_id: str,
        use_fallback: bool = False
    ) -> List[str]:
        """
        Generate all scene images for a video.
        Returns list of image file paths in order.
        Uses parallel generation for speed.
        """
        global _hf_unavailable, _gemini_unavailable
        _hf_unavailable = False  # Reset for each job
        _gemini_unavailable = False

        total = len(scene_prompts)

        if use_fallback:
            image_paths = []
            for i, scene_prompt in enumerate(scene_prompts):
                output_path = str(self.output_dir / f"{job_id}_scene_{i:03d}.png")
                path = self._generate_fallback_image(scene_prompt, output_path, i)
                image_paths.append(path)
            logger.info(f"Generated {len(image_paths)}/{total} fallback images")
            return image_paths

        # Parallel generation with ThreadPoolExecutor
        args_list = [
            (i, prompt, job_id, total)
            for i, prompt in enumerate(scene_prompts)
        ]

        results = {}
        workers = min(self.max_workers, total)
        logger.info(f"Generating {total} images with {workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._generate_single_scene, args): args[0]
                for args in args_list
            }
            for future in as_completed(futures):
                idx, path = future.result()
                if path:
                    results[idx] = path

        # Return paths in order
        image_paths = [results[i] for i in sorted(results.keys())]
        logger.info(f"Generated {len(image_paths)}/{total} images")
        return image_paths

    def generate_thumbnail(
        self, thumbnail_prompt: str, job_id: str
    ) -> Optional[str]:
        """Generate a custom thumbnail image."""
        output_path = str(self.output_dir / f"{job_id}_thumbnail.png")
        full_prompt = (
            f"{self.style_prefix}, {thumbnail_prompt}, "
            "eye-catching, vibrant colors, YouTube thumbnail style"
        )
        # Try HF → Gemini → PIL fallback
        try:
            return self._generate_image_hf(full_prompt, output_path)
        except Exception:
            pass

        try:
            path = self._generate_image_gemini(full_prompt, output_path)
            if path:
                return path
        except Exception:
            pass

        logger.warning("All image providers failed for thumbnail, using fallback")
        return self._generate_fallback_image(thumbnail_prompt, output_path, 0)
