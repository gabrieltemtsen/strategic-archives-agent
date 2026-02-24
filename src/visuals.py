"""
Visuals Generator - AI image generation for video scenes
Primary: Hugging Face Inference API (free tier)
Uses SDXL or similar models for high quality kids illustrations
"""

import os
import time
import logging
import requests
from pathlib import Path
from typing import List, Optional
from io import BytesIO

logger = logging.getLogger(__name__)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"


class VisualsGenerator:
    def __init__(self, config: dict, output_dir: str = "./output"):
        self.config = config.get("visuals", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.model = self.config.get(
            "model", "black-forest-labs/FLUX.1-schnell"
        )
        self.style_prefix = self.config.get(
            "style",
            "children's book illustration, colorful, cute, safe for kids, "
            "bright colors, cartoon style, watercolor"
        )
        self.negative_prompt = self.config.get(
            "negative_prompt",
            "scary, dark, violent, adult content, realistic, photographic, "
            "ugly, deformed, blurry, low quality"
        )

    def _build_prompt(self, scene_prompt: str) -> str:
        """Combine scene prompt with style prefix."""
        return f"{self.style_prefix}, {scene_prompt}"

    def _generate_image_hf(
        self, prompt: str, output_path: str, retries: int = 3
    ) -> Optional[str]:
        """Generate image via Hugging Face Inference API."""
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "width": 1280,
                "height": 720,
            }
        }

        url = HF_API_URL + self.model
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                if response.status_code == 503:
                    # Model loading, wait and retry
                    wait_time = 20 * (attempt + 1)
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                if not response.ok:
                    logger.warning(f"HF API error {response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Image saved: {output_path}")
                return output_path
            except requests.exceptions.RequestException as e:
                logger.warning(f"Image generation attempt {attempt+1} failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(5)
        return None

    def _generate_fallback_image(
        self, scene_prompt: str, output_path: str, index: int
    ) -> str:
        """
        Fallback: create a simple solid-color image with PIL
        when HF API is unavailable. Used in dev/testing.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            colors = [
                (255, 179, 102), (102, 179, 255), (179, 255, 102),
                (255, 102, 179), (179, 102, 255), (102, 255, 179),
                (255, 220, 100), (100, 220, 255), (220, 100, 255)
            ]
            bg_color = colors[index % len(colors)]
            img = Image.new("RGB", (1280, 720), bg_color)
            draw = ImageDraw.Draw(img)
            # Add scene text
            words = scene_prompt[:80]
            draw.text((50, 300), words, fill=(50, 50, 50))
            img.save(output_path, "PNG")
            return output_path
        except ImportError:
            logger.error("PIL not available for fallback image generation")
            raise

    def generate_scene_images(
        self,
        scene_prompts: List[str],
        job_id: str,
        use_fallback: bool = False
    ) -> List[str]:
        """
        Generate all scene images for a video.
        Returns list of image file paths in order.
        """
        image_paths = []
        total = len(scene_prompts)

        for i, scene_prompt in enumerate(scene_prompts):
            output_path = str(self.output_dir / f"{job_id}_scene_{i:03d}.png")

            logger.info(f"Generating image {i+1}/{total}: {scene_prompt[:60]}...")

            try:
                if use_fallback:
                    path = self._generate_fallback_image(scene_prompt, output_path, i)
                else:
                    full_prompt = self._build_prompt(scene_prompt)
                    path = self._generate_image_hf(full_prompt, output_path)

                if path:
                    image_paths.append(path)
                    # Rate limit: be gentle with free tier
                    if i < total - 1:
                        time.sleep(2)
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                # Use fallback if HF fails
                try:
                    path = self._generate_fallback_image(scene_prompt, output_path, i)
                    image_paths.append(path)
                except Exception as fe:
                    logger.error(f"Fallback also failed: {fe}")

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
        try:
            return self._generate_image_hf(full_prompt, output_path)
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return self._generate_fallback_image(thumbnail_prompt, output_path, 0)
