"""
Character Generator
Generates consistent character images for each channel using FLUX via HuggingFace.
Characters are defined per channel and cached — same look every episode.
"""

import os
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

HF_API = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

# ── Per-channel character definitions ────────────────────────────────────────
# Each channel has a primary character with a fixed visual identity.
# The base_prompt is used for every scene — scene_suffix is added per scene.

CHANNEL_CHARACTERS = {
    "kids_universe": {
        "name": "Sunny",
        "style": "Pixar 3D animated movie style, vibrant colors, expressive eyes",
        "base_prompt": (
            "Sunny, a cheerful 8-year-old cartoon girl with bright big eyes, "
            "round face, curly hair with a yellow bow, wearing a colorful explorer outfit, "
            "Pixar 3D animated movie style, warm vibrant colors, friendly expression, "
            "full body visible, white background removed, facing camera"
        ),
    },
    "strategic_archives": {
        "name": "The General",
        "style": "Epic animated movie style, dramatic lighting, historical realism",
        "base_prompt": (
            "The General, a battle-hardened Roman commander in full bronze armor, "
            "strong jaw, piercing eyes, short dark hair, red cape flowing, "
            "epic animated movie style, dramatic cinematic lighting, "
            "waist-up portrait facing camera, intense determined expression"
        ),
    },
    "stories_tales": {
        "name": "Shadow",
        "style": "Dark animated gothic style, mysterious lighting, eerie atmosphere",
        "base_prompt": (
            "Shadow, a mysterious hooded storyteller figure, shadowed face with glowing eyes, "
            "dark flowing cloak, candle light flickering, "
            "dark gothic animated movie style, eerie atmospheric lighting, "
            "waist-up portrait facing camera, mysterious expression"
        ),
    },
    "gabe_dev_codes": {
        "name": "Dev",
        "style": "Futuristic sci-fi animated style, neon accents, sleek design",
        "base_prompt": (
            "Dev, a futuristic AI robot character with a friendly glowing face display, "
            "sleek metallic body with neon blue accents, expressive LED eyes, "
            "futuristic animated movie style, clean sci-fi aesthetic, "
            "waist-up portrait facing camera, curious helpful expression"
        ),
    },
}


class CharacterGenerator:
    def __init__(self, channel_key: str, output_dir: str = "./output"):
        self.channel_key = channel_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.char = CHANNEL_CHARACTERS.get(channel_key, CHANNEL_CHARACTERS["strategic_archives"])

    def get_character_info(self) -> dict:
        return self.char

    def _generate_image(self, prompt: str, out_path: Path) -> str:
        """Generate image via HuggingFace FLUX."""
        if not self.hf_token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN not set")

        logger.info(f"Generating image: {prompt[:60]}...")
        for attempt in range(3):
            try:
                r = requests.post(
                    HF_API,
                    headers={"Authorization": f"Bearer {self.hf_token}"},
                    json={"inputs": prompt},
                    timeout=120,
                )
                if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                    out_path.write_bytes(r.content)
                    logger.info(f"  ✅ Image saved: {out_path} ({len(r.content)//1024}KB)")
                    return str(out_path)
                elif r.status_code == 503:
                    logger.warning(f"  HF model loading, waiting 20s... (attempt {attempt+1})")
                    import time; time.sleep(20)
                else:
                    logger.error(f"  HF error {r.status_code}: {r.text[:200]}")
                    if attempt < 2:
                        import time; time.sleep(5)
            except Exception as e:
                logger.error(f"  Image gen error: {e}")
                if attempt < 2:
                    import time; time.sleep(5)

        raise RuntimeError(f"Failed to generate image after 3 attempts: {prompt[:60]}")

    def generate_scene_image(self, scene: dict, job_id: str, scene_idx: int) -> str:
        """
        Generate a cinematic scene image for the given scene.
        For character scenes: character is embedded in the environment.
        For establishing scenes: environment only, no character needed.
        """
        scene_type = scene.get("type", "default")
        image_prompt = scene.get("image_prompt", "")
        char = self.char

        if scene_type == "establishing":
            # Wide establishing shot — no character
            prompt = (
                f"{image_prompt}, "
                f"cinematic wide shot, {char['style']}, "
                f"8K resolution, movie quality, epic composition, "
                f"no text, no watermarks"
            )
        elif scene_type == "character":
            # Character embedded in the scene environment
            prompt = (
                f"{char['base_prompt']}, "
                f"set in: {image_prompt}, "
                f"full cinematic scene, character in environment, "
                f"{char['style']}, 8K resolution, movie quality, "
                f"no text, no watermarks"
            )
        else:
            # Action / montage / default
            prompt = (
                f"{image_prompt}, "
                f"{char['style']}, "
                f"cinematic composition, dynamic action shot, "
                f"8K resolution, movie quality, no text, no watermarks"
            )

        out_path = self.output_dir / f"{job_id}_scene{scene_idx:02d}_img.png"

        # Use cached version if exists
        if out_path.exists():
            logger.info(f"  Using cached scene image: {out_path}")
            return str(out_path)

        return self._generate_image(prompt, out_path)

    def generate_all_scenes(self, scenes: list, job_id: str) -> list:
        """Generate images for all scenes. Returns list of image paths."""
        image_paths = []
        for i, scene in enumerate(scenes):
            try:
                path = self.generate_scene_image(scene, job_id, i)
                image_paths.append(path)
            except Exception as e:
                logger.error(f"Scene {i} image gen failed: {e}")
                # Use a blank fallback image
                fallback = self._create_fallback_image(scene, job_id, i)
                image_paths.append(fallback)
        return image_paths

    def _create_fallback_image(self, scene: dict, job_id: str, idx: int) -> str:
        """Create a simple colored fallback image when FLUX fails."""
        from PIL import Image, ImageDraw, ImageFont
        out_path = self.output_dir / f"{job_id}_scene{idx:02d}_img.png"
        img = Image.new("RGB", (1920, 1080), color=(20, 20, 40))
        draw = ImageDraw.Draw(img)
        text = scene.get("setting", f"Scene {idx+1}")[:60]
        draw.text((960, 540), text, fill=(200, 200, 255), anchor="mm")
        img.save(str(out_path))
        return str(out_path)
