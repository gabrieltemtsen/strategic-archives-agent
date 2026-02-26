"""
Script Generator - Niche-driven content generation via Gemini.
The channel's niche string drives everything — no hardcoded content types.
Gemini infers tone, format, style, and structure from the niche description.
"""

import os
import json
import random
import logging
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

SUPPORTED_LANGUAGES = {
    "en": "English", "fr": "French", "yo": "Yoruba",
    "ha": "Hausa",   "ig": "Igbo",   "pt": "Portuguese", "es": "Spanish"
}

MASTER_PROMPT = """
You are an expert YouTube content creator and scriptwriter.

Channel niche: {niche}
Language: {language}
Target video length: {min_duration}–{max_duration} seconds (~{word_count_min}–{word_count_max} words of narration)

Your task: Write ONE complete, high-quality YouTube video script perfectly suited to this niche.

Guidelines:
- Open with a strong hook that grabs attention in the first 10 seconds
- Structure for good retention: hook → build-up → payoff → CTA
- Match the tone, pacing, and vocabulary to the niche audience
- Use natural spoken language (this is a voiceover script)
- Mark natural pauses with [PAUSE] and section breaks with [SECTION]
- End with a clear call-to-action (like/subscribe/comment prompt)
- Generate 12–15 scene image prompts that visually match the narration

Return ONLY a valid JSON object with this exact structure:
{{
  "title": "Compelling YouTube title (under 70 chars)",
  "description": "2–3 sentence YouTube description with keywords",
  "content_type": "the type of content you chose (e.g. bedtime_story, tutorial, horror_story, etc.)",
  "script": "Full narration script with [PAUSE] and [SECTION] markers",
  "tags": ["tag1", "tag2", ...],
  "thumbnail_prompt": "Detailed prompt for generating an eye-catching thumbnail image",
  "scene_prompts": ["scene image prompt 1", "scene image prompt 2", ...],
  "language": "{language}",
  "language_code": "{language_code}"
}}
"""


class ScriptGenerator:
    def __init__(self, channel: dict):
        """channel: enriched channel dict from channel_loader.load_channels()"""
        self.channel = channel
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call_gemini(self, prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    top_p=0.95,
                    max_output_tokens=4096,
                )
            )
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"):     text = text[3:]
            if text.endswith("```"):       text = text[:-3]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Gemini JSON parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _pick_language(self) -> tuple[str, str]:
        supported = self.channel.get("content", {}).get(
            "languages", {}).get("supported", ["en"])
        weighted = ["en"] * 3 + ["fr"] * 2 + [
            l for l in supported if l not in ("en", "fr")]
        code = random.choice(weighted)
        return code, SUPPORTED_LANGUAGES.get(code, "English")

    def _estimate_word_count(self) -> tuple[int, int]:
        video_cfg = self.channel.get("content", {}).get("video", {})
        min_s = video_cfg.get("min_duration", 300)
        max_s = video_cfg.get("max_duration", 600)
        # ~130 words/min speaking rate
        return int(min_s * 130 / 60), int(max_s * 130 / 60)

    def generate(self,
                 content_type: Optional[str] = None,
                 language: Optional[str] = None,
                 **kwargs) -> dict:
        """
        Generate a video script for the channel.
        Niche drives everything — content_type hint is optional.
        """
        lang_code, lang_name = (
            (language, SUPPORTED_LANGUAGES.get(language, "English"))
            if language else self._pick_language()
        )
        wc_min, wc_max = self._estimate_word_count()
        video_cfg = self.channel.get("content", {}).get("video", {})

        niche = self.channel.get("niche", "general YouTube content")
        if content_type:
            niche = f"{niche} — specifically: {content_type.replace('_', ' ')}"

        prompt = MASTER_PROMPT.format(
            niche=niche,
            language=lang_name,
            language_code=lang_code,
            min_duration=video_cfg.get("min_duration", 300),
            max_duration=video_cfg.get("max_duration", 600),
            word_count_min=wc_min,
            word_count_max=wc_max,
        )

        logger.info(
            f"Generating script | channel: {self.channel.get('name')} "
            f"| niche: \"{niche[:60]}\" | lang: {lang_name}"
        )
        result = self._call_gemini(prompt)
        result["language_code"] = lang_code
        result["channel_key"] = self.channel.get("key", "")
        result["channel_name"] = self.channel.get("name", "")
        return result
