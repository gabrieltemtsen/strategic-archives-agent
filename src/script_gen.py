"""
Script Generator - Niche-driven, scene-synchronized content generation via Gemini.
Produces a cohesive visual_style_guide + structured scenes so narration,
visuals, camera motion, and burned-in captions stay 100% in sync.
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

VALID_MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right")

MASTER_PROMPT = """
You are an elite YouTube retention strategist, scriptwriter, and visual director.

Channel niche: {niche}
Format: {format_label}
Language: {language}
Target duration: {min_duration}–{max_duration} seconds (~{word_count_min}–{word_count_max} words total narration)
Target scene count: {scene_count_min}–{scene_count_max} scenes

Your task: Write ONE complete, high-retention YouTube video script with tightly synchronized visual scenes.

Retention & Quality Rules:
1. HOOK (Scene 1): Open with an immediate pattern-interrupt, bold question, or vivid moment in the first 8 seconds. No "Hey guys, welcome back to the channel".
2. PACING: Every scene must advance the story or reveal a new insight. Use open loops ("Wait until you hear what happened next...") to hold attention.
3. VOICEOVER STYLE: Write natural spoken language meant to be read aloud. Use [PAUSE] sparingly for dramatic beats.
4. VISUAL CONSISTENCY: Define ONE cohesive `visual_style_guide` (art style, color palette, lighting, recurring character/setting details) that applies to ALL scenes so the video looks like one unified film, not random stock images.
5. SCENE SYNC: Each scene's `visual_prompt` must depict the exact moment in that scene's `narration`. Do NOT repeat style keywords inside `visual_prompt` (the style guide is prepended automatically).
6. PAYOFF & CTA: Deliver a satisfying climax/takeaway in the final scenes and end with a natural subscribe/comment prompt.

Return ONLY a valid JSON object with this exact schema:
{{
  "title": "Compelling, curiosity-driven YouTube title (under 70 chars)",
  "description": "2–3 sentence SEO-friendly YouTube description",
  "content_type": "specific content type (e.g. bedtime_story, war_history, horror_story, coding_tutorial)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "visual_style_guide": "Single cohesive art direction string: medium, lighting, color palette, character/world consistency rules",
  "thumbnail_prompt": "High-contrast, eye-catching YouTube thumbnail scene with clear focal subject and dramatic lighting",
  "scenes": [
    {{
      "narration": "Exact spoken narration for this scene...",
      "visual_prompt": "Specific subject, action, composition, and mood for this scene",
      "camera_motion": "zoom_in | zoom_out | pan_left | pan_right",
      "caption_highlight": "Optional 2-5 word on-screen hook or chapter label (or empty string)"
    }}
  ],
  "language": "{language}",
  "language_code": "{language_code}"
}}
"""


class ScriptGenerator:
    def __init__(self, channel: dict):
        """channel: enriched channel dict from channel_loader.load_channels()"""
        self.channel = channel
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call_gemini(self, prompt: str, retries: int = 2) -> dict:
        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        top_p=0.95,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                return json.loads(text.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    logger.info("Retrying with stricter JSON instruction...")
                    prompt += "\n\nIMPORTANT: Return ONLY valid, complete JSON. No markdown fences or extra text."
                else:
                    logger.error("All retries exhausted — JSON still invalid")
                    raise
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise

    def _pick_language(self) -> tuple[str, str]:
        supported = self.channel.get("content", {}).get("languages", {}).get("supported", ["en"])
        weighted = ["en"] * 3 + ["fr"] * 2 + [l for l in supported if l not in ("en", "fr")]
        code = random.choice(weighted)
        return code, SUPPORTED_LANGUAGES.get(code, "English")

    def _estimate_word_count(self, min_s: int, max_s: int) -> tuple[int, int]:
        # ~135 words/min natural narration pace
        return int(min_s * 135 / 60), int(max_s * 135 / 60)

    def _normalize_output(self, data: dict, lang_code: str) -> dict:
        """
        Ensure structured `scenes` exist and populate backward-compatible
        `script` and `scene_prompts` fields.
        """
        scenes = data.get("scenes") or []
        if not scenes and data.get("script"):
            # Fallback if model returned legacy format
            raw_prompts = data.get("scene_prompts") or ["cinematic establishing shot"]
            sentences = [s.strip() for s in data["script"].split("[SECTION]") if s.strip()]
            if len(sentences) < 2:
                sentences = [data["script"]]
            scenes = []
            for idx, text in enumerate(sentences):
                scenes.append({
                    "narration": text,
                    "visual_prompt": raw_prompts[idx % len(raw_prompts)],
                    "camera_motion": VALID_MOTIONS[idx % len(VALID_MOTIONS)],
                    "caption_highlight": "",
                })

        normalized_scenes = []
        for idx, sc in enumerate(scenes):
            narration = (sc.get("narration") or "").strip()
            if not narration:
                continue
            motion = sc.get("camera_motion", VALID_MOTIONS[idx % len(VALID_MOTIONS)])
            if motion not in VALID_MOTIONS:
                motion = VALID_MOTIONS[idx % len(VALID_MOTIONS)]
            normalized_scenes.append({
                "narration": narration,
                "visual_prompt": (sc.get("visual_prompt") or "cinematic scene").strip(),
                "camera_motion": motion,
                "caption_highlight": (sc.get("caption_highlight") or "").strip(),
            })

        data["scenes"] = normalized_scenes
        data["script"] = "\n[SECTION]\n".join(s["narration"] for s in normalized_scenes)
        data["scene_prompts"] = [s["visual_prompt"] for s in normalized_scenes]
        data["visual_style_guide"] = (
            data.get("visual_style_guide")
            or self.channel.get("visuals", {}).get("style", "cinematic digital illustration, cohesive lighting, 4K")
        ).strip()
        data["language_code"] = lang_code
        data["channel_key"] = self.channel.get("key", "")
        data["channel_name"] = self.channel.get("name", "")
        return data

    def generate(
        self,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        video_format: str = "long",
        **kwargs,
    ) -> dict:
        """Generate a scene-synchronized video script for the channel."""
        lang_code, lang_name = (
            (language, SUPPORTED_LANGUAGES.get(language, "English"))
            if language else self._pick_language()
        )
        video_cfg = self.channel.get("content", {}).get("video", {})

        if video_format == "short":
            min_s, max_s = 40, 55
            sc_min, sc_max = 5, 7
            format_label = "YouTube Short (9:16 vertical, fast-paced, under 58 seconds)"
        else:
            min_s = video_cfg.get("min_duration", 240)
            max_s = video_cfg.get("max_duration", 480)
            sc_min, sc_max = 12, 16
            format_label = "YouTube Long-form (16:9 horizontal, engaging storytelling)"

        wc_min, wc_max = self._estimate_word_count(min_s, max_s)

        niche = self.channel.get("niche", "general YouTube content")
        if content_type:
            niche = f"{niche} — specifically: {content_type.replace('_', ' ')}"

        prompt = MASTER_PROMPT.format(
            niche=niche,
            format_label=format_label,
            language=lang_name,
            language_code=lang_code,
            min_duration=min_s,
            max_duration=max_s,
            word_count_min=wc_min,
            word_count_max=wc_max,
            scene_count_min=sc_min,
            scene_count_max=sc_max,
        )

        logger.info(
            f"Generating script | channel: {self.channel.get('name')} "
            f"| format: {video_format} | lang: {lang_name}"
        )
        raw_result = self._call_gemini(prompt)
        result = self._normalize_output(raw_result, lang_code)
        logger.info(f"Generated {len(result['scenes'])} synchronized scenes")
        return result
