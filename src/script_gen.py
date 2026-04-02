"""
Script Generator - Niche-driven content generation via Gemini.
The channel's niche string drives everything — no hardcoded content types.
Gemini infers tone, format, style, and structure from the niche description.
"""

import os
import json
import random
import logging
import time
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
You are an expert YouTube content creator, screenwriter, and cinematic director.

Channel niche: {niche}
Character name: {character_name}
Character style: {character_style}
Language: {language}
Target video length: {min_duration}–{max_duration} seconds
Made for kids: {made_for_kids}

{content_rules}

Your task: Write ONE complete cinematic YouTube video script — structured like a MOVIE, not a slideshow.
The video will be animated using AI video generation (Higgsfield + FLUX) so every scene needs cinematic detail.

Scene types you can use:
- "establishing": Wide cinematic shot of the environment/setting. No character dialogue. Camera moves through the world.
- "character": The main character ({character_name}) is in the scene speaking/acting. Use for narration + key moments.
- "action": Dynamic action shot — something is happening. High energy.
- "montage": Quick atmospheric shot used for transitions between major beats.

Guidelines:
- Start with a dramatic establishing shot that sets the world
- Alternate between establishing, character, and action scenes naturally
- Each scene should feel like a movie shot — think cinematographer, not slideshow maker
- The character speaks the narration — write it as their voice
- Motion prompts MUST describe MOVEMENT: camera direction, character action, environmental motion
- Keep each scene 5–8 seconds. Total 8–12 scenes.
- End with a strong call-to-action scene

Return ONLY a valid JSON object:
{{
  "title": "Compelling YouTube title (under 70 chars)",
  "description": "2–3 sentence YouTube description with keywords",
  "content_type": "e.g. documentary, story, tutorial, horror etc.",
  "tags": ["tag1", "tag2", ...],
  "thumbnail_prompt": "Vivid detailed prompt for eye-catching thumbnail, {character_style}",
  "language": "{language}",
  "language_code": "{language_code}",
  "scenes": [
    {{
      "type": "establishing",
      "setting": "Brief setting description",
      "narration": "What the character says during this scene (empty string if none)",
      "image_prompt": "Detailed FLUX image generation prompt — describe the full scene visually",
      "motion_prompt": "Cinematic motion description for Higgsfield — describe camera movement + action",
      "duration": 6
    }},
    {{
      "type": "character",
      "setting": "Brief setting description",
      "narration": "Character dialogue/narration for this scene",
      "image_prompt": "Detailed prompt — MUST include the character ({character_name}) in the scene",
      "motion_prompt": "Character action + camera movement for Higgsfield",
      "duration": 7
    }}
  ]
}}

Motion prompt examples:
- "Slow cinematic dolly push forward through ancient Roman city gates, golden dust particles floating, epic scale"
- "The General raises his sword dramatically, red cape billowing in the wind, camera slowly pulls back revealing the battlefield"
- "Camera sweeps low across burning ships, smoke rising, dramatic orchestral feel"
- "Sunny turns excitedly to face camera, eyes wide with wonder, bright sparkles appear around her"
"""

# Per-channel content rules injected into the prompt
CONTENT_RULES = {
    "kids": """
⚠️ STRICT KIDS CONTENT RULES (made_for_kids = true):
- ONLY generate content suitable for children aged 3–10
- Topics: fairy tales, fun facts about animals/nature/space, simple science, positive life lessons, bedtime stories
- NO violence, NO scary content, NO adult themes, NO conflict, NO weapons
- Language must be simple, warm, encouraging, and age-appropriate
- Characters must be friendly, safe, and wholesome
- Every scene must be bright, colorful, and visually joyful
""",
    "current_events": """
⚠️ CURRENT AFFAIRS CONTENT RULES:
- Focus on CURRENT, ONGOING, or RECENT events (within the last 1–3 years)
- Topics: ongoing wars (Ukraine-Russia, Middle East, Sudan, etc.), geopolitical tensions,
  global controversies, international conflicts, political power shifts, economic wars,
  sanctions, proxy conflicts, NATO/alliances dynamics, emerging threats
- NO ancient history or medieval battles unless directly relevant to a current situation
- Be factual, analytical, and journalistic in tone
- Each video should feel like breaking news meets documentary
- Always present multiple perspectives — do not take political sides
""",
}


def _get_content_rules(channel: dict) -> str:
    if channel.get("made_for_kids"):
        return CONTENT_RULES["kids"]
    niche = channel.get("niche", "").lower()
    if any(w in niche for w in ["war", "conflict", "geopolit", "current", "controversy"]):
        return CONTENT_RULES["current_events"]
    return ""  # no special rules for other channels


class GeminiQuotaExceededError(RuntimeError):
    pass


def _fallback_script(channel: dict, lang_code: str, lang_name: str, niche: str) -> dict:
    """Ultra-simple fallback when Gemini is down/out of quota.

    This keeps the pipeline alive (so we can still generate + upload videos) but quality
    will be lower than Gemini.
    """
    from src.character_gen import CHANNEL_CHARACTERS

    channel_key = channel.get("key", "strategic_archives")
    char = CHANNEL_CHARACTERS.get(channel_key, CHANNEL_CHARACTERS["strategic_archives"])
    title = f"{niche.title()[:55]} — Quick Explained"

    scenes = []
    base_settings = [
        "wide cinematic overview of the topic",
        "close-up character explanation",
        "simple illustrative example",
        "summary and key takeaway",
    ]
    for i, setting in enumerate(base_settings):
        scene_type = "establishing" if i == 0 else "character"
        narration = (
            f"Hi, I’m {char['name']}. Today we’ll learn about {niche}. "
            f"Here’s the key idea in simple terms." if i == 1 else
            (f"Example: imagine this happening in real life — that’s how {niche} works." if i == 2 else
             (f"Let’s recap the main points and what you should remember about {niche}." if i == 3 else ""))
        )
        scenes.append({
            "type": scene_type,
            "setting": setting,
            "narration": narration,
            "image_prompt": f"{char['style']}. {setting}. theme: {niche}. cinematic lighting, high quality",
            "motion_prompt": "Slow cinematic dolly push in, gentle parallax, subtle character motion",
            "duration": 7,
        })

    return {
        "title": title,
        "description": f"A quick breakdown of {niche}.",
        "content_type": "explainer",
        "tags": ["explained", "education", niche.replace(" ", "-")[:25]],
        "thumbnail_prompt": f"{char['style']}. Bold title text: {niche}. bright, high contrast, clean composition",
        "language": lang_name,
        "language_code": lang_code,
        "scenes": scenes,
        "scene_prompts": [s["image_prompt"] for s in scenes],
        "script": " ".join(s.get("narration", "") for s in scenes if s.get("narration")),
    }


class ScriptGenerator:
    def __init__(self, channel: dict):
        """channel: enriched channel dict from channel_loader.load_channels()"""
        self.channel = channel
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call_gemini(self, prompt: str, retries: int = 2) -> dict:
        """Call Gemini with basic retries.

        Notes:
        - 429 RESOURCE_EXHAUSTED often means either rate-limit OR hard quota/billing exhaustion.
          Retries only help for rate-limit. If quota is exhausted we raise a clear error.
        """
        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.85,
                        top_p=0.95,
                        # Keep this sane to reduce cost/quota burn.
                        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
                        response_mime_type="application/json",
                    )
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
                    prompt = prompt + "\n\nIMPORTANT: Return ONLY valid, complete JSON. No extra text."
                    time.sleep(1 + attempt)
                else:
                    logger.error("All retries exhausted — JSON still invalid")
                    raise

            except Exception as e:
                msg = str(e)
                # Quota/rate-limit
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    # brief backoff; if still failing after retries, treat as hard exhaustion
                    if attempt < retries:
                        sleep_s = min(20, 2 ** attempt * 3)
                        logger.warning(f"Gemini 429/RESOURCE_EXHAUSTED (attempt {attempt + 1}) — backoff {sleep_s}s")
                        time.sleep(sleep_s)
                        continue
                    raise GeminiQuotaExceededError(
                        "Gemini quota exhausted (429 RESOURCE_EXHAUSTED). "
                        "Check plan/billing + rate limits: https://ai.google.dev/gemini-api/docs/rate-limits"
                    )

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
        Generate a cinematic video script for the channel.
        Niche + character drive the output — content_type hint is optional.
        """
        from src.character_gen import CHANNEL_CHARACTERS

        lang_code, lang_name = (
            (language, SUPPORTED_LANGUAGES.get(language, "English"))
            if language else self._pick_language()
        )
        video_cfg = self.channel.get("content", {}).get("video", {})

        niche = self.channel.get("niche", "general YouTube content")
        if content_type:
            niche = f"{niche} — specifically: {content_type.replace('_', ' ')}"

        channel_key = self.channel.get("key", "strategic_archives")
        char = CHANNEL_CHARACTERS.get(channel_key, CHANNEL_CHARACTERS["strategic_archives"])

        prompt = MASTER_PROMPT.format(
            niche=niche,
            character_name=char["name"],
            character_style=char["style"],
            language=lang_name,
            language_code=lang_code,
            min_duration=video_cfg.get("min_duration", 60),
            max_duration=video_cfg.get("max_duration", 90),
            made_for_kids=str(self.channel.get("made_for_kids", False)).lower(),
            content_rules=_get_content_rules(self.channel),
        )

        logger.info(
            f"Generating script | channel: {self.channel.get('name')} "
            f"| character: {char['name']} | niche: \"{niche[:50]}\" | lang: {lang_name}"
        )

        try:
            result = self._call_gemini(prompt)
        except GeminiQuotaExceededError as e:
            if os.getenv("ALLOW_FALLBACK_SCRIPT", "true").lower() in ("1", "true", "yes"):
                logger.warning(f"{e} — using fallback script generator")
                result = _fallback_script(self.channel, lang_code, lang_name, niche)
            else:
                raise
        result["language_code"] = lang_code
        result["channel_key"] = channel_key
        result["channel_name"] = self.channel.get("name", "")

        # Build legacy scene_prompts list from scenes for backward compat
        scenes = result.get("scenes", [])
        result["scene_prompts"] = [s.get("image_prompt", "") for s in scenes]

        # Build full script text from scene narrations for TTS
        if not result.get("script") and scenes:
            result["script"] = " ".join(
                s.get("narration", "") for s in scenes if s.get("narration")
            )

        return result
