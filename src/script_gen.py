"""
Script Generator - Channel-aware content generation via Gemini
Supports all channel niches: kids, horror, african folklore, motivational
Uses: google-genai SDK
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

GEMINI_MODEL = "gemini-2.0-flash"

# ─── Prompt Templates ────────────────────────────────────────────────────────

BEDTIME_STORY_PROMPT = """
You are a children's storyteller creating a soothing bedtime story for kids aged {age_range}.
Topic/Theme: {theme} | Language: {language} | Words: {word_count_min}-{word_count_max} | Tone: {tone}

Requirements:
- Begin with a calming, inviting opening
- Gentle moral lesson, simple vocabulary, vivid but peaceful imagery
- End with a sleepy, satisfying conclusion
- Include natural pause points marked with [PAUSE]
- No scary elements, violence, or adult themes

Return JSON:
{{"title": "...", "description": "2-3 sentence YouTube description",
  "script": "full story with [PAUSE] markers", "moral": "one-line moral",
  "tags": ["tag1",...], "thumbnail_prompt": "detailed image prompt",
  "scene_prompts": ["prompt1",...],
  "language": "{language}", "content_type": "bedtime_story"}}
"""

FUN_FACTS_PROMPT = """
You are an enthusiastic kids educator. Create a fun facts video about: {topic}
Language: {language} | Facts: {facts_count} | Age: {age_range} | Tone: {tone}

Requirements:
- Exciting hook/intro, surprising facts, simple analogies kids relate to
- Mark each fact with [FACT_START] and [FACT_END], transitions with [TRANSITION]
- End with a fun quiz question

Return JSON:
{{"title": "...", "description": "...", "script": "full script with markers",
  "facts": ["fact1",...], "quiz_question": "...", "quiz_answer": "...",
  "tags": ["tag1",...], "thumbnail_prompt": "...", "scene_prompts": ["prompt1",...],
  "language": "{language}", "content_type": "fun_facts"}}
"""

HORROR_STORY_PROMPT = """
You are a master horror storyteller. Write a chilling {content_subtype} in {language}.
Theme: {theme} | Words: {word_count_min}-{word_count_max} | Tone: {tone}

Requirements:
- Open with an unsettling hook that grabs immediately
- Build tension gradually, use atmospheric descriptions
- Include a shocking or ambiguous ending
- Mark scene breaks with [SCENE_BREAK], tense moments with [TENSION]
- Suitable for adult audiences — no children's content

Return JSON:
{{"title": "...", "description": "...", "script": "full story with markers",
  "hook": "opening line", "twist": "the twist or ending description",
  "tags": ["tag1",...], "thumbnail_prompt": "dark, atmospheric image prompt",
  "scene_prompts": ["dark scene prompt1",...],
  "language": "{language}", "content_type": "{content_subtype}"}}
"""

FOLKTALE_PROMPT = """
You are a master African storyteller bringing ancient wisdom to life.
Create a {content_subtype} in {language} from {region} Africa.
Theme: {theme} | Words: {word_count_min}-{word_count_max} | Tone: {tone}

Requirements:
- Open with a traditional storytelling phrase appropriate to the culture
- Weave in authentic cultural elements, proverbs, or spiritual beliefs
- Characters can be humans, animals, spirits, or deities
- Include a moral or cultural wisdom takeaway
- Mark story sections with [SCENE] and proverbs with [PROVERB]
- Use rich, rhythmic language that echoes oral tradition

Return JSON:
{{"title": "...", "description": "...", "script": "full story with markers",
  "moral": "cultural wisdom", "proverb": "relevant African proverb",
  "culture": "specific culture/ethnic group",
  "tags": ["tag1",...], "thumbnail_prompt": "african art style image prompt",
  "scene_prompts": ["african cultural scene prompt1",...],
  "language": "{language}", "content_type": "{content_subtype}"}}
"""

MOTIVATIONAL_PROMPT = """
You are a powerful motivational storyteller. Create a {content_subtype} in {language}.
Theme: {theme} | Words: {word_count_min}-{word_count_max} | Tone: {tone}

Requirements:
- Open with a powerful question or bold statement
- Tell a compelling story (real or composite) that illustrates the theme
- Include specific, vivid details — not generic platitudes
- Build to an emotional climax then a clear actionable takeaway
- Mark key moments with [KEY_MOMENT] and the lesson with [LESSON]

Return JSON:
{{"title": "...", "description": "...", "script": "full story with markers",
  "key_lesson": "one-line takeaway", "call_to_action": "closing CTA",
  "tags": ["tag1",...], "thumbnail_prompt": "cinematic, inspirational image prompt",
  "scene_prompts": ["cinematic scene prompt1",...],
  "language": "{language}", "content_type": "{content_subtype}"}}
"""

# ─── Random theme/topic pools ─────────────────────────────────────────────────

THEMES = {
    "bedtime_stories":   ["a little bunny who can't sleep", "a baby elephant learning to fly",
                          "a star that fell from the sky", "a dragon who only breathes rainbows",
                          "a child who discovers a magic garden"],
    "fun_facts":         ["amazing animals", "space and planets", "the deep ocean",
                          "dinosaurs", "the human body", "insects and bugs",
                          "volcanoes", "rainforests", "robots", "ancient civilizations"],
    "horror_story":      ["paranormal", "urban_legends", "supernatural",
                          "psychological", "folklore horror"],
    "creepypasta":       ["internet horror", "found footage", "ritual gone wrong",
                          "haunted place", "cursed object"],
    "folktale":          ["trickster tales", "animal wisdom", "ancestor spirits",
                          "hero journey", "moral lesson", "nature spirits"],
    "myth_legend":       ["orishas", "kings and warriors", "ancient kingdoms",
                          "sacred rivers", "legendary creatures", "origin stories"],
    "motivational_story":["overcoming adversity", "self belief", "discipline",
                          "resilience", "finding purpose", "legacy"],
    "success_story":     ["entrepreneurs", "athletes", "scientists",
                          "artists", "everyday heroes"],
}

AFRICAN_REGIONS = ["West African", "East African", "Southern African",
                   "Central African", "North African", "Pan-African"]

SUPPORTED_LANGUAGES = {
    "en": "English", "fr": "French", "yo": "Yoruba",
    "ha": "Hausa", "ig": "Igbo", "pt": "Portuguese", "es": "Spanish"
}


class ScriptGenerator:
    def __init__(self, channel_config: dict):
        """
        channel_config: the config for a specific channel
        e.g. config['channels']['kids_universe']
        """
        self.channel_config = channel_config
        self.content_config = channel_config.get("content", {})
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _call_gemini(self, prompt: str) -> dict:
        """Call Gemini and parse JSON response."""
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
            logger.error(f"Failed to parse Gemini JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _pick_language(self) -> tuple[str, str]:
        """Pick a language code + name based on channel config."""
        supported = self.content_config.get("languages", {}).get("supported", ["en"])
        weighted = ["en"] * 3 + ["fr"] * 2 + [l for l in supported if l not in ("en", "fr")]
        code = random.choice(weighted)
        return code, SUPPORTED_LANGUAGES.get(code, "English")

    # ── Kids content ─────────────────────────────────────────────────────────

    def generate_bedtime_story(self, theme: Optional[str] = None, language: str = "en") -> dict:
        if not theme:
            theme = random.choice(THEMES["bedtime_stories"])
        cfg = self.content_config.get("bedtime_stories", {})
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        prompt = BEDTIME_STORY_PROMPT.format(
            age_range=cfg.get("target_age", "3-8"), theme=theme, language=lang_name,
            word_count_min=cfg.get("word_count_min", 800),
            word_count_max=cfg.get("word_count_max", 1200),
            tone=cfg.get("tone", "warm, soothing, magical")
        )
        result = self._call_gemini(prompt)
        result.update({"theme": theme, "language_code": language})
        return result

    def generate_fun_facts(self, topic: Optional[str] = None, language: str = "en") -> dict:
        if not topic:
            topic = random.choice(THEMES["fun_facts"])
        cfg = self.content_config.get("fun_facts", {})
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        prompt = FUN_FACTS_PROMPT.format(
            age_range=cfg.get("target_age", "4-10"), topic=topic, language=lang_name,
            facts_count=random.randint(cfg.get("facts_count_min", 10), cfg.get("facts_count_max", 15)),
            tone=cfg.get("tone", "energetic, curious, fun")
        )
        result = self._call_gemini(prompt)
        result.update({"topic": topic, "language_code": language})
        return result

    # ── Horror content ────────────────────────────────────────────────────────

    def generate_horror_story(self, subtype: str = "horror_story",
                               theme: Optional[str] = None, language: str = "en") -> dict:
        if not theme:
            theme = random.choice(THEMES.get(subtype, THEMES["horror_story"]))
        cfg = self.content_config.get(subtype, self.content_config.get("horror_story", {}))
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        prompt = HORROR_STORY_PROMPT.format(
            content_subtype=subtype, theme=theme, language=lang_name,
            word_count_min=cfg.get("word_count_min", 1000),
            word_count_max=cfg.get("word_count_max", 1800),
            tone=cfg.get("tone", "eerie, suspenseful, chilling")
        )
        result = self._call_gemini(prompt)
        result.update({"theme": theme, "language_code": language})
        return result

    # ── African Folklore ──────────────────────────────────────────────────────

    def generate_folktale(self, subtype: str = "folktale",
                           theme: Optional[str] = None, language: str = "en") -> dict:
        if not theme:
            theme = random.choice(THEMES.get(subtype, THEMES["folktale"]))
        cfg = self.content_config.get(subtype, self.content_config.get("folktale", {}))
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        region = random.choice(AFRICAN_REGIONS)
        prompt = FOLKTALE_PROMPT.format(
            content_subtype=subtype, theme=theme, language=lang_name, region=region,
            word_count_min=cfg.get("word_count_min", 900),
            word_count_max=cfg.get("word_count_max", 1400),
            tone=cfg.get("tone", "cultural, storytelling, rhythmic")
        )
        result = self._call_gemini(prompt)
        result.update({"theme": theme, "language_code": language, "region": region})
        return result

    # ── Motivational content ──────────────────────────────────────────────────

    def generate_motivational(self, subtype: str = "motivational_story",
                               theme: Optional[str] = None, language: str = "en") -> dict:
        if not theme:
            theme = random.choice(THEMES.get(subtype, THEMES["motivational_story"]))
        cfg = self.content_config.get(subtype, self.content_config.get("motivational_story", {}))
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        prompt = MOTIVATIONAL_PROMPT.format(
            content_subtype=subtype, theme=theme, language=lang_name,
            word_count_min=cfg.get("word_count_min", 700),
            word_count_max=cfg.get("word_count_max", 1200),
            tone=cfg.get("tone", "inspiring, powerful, uplifting")
        )
        result = self._call_gemini(prompt)
        result.update({"theme": theme, "language_code": language})
        return result

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(self, content_type: Optional[str] = None,
                 language: Optional[str] = None, **kwargs) -> dict:
        """
        Generate content for this channel.
        Auto-selects type and language from channel config if not specified.
        """
        types_pool = self.content_config.get("types", ["bedtime_stories"])
        if not content_type:
            content_type = random.choice(types_pool)

        lang_code, lang_name = (language, SUPPORTED_LANGUAGES.get(language, "English")) \
            if language else self._pick_language()

        logger.info(f"Generating [{content_type}] in {lang_name} for channel: "
                    f"{self.channel_config.get('name', 'unknown')}")

        dispatch = {
            "bedtime_stories": self.generate_bedtime_story,
            "bedtime_story":   self.generate_bedtime_story,
            "fun_facts":       self.generate_fun_facts,
            "horror_story":    lambda **kw: self.generate_horror_story(subtype="horror_story", **kw),
            "creepypasta":     lambda **kw: self.generate_horror_story(subtype="creepypasta", **kw),
            "folktale":        lambda **kw: self.generate_folktale(subtype="folktale", **kw),
            "myth_legend":     lambda **kw: self.generate_folktale(subtype="myth_legend", **kw),
            "motivational_story": lambda **kw: self.generate_motivational(subtype="motivational_story", **kw),
            "success_story":   lambda **kw: self.generate_motivational(subtype="success_story", **kw),
        }

        fn = dispatch.get(content_type)
        if not fn:
            logger.warning(f"Unknown content type '{content_type}', defaulting to bedtime_story")
            fn = self.generate_bedtime_story

        return fn(language=lang_code, **kwargs)
