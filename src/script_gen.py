"""
Script Generator - Uses Gemini API to generate kids video scripts
Supports: Bedtime Stories, Fun Facts, and extensible content types
Uses: google-genai SDK (new, replaces deprecated google-generativeai)
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


BEDTIME_STORY_PROMPT = """
You are a children's storyteller creating a soothing bedtime story for kids aged {age_range}.
Write a complete, engaging bedtime story with the following:

Topic/Theme: {theme}
Language: {language}
Word count: {word_count_min} - {word_count_max} words
Tone: {tone}

Requirements:
- Begin with a calming, inviting opening
- Include a gentle moral lesson
- Use simple, age-appropriate vocabulary
- Create vivid but peaceful imagery
- End with a sleepy, satisfying conclusion
- Include natural pause points (marked with [PAUSE])
- No scary elements, violence, or adult themes

Return a JSON object with:
{{
  "title": "Story title",
  "description": "2-3 sentence YouTube description",
  "script": "Full story text with [PAUSE] markers",
  "moral": "One-line moral of the story",
  "tags": ["tag1", "tag2", ...],
  "thumbnail_prompt": "Detailed prompt for generating thumbnail image",
  "scene_prompts": ["prompt1", "prompt2", ...],  // 12-15 image prompts for scenes
  "language": "{language}",
  "content_type": "bedtime_story"
}}
"""

FUN_FACTS_PROMPT = """
You are an enthusiastic kids educator creating an exciting fun facts video for children aged {age_range}.
Create a fun facts video script about: {topic}
Language: {language}
Number of facts: {facts_count}
Tone: {tone}

Requirements:
- Start with an exciting hook/intro
- Each fact should be surprising and easy to understand
- Use simple analogies kids can relate to
- Include "WOW!" moments and enthusiasm
- Add a fun quiz question at the end
- Mark each fact with [FACT_START] and [FACT_END]
- Mark transitions with [TRANSITION]

Return a JSON object with:
{{
  "title": "Video title",
  "description": "2-3 sentence YouTube description",
  "script": "Full script with markers",
  "facts": ["fact1", "fact2", ...],  // clean list of facts
  "quiz_question": "Fun quiz question",
  "quiz_answer": "Answer",
  "tags": ["tag1", "tag2", ...],
  "thumbnail_prompt": "Detailed prompt for generating thumbnail image",
  "scene_prompts": ["prompt1", "prompt2", ...],  // 12-15 image prompts
  "language": "{language}",
  "content_type": "fun_facts"
}}
"""

SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
    "pt": "Portuguese",
    "es": "Spanish"
}

BEDTIME_THEMES = [
    "a little bunny who can't sleep",
    "a baby elephant learning to use their trunk",
    "a star that fell from the sky",
    "a dragon who only breathes rainbow bubbles",
    "a child who discovers a magic garden",
    "a bear family preparing for winter",
    "a lighthouse keeper's daughter and the sea",
    "a tiny seed growing into a big tree",
    "a cloud that learns to make rain",
    "a moon who wanted to see the daytime"
]

FUN_FACT_TOPICS = [
    "amazing animals",
    "space and planets",
    "the deep ocean",
    "dinosaurs",
    "the human body",
    "insects and bugs",
    "volcanoes and earthquakes",
    "rainforests",
    "robots and technology",
    "ancient civilizations"
]


GEMINI_MODEL = "gemini-2.5-flash"


class ScriptGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.content_config = config.get("content", {})

    def _call_gemini(self, prompt: str, _retry: int = 0) -> dict:
        """Call Gemini API and parse JSON response."""
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt + "\n\nIMPORTANT: Return ONLY valid, compact JSON. Keep the script concise. No markdown formatting.",
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    top_p=0.95,
                    max_output_tokens=32768,
                )
            )
            # Extract JSON from response
            text = response.text.strip()
            # Handle markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Try direct parse first
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Attempt to repair: fix unescaped newlines inside JSON strings
            import re
            repaired = re.sub(r'(?<=": ")(.*?)(?="[,\}])', lambda m: m.group(0).replace('\n', '\\n'), text, flags=re.DOTALL)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

            # Attempt to repair: truncated JSON — close any open strings/objects
            if not text.rstrip().endswith('}'):
                # Try to close the JSON
                fixed = text.rstrip().rstrip(',')
                # Count open braces/brackets
                open_braces = fixed.count('{') - fixed.count('}')
                open_brackets = fixed.count('[') - fixed.count(']')
                # Check if we're in an unclosed string
                in_string = fixed.count('"') % 2 != 0
                if in_string:
                    fixed += '"'
                fixed += ']' * max(0, open_brackets)
                fixed += '}' * max(0, open_braces)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

            raise json.JSONDecodeError("Could not parse or repair JSON", text[:100], 0)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {response.text[:500]}")
            if _retry < 2:
                logger.info(f"Retrying Gemini call (attempt {_retry + 2})...")
                return self._call_gemini(prompt, _retry=_retry + 1)
            raise
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def generate_bedtime_story(
        self,
        theme: Optional[str] = None,
        language: str = "en"
    ) -> dict:
        """Generate a bedtime story script."""
        if not theme:
            theme = random.choice(BEDTIME_THEMES)

        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        story_config = self.content_config.get("bedtime_stories", {})

        prompt = BEDTIME_STORY_PROMPT.format(
            age_range=story_config.get("target_age", "3-8"),
            theme=theme,
            language=lang_name,
            word_count_min=story_config.get("word_count_min", 800),
            word_count_max=story_config.get("word_count_max", 1200),
            tone=story_config.get("tone", "warm, soothing, magical")
        )

        logger.info(f"Generating bedtime story: '{theme}' in {lang_name}")
        result = self._call_gemini(prompt)
        result["theme"] = theme
        result["language_code"] = language
        return result

    def generate_fun_facts(
        self,
        topic: Optional[str] = None,
        language: str = "en"
    ) -> dict:
        """Generate a fun facts script."""
        if not topic:
            topic = random.choice(FUN_FACT_TOPICS)

        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        facts_config = self.content_config.get("fun_facts", {})

        prompt = FUN_FACTS_PROMPT.format(
            age_range=facts_config.get("target_age", "4-10"),
            topic=topic,
            language=lang_name,
            facts_count=random.randint(
                facts_config.get("facts_count_min", 10),
                facts_config.get("facts_count_max", 15)
            ),
            tone=facts_config.get("tone", "energetic, curious, fun")
        )

        logger.info(f"Generating fun facts: '{topic}' in {lang_name}")
        result = self._call_gemini(prompt)
        result["topic"] = topic
        result["language_code"] = language
        return result

    def generate(
        self,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> dict:
        """
        Main entry point. Auto-picks content type and language if not specified.
        content_type: 'bedtime_story' | 'fun_facts' | None (random)
        """
        content_types = self.content_config.get("types", ["bedtime_stories", "fun_facts"])
        languages = self.content_config.get("languages", {})

        if not content_type:
            content_type = random.choice(content_types)

        if not language:
            supported = languages.get("supported", ["en"])
            # Weight English and French higher for broader reach
            weighted = ["en"] * 3 + ["fr"] * 2 + [l for l in supported if l not in ["en", "fr"]]
            language = random.choice(weighted)

        if content_type in ("bedtime_stories", "bedtime_story"):
            return self.generate_bedtime_story(language=language, **kwargs)
        elif content_type in ("fun_facts",):
            return self.generate_fun_facts(language=language, **kwargs)
        else:
            logger.warning(f"Unknown content type '{content_type}', defaulting to bedtime story")
            return self.generate_bedtime_story(language=language)
