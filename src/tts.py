"""
Text-to-Speech Module
Primary: Google Cloud TTS (free 4M chars/month)
Fallback: gTTS (Google Translate TTS - fully free)
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, config: dict, output_dir: str = "./output"):
        self.config = config.get("tts", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider = self.config.get("provider", "google")

    def _clean_script(self, script: str) -> str:
        """Remove markers and clean script for TTS."""
        script = re.sub(r'\[PAUSE\]', '...', script)
        script = re.sub(r'\[FACT_START\]', '', script)
        script = re.sub(r'\[FACT_END\]', '...', script)
        script = re.sub(r'\[TRANSITION\]', '...', script)
        script = re.sub(r'\[.*?\]', '', script)  # Remove any remaining markers
        script = re.sub(r'\s+', ' ', script).strip()
        return script

    def _split_into_chunks(self, text: str, max_chars: int = 4500) -> list:
        """Split long text into chunks for API limits."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) < max_chars:
                current += " " + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks

    def synthesize_google_cloud(
        self, text: str, output_path: str, language_code: str = "en-US"
    ) -> str:
        """Use Google Cloud TTS API."""
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()

        # Map language codes
        lang_map = {
            "en": "en-US", "fr": "fr-FR", "es": "es-ES",
            "pt": "pt-BR", "yo": "yo-NG", "ha": "ha-NE", "ig": "ig-NG"
        }
        lang_code = lang_map.get(language_code, "en-US")

        # Kids-friendly voice selection
        voice_map = {
            "en-US": "en-US-Neural2-H",  # Warm, friendly voice
            "fr-FR": "fr-FR-Neural2-A",
            "es-ES": "es-ES-Neural2-A",
            "pt-BR": "pt-BR-Neural2-A",
        }

        voice_name = voice_map.get(lang_code)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name if voice_name else None,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=self.config.get("speaking_rate", 0.85),
            pitch=self.config.get("pitch", 2.0),
            volume_gain_db=self.config.get("volume_gain_db", 2.0),
            effects_profile_id=["headphone-class-device"]
        )

        chunks = self._split_into_chunks(text)
        audio_files = []

        for i, chunk in enumerate(chunks):
            synthesis_input = texttospeech.SynthesisInput(text=chunk)
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            chunk_path = output_path.replace(".mp3", f"_chunk_{i}.mp3")
            with open(chunk_path, "wb") as f:
                f.write(response.audio_content)
            audio_files.append(chunk_path)
            logger.info(f"TTS chunk {i+1}/{len(chunks)} synthesized")

        # Merge chunks if multiple
        if len(audio_files) == 1:
            import shutil
            shutil.move(audio_files[0], output_path)
        else:
            self._merge_audio(audio_files, output_path)
            for f in audio_files:
                os.remove(f)

        return output_path

    def synthesize_gtts(
        self, text: str, output_path: str, language_code: str = "en"
    ) -> str:
        """Use gTTS (Google Translate TTS) - fully free fallback."""
        from gtts import gTTS
        import subprocess

        # gTTS language codes
        lang_map = {
            "en": "en", "fr": "fr", "es": "es",
            "pt": "pt", "yo": "yo", "ha": "ha", "ig": "ig"
        }
        lang = lang_map.get(language_code, "en")

        chunks = self._split_into_chunks(text, max_chars=3000)
        audio_files = []

        for i, chunk in enumerate(chunks):
            chunk_path = output_path.replace(".mp3", f"_chunk_{i}.mp3")
            tts = gTTS(text=chunk, lang=lang, slow=False)
            tts.save(chunk_path)
            audio_files.append(chunk_path)
            logger.info(f"gTTS chunk {i+1}/{len(chunks)} saved")

        if len(audio_files) == 1:
            import shutil
            shutil.move(audio_files[0], output_path)
        else:
            self._merge_audio(audio_files, output_path)
            for f in audio_files:
                os.remove(f)

        return output_path

    def _merge_audio(self, audio_files: list, output_path: str):
        """Merge multiple audio files using ffmpeg."""
        import subprocess
        # Create a file list for ffmpeg
        list_path = output_path.replace(".mp3", "_list.txt")
        with open(list_path, "w") as f:
            for audio in audio_files:
                f.write(f"file '{os.path.abspath(audio)}'\n")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", output_path, "-y", "-loglevel", "quiet"
        ], check=True)
        os.remove(list_path)

    def synthesize(
        self, script: str, job_id: str, language_code: str = "en"
    ) -> str:
        """
        Main TTS entry point.
        Returns path to generated audio file.
        """
        clean_text = self._clean_script(script)
        output_path = str(self.output_dir / f"{job_id}_audio.mp3")

        logger.info(f"Synthesizing speech ({len(clean_text)} chars) via {self.provider}")

        try:
            if self.provider == "google":
                return self.synthesize_google_cloud(clean_text, output_path, language_code)
            else:
                return self.synthesize_gtts(clean_text, output_path, language_code)
        except Exception as e:
            logger.warning(f"Primary TTS provider failed: {e}. Falling back to gTTS...")
            return self.synthesize_gtts(clean_text, output_path, language_code)

    def get_duration_estimate(self, script: str) -> float:
        """Estimate audio duration in seconds (avg 130 words/min for kids)."""
        clean = self._clean_script(script)
        word_count = len(clean.split())
        words_per_minute = 130 * self.config.get("speaking_rate", 0.85)
        return (word_count / words_per_minute) * 60
