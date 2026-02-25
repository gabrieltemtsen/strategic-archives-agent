"""
Text-to-Speech Module
Primary: Gemini TTS (natural, expressive voices via Gemini API)
Secondary: Google Cloud TTS (free 4M chars/month)
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
        self.provider = self.config.get("provider", "gemini")

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

    def synthesize_gemini(
        self, text: str, output_path: str, language_code: str = "en"
    ) -> str:
        """
        Use Gemini TTS for natural, expressive voice synthesis.
        Uses the same GEMINI_API_KEY — no extra credentials needed.
        """
        from google import genai
        from google.genai import types
        import wave
        import struct
        import io

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Map language codes to Gemini voice names
        voice_map = {
            "en": "Kore",    # Warm, friendly English voice
            "fr": "Kore",
            "es": "Kore",
            "pt": "Kore",
            "yo": "Kore",
            "ha": "Kore",
            "ig": "Kore",
        }
        voice_name = voice_map.get(language_code, "Kore")

        chunks = self._split_into_chunks(text, max_chars=4000)
        audio_files = []

        for i, chunk in enumerate(chunks):
            chunk_path = output_path.replace(".mp3", f"_chunk_{i}.wav")

            # Use timeout to prevent hanging API calls
            import concurrent.futures
            def _call_tts():
                return client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=chunk,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                    ),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_tts)
                try:
                    response = future.result(timeout=120)  # 2 min max per chunk
                except concurrent.futures.TimeoutError:
                    raise RuntimeError(f"Gemini TTS timed out on chunk {i+1}/{len(chunks)}")

            # Extract audio data from response
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            mime_type = response.candidates[0].content.parts[0].inline_data.mime_type or ""

            # Gemini returns audio/L16 (raw PCM) or WAV — handle both
            if audio_data[:4] == b'RIFF':
                # Already a proper WAV file
                with open(chunk_path, "wb") as f:
                    f.write(audio_data)
            else:
                # Raw PCM data - wrap in WAV container
                # Gemini TTS typically outputs 24kHz, 16-bit, mono PCM
                sample_rate = 24000
                if "rate=" in mime_type:
                    try:
                        sample_rate = int(mime_type.split("rate=")[1].split(";")[0].strip())
                    except (ValueError, IndexError):
                        pass
                with wave.open(chunk_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

            audio_files.append(chunk_path)
            logger.info(f"Gemini TTS chunk {i+1}/{len(chunks)} synthesized")

        # Merge and convert to MP3
        if len(audio_files) == 1:
            self._convert_wav_to_mp3(audio_files[0], output_path)
            os.remove(audio_files[0])
        else:
            # Merge WAV files using Python wave module (no ffmpeg concat issues)
            merged_wav = output_path.replace(".mp3", "_merged.wav")
            self._merge_wav_python(audio_files, merged_wav)
            self._convert_wav_to_mp3(merged_wav, output_path)
            os.remove(merged_wav)
            for f in audio_files:
                if os.path.exists(f):
                    os.remove(f)

        return output_path

    def _merge_wav_python(self, wav_files: list, output_path: str):
        """Merge WAV files using Python wave module — no ffmpeg dependency."""
        import wave

        # Read first file to get params
        with wave.open(wav_files[0], "rb") as first:
            params = first.getparams()

        with wave.open(output_path, "wb") as output:
            output.setparams(params)
            for wav_file in wav_files:
                with wave.open(wav_file, "rb") as wf:
                    output.writeframes(wf.readframes(wf.getnframes()))
        logger.info(f"Merged {len(wav_files)} audio chunks → {output_path}")

    def _convert_wav_to_mp3(self, wav_path: str, mp3_path: str):
        """Convert WAV to MP3 using ffmpeg."""
        import subprocess
        subprocess.run([
            "ffmpeg", "-i", wav_path,
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            mp3_path, "-y", "-loglevel", "quiet"
        ], check=True)

    def _merge_audio_wav(self, audio_files: list, output_path: str):
        """Merge multiple WAV/audio files using ffmpeg re-encode (fallback)."""
        import subprocess
        # Use filter_complex to concatenate with re-encoding (avoids codec issues)
        inputs = []
        for f in audio_files:
            inputs.extend(["-i", f])
        filter_str = "".join(f"[{i}:a]" for i in range(len(audio_files)))
        filter_str += f"concat=n={len(audio_files)}:v=0:a=1[out]"
        subprocess.run(
            ["ffmpeg"] + inputs +
            ["-filter_complex", filter_str, "-map", "[out]",
             output_path, "-y", "-loglevel", "quiet"],
            check=True
        )

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

        # Try providers in order: gemini → google cloud → gtts
        providers = []
        if self.provider == "gemini":
            providers = [
                ("gemini", lambda: self.synthesize_gemini(clean_text, output_path, language_code)),
                ("gtts", lambda: self.synthesize_gtts(clean_text, output_path, language_code)),
            ]
        elif self.provider == "google":
            providers = [
                ("google", lambda: self.synthesize_google_cloud(clean_text, output_path, language_code)),
                ("gtts", lambda: self.synthesize_gtts(clean_text, output_path, language_code)),
            ]
        else:
            providers = [
                ("gtts", lambda: self.synthesize_gtts(clean_text, output_path, language_code)),
            ]

        for name, synth_fn in providers:
            try:
                return synth_fn()
            except Exception as e:
                logger.warning(f"TTS provider '{name}' failed: {e}.")
                if name != providers[-1][0]:
                    logger.info(f"Falling back to next provider...")
                else:
                    raise

    def get_duration_estimate(self, script: str) -> float:
        """Estimate audio duration in seconds (avg 130 words/min for kids)."""
        clean = self._clean_script(script)
        word_count = len(clean.split())
        words_per_minute = 130 * self.config.get("speaking_rate", 0.85)
        return (word_count / words_per_minute) * 60
