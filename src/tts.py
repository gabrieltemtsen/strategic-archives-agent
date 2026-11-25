"""
Text-to-Speech Module
Primary:   Gemini TTS (natural, expressive voices via GEMINI_API_KEY)
Secondary: Edge TTS (free, unlimited Microsoft Neural voices — no API key)
Fallback:  gTTS (Google Translate TTS - free)

Supports per-scene synthesis so every visual scene and subtitle stays 100% synced
with the exact duration of its spoken narration.
"""

import os
import re
import wave
import asyncio
import logging
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _probe_duration(audio_path: str) -> float:
    """Get exact audio duration in seconds via ffprobe."""
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(res.stdout.strip())
    except Exception:
        # Fallback for WAV files
        try:
            with wave.open(audio_path, "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 5.0


class TTSEngine:
    def __init__(self, config: dict, output_dir: str = "./output"):
        self.channel_config = config
        self.config = config.get("tts", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider = self.config.get("provider", "gemini")
        self.niche = (config.get("niche") or "").lower()

    def _clean_script(self, script: str) -> str:
        """Remove markers and clean script for TTS."""
        script = re.sub(r'\[PAUSE\]', '...', script)
        script = re.sub(r'\[FACT_START\]', '', script)
        script = re.sub(r'\[FACT_END\]', '...', script)
        script = re.sub(r'\[TRANSITION\]', '...', script)
        script = re.sub(r'\[.*?\]', '', script)
        script = re.sub(r'\s+', ' ', script).strip()
        return script

    def _split_into_chunks(self, text: str, max_chars: int = 4000) -> list:
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
        return chunks or [text]

    def _pick_gemini_voice(self) -> str:
        """Select a Gemini voice that matches the channel's niche."""
        explicit = self.config.get("voice")
        if explicit:
            return explicit
        if self._is_kids_channel():
            return "Leda"    # Bright, youthful voice if Gemini fallback is used
        if any(k in self.niche for k in ("war", "history", "horror", "scary", "crime", "creepy")):
            return "Charon"  # Deeper, dramatic storytelling voice
        if any(k in self.niche for k in ("code", "coding", "dev", "ai", "tech", "tutorial")):
            return "Puck"    # Crisp, upbeat explainer voice
        return "Kore"        # Warm, friendly voice

    def _is_drama_channel(self) -> bool:
        key = (self.channel_config.get("key") or "").lower()
        return key == "stories_tales" or any(
            k in self.niche for k in ("nollywood", "nigerian movie", "market drama")
        )

    def _is_kids_channel(self) -> bool:
        key = (self.channel_config.get("key") or "").lower()
        made_for_kids = bool(
            self.channel_config.get("made_for_kids")
            or self.channel_config.get("youtube", {}).get("made_for_kids")
        )
        return key == "kids_universe" or made_for_kids or any(
            k in self.niche for k in ("kids", "children", "nursery", "bedtime")
        )

    def _build_speaker_voices(
        self, characters: Optional[List[dict]], scenes: List[dict], language_code: str
    ) -> dict:
        """Map each character name to a consistent (gemini_voice, edge_voice) pair."""
        female_gemini = ["Kore", "Aoede", "Leda", "Zephyr"]
        male_gemini = ["Charon", "Fenrir", "Puck", "Orus"]
        female_edge_ng = ["en-NG-EzinneNeural"]
        male_edge_ng = ["en-NG-AbeoNeural"]

        char_gender = {}
        for ch in characters or []:
            name = (ch.get("name") or "").strip().lower()
            gender = (ch.get("gender") or "").strip().lower()
            if name:
                char_gender[name] = gender

        f_idx = 0
        m_idx = 0
        voice_map = {}
        for sc in scenes:
            speaker = (sc.get("speaker") or "").strip()
            key = speaker.lower()
            if not key or key in voice_map:
                continue
            gender = char_gender.get(key, "")
            if not gender:
                # Infer from common titles/names or alternate
                if any(w in key for w in ("mama", "madam", "aunty", "sister", "queen", "girl", "woman")):
                    gender = "female"
                elif any(w in key for w in ("chief", "papa", "baba", "bro", "pastor", "man", "boy", "mr")):
                    gender = "male"
                else:
                    gender = "female" if (f_idx <= m_idx) else "male"

            if gender == "female":
                g_voice = female_gemini[f_idx % len(female_gemini)]
                e_voice = female_edge_ng[0] if language_code == "en" else self._pick_edge_voice(language_code)
                f_idx += 1
            else:
                g_voice = male_gemini[m_idx % len(male_gemini)]
                e_voice = male_edge_ng[0] if language_code == "en" else self._pick_edge_voice(language_code)
                m_idx += 1

            voice_map[key] = (g_voice, e_voice)
        return voice_map

    def _pick_edge_voice(self, language_code: str) -> str:
        """Select a free Microsoft Edge Neural voice by language and niche."""
        if self._is_kids_channel():
            kids_map = {
                "en": "en-US-AnaNeural",      # Authentic child voice
                "fr": "fr-FR-EloiseNeural",   # Child voice
                "es": "es-ES-ElviraNeural",
                "pt": "pt-BR-FranciscaNeural",
                "yo": "en-NG-EzinneNeural",
                "ha": "en-NG-AbeoNeural",
                "ig": "en-NG-EzinneNeural",
            }
            return kids_map.get(language_code, "en-US-AnaNeural")
        dramatic = any(k in self.niche for k in ("war", "history", "horror", "scary", "crime"))
        if language_code == "en" and self._is_drama_channel():
            return "en-NG-EzinneNeural"
        voice_map = {
            "en": "en-US-ChristopherNeural" if dramatic else "en-US-AriaNeural",
            "fr": "fr-FR-HenriNeural" if dramatic else "fr-FR-DeniseNeural",
            "es": "es-ES-AlvaroNeural" if dramatic else "es-ES-ElviraNeural",
            "pt": "pt-BR-AntonioNeural" if dramatic else "pt-BR-FranciscaNeural",
            "yo": "en-NG-AbeoNeural" if dramatic else "en-NG-EzinneNeural",
            "ha": "en-NG-AbeoNeural",
            "ig": "en-NG-EzinneNeural",
        }
        return voice_map.get(language_code, "en-US-AriaNeural")

    def _gemini_tts_chunk(self, text: str, wav_path: str, voice_name: str):
        """Synthesize a single chunk via Gemini TTS to WAV."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
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
            future = executor.submit(_call)
            try:
                response = future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                raise RuntimeError("Gemini TTS timed out")

        part = response.candidates[0].content.parts[0]
        audio_data = part.inline_data.data
        mime_type = part.inline_data.mime_type or ""

        if audio_data[:4] == b'RIFF':
            with open(wav_path, "wb") as f:
                f.write(audio_data)
        else:
            sample_rate = 24000
            if "rate=" in mime_type:
                try:
                    sample_rate = int(mime_type.split("rate=")[1].split(";")[0].strip())
                except (ValueError, IndexError):
                    pass
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

    def _edge_tts_chunk(
        self, text: str, mp3_path: str, language_code: str, voice_override: str = None
    ):
        """Synthesize a single chunk via free Edge Neural TTS."""
        import edge_tts

        voice = voice_override or self._pick_edge_voice(language_code)
        rate_val = self.config.get("speaking_rate", 1.0)
        pct = int((rate_val - 1.0) * 100)
        rate_str = f"{pct:+d}%"

        async def _run():
            comm = edge_tts.Communicate(text, voice, rate=rate_str)
            await comm.save(mp3_path)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an async loop (e.g. FastAPI) — run in worker thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(lambda: asyncio.run(_run())).result(timeout=120)
        else:
            asyncio.run(_run())

    def _gtts_chunk(self, text: str, mp3_path: str, language_code: str):
        """Synthesize a single chunk via gTTS (last-resort free fallback)."""
        from gtts import gTTS

        lang_map = {"en": "en", "fr": "fr", "es": "es", "pt": "pt", "yo": "en", "ha": "en", "ig": "en"}
        tts = gTTS(text=text, lang=lang_map.get(language_code, "en"), slow=False)
        tts.save(mp3_path)

    def _synthesize_segment(
        self,
        text: str,
        out_mp3: str,
        language_code: str,
        gemini_voice: str = None,
        edge_voice: str = None,
    ) -> str:
        """Synthesize a short segment (scene) with provider fallback."""
        clean = self._clean_script(text)
        if not clean:
            clean = "..."

        providers = []
        if self._is_kids_channel():
            # Edge TTS has real child voices (en-US-AnaNeural / fr-FR-EloiseNeural)
            providers.append("edge")
            if self.provider == "gemini" and os.getenv("GEMINI_API_KEY"):
                providers.append("gemini")
            providers.append("gtts")
        else:
            if self.provider == "gemini" and os.getenv("GEMINI_API_KEY"):
                providers.append("gemini")
            providers.extend(["edge", "gtts"])

        last_err = None
        for prov in providers:
            try:
                if prov == "gemini":
                    wav_tmp = out_mp3.replace(".mp3", ".wav")
                    self._gemini_tts_chunk(clean, wav_tmp, gemini_voice or self._pick_gemini_voice())
                    self._convert_to_mp3(wav_tmp, out_mp3)
                    if os.path.exists(wav_tmp):
                        os.remove(wav_tmp)
                    return out_mp3
                elif prov == "edge":
                    self._edge_tts_chunk(clean, out_mp3, language_code, voice_override=edge_voice)
                    return out_mp3
                elif prov == "gtts":
                    self._gtts_chunk(clean, out_mp3, language_code)
                    return out_mp3
            except Exception as e:
                last_err = e
                logger.warning(f"TTS provider '{prov}' failed for segment: {e}")

        raise RuntimeError(f"All TTS providers failed: {last_err}")

    def _convert_to_mp3(self, src_path: str, mp3_path: str):
        """Normalize audio to 44.1kHz stereo MP3 for uniform concatenation."""
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", src_path,
                "-ar", "44100", "-ac", "2",
                "-codec:a", "libmp3lame", "-b:a", "192k",
                mp3_path,
            ],
            check=True,
        )

    def _concat_with_pauses(self, segment_paths: List[str], output_path: str, pause_sec: float = 0.35) -> List[float]:
        """
        Normalize all segments, insert a brief natural pause between scenes,
        and concatenate into a single MP3. Returns exact duration of each scene
        (including its trailing pause, except the last scene).
        """
        silence_mp3 = str(self.output_dir / f"{Path(output_path).stem}_silence.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", str(pause_sec),
                "-codec:a", "libmp3lame", "-b:a", "192k",
                silence_mp3,
            ],
            check=True,
        )

        norm_files = []
        scene_durations = []
        concat_list = []

        try:
            for idx, seg in enumerate(segment_paths):
                norm = seg.replace(".mp3", "_norm.mp3")
                self._convert_to_mp3(seg, norm)
                norm_files.append(norm)
                dur = _probe_duration(norm)
                concat_list.append(norm)
                if idx < len(segment_paths) - 1:
                    concat_list.append(silence_mp3)
                    scene_durations.append(dur + pause_sec)
                else:
                    scene_durations.append(dur)

            list_txt = output_path.replace(".mp3", "_concat.txt")
            with open(list_txt, "w") as f:
                for p in concat_list:
                    f.write(f"file '{os.path.abspath(p)}'\n")

            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0",
                    "-i", list_txt,
                    "-c", "copy", output_path,
                ],
                check=True,
            )
            os.remove(list_txt)
        finally:
            if os.path.exists(silence_mp3):
                os.remove(silence_mp3)
            for p in norm_files + segment_paths:
                if os.path.exists(p):
                    os.remove(p)

        return scene_durations

    def synthesize_scenes(
        self,
        scenes: List[dict],
        job_id: str,
        language_code: str = "en",
        characters: List[dict] = None,
    ) -> Tuple[str, List[dict]]:
        """
        Synthesize each scene separately so we get exact per-scene audio timings.
        Supports multi-character voice mapping when scenes include `speaker`.
        Returns (merged_audio_path, scene_timings).
        """
        output_path = str(self.output_dir / f"{job_id}_audio.mp3")
        segment_paths = []
        speaker_voices = self._build_speaker_voices(characters, scenes, language_code)
        if speaker_voices:
            logger.info(f"Multi-character voices mapped: {list(speaker_voices.keys())}")
        logger.info(f"Synthesizing {len(scenes)} scenes (primary: {self.provider})...")

        for i, sc in enumerate(scenes):
            seg_path = str(self.output_dir / f"{job_id}_scene_audio_{i:03d}.mp3")
            speaker_key = (sc.get("speaker") or "").strip().lower()
            g_voice, e_voice = speaker_voices.get(speaker_key, (None, None))
            self._synthesize_segment(
                sc.get("narration", ""),
                seg_path,
                language_code,
                gemini_voice=g_voice,
                edge_voice=e_voice,
            )
            segment_paths.append(seg_path)
            logger.info(f"TTS scene {i + 1}/{len(scenes)} ready")

        durations = self._concat_with_pauses(segment_paths, output_path, pause_sec=0.35)

        timings = []
        cursor = 0.0
        for i, (sc, dur) in enumerate(zip(scenes, durations)):
            timings.append({
                "index": i,
                "start": cursor,
                "end": cursor + dur,
                "duration": dur,
                "narration": self._clean_script(sc.get("narration", "")),
                "camera_motion": sc.get("camera_motion", "zoom_in"),
                "caption_highlight": sc.get("caption_highlight", ""),
            })
            cursor += dur

        logger.info(f"Full narration ready: {cursor:.1f}s across {len(timings)} synced scenes")
        return output_path, timings

    def synthesize(self, script: str, job_id: str, language_code: str = "en") -> str:
        """Legacy entry point: synthesize a flat script string into a single MP3."""
        clean_text = self._clean_script(script)
        chunks = self._split_into_chunks(clean_text, max_chars=3500)
        fake_scenes = [{"narration": c} for c in chunks]
        audio_path, _ = self.synthesize_scenes(fake_scenes, job_id, language_code)
        return audio_path

    def get_duration_estimate(self, script: str) -> float:
        clean = self._clean_script(script)
        word_count = len(clean.split())
        words_per_minute = 135 * self.config.get("speaking_rate", 0.9)
        return (word_count / words_per_minute) * 60
