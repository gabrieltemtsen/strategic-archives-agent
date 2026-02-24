# 🎬 Strategic Archives Agent

Automated daily YouTube content pipeline for **@strategicarchivesHQ** kids channel.

Generates, reviews, compiles, and uploads videos **daily at 6pm WAT** — fully on autopilot after a one-time setup.

---

## ✨ What It Does

```
Gemini generates script
       ↓
Telegram approval (you review here)
       ↓
Google TTS narration
       ↓
AI image generation (Hugging Face)
       ↓
FFmpeg video compilation
       ↓
YouTube upload (scheduled 6pm WAT)
       ↓
Telegram confirmation with video link
```

---

## 🎯 Content Types

| Type | Target Age | Length |
|---|---|---|
| Bedtime Stories | 3–8 years | 5–10 min |
| Fun Facts | 4–10 years | 5–10 min |

**Languages:** English, French, Yoruba, Hausa, Igbo, Portuguese, Spanish

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/gabrieltemtsen/strategic-archives-agent
cd strategic-archives-agent
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. YouTube OAuth Setup
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Enable **YouTube Data API v3**
- Create OAuth2 credentials → Download as `client_secrets.json`
- Place in project root

### 4. Run
```bash
# Start daily scheduler (6pm WAT)
python -m src.main --schedule

# Run immediately (one-off)
python -m src.main --run-now

# Dry run (stop after script approval, no video/upload)
python -m src.main --dry-run

# Override content type or language
python -m src.main --run-now --type fun_facts --lang fr
```

---

## 🔑 Required API Keys

| Service | Where to Get | Cost |
|---|---|---|
| Gemini API | [aistudio.google.com](https://aistudio.google.com) | Free tier |
| Google Cloud TTS | [console.cloud.google.com](https://console.cloud.google.com) | Free (4M chars/mo) |
| YouTube Data API | [console.cloud.google.com](https://console.cloud.google.com) | Free |
| Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free tier |
| Telegram Bot | [@BotFather](https://t.me/BotFather) | Free |

---

## 📁 Project Structure

```
strategic-archives-agent/
├── src/
│   ├── main.py          # Orchestrator + scheduler
│   ├── script_gen.py    # Gemini script generation
│   ├── tts.py           # Google TTS / gTTS voiceover
│   ├── visuals.py       # Hugging Face image generation
│   ├── video.py         # FFmpeg/MoviePy video compiler
│   ├── upload.py        # YouTube Data API uploader
│   └── approval.py      # Telegram approval flow
├── config/
│   └── config.yaml      # All configuration
├── assets/
│   ├── music/           # Background music (MP3/WAV)
│   └── fonts/           # Subtitle fonts (optional)
├── output/              # Generated files (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🎵 Adding Background Music

Drop royalty-free MP3/WAV files into `assets/music/`. The agent picks one randomly per video.

Free sources:
- [YouTube Audio Library](https://studio.youtube.com/channel/music)
- [Free Music Archive](https://freemusicarchive.org)
- [Pixabay Music](https://pixabay.com/music)

---

## 🔄 Approval Flow

Every day before generation starts, you get a Telegram message:

```
🎬 New Video Ready for Review

📌 Type: Bedtime Story
🌍 Language: English
📝 Title: The Little Star Who Couldn't Sleep

Script Preview (first 800 chars):
Once upon a time, high above the clouds...

[✅ Approve & Generate]  [❌ Reject & Regenerate]
[✏️ Edit Title]          [📋 View Full Script]
```

- **Approve** → video gets made and uploaded
- **Reject** → new script is generated and sent again
- **Edit Title** → you send a new title, then approve
- **No response in 1 hour** → skipped (retries next day)

---

## 🛣️ Roadmap

- [ ] Scary Tales channel support
- [ ] Custom intro/outro overlays
- [ ] Subtitle/captions generation
- [ ] Analytics tracking
- [ ] Multi-channel support
- [ ] Video upgrade to 15–30 min
