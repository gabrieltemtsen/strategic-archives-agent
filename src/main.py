"""
Strategic Archives Agent - Multi-Channel Orchestrator
Supports: kids, horror, african folklore, motivational channels
"""

import os
import uuid
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log"),
    ]
)
logger = logging.getLogger("main")


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_channel_config(config: dict, channel_key: str) -> dict:
    """Get a specific channel's config. Falls back to default_channel."""
    channels = config.get("channels", {})
    if channel_key not in channels:
        default = config.get("app", {}).get("default_channel", "kids_universe")
        logger.warning(f"Channel '{channel_key}' not found — using '{default}'")
        channel_key = default
    ch = channels[channel_key].copy()
    ch["_key"] = channel_key
    ch["_timezone"] = config.get("app", {}).get("timezone", "Africa/Lagos")
    # Resolve YouTube channel ID from env var
    env_key = f"YOUTUBE_CHANNEL_ID_{channel_key.upper()}"
    ch["_youtube_channel_id"] = os.getenv(env_key, ch.get("youtube_channel_id", ""))
    return ch


def get_active_channels(config: dict) -> list:
    """Return list of (key, channel_config) for all active channels."""
    return [
        (key, get_channel_config(config, key))
        for key, ch in config.get("channels", {}).items()
        if ch.get("active", False)
    ]


def cleanup_job_files(job_id: str, output_dir: str = "./output"):
    output = Path(output_dir)
    for pattern in [f"{job_id}_scene_*.png", f"{job_id}_audio*.mp3",
                    f"{job_id}_final.mp4", f"{job_id}_thumbnail.png"]:
        for f in output.glob(pattern):
            f.unlink(missing_ok=True)


def run_daily_job(
    config: dict,
    channel_key: str = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    dry_run: bool = False
):
    """Main pipeline: Generate → Approve → Voice → Visuals → Compile → Upload"""
    from src.script_gen import ScriptGenerator
    from src.tts import TTSEngine
    from src.visuals import VisualsGenerator
    from src.video import VideoCompiler
    from src.upload import YouTubeUploader
    from src.approval import TelegramApproval

    # Resolve channel
    channel_key = channel_key or config.get("app", {}).get("default_channel", "kids_universe")
    channel = get_channel_config(config, channel_key)

    job_id = f"{channel_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    assets_dir = os.getenv("ASSETS_DIR", "./assets")

    approval = TelegramApproval(config)
    approval.notify(
        f"🤖 *{channel['name']}* — starting job\n"
        f"Channel: `{channel_key}` | Niche: `{channel.get('niche', 'N/A')}`\n"
        f"Job: `{job_id}`"
    )

    logger.info("=" * 60)
    logger.info(f"Job: {job_id} | Channel: {channel['name']} | Niche: {channel.get('niche')}")
    logger.info("=" * 60)

    try:
        # ── 1. Generate Script ────────────────────────────────────────
        logger.info("[1/5] Generating script...")
        gen = ScriptGenerator(channel)
        script_data = gen.generate(content_type=content_type, language=language)
        logger.info(f"Script: '{script_data['title']}' [{script_data.get('language_code', 'en')}]")

        # ── 2. Telegram Approval ──────────────────────────────────────
        logger.info("[2/5] Awaiting approval...")
        result = approval.send_for_approval(script_data)

        if result["status"] == "rejected":
            logger.info("Rejected — regenerating...")
            script_data = gen.generate(content_type=content_type, language=language)
            result = approval.send_for_approval(script_data)

        if result["status"] in ("rejected", "timeout"):
            logger.warning(f"Job aborted: {result['status']}")
            return

        script_data = result["script_data"]
        logger.info("Approved ✓")

        if dry_run:
            approval.notify(f"🧪 *Dry run complete* — `{script_data['title']}`")
            return

        # ── 3. TTS Audio ──────────────────────────────────────────────
        logger.info("[3/5] Generating voiceover...")
        tts = TTSEngine(channel, output_dir=output_dir)
        audio_path = tts.synthesize(
            script=script_data["script"],
            job_id=job_id,
            language_code=script_data.get("language_code", "en")
        )

        # ── 4. Scene Images ───────────────────────────────────────────
        logger.info("[4/5] Generating visuals...")
        vis = VisualsGenerator(channel, output_dir=output_dir)
        scene_prompts = script_data.get("scene_prompts", [])
        if len(scene_prompts) < 10:
            base = scene_prompts or [channel.get("visuals", {}).get("style", "cinematic scene")]
            while len(scene_prompts) < 14:
                scene_prompts.append(base[len(scene_prompts) % len(base)])

        image_paths = vis.generate_scene_images(scene_prompts, job_id)
        thumbnail_path = vis.generate_thumbnail(
            script_data.get("thumbnail_prompt", script_data["title"]), job_id
        )
        if not image_paths:
            raise RuntimeError("No images generated — cannot compile")

        # ── 5a. Compile Video ─────────────────────────────────────────
        logger.info("[5a/5] Compiling video...")
        compiler = VideoCompiler(channel, output_dir=output_dir, assets_dir=assets_dir)
        video_path = compiler.compile(
            image_paths=image_paths, audio_path=audio_path,
            job_id=job_id, title=script_data["title"]
        )
        info = compiler.get_video_info(video_path)
        logger.info(f"Video: {info['duration']:.0f}s, {info['size_mb']:.1f}MB")

        # ── 5b. Upload to YouTube ─────────────────────────────────────
        logger.info("[5b/5] Uploading to YouTube...")
        uploader = YouTubeUploader(config, channel)
        upload_result = uploader.upload(
            video_path=video_path,
            title=script_data["title"],
            description=script_data.get("description", ""),
            tags=script_data.get("tags", []),
            thumbnail_path=thumbnail_path,
            language=script_data.get("language_code", "en"),
            schedule=True
        )

        approval.notify(
            f"✅ *Uploaded!*\n\n"
            f"📺 *Channel:* {channel['name']}\n"
            f"🎬 *Title:* {upload_result['title']}\n"
            f"🔗 {upload_result['video_url']}\n"
            f"📅 *Goes live:* {upload_result.get('scheduled_for', 'N/A')}"
        )
        logger.info(f"Done: {upload_result['video_url']}")

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        approval.notify(f"❌ *Job failed* on `{channel['name']}`\n`{str(e)[:200]}`")
    finally:
        cleanup_job_files(job_id, output_dir=output_dir)


def start_scheduler(config: dict):
    """Schedule all active channels at their configured upload times."""
    active = get_active_channels(config)
    if not active:
        logger.warning("No active channels found in config. Set active: true to enable.")
        return

    for key, channel in active:
        upload_time = channel.get("upload_time", "18:00")
        logger.info(f"Scheduling '{channel['name']}' at {upload_time} WAT")
        schedule.every().day.at(upload_time).do(
            run_daily_job, config=config, channel_key=key
        )

    logger.info(f"{len(active)} channel(s) scheduled. Running...")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategic Archives Multi-Channel Agent")
    parser.add_argument("--run-now", action="store_true", help="Run a job immediately")
    parser.add_argument("--dry-run", action="store_true", help="Stop after approval (no upload)")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    parser.add_argument("--channel", default=None,
                        help="Channel key to run (kids_universe, scary_tales, african_folklore, mind_fuel)")
    parser.add_argument("--type", default=None, help="Content type override")
    parser.add_argument("--lang", default=None, help="Language code override")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.dashboard:
        from src.startup import bootstrap
        bootstrap()
        from src.dashboard import start_dashboard
        port = args.port or int(os.getenv("PORT", 8000))
        start_dashboard(port=port)
    elif args.run_now or args.dry_run:
        run_daily_job(
            config=config,
            channel_key=args.channel,
            content_type=args.type,
            language=args.lang,
            dry_run=args.dry_run
        )
    else:
        start_scheduler(config)
