"""
Strategic Archives Agent - Main Orchestrator
Daily YouTube automation for @strategicarchivesHQ kids channel
"""

import os
import sys
import uuid
import shutil
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

# Setup logging
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


def cleanup_job_files(job_id: str, output_dir: str = "./output", keep_final: bool = False):
    """Remove intermediate files after job completes."""
    output = Path(output_dir)
    patterns = [f"{job_id}_scene_*.png", f"{job_id}_audio*.mp3"]
    if not keep_final:
        patterns.append(f"{job_id}_final.mp4")
        patterns.append(f"{job_id}_thumbnail.png")

    for pattern in patterns:
        for f in output.glob(pattern):
            f.unlink(missing_ok=True)
    logger.debug(f"Cleaned up intermediate files for job {job_id}")


def run_daily_job(
    config: dict,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    dry_run: bool = False
):
    """
    Main job: Generate → Approve → Create → Upload
    """
    from src.script_gen import ScriptGenerator
    from src.tts import TTSEngine
    from src.visuals import VisualsGenerator
    from src.video import VideoCompiler
    from src.upload import YouTubeUploader
    from src.approval import TelegramApproval

    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    assets_dir = os.getenv("ASSETS_DIR", "./assets")

    approval = TelegramApproval(config)
    approval.notify(f"🤖 *Strategic Archives Agent* starting job `{job_id}`...")

    logger.info(f"{'='*60}")
    logger.info(f"Job started: {job_id}")
    logger.info(f"{'='*60}")

    try:
        # ── STEP 1: Generate Script ─────────────────────────────────
        logger.info("[1/5] Generating script...")
        gen = ScriptGenerator(config)
        script_data = gen.generate(content_type=content_type, language=language)
        logger.info(f"Script generated: '{script_data['title']}' ({script_data.get('language', 'en')})")

        # ── STEP 2: Telegram Approval ───────────────────────────────
        logger.info("[2/5] Sending for approval...")
        result = approval.send_for_approval(script_data)

        if result["status"] == "rejected":
            logger.info("Script rejected. Regenerating...")
            script_data = gen.generate(content_type=content_type, language=language)
            result = approval.send_for_approval(script_data)

        if result["status"] in ("rejected", "timeout"):
            logger.warning(f"Job {job_id} aborted: {result['status']}")
            return

        script_data = result["script_data"]  # May have edits
        logger.info("Script approved ✓")

        if dry_run:
            logger.info("[DRY RUN] Stopping after script approval.")
            approval.notify(f"🧪 *Dry run complete* — script approved for `{script_data['title']}`")
            return

        # ── STEP 3: Generate TTS Audio ──────────────────────────────
        logger.info("[3/5] Generating voiceover...")
        tts = TTSEngine(config, output_dir=output_dir)
        audio_path = tts.synthesize(
            script=script_data["script"],
            job_id=job_id,
            language_code=script_data.get("language_code", "en")
        )
        duration = tts.get_duration_estimate(script_data["script"])
        logger.info(f"Audio ready: {duration:.0f}s estimated duration")

        # ── STEP 4: Generate Images ─────────────────────────────────
        logger.info("[4/5] Generating scene images...")
        vis = VisualsGenerator(config, output_dir=output_dir)
        scene_prompts = script_data.get("scene_prompts", [])

        # Ensure we have enough prompts
        if len(scene_prompts) < 10:
            logger.warning(f"Only {len(scene_prompts)} scene prompts, padding...")
            base = scene_prompts or ["colorful children's storybook scene"]
            while len(scene_prompts) < 15:
                scene_prompts.append(base[len(scene_prompts) % len(base)])

        image_paths = vis.generate_scene_images(scene_prompts, job_id)
        thumbnail_path = vis.generate_thumbnail(
            script_data.get("thumbnail_prompt", script_data["title"]),
            job_id
        )

        if not image_paths:
            raise RuntimeError("No images were generated — cannot compile video")

        # ── STEP 5: Compile Video ───────────────────────────────────
        logger.info("[5a/5] Compiling video...")
        compiler = VideoCompiler(config, output_dir=output_dir, assets_dir=assets_dir)
        video_path = compiler.compile(
            image_paths=image_paths,
            audio_path=audio_path,
            job_id=job_id,
            title=script_data["title"],
        )

        info = compiler.get_video_info(video_path)
        logger.info(f"Video compiled: {info['duration']:.0f}s, {info['size_mb']:.1f}MB")

        # ── STEP 6: Upload to YouTube ───────────────────────────────
        logger.info("[5b/5] Uploading to YouTube...")
        uploader = YouTubeUploader(config)
        result = uploader.upload(
            video_path=video_path,
            title=script_data["title"],
            description=script_data.get("description", ""),
            tags=script_data.get("tags", []),
            thumbnail_path=thumbnail_path,
            language=script_data.get("language_code", "en"),
            schedule=True
        )

        # ── DONE ────────────────────────────────────────────────────
        scheduled_for = result.get("scheduled_for", "N/A")
        approval.notify(
            f"✅ *Video uploaded successfully!*\n\n"
            f"🎬 *Title:* {result['title']}\n"
            f"🔗 *URL:* {result['video_url']}\n"
            f"📅 *Goes live:* {scheduled_for}\n\n"
            f"Job: `{job_id}`"
        )
        logger.info(f"Job {job_id} completed: {result['video_url']}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        approval.notify(
            f"❌ *Job failed*\n\n"
            f"Job: `{job_id}`\n"
            f"Error: `{str(e)[:200]}`"
        )
    finally:
        cleanup_job_files(job_id, output_dir=output_dir)


def start_scheduler(config: dict):
    """Start the daily scheduler at 6pm WAT."""
    app_config = config.get("app", {})
    upload_time = app_config.get("upload_time", "18:00")
    tz = app_config.get("timezone", "Africa/Lagos")

    logger.info(f"Scheduler started — daily job at {upload_time} {tz}")
    schedule.every().day.at(upload_time).do(run_daily_job, config=config)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategic Archives YouTube Agent")
    parser.add_argument("--run-now", action="store_true", help="Run job immediately")
    parser.add_argument("--dry-run", action="store_true", help="Stop after script approval (no video/upload)")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard at http://localhost:8000")
    parser.add_argument("--type", choices=["bedtime_story", "fun_facts"], help="Content type override")
    parser.add_argument("--lang", help="Language code override (en, fr, yo, ha, ig, pt, es)")
    parser.add_argument("--schedule", action="store_true", default=True, help="Start daily scheduler")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port (default: $PORT env or 8000)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.dashboard:
        from src.startup import bootstrap
        bootstrap()  # Decode Railway secrets → files before anything starts
        from src.dashboard import start_dashboard
        # Railway injects $PORT — respect it, fall back to --port arg or 8000
        port = args.port or int(os.getenv("PORT", 8000))
        start_dashboard(port=port)
    elif args.run_now or args.dry_run:
        run_daily_job(
            config=config,
            content_type=args.type,
            language=args.lang,
            dry_run=args.dry_run
        )
    else:
        start_scheduler(config)
