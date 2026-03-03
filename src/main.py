"""
Strategic Archives Agent - Multi-Channel Orchestrator
Channels driven by ACTIVE_CHANNELS env var.
Flow: Pick Channel → Pick Type → Generate Script → Approve → Voice → Visuals → Compile → Upload
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
    dry_run: bool = False,
    slot_time: Optional[str] = None,   # e.g. "08:00" — overrides channel upload_time for this run
):
    """
    Full pipeline:
    Pick Channel → Pick Type → Generate → Approve → TTS → Visuals → Compile → Upload
    """
    from src.script_gen import ScriptGenerator
    from src.tts import TTSEngine
    from src.visuals import VisualsGenerator
    from src.video import VideoCompiler
    from src.upload import YouTubeUploader
    from src.approval import TelegramApproval
    from src.channel_loader import load_active_channels

    approval = TelegramApproval(config)
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    assets_dir = os.getenv("ASSETS_DIR", "./assets")

    # ── 0a. Load channels from ACTIVE_CHANNELS env ────────────────
    try:
        active_channels = load_active_channels(config)
        logger.info(f"Active channels: {list(active_channels.keys())}")
    except ValueError as e:
        logger.error(str(e))
        approval.notify(f"❌ Channel config error:\n{e}")
        return

    # ── 0b. Channel selection ─────────────────────────────────────
    if channel_key and channel_key in active_channels:
        # Explicit override (e.g. from scheduler)
        channel = active_channels[channel_key]
        logger.info(f"Channel override → {channel_key}")
    elif len(active_channels) == 1:
        # Only one active — skip the picker
        channel_key = list(active_channels.keys())[0]
        channel = active_channels[channel_key]
    else:
        # Let user pick via Telegram (or CLI fallback)
        channel_key = approval.pick_channel(active_channels)
        if not channel_key:
            logger.warning("No channel selected — aborting")
            return
        channel = active_channels[channel_key]

    # ── 0c. Content type selection ────────────────────────────────
    if not content_type:
        content_type = approval.pick_content_type(channel)
        # None means random — ScriptGenerator will pick

    # ── 0d-pre. Slot time override ────────────────────────────────
    # When scheduler fires for a specific time slot (e.g. "08:00"), override
    # channel upload_time so the video is scheduled to go public at that slot.
    if slot_time:
        channel = dict(channel)          # shallow copy — don't mutate the original
        channel["upload_time"] = slot_time
        logger.info(f"Slot time override → video will publish at {slot_time} WAT")

    # ── 0d. Pre-flight YouTube auth check ────────────────────────
    # Validate BEFORE generating anything — no wasted TTS/video work on bad tokens
    logger.info(f"[0/5] Validating YouTube auth for '{channel.get('name')}'...")
    uploader_check = YouTubeUploader(config, channel)
    auth_ok, auth_err = uploader_check.validate_auth()
    if not auth_ok:
        msg = (
            f"❌ <b>Auth failed for {channel.get('name','?')}</b> — job aborted.\n\n"
            f"<code>{auth_err[:400]}</code>\n\n"
            f"Fix: re-run <code>python scripts/auth_channel.py --channel {channel_key}</code> "
            f"locally, then update the Railway env var."
        )
        logger.error(f"Auth check failed for '{channel_key}': {auth_err}")
        approval.notify(msg)
        return
    logger.info(f"Auth check passed ✓ for '{channel.get('name')}'")

    job_id = f"{channel_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    approval.notify(
        f"🤖 Starting job for <b>{channel['name']}</b>\n"
        f"Niche: <i>{channel.get('niche','').replace('_',' ')}</i>\n"
        f"Type: <i>{content_type or 'random'}</i>\n"
        f"Job: <code>{job_id}</code>"
    )

    logger.info("=" * 60)
    logger.info(f"Job: {job_id} | {channel['name']} | {channel.get('niche')}")
    logger.info("=" * 60)

    try:
        # ── 1. Generate Script ────────────────────────────────────
        logger.info("[1/5] Generating script...")
        gen = ScriptGenerator(channel)
        script_data = gen.generate(content_type=content_type, language=language)
        logger.info(f"Script: '{script_data['title']}' [{script_data.get('language_code','en')}]")

        # ── 2. Telegram Script Approval ───────────────────────────
        logger.info("[2/5] Awaiting script approval...")
        result = approval.send_for_approval(script_data)

        if result["status"] == "rejected":
            logger.info("Rejected — regenerating...")
            script_data = gen.generate(content_type=content_type, language=language)
            result = approval.send_for_approval(script_data)

        if result["status"] in ("rejected", "timeout"):
            logger.warning(f"Job aborted: {result['status']}")
            return

        script_data = result["script_data"]
        logger.info("Script approved ✓")

        if dry_run:
            approval.notify(f"🧪 <b>Dry run done</b> — {script_data['title']}")
            return

        scenes = script_data.get("scenes", [])

        # ── 3. TTS Voiceover ──────────────────────────────────────
        logger.info("[3/6] Generating voiceover...")
        tts = TTSEngine(channel, output_dir=output_dir)
        # Concatenate all scene narrations for a single TTS pass
        full_script = script_data.get("script") or " ".join(
            s.get("narration", "") for s in scenes if s.get("narration")
        )
        audio_path = tts.synthesize(
            script=full_script,
            job_id=job_id,
            language_code=script_data.get("language_code", "en")
        )

        # ── 4. Scene Images ───────────────────────────────────────
        logger.info("[4/6] Generating scene images...")
        from src.character_gen import CharacterGenerator
        char_gen = CharacterGenerator(channel_key=channel_key, output_dir=output_dir)
        image_paths = char_gen.generate_all_scenes(scenes, job_id)

        # Generate thumbnail separately
        from src.visuals import VisualsGenerator
        vis = VisualsGenerator(channel, output_dir=output_dir)
        thumbnail_path = vis.generate_thumbnail(
            script_data.get("thumbnail_prompt", script_data["title"]), job_id
        )

        if not image_paths:
            raise RuntimeError("No images generated")

        # ── 5. Animate Scenes (Higgsfield) ────────────────────────
        logger.info("[5/6] Animating scenes with Higgsfield...")
        try:
            from src.scene_animator import SceneAnimator
            animator = SceneAnimator(output_dir=output_dir)
            clip_paths = animator.animate_scenes(scenes, image_paths, job_id)
            logger.info(f"Animated {len(clip_paths)} clips ✓")
        except Exception as e:
            logger.warning(f"Higgsfield animation failed: {e} — falling back to Ken Burns")
            clip_paths = None

        # ── 6a. Compile Video ─────────────────────────────────────
        logger.info("[6a/6] Compiling video...")
        compiler = VideoCompiler(channel, output_dir=output_dir, assets_dir=assets_dir)
        if clip_paths:
            video_path = compiler.compile_from_clips(
                clip_paths=clip_paths, audio_path=audio_path,
                job_id=job_id, title=script_data["title"]
            )
        else:
            video_path = compiler.compile(
                image_paths=image_paths, audio_path=audio_path,
                job_id=job_id, title=script_data["title"]
            )

        # ── 6b. Mix Background Music ──────────────────────────────
        logger.info("[6b/6] Mixing background music...")
        from src.music_mixer import MusicMixer
        mixer = MusicMixer(channel_key=channel_key, assets_dir=assets_dir)
        final_video = mixer.mix(
            video_path=video_path,
            output_path=video_path.replace("_graded.mp4", "_final.mp4").replace("_final.mp4", "_final.mp4"),
        )
        video_path = final_video

        info = compiler.get_video_info(video_path)
        logger.info(f"Video: {info['duration']:.0f}s, {info['size_mb']:.1f}MB")

        # ── 5b. Upload ────────────────────────────────────────────
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
            f"✅ <b>Uploaded!</b>\n\n"
            f"📺 <b>Channel:</b> {channel['name']}\n"
            f"🎬 <b>Title:</b> {upload_result['title']}\n"
            f"🔗 {upload_result['video_url']}\n"
            f"📅 <b>Goes live:</b> {upload_result.get('scheduled_for', 'N/A')}"
        )
        logger.info(f"Done → {upload_result['video_url']}")

    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        approval.notify(f"❌ <b>Job failed</b> on {channel.get('name','?')}\n<code>{str(e)[:300]}</code>")
    finally:
        cleanup_job_files(job_id, output_dir=output_dir)


def start_scheduler(config: dict):
    """Schedule each active channel at its own upload_time."""
    from src.channel_loader import load_active_channels
    from src.upload import validate_all_channel_auth
    from src.approval import TelegramApproval

    try:
        active = load_active_channels(config)
    except ValueError as e:
        logger.error(f"Scheduler aborted: {e}")
        return

    # ── Startup auth validation ──────────────────────────────────
    # Check every channel token NOW so Railway logs show auth status at deploy time.
    # Sends a Telegram summary with any broken channels.
    logger.info("=" * 60)
    logger.info("Pre-flight YouTube auth check for all channels...")
    logger.info("=" * 60)
    auth_results = validate_all_channel_auth(config, active)
    broken = [(k, r["error"]) for k, r in auth_results.items() if not r["ok"]]
    healthy = [k for k, r in auth_results.items() if r["ok"]]

    notifier = TelegramApproval(config)
    if broken:
        lines = "\n".join(
            f"❌ <b>{active[k].get('name', k)}</b>: <code>{(err or '')[:200]}</code>"
            for k, err in broken
        )
        ok_line = f"✅ Healthy: {', '.join(active[k].get('name', k) for k in healthy)}" if healthy else ""
        notifier.notify(
            f"⚠️ <b>Startup Auth Warning</b>\n\n"
            f"{lines}\n\n"
            f"{ok_line}\n\n"
            f"Fix: run <code>python scripts/auth_channel.py --channel &lt;key&gt;</code> "
            f"locally and update Railway env vars."
        )
    else:
        notifier.notify(
            f"✅ <b>All channels authenticated</b>\n"
            + "\n".join(f"  • {active[k].get('name', k)}" for k in healthy)
        )

    # ── Schedule jobs ────────────────────────────────────────────
    total_jobs = 0
    for key, ch in active.items():
        # Support both upload_times (list) and legacy upload_time (string)
        times = ch.get("upload_times") or [ch.get("upload_time", "18:00")]
        if isinstance(times, str):
            times = [t.strip() for t in times.split(",")]

        for t in times:
            logger.info(f"Scheduling '{ch['name']}' [{key}] at {t} WAT")
            schedule.every().day.at(t).do(
                run_daily_job, config=config, channel_key=key, slot_time=t
            )
            total_jobs += 1

    logger.info(f"{len(active)} channel(s), {total_jobs} daily job(s) scheduled. Running...")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strategic Archives Multi-Channel Agent")
    parser.add_argument("--run-now",  action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--channel",  default=None,
                        help="Channel key (e.g. kids_universe, ai_tools). "
                             "If omitted and >1 active channel, Telegram picker is shown.")
    parser.add_argument("--type",     default=None)
    parser.add_argument("--lang",     default=None)
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--port",     type=int, default=None)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.dashboard:
        from src.startup import bootstrap
        bootstrap()

        # Start the daily scheduler in a background daemon thread
        # so it keeps firing even while the dashboard (uvicorn) is running.
        import threading
        scheduler_thread = threading.Thread(
            target=start_scheduler,
            args=(config,),
            daemon=True,
            name="scheduler"
        )
        scheduler_thread.start()
        logger.info("Scheduler thread started alongside dashboard")

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
