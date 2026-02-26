"""
Dashboard - FastAPI web server for controlling the video pipeline
Features: Job triggering, live log streaming (SSE), job history, YouTube analytics,
          live image preview, job cancellation
"""

import os
import json
import glob
import logging
import threading
import asyncio
from datetime import datetime
from typing import Optional
from queue import Empty

import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

from src.job_store import get_store, JobLogHandler
from src.analytics import YouTubeAnalytics

logger = logging.getLogger(__name__)

app = FastAPI(title="Strategic Archives Dashboard", version="1.0.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global state
_current_job_thread: Optional[threading.Thread] = None
_current_job_id: Optional[str] = None
_cancel_event = threading.Event()  # Signal to cancel the running job


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_job_in_background(job_id: str, config: dict, cancel_event: threading.Event, content_type: str = None, language: str = None, video_format: str = "long", channel_key: str = None):
    """Run a video generation job in a background thread with log capture."""
    global _current_job_id

    import uuid
    from src.script_gen import ScriptGenerator
    from src.tts import TTSEngine
    from src.visuals import VisualsGenerator
    from src.video import VideoCompiler
    from src.upload import YouTubeUploader
    from src.approval import TelegramApproval
    from src.channel_loader import load_channels

    store = get_store()
    store.create_job(job_id, content_type or "", language or "", video_format=video_format)

    # Resolve channel from CHANNELS env var
    try:
        channels = load_channels(config)
    except ValueError as e:
        logger.error(f"Channel load failed: {e}")
        store.update_job(job_id, status="failed", error=str(e), completed_at=datetime.now().isoformat())
        return

    if channel_key and channel_key in channels:
        channel = channels[channel_key]
    else:
        channel_key, channel = next(iter(channels.items()))

    logger.info(f"Running job for channel: {channel.get('name')} [{channel_key}]")
    store.update_job(job_id, title=f"[{channel.get('name', channel_key)}] generating...")

    # Attach log handler to capture all logs for this job
    root_logger = logging.getLogger()
    handler = JobLogHandler(store, job_id)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    output_dir = os.getenv("OUTPUT_DIR", "./output")
    assets_dir = os.getenv("ASSETS_DIR", "./assets")

    def check_cancelled():
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled by user")

    try:
        approval = TelegramApproval(config)

        # Step 1: Script
        check_cancelled()
        store.update_job(job_id, step="Generating script...", step_number=1)
        gen = ScriptGenerator(channel)
        script_data = gen.generate(content_type=content_type, language=language)
        store.update_job(job_id, title=script_data["title"])
        logger.info(f"Script generated: '{script_data['title']}'")

        # Step 2: Approval
        check_cancelled()
        store.update_job(job_id, step="Waiting for Telegram approval...", step_number=2)
        result = approval.send_for_approval(script_data)

        if result["status"] == "rejected":
            logger.info("Script rejected. Regenerating...")
            script_data = gen.generate(content_type=content_type, language=language)
            result = approval.send_for_approval(script_data)

        if result["status"] in ("rejected", "timeout"):
            store.update_job(job_id, status="failed", error=f"Script {result['status']}", completed_at=datetime.now().isoformat())
            return

        script_data = result["script_data"]
        store.update_job(job_id, title=script_data["title"])
        logger.info("Script approved ✓")

        # Step 3: TTS
        check_cancelled()
        store.update_job(job_id, step="Generating voiceover...", step_number=3)
        tts = TTSEngine(channel, output_dir=output_dir)
        audio_path = tts.synthesize(
            script=script_data["script"],
            job_id=job_id,
            language_code=script_data.get("language_code", "en")
        )

        # Step 4: Images
        check_cancelled()
        store.update_job(job_id, step="Generating scene images...", step_number=4)
        is_short = video_format == "short"
        aspect_ratio = "9:16" if is_short else "16:9"
        vis = VisualsGenerator(channel, output_dir=output_dir, aspect_ratio=aspect_ratio)
        scene_prompts = script_data.get("scene_prompts", [])

        # Shorts need 3-5 images, long-form needs 12-15
        if is_short:
            if len(scene_prompts) < 3:
                base = scene_prompts or ["colorful children's vertical scene"]
                while len(scene_prompts) < 3:
                    scene_prompts.append(base[len(scene_prompts) % len(base)])
            scene_prompts = scene_prompts[:5]  # Cap at 5 for shorts
        else:
            if len(scene_prompts) < 10:
                base = scene_prompts or ["colorful children's storybook scene"]
                while len(scene_prompts) < 15:
                    scene_prompts.append(base[len(scene_prompts) % len(base)])

        image_paths = vis.generate_scene_images(scene_prompts, job_id)
        check_cancelled()
        thumbnail_path = vis.generate_thumbnail(
            script_data.get("thumbnail_prompt", script_data["title"]), job_id
        )

        if not image_paths:
            raise RuntimeError("No images generated")

        # Step 5: Compile + Upload
        check_cancelled()
        store.update_job(job_id, step="Compiling video...", step_number=5)
        compiler = VideoCompiler(channel, output_dir=output_dir, assets_dir=assets_dir, video_format=video_format)
        video_path = compiler.compile(
            image_paths=image_paths,
            audio_path=audio_path,
            job_id=job_id,
            title=script_data["title"],
        )

        check_cancelled()
        store.update_job(job_id, step="Uploading to YouTube...")
        uploader = YouTubeUploader(config, channel)
        result = uploader.upload(
            video_path=video_path,
            title=script_data["title"],
            description=script_data.get("description", ""),
            tags=script_data.get("tags", []),
            thumbnail_path=thumbnail_path,
            language=script_data.get("language_code", "en"),
            schedule=(video_format != "short"),
            video_format=video_format,
        )

        store.update_job(
            job_id,
            status="completed",
            step="Done ✓",
            youtube_url=result["video_url"],
            youtube_id=result["video_id"],
            completed_at=datetime.now().isoformat()
        )
        logger.info(f"Job completed: {result['video_url']}")

        # Notify on Telegram
        approval.notify(
            f"✅ *Video uploaded!*\n🎬 {result['title']}\n🔗 {result['video_url']}"
        )

    except InterruptedError:
        logger.warning(f"Job {job_id} cancelled by user")
        store.update_job(
            job_id,
            status="cancelled",
            step="Cancelled",
            error="Cancelled by user",
            completed_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        store.update_job(
            job_id,
            status="failed",
            step="Failed ✗",
            error=str(e)[:500],
            completed_at=datetime.now().isoformat()
        )
    finally:
        root_logger.removeHandler(handler)
        _current_job_id = None
        # Cleanup intermediate files
        from src.main import cleanup_job_files
        cleanup_job_files(job_id, output_dir=output_dir)


# ─── API Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard UI."""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard not found. Place index.html in /static/</h1>")


@app.post("/api/jobs/trigger")
async def trigger_job(request: Request):
    """Trigger a new video generation job."""
    global _current_job_thread, _current_job_id, _cancel_event

    store = get_store()
    running = store.get_running_job()
    if running:
        return JSONResponse(
            {"error": "A job is already running", "job_id": running["job_id"]},
            status_code=409
        )

    body = await request.json() if await request.body() else {}
    content_type = body.get("content_type")
    language = body.get("language")
    video_format = body.get("format", "long")
    channel_key = body.get("channel_key")

    import uuid
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    _current_job_id = job_id
    _cancel_event = threading.Event()

    config = load_config()
    _current_job_thread = threading.Thread(
        target=run_job_in_background,
        args=(job_id, config, _cancel_event, content_type, language, video_format, channel_key),
        daemon=True
    )
    _current_job_thread.start()

    return {"job_id": job_id, "status": "started", "format": video_format, "channel_key": channel_key}


@app.get("/api/channels")
async def get_channels():
    """Return active channels loaded from CHANNELS env var."""
    from src.channel_loader import load_channels
    try:
        config = load_config()
        channels = load_channels(config)
        return {
            "channels": [
                {
                    "key":         key,
                    "name":        ch["name"],
                    "niche":       ch.get("niche", ""),
                    "emoji":       ch.get("emoji", "📺"),
                    "upload_time": ch.get("upload_time", "18:00"),
                    "youtube_id":  ch.get("_youtube_channel_id", ""),
                }
                for key, ch in channels.items()
            ]
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/jobs/cancel")
async def cancel_job():
    """Cancel the currently running job."""
    global _cancel_event, _current_job_id

    store = get_store()
    running = store.get_running_job()
    if not running:
        return JSONResponse({"error": "No job is running"}, status_code=404)

    _cancel_event.set()
    logger.info(f"Cancel signal sent for job {running['job_id']}")
    return {"status": "cancelling", "job_id": running["job_id"]}


@app.get("/api/jobs")
async def list_jobs():
    """List job history."""
    store = get_store()
    return store.get_jobs(limit=50)


@app.get("/api/jobs/current")
async def current_job():
    """Get the currently running job."""
    store = get_store()
    running = store.get_running_job()
    return running or {"status": "idle"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get details for a specific job."""
    store = get_store()
    job = store.get_job(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.get("/api/jobs/{job_id}/images")
async def get_job_images(job_id: str):
    """Get list of generated images for a job (for live preview)."""
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    pattern = os.path.join(output_dir, f"{job_id}_scene_*.png")
    files = sorted(glob.glob(pattern))
    images = []
    for f in files:
        basename = os.path.basename(f)
        # Extract scene number from filename like job_xxx_scene_001.png
        try:
            scene_num = int(basename.split("_scene_")[1].split(".")[0])
        except (IndexError, ValueError):
            scene_num = len(images)
        images.append({
            "url": f"/api/images/{basename}",
            "scene": scene_num,
            "filename": basename
        })
    # Also check for thumbnail
    thumb_path = os.path.join(output_dir, f"{job_id}_thumbnail.png")
    thumbnail = None
    if os.path.exists(thumb_path):
        thumbnail = {"url": f"/api/images/{job_id}_thumbnail.png"}
    return {"images": images, "thumbnail": thumbnail, "total": len(images)}


@app.get("/api/images/{filename}")
async def serve_image(filename: str):
    """Serve a generated image file."""
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "Image not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png")


@app.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """SSE endpoint for live log streaming."""
    store = get_store()

    async def event_generator():
        # First, send existing logs
        existing = store.get_logs(job_id)
        for log in existing:
            yield {"event": "log", "data": json.dumps(log)}

        # Then stream new logs
        q = store.subscribe(job_id)
        try:
            while True:
                try:
                    entry = q.get(timeout=0.5)
                    yield {"event": "log", "data": json.dumps(entry)}
                except Empty:
                    # Check if job is still running
                    job = store.get_job(job_id)
                    if job and job["status"] != "running":
                        yield {"event": "done", "data": json.dumps({"status": job["status"]})}
                        break
                    yield {"event": "ping", "data": ""}
                await asyncio.sleep(0.1)
        finally:
            store.unsubscribe(job_id, q)

    return EventSourceResponse(event_generator())


@app.get("/api/analytics")
async def get_analytics():
    """Get YouTube video analytics."""
    store = get_store()
    jobs = store.get_jobs(limit=50)

    # Collect video IDs from completed jobs
    video_ids = [j["youtube_id"] for j in jobs if j.get("youtube_id")]

    if not video_ids:
        return {"videos": [], "channel": None}

    analytics = YouTubeAnalytics()
    videos = analytics.get_video_stats(video_ids)
    channel = analytics.get_channel_stats()

    return {"videos": videos, "channel": channel}


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/health")
async def health_railway():
    """Railway healthcheck endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def start_dashboard(host: str = "0.0.0.0", port: int = None):
    """Start the dashboard server. Reads $PORT from Railway env if not provided."""
    port = port or int(os.getenv("PORT", 8000))
    logger.info(f"Starting dashboard at http://0.0.0.0:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
