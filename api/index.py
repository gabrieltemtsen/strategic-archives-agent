"""
Vercel Serverless Dashboard & Dispatcher (0-dollar hosting)
==========================================================
Runs the lightweight FastAPI dashboard + API on Vercel Serverless Functions.
Heavy FFmpeg video rendering & YouTube uploads are dispatched to GitHub Actions
(free tier: 2,000 mins/month private, unlimited public) so there is zero 24/7
Railway/container cost.

Required Vercel Env Vars:
  CHANNELS            - JSON array of channels
  GITHUB_TOKEN        - GitHub PAT with `actions:write` / `repo` scope
  GITHUB_REPO         - e.g. "gabrieltemtsen/strategic-archives-agent"
Optional:
  GITHUB_WORKFLOW_ID  - defaults to "generate-video.yml"
  GITHUB_REF          - branch name, defaults to "main"
  CRON_SECRET         - Vercel Cron bearer secret
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import requests
from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from src.channel_loader import load_channels

logger = logging.getLogger("vercel_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

app = FastAPI(title="Strategic Archives Dashboard (Vercel)", version="2.0.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN env var is not set on Vercel")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_repo() -> str:
    repo = os.getenv("GITHUB_REPO", "").strip()
    if not repo:
        raise RuntimeError("GITHUB_REPO env var is not set on Vercel (e.g. user/repo)")
    return repo


def _dispatch_github_workflow(
    channel_key: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    video_format: str = "long",
    auto_approve: bool = True,
) -> dict:
    repo = _gh_repo()
    workflow = os.getenv("GITHUB_WORKFLOW_ID", "generate-video.yml")
    ref = os.getenv("GITHUB_REF", "main")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"

    inputs = {
        "channel_key": channel_key or "",
        "content_type": content_type or "",
        "language": language or "",
        "video_format": video_format or "long",
        "auto_approve": "true" if auto_approve else "false",
    }
    resp = requests.post(
        url,
        headers=_gh_headers(),
        json={"ref": ref, "inputs": inputs},
        timeout=15,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"GitHub dispatch failed ({resp.status_code}): {resp.text[:300]}")
    return {"dispatched": True, "workflow": workflow, "ref": ref, "inputs": inputs}


def _fetch_github_runs(limit: int = 20) -> list:
    repo = os.getenv("GITHUB_REPO", "").strip()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not repo or not token:
        return []
    workflow = os.getenv("GITHUB_WORKFLOW_ID", "generate-video.yml")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?per_page={limit}"
    resp = requests.get(url, headers=_gh_headers(), timeout=15)
    if not resp.ok:
        logger.warning(f"Failed to fetch GitHub runs: {resp.status_code}")
        return []

    runs = resp.json().get("workflow_runs", [])
    mapped = []
    for r in runs:
        gh_status = r.get("status", "")
        conclusion = r.get("conclusion")
        if gh_status in ("queued", "in_progress", "waiting"):
            status = "running"
        elif conclusion == "success":
            status = "completed"
        elif conclusion == "cancelled":
            status = "cancelled"
        else:
            status = "failed"

        mapped.append({
            "job_id": f"gh_{r['id']}",
            "status": status,
            "step": f"GitHub Actions: {gh_status}" + (f" ({conclusion})" if conclusion else ""),
            "step_number": 5 if status == "completed" else (2 if status == "running" else 0),
            "total_steps": 5,
            "title": r.get("display_title") or f"Workflow Run #{r.get('run_number')}",
            "content_type": "",
            "language": "",
            "video_format": "long",
            "youtube_url": "",
            "html_url": r.get("html_url", ""),
            "created_at": r.get("created_at", ""),
            "completed_at": r.get("updated_at", "") if status != "running" else "",
        })
    return mapped


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/api/channels")
async def get_channels():
    try:
        config = _load_config()
        channels = load_channels(config)
        return {
            "channels": [
                {
                    "key": key,
                    "name": ch["name"],
                    "niche": ch.get("niche", ""),
                    "emoji": ch.get("emoji", "📺"),
                    "upload_time": ch.get("upload_time", "18:00"),
                    "youtube_id": ch.get("_youtube_channel_id", ""),
                }
                for key, ch in channels.items()
            ]
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/jobs/trigger")
async def trigger_job(request: Request):
    body = await request.json() if await request.body() else {}
    try:
        res = _dispatch_github_workflow(
            channel_key=body.get("channel_key"),
            content_type=body.get("content_type"),
            language=body.get("language"),
            video_format=body.get("format", "long"),
            auto_approve=body.get("auto_approve", True),
        )
        return {
            "job_id": f"gh_pending_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "status": "started",
            "runner": "github_actions",
            **res,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/cron/trigger")
async def cron_trigger(
    channel: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    """Vercel Cron entrypoint — dispatches GitHub Actions worker at $0 idle cost."""
    cron_secret = os.getenv("CRON_SECRET", "").strip()
    if cron_secret and authorization != f"Bearer {cron_secret}":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        config = _load_config()
        channels = load_channels(config)
        target_keys = [channel] if channel and channel in channels else list(channels.keys())
        dispatched = []
        for key in target_keys:
            res = _dispatch_github_workflow(channel_key=key, auto_approve=True)
            dispatched.append({"channel": key, **res})
        return {"status": "ok", "dispatched": dispatched}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/jobs/cancel")
async def cancel_job():
    runs = _fetch_github_runs(limit=5)
    running = next((r for r in runs if r["status"] == "running"), None)
    if not running:
        return JSONResponse({"error": "No running GitHub Actions job found"}, status_code=404)
    run_id = running["job_id"].replace("gh_", "")
    repo = _gh_repo()
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/cancel"
    resp = requests.post(url, headers=_gh_headers(), timeout=15)
    if not resp.ok:
        return JSONResponse({"error": resp.text[:300]}, status_code=resp.status_code)
    return {"status": "cancelling", "job_id": running["job_id"]}


@app.get("/api/jobs")
async def list_jobs():
    return _fetch_github_runs(limit=25)


@app.get("/api/jobs/current")
async def current_job():
    runs = _fetch_github_runs(limit=5)
    running = next((r for r in runs if r["status"] == "running"), None)
    return running or {"status": "idle"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    runs = _fetch_github_runs(limit=25)
    job = next((r for r in runs if r["job_id"] == job_id), None)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.get("/api/jobs/{job_id}/images")
async def get_job_images(job_id: str):
    return {"images": [], "thumbnail": None, "total": 0}


@app.get("/api/analytics")
async def get_analytics():
    return {"videos": [], "channel": None}


@app.get("/api/health")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "runtime": "vercel",
        "timestamp": datetime.utcnow().isoformat(),
    }
