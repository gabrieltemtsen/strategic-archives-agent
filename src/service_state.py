"""Service state toggles for the Strategic Archives Agent.

Goal: allow an operator (via dashboard API) to quickly pause background activity
(scheduler + job triggers) without redeploying.

State is stored in a small JSON file on disk so it persists across module reloads
within the same container/runtime.

Security note:
- This is *not* an auth system. Protect the dashboard with Railway/Cloudflare
  auth, basic auth, or a private network if needed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


DEFAULT_STATE: Dict[str, Any] = {
    "scheduler_enabled": True,
    "jobs_enabled": True,
    "reason": "",
    "updated_at": None,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> str:
    # Prefer /tmp on Railway (writable). Allow override.
    return os.getenv("SERVICE_STATE_PATH", "/tmp/strategic_archives_service_state.json")


def get_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return dict(DEFAULT_STATE)
    except Exception:
        # If file is corrupted, fall back to defaults (safer: keep things enabled)
        return dict(DEFAULT_STATE)

    state = dict(DEFAULT_STATE)
    if isinstance(data, dict):
        state.update({k: data.get(k, state[k]) for k in state.keys()})
    return state


def set_state(**patch: Any) -> Dict[str, Any]:
    state = get_state()
    for k, v in patch.items():
        if k in DEFAULT_STATE:
            state[k] = v
    state["updated_at"] = _utc_now_iso()

    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return state


def disable_all(reason: str = "") -> Dict[str, Any]:
    return set_state(scheduler_enabled=False, jobs_enabled=False, reason=reason or "paused")


def enable_all(reason: str = "") -> Dict[str, Any]:
    # Keep reason for auditing, but allow clearing.
    return set_state(scheduler_enabled=True, jobs_enabled=True, reason=reason)
