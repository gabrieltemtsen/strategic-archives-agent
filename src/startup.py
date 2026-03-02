"""
Railway Startup Bootstrap
=========================
Decodes base64-encoded secrets from env vars → writes them to disk at startup.

Required env vars:
  YOUTUBE_CLIENT_SECRET_B64           → base64(client_secret.json)

Per-channel token env vars (one per channel):
  YOUTUBE_TOKEN_B64_KIDS_UNIVERSE     → base64(.youtube_token_kids_universe.pkl)
  YOUTUBE_TOKEN_B64_STRATEGIC_ARCHIVES→ base64(.youtube_token_strategic_archives.pkl)
  YOUTUBE_TOKEN_B64_STORIES_TALES     → base64(.youtube_token_stories_tales.pkl)
  YOUTUBE_TOKEN_B64_GABE_DEV_CODES    → base64(.youtube_token_gabe_dev_codes.pkl)

Legacy fallback (single-channel):
  YOUTUBE_TOKEN_B64                   → base64(.youtube_token.pkl)

How to generate per-channel tokens:
  python scripts/auth_channel.py --channel kids_universe
  python scripts/auth_channel.py --channel strategic_archives
  ... etc.

How to encode for Railway:
  base64 -i .youtube_token_kids_universe.pkl | tr -d '\\n'
"""

import os
import base64
import logging
import pickle
import json as _json

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

# All known channel keys — add new channels here
CHANNEL_KEYS = [
    "kids_universe",
    "strategic_archives",
    "stories_tales",
    "gabe_dev_codes",
]


def _decode_client_secret():
    """Decode YOUTUBE_CLIENT_SECRET_B64 → client_secret.json"""
    value = os.getenv("YOUTUBE_CLIENT_SECRET_B64", "").strip()
    if not value:
        logger.warning("⚠️  YOUTUBE_CLIENT_SECRET_B64 not set — YouTube upload will fail")
        return
    if os.path.exists("client_secret.json"):
        logger.debug("client_secret.json already exists — skipping")
        return
    try:
        decoded = base64.b64decode(value)
        with open("client_secret.json", "wb") as f:
            f.write(decoded)
        logger.info("✅ Decoded YOUTUBE_CLIENT_SECRET_B64 → client_secret.json")
    except Exception as e:
        logger.error(f"Failed to decode YOUTUBE_CLIENT_SECRET_B64: {e}")
        raise


def _decode_token_value(value: str, token_path: str):
    """
    Decode a base64 token string → pickle file.
    Accepts JSON format (Railway-friendly) or raw pickle bytes.
    """
    raw = base64.b64decode(value)
    try:
        token_data = _json.loads(raw.decode("utf-8"))
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes", SCOPES),
        )
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)
        logger.info(f"✅ Decoded JSON token → {token_path}")
    except (_json.JSONDecodeError, UnicodeDecodeError):
        with open(token_path, "wb") as f:
            f.write(raw)
        logger.info(f"✅ Decoded pickle token → {token_path}")


def _decode_channel_tokens():
    """
    Decode per-channel tokens from YOUTUBE_TOKEN_B64_<CHANNEL_KEY_UPPER>.
    Falls back to legacy YOUTUBE_TOKEN_B64 → .youtube_token.pkl.
    Logs clearly which tokens are ready and which are missing.
    """
    found, missing = [], []

    # Per-channel tokens
    for key in CHANNEL_KEYS:
        env_var = f"YOUTUBE_TOKEN_B64_{key.upper()}"
        token_path = f".youtube_token_{key}.pkl"
        value = os.getenv(env_var, "").strip()

        if not value:
            missing.append((key, env_var))
            continue

        if os.path.exists(token_path):
            logger.info(f"  ✅ Token ready (cached): {token_path}")
            found.append(key)
            continue

        try:
            _decode_token_value(value, token_path)
            found.append(key)
        except Exception as e:
            logger.error(f"  ❌ Failed to decode {env_var}: {e}")
            missing.append((key, env_var))

    # Summary
    if found:
        logger.info(f"YouTube tokens ready: {found}")
    if missing:
        logger.warning(
            f"⚠️  Missing YouTube tokens for: {[k for k, _ in missing]}\n"
            + "\n".join(
                f"  → Set Railway env var: {ev}\n"
                f"    Generate with: python scripts/auth_channel.py --channel {k}\n"
                f"    Then encode:   base64 -i .youtube_token_{k}.pkl | tr -d '\\n'"
                for k, ev in missing
            )
        )

    # Legacy fallback
    legacy_value = os.getenv("YOUTUBE_TOKEN_B64", "").strip()
    if legacy_value and not os.path.exists(".youtube_token.pkl"):
        try:
            _decode_token_value(legacy_value, ".youtube_token.pkl")
            logger.warning(
                "⚠️  Legacy YOUTUBE_TOKEN_B64 decoded → .youtube_token.pkl\n"
                "    This uploads to your DEFAULT channel only.\n"
                "    Set per-channel tokens to fix channel targeting."
            )
        except Exception as e:
            logger.error(f"Failed to decode YOUTUBE_TOKEN_B64: {e}")

    if not found and not legacy_value:
        logger.error(
            "❌ NO YouTube tokens found at all — all uploads will fail.\n"
            "   Run scripts/auth_channel.py for each channel, then set\n"
            "   YOUTUBE_TOKEN_B64_<CHANNEL_KEY_UPPER> in Railway."
        )


def bootstrap():
    """Decode all base64 env vars → files at startup."""
    _decode_client_secret()
    _decode_channel_tokens()


if __name__ == "__main__":
    print("\nChannel token env vars to set in Railway:\n")
    for key in CHANNEL_KEYS:
        env_var = f"YOUTUBE_TOKEN_B64_{key.upper()}"
        token_path = f".youtube_token_{key}.pkl"
        print(f"  {env_var}")
        print(f"    base64 -i {token_path} | tr -d '\\n'\n")
