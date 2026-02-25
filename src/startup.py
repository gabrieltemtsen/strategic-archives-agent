"""
Railway Startup Bootstrap
Decodes base64-encoded secrets from env vars and writes them to files.

Railway doesn't support file mounts like docker-compose does, so sensitive
files (client_secret.json, .youtube_token.pkl) are stored as base64 env vars
and decoded to disk at startup.

Required env vars on Railway:
  YOUTUBE_CLIENT_SECRET_B64  → base64(client_secret.json)
  YOUTUBE_TOKEN_B64          → base64(.youtube_token.pkl)  [optional on first run]

How to encode locally:
  base64 -i client_secret.json | tr -d '\\n'   → copy output to Railway env var
  base64 -i .youtube_token.pkl | tr -d '\\n'   → copy output to Railway env var
"""

import os
import base64
import logging

logger = logging.getLogger(__name__)

def _decode_client_secret():
    """Decode YOUTUBE_CLIENT_SECRET_B64 → client_secret.json"""
    value = os.getenv("YOUTUBE_CLIENT_SECRET_B64", "").strip()
    if not value:
        logger.warning(
            "⚠️  YOUTUBE_CLIENT_SECRET_B64 not set — YouTube upload will fail. "
            "Encode with: python3 -c \"import base64,json; "
            "print(base64.b64encode(json.dumps(YOUR_SECRET_DICT).encode()).decode())\""
        )
        return
    if os.path.exists("client_secret.json"):
        logger.debug("client_secret.json already exists — skipping decode")
        return
    try:
        decoded = base64.b64decode(value)
        with open("client_secret.json", "wb") as f:
            f.write(decoded)
        logger.info("✅ Decoded YOUTUBE_CLIENT_SECRET_B64 → client_secret.json")
    except Exception as e:
        logger.error(f"Failed to decode YOUTUBE_CLIENT_SECRET_B64: {e}")
        raise


def _decode_youtube_token():
    """
    Decode YOUTUBE_TOKEN_B64 → .youtube_token.pkl
    Accepts two formats:
      1. JSON  {"token":..., "refresh_token":..., ...}  (Railway-friendly)
      2. Raw pickle bytes (local dev export)
    Converts JSON → Credentials object → pickle so upload.py needs no changes.
    """
    value = os.getenv("YOUTUBE_TOKEN_B64", "").strip()
    if not value:
        logger.debug("YOUTUBE_TOKEN_B64 not set — YouTube token not pre-loaded")
        return
    if os.path.exists(".youtube_token.pkl"):
        logger.debug(".youtube_token.pkl already exists — skipping decode")
        return
    try:
        import pickle, json as _json
        raw = base64.b64decode(value)

        # Try JSON format first (Railway-native)
        try:
            token_data = _json.loads(raw.decode("utf-8"))
            from google.oauth2.credentials import Credentials
            creds = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes", [
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube"
                ])
            )
            with open(".youtube_token.pkl", "wb") as f:
                pickle.dump(creds, f)
            logger.info("✅ Decoded YOUTUBE_TOKEN_B64 (JSON) → .youtube_token.pkl")
        except (_json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to raw pickle bytes
            with open(".youtube_token.pkl", "wb") as f:
                f.write(raw)
            logger.info("✅ Decoded YOUTUBE_TOKEN_B64 (pickle) → .youtube_token.pkl")

    except Exception as e:
        logger.error(f"Failed to decode YOUTUBE_TOKEN_B64: {e}")


def bootstrap():
    """
    Decode base64 env vars → files at startup.
    Call before any YouTube API calls.
    """
    _decode_client_secret()
    _decode_youtube_token()


def encode_file_instructions():
    """Print instructions for encoding files to base64 (for Railway setup)."""
    print("\n" + "="*60)
    print("RAILWAY SETUP — Encoding secrets to base64")
    print("="*60)
    print("\nRun these commands locally and copy output to Railway env vars:\n")
    for secret in SECRET_FILES:
        fname = secret["output_path"].lstrip(".")
        print(f"  {secret['env_var']}:")
        print(f"    base64 -i {secret['output_path']} | tr -d '\\n'")
        print()
    print("Then add each as an env var in Railway → your service → Variables\n")


if __name__ == "__main__":
    encode_file_instructions()
