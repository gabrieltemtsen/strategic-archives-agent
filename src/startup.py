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

SECRET_FILES = [
    {
        "env_var": "YOUTUBE_CLIENT_SECRET_B64",
        "output_path": "client_secret.json",
        "required": True,
        "description": "YouTube OAuth client secret",
    },
    {
        "env_var": "YOUTUBE_TOKEN_B64",
        "output_path": ".youtube_token.pkl",
        "required": False,
        "description": "YouTube OAuth token (generated after first auth)",
    },
]


def bootstrap():
    """
    Decode base64 env vars → files.
    Call this once at app startup before any YouTube API calls.
    """
    for secret in SECRET_FILES:
        env_var = secret["env_var"]
        output_path = secret["output_path"]
        required = secret["required"]
        description = secret["description"]

        value = os.getenv(env_var, "").strip()

        if not value:
            if required:
                logger.warning(
                    f"⚠️  {env_var} not set — {description} unavailable. "
                    f"YouTube upload will fail. "
                    f"Run: base64 -i client_secret.json | tr -d '\\n'  "
                    f"and set it as a Railway env var."
                )
            else:
                logger.debug(f"{env_var} not set — skipping {output_path}")
            continue

        # Skip if file already exists (local dev with real files)
        if os.path.exists(output_path):
            logger.debug(f"{output_path} already exists — skipping decode")
            continue

        try:
            decoded = base64.b64decode(value)
            with open(output_path, "wb") as f:
                f.write(decoded)
            logger.info(f"✅ Decoded {env_var} → {output_path}")
        except Exception as e:
            logger.error(f"Failed to decode {env_var}: {e}")
            if required:
                raise RuntimeError(
                    f"Could not decode {env_var}. "
                    "Make sure it's valid base64 (no newlines)."
                ) from e


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
