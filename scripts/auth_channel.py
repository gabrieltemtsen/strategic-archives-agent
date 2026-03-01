"""
Per-Channel YouTube OAuth Token Generator
==========================================
Run this ONCE per channel to generate a separate token for each.

Usage:
    python scripts/auth_channel.py --channel kids_universe
    python scripts/auth_channel.py --channel strategic_archives
    python scripts/auth_channel.py --channel stories_tales
    python scripts/auth_channel.py --channel gabe_dev_codes

BEFORE running for each channel:
    1. Open YouTube Studio (studio.youtube.com)
    2. Click your profile icon (top right)
    3. Switch to the TARGET channel
    4. Then run this script — the OAuth flow will authenticate AS that channel
"""

import argparse
import os
import pickle
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

CLIENT_SECRETS_PATH = "client_secret.json"

CHANNELS = {
    "kids_universe":      "KidsFirstUniverse",
    "strategic_archives": "StrategicArchives",
    "stories_tales":      "yt-storiesandtales",
    "gabe_dev_codes":     "gabedevcodes",
}


def main():
    parser = argparse.ArgumentParser(description="Generate per-channel YouTube OAuth token")
    parser.add_argument("--channel", required=True, choices=CHANNELS.keys(),
                        help="Channel key to authenticate for")
    args = parser.parse_args()

    channel_key = args.channel
    channel_name = CHANNELS[channel_key]
    token_path = f".youtube_token_{channel_key}.pkl"

    print(f"\n{'='*60}")
    print(f"  Authenticating for: {channel_name} ({channel_key})")
    print(f"{'='*60}")
    print(f"""
⚠️  IMPORTANT — Do this BEFORE clicking the link below:
    1. Open https://studio.youtube.com in your browser
    2. Click your profile icon (top-right corner)
    3. Switch to the channel: "{channel_name}"
    4. Then come back here and press ENTER to continue
""")
    input("Press ENTER when you've switched to the correct channel...")

    if not os.path.exists(CLIENT_SECRETS_PATH):
        print(f"\n❌ Error: {CLIENT_SECRETS_PATH} not found.")
        print("   Download it from Google Cloud Console → APIs & Services → Credentials")
        sys.exit(1)

    print(f"\n🔐 Opening browser for OAuth flow...")
    print(f"   Sign in with the Google account that owns '{channel_name}'\n")

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
    creds = flow.run_local_server(port=8080, open_browser=True)

    with open(token_path, "wb") as f:
        pickle.dump(creds, f)

    print(f"\n✅ Token saved: {token_path}")
    print(f"\n📋 Now encode it for Railway:")
    print(f"   base64 -i {token_path} | tr -d '\\n'")
    print(f"\n   Set this as Railway env var:")
    print(f"   YOUTUBE_TOKEN_B64_{channel_key.upper()} = <paste output above>")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
