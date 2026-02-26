"""
Telegram Approval Flow
Sends script to Gabe for review before video generation starts.
Supports: Approve, Edit, Reject (regenerate)
"""

import os
import time
import html as html_mod
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramApproval:
    def __init__(self, config: dict):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.timeout = config.get("telegram", {}).get("approval_timeout", 3600)

        if self.token and self.chat_id:
            # chat_id must be a numeric integer for personal chats
            # e.g. 123456789  NOT  @username
            if self.chat_id.startswith("@"):
                logger.warning(
                    "TELEGRAM_CHAT_ID looks like a username (@...). "
                    "For personal/DM chats use your numeric user ID instead. "
                    "Get it by messaging @userinfobot on Telegram."
                )
            self._validated = True
        else:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "Telegram notifications disabled."
            )
            self._validated = False

    def _api(self, method: str, **kwargs) -> dict:
        url = TELEGRAM_API.format(token=self.token, method=method)
        response = requests.post(url, json=kwargs, timeout=30)
        if not response.ok:
            logger.error(f"Telegram API error: {response.status_code} {response.text}")
        response.raise_for_status()
        return response.json()

    def _send_message(self, text: str, reply_markup: Optional[dict] = None) -> int:
        """Send a message and return message_id."""
        kwargs = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML", # Changed from "Markdown" to "HTML"
        }
        if reply_markup:
            kwargs["reply_markup"] = reply_markup
        result = self._api("sendMessage", **kwargs)
        return result["result"]["message_id"]

    def _get_updates(self, offset: int = 0) -> list:
        result = self._api("getUpdates", offset=offset, timeout=10, allowed_updates=["callback_query", "message"])
        return result.get("result", [])

    def send_for_approval(self, script_data: dict) -> dict:
        """
        Send script to Telegram for approval.
        Returns: {'status': 'approved'|'edited'|'rejected', 'script_data': dict}
        Falls back to CLI prompt if Telegram is not configured.
        """
        if not self._validated:
            # CLI fallback when Telegram is not set up
            logger.info("Telegram not configured — falling back to CLI approval")
            print("\n" + "="*60)
            print(f"📋 SCRIPT READY FOR REVIEW")
            print("="*60)
            print(f"Title   : {script_data.get('title', 'Untitled')}")
            print(f"Type    : {script_data.get('content_type', 'N/A')}")
            print(f"Language: {script_data.get('language', 'English')}")
            print("-"*60)
            print(script_data.get("script", "")[:600] + "...\n")
            choice = input("Approve? [y/N/r(egenerate)]: ").strip().lower()
            if choice == "y":
                return {"status": "approved", "script_data": script_data}
            elif choice == "r":
                return {"status": "rejected", "script_data": script_data}
            else:
                return {"status": "timeout", "script_data": script_data}

        content_type = script_data.get("content_type", "video")
        title = script_data.get("title", "Untitled")
        language = script_data.get("language", "English")
        script_preview = script_data.get("script", "")[:800]

        # Escape HTML special chars in user content
        safe_title = html_mod.escape(title)
        safe_preview = html_mod.escape(script_preview)

        message = (
            f"🎬 <b>New Video Ready for Review</b>\n\n"
            f"📌 <b>Type:</b> {content_type.replace('_', ' ').title()}\n"
            f"🌍 <b>Language:</b> {language}\n"
            f"📝 <b>Title:</b> {safe_title}\n\n"
            f"<b>Script Preview (first 800 chars):</b>\n"
            f"<i>{safe_preview}...</i>\n\n"
            f"👇 What do you want to do?"
        )

        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve & Generate", "callback_data": "approve"},
                    {"text": "❌ Reject & Regenerate", "callback_data": "reject"},
                ],
                [
                    {"text": "✏️ Edit Title", "callback_data": "edit_title"},
                    {"text": "📋 View Full Script", "callback_data": "view_full"},
                ]
            ]
        }

        msg_id = self._send_message(message, reply_markup)
        logger.info(f"Approval request sent (msg_id={msg_id}), waiting...")

        # Poll for response
        start = time.time()
        offset = 0

        while time.time() - start < self.timeout:
            updates = self._get_updates(offset=offset)
            for update in updates:
                offset = update["update_id"] + 1

                # Handle callback query (button press)
                if "callback_query" in update:
                    cq = update["callback_query"]
                    data = cq.get("data", "")
                    user_id = cq["from"]["id"]

                    # Answer the callback
                    self._api("answerCallbackQuery", callback_query_id=cq["id"])

                    if data == "approve":
                        self._send_message("✅ <b>Approved!</b> Starting video generation now... 🎬")
                        return {"status": "approved", "script_data": script_data}

                    elif data == "reject":
                        self._send_message("🔄 <b>Rejected.</b> Regenerating a new script...")
                        return {"status": "rejected", "script_data": script_data}

                    elif data == "edit_title":
                        self._send_message(
                            "✏️ Send the new title as a message now:"
                        )
                        # Wait for text reply
                        title_response = self._wait_for_text(offset, timeout=300)
                        if title_response:
                            script_data["title"] = title_response
                            self._send_message(
                                f"✅ Title updated to: <b>{html_mod.escape(title_response)}</b>\n\nApprove now?",
                                reply_markup={
                                    "inline_keyboard": [[
                                        {"text": "✅ Approve", "callback_data": "approve"},
                                        {"text": "❌ Reject", "callback_data": "reject"},
                                    ]]
                                }
                            )

                    elif data == "view_full":
                        full_script = script_data.get("script", "")
                        # Split into chunks (Telegram 4096 char limit)
                        for i in range(0, len(full_script), 4000):
                            chunk = full_script[i:i+4000]
                            self._send_message(f"<pre>{html_mod.escape(chunk)}</pre>")
                        self._send_message(
                            "👆 Full script above. Approve or reject?",
                            reply_markup={
                                "inline_keyboard": [[
                                    {"text": "✅ Approve", "callback_data": "approve"},
                                    {"text": "❌ Reject", "callback_data": "reject"},
                                ]]
                            }
                        )

            time.sleep(3)

        # Timeout — auto-skip
        logger.warning(f"Approval timeout after {self.timeout}s. Skipping today's video.")
        self._send_message(
            f"⏰ <b>Approval timed out</b> after {self.timeout//60} minutes. "
            "Today's video was skipped. I'll try again tomorrow!"
        )
        return {"status": "timeout", "script_data": script_data}

    def _wait_for_text(self, offset: int, timeout: int = 300) -> Optional[str]:
        """Wait for a plain text message reply."""
        start = time.time()
        while time.time() - start < timeout:
            updates = self._get_updates(offset=offset)
            for update in updates:
                if "message" in update and "text" in update["message"]:
                    return update["message"]["text"]
            time.sleep(2)
        return None

    def pick_channel(self, active_channels: dict) -> Optional[str]:
        """
        Step 0 of the flow: ask user to pick a channel.
        Returns selected channel key, or None on timeout.

        Falls back to CLI prompt if Telegram not configured.
        """
        if not self._validated:
            # CLI fallback
            print("\n📺 Available channels:")
            for i, (key, ch) in enumerate(active_channels.items()):
                emoji = ch.get("emoji", "📺")
                print(f"  {i+1}. [{key}] {emoji} {ch['name']} — {ch.get('niche', '').replace('_', ' ')}")
            try:
                choice = input("\nPick channel number: ").strip()
                keys = list(active_channels.keys())
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    return keys[idx]
            except (ValueError, IndexError):
                pass
            return list(active_channels.keys())[0]

        # Build Telegram inline keyboard — 2 channels per row
        buttons = []
        row = []
        for key, ch in active_channels.items():
            emoji = ch.get("emoji", "📺")
            row.append({"text": f"{emoji} {ch['name']}", "callback_data": f"ch:{key}"})
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        channel_list = "\n".join(
            f"{ch.get('emoji','📺')} <b>{ch['name']}</b> — <i>{ch.get('niche','').replace('_',' ')}</i>"
            for ch in active_channels.values()
        )
        self._send_message(
            f"🎬 <b>New Video Session</b>\n\n"
            f"Which channel are we posting to today?\n\n"
            f"{channel_list}",
            reply_markup={"inline_keyboard": buttons}
        )

        logger.info(f"Channel picker sent — waiting for selection...")
        start = time.time()
        offset = 0

        while time.time() - start < self.timeout:
            updates = self._get_updates(offset=offset)
            for update in updates:
                offset = update["update_id"] + 1
                if "callback_query" in update:
                    cq = update["callback_query"]
                    data = cq.get("data", "")
                    self._api("answerCallbackQuery", callback_query_id=cq["id"])
                    if data.startswith("ch:"):
                        key = data[3:]
                        if key in active_channels:
                            ch = active_channels[key]
                            self._send_message(
                                f"{ch.get('emoji','📺')} <b>{ch['name']}</b> selected!\n"
                                f"Niche: <i>{ch.get('niche','').replace('_',' ')}</i>\n\n"
                                f"⏳ Generating script..."
                            )
                            return key
            time.sleep(2)

        logger.warning("Channel pick timed out — using default")
        return list(active_channels.keys())[0]

    def pick_content_type(self, channel: dict) -> Optional[str]:
        """
        Optional step: let user pick a content type for the channel,
        or tap 🎲 Random to let the agent decide.
        Falls back to None (random) if Telegram not configured.
        """
        content_types = channel.get("content", {}).get("types", [])
        if not content_types or len(content_types) == 1:
            return content_types[0] if content_types else None

        if not self._validated:
            return None  # Let agent pick randomly

        # Build keyboard
        buttons = [[{"text": f"🎲 Random", "callback_data": "ct:random"}]]
        row = []
        for ct in content_types:
            label = ct.replace("_", " ").title()
            row.append({"text": label, "callback_data": f"ct:{ct}"})
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        ch_name = channel.get("name", "channel")
        self._send_message(
            f"📝 <b>Content type for {ch_name}?</b>\n"
            f"Or let me pick randomly:",
            reply_markup={"inline_keyboard": buttons}
        )

        start = time.time()
        offset = 0
        while time.time() - start < 120:  # 2 min to pick type
            updates = self._get_updates(offset=offset)
            for update in updates:
                offset = update["update_id"] + 1
                if "callback_query" in update:
                    cq = update["callback_query"]
                    data = cq.get("data", "")
                    self._api("answerCallbackQuery", callback_query_id=cq["id"])
                    if data.startswith("ct:"):
                        ct = data[3:]
                        return None if ct == "random" else ct
            time.sleep(2)

        return None  # timeout → random

    def notify(self, message: str):
        """Send a simple notification message."""
        if not self._validated:
            logger.info(f"[Telegram disabled] {message}")
            return
        try:
            self._send_message(message)
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
