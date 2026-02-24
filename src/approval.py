"""
Telegram Approval Flow
Sends script to Gabe for review before video generation starts.
Supports: Approve, Edit, Reject (regenerate)
"""

import os
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramApproval:
    def __init__(self, config: dict):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = config.get("telegram", {}).get("approval_timeout", 3600)

    def _api(self, method: str, **kwargs) -> dict:
        url = TELEGRAM_API.format(token=self.token, method=method)
        response = requests.post(url, json=kwargs, timeout=30)
        response.raise_for_status()
        return response.json()

    def _send_message(self, text: str, reply_markup: Optional[dict] = None) -> int:
        """Send a message and return message_id."""
        kwargs = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
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
        """
        content_type = script_data.get("content_type", "video")
        title = script_data.get("title", "Untitled")
        language = script_data.get("language", "English")
        script_preview = script_data.get("script", "")[:800]

        message = (
            f"🎬 *New Video Ready for Review*\n\n"
            f"📌 *Type:* {content_type.replace('_', ' ').title()}\n"
            f"🌍 *Language:* {language}\n"
            f"📝 *Title:* {title}\n\n"
            f"*Script Preview (first 800 chars):*\n"
            f"_{script_preview}..._\n\n"
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
                        self._send_message("✅ *Approved!* Starting video generation now... 🎬")
                        return {"status": "approved", "script_data": script_data}

                    elif data == "reject":
                        self._send_message("🔄 *Rejected.* Regenerating a new script...")
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
                                f"✅ Title updated to: *{title_response}*\n\nApprove now?",
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
                            self._send_message(f"```\n{chunk}\n```")
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
            f"⏰ *Approval timed out* after {self.timeout//60} minutes. "
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

    def notify(self, message: str):
        """Send a simple notification message."""
        if self.token and self.chat_id:
            try:
                self._send_message(message)
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")
