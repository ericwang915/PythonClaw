"""
WhatsApp channel for PythonClaw (via the WhatsApp Cloud API).

Uses the ``pywa`` library which hooks directly into the existing FastAPI app
to receive incoming webhook events from Meta's WhatsApp Cloud API.

Session IDs used by this channel: ``whatsapp:{wa_id}``

Requirements
------------
* A **Meta Business** account with WhatsApp Cloud API access.
* A **Phone Number ID**, permanent **Access Token**, and a **Verify Token**
  (arbitrary string you choose for webhook verification).
* A publicly reachable HTTPS callback URL pointing to ``/whatsapp/webhook``
  on the PythonClaw web server (use ngrok for local development).

Commands
--------
  !reset          -- discard and recreate the current session
  !status         -- show session info
  !compact [hint] -- compact conversation history
  <text>          -- forwarded to Agent.chat(), reply sent back
  <image>         -- image sent to LLM for analysis

Access control
--------------
Set ``channels.whatsapp.allowedNumbers`` in ``pythonclaw.json`` to a list of
E.164 phone numbers (without "+") to restrict access.  Leave empty to allow
everyone.

Group behaviour
---------------
Set ``channels.whatsapp.requireMention`` to ``true`` to require @bot mention
in group chats.  DMs always respond.
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import TYPE_CHECKING

from .. import config

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)


class WhatsAppBot:
    """
    WhatsApp channel — pure I/O layer.

    Routes messages to the appropriate Agent via the shared SessionManager.
    The PyWa ``WhatsApp`` client registers its own webhook route on the
    FastAPI app so no extra endpoint wiring is needed.
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        phone_id: str,
        token: str,
        verify_token: str,
        callback_url: str | None = None,
        allowed_numbers: list[str] | None = None,
        require_mention: bool = False,
    ) -> None:
        self._sm = session_manager
        self._phone_id = phone_id
        self._token = token
        self._verify_token = verify_token
        self._callback_url = callback_url
        self._allowed_numbers: set[str] = set(allowed_numbers) if allowed_numbers else set()
        self._require_mention = require_mention
        self._wa = None  # set in mount()
        self._locks: dict[str, threading.Lock] = {}

    # ── Session ID convention ─────────────────────────────────────────────────

    @staticmethod
    def _session_id(wa_id: str) -> str:
        return f"whatsapp:{wa_id}"

    # ── Access control ────────────────────────────────────────────────────────

    def _is_allowed(self, wa_id: str) -> bool:
        if not self._allowed_numbers:
            return True
        return wa_id in self._allowed_numbers

    def _get_lock(self, session_id: str) -> threading.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = threading.Lock()
        return self._locks[session_id]

    # ── File sending ──────────────────────────────────────────────────────────

    def _register_file_sender(self, client, wa_id: str) -> None:
        """Register a sync callback so the Agent can send files via WhatsApp."""
        from ..core.tools import set_file_sender

        def _sender(path: str, caption: str = "") -> None:
            try:
                client.send_document(
                    to=wa_id,
                    document=path,
                    caption=caption[:1024] if caption else None,
                )
            except Exception as exc:
                logger.warning("[WhatsApp] send_file failed: %s", exc)

        set_file_sender(_sender)

    # ── Mount on FastAPI ──────────────────────────────────────────────────────

    def mount(self, app: "FastAPI") -> None:
        """Attach the PyWa webhook handler to *app*."""
        try:
            from pywa import WhatsApp, types
        except ImportError:
            logger.warning(
                "[WhatsApp] pywa is not installed.  "
                "Install with: pip install 'pywa[fastapi]'"
            )
            return

        wa_kwargs: dict = {
            "phone_id": self._phone_id,
            "token": self._token,
            "server": app,
            "verify_token": self._verify_token,
        }
        if self._callback_url:
            wa_kwargs["callback_url"] = self._callback_url

        wa = WhatsApp(**wa_kwargs)
        self._wa = wa
        bot = self

        @wa.on_message
        def _on_message(client: WhatsApp, msg: types.Message) -> None:
            wa_id = msg.from_user.wa_id
            if not bot._is_allowed(wa_id):
                msg.reply("Sorry, you are not authorised to use this bot.")
                return

            text = (msg.text or "").strip()
            has_image = msg.has_media and getattr(msg, "image", None) is not None
            has_audio = msg.has_media and (
                getattr(msg, "audio", None) is not None
                or getattr(msg, "voice", None) is not None
            )

            # Group mention check
            is_group = getattr(msg, "is_group", False)
            if is_group and bot._require_mention:
                mentioned = False
                if hasattr(msg, "mentioned") and msg.mentioned:
                    mentioned = True
                elif text and bot._phone_id:
                    mentioned = bot._phone_id in text
                if not mentioned:
                    return

            if has_audio and not text:
                transcript = _transcribe_wa_audio(client, msg)
                if transcript is None:
                    return
                text = transcript

            if not text and not has_image:
                return

            sid = bot._session_id(wa_id)
            agent = bot._sm.get_or_create(sid)

            if text.lower() == "!reset":
                bot._sm.reset(sid)
                msg.reply("Session reset. Starting fresh!")
                return

            if text.lower() == "!status":
                from ..core.compaction import estimate_tokens
                msg.reply(
                    f"*Session Status*\n"
                    f"Session: {sid}\n"
                    f"Skills: {len(agent.loaded_skill_names)} loaded\n"
                    f"Memories: {len(agent.memory.list_all())} entries\n"
                    f"History: {len(agent.messages)} messages\n"
                    f"Tokens: ~{estimate_tokens(agent.messages):,}\n"
                    f"Compactions: {agent.compaction_count}"
                )
                return

            if text.lower().startswith("!compact"):
                hint = text[len("!compact"):].strip() or None
                msg.reply("Compacting conversation history...")
                try:
                    result = agent.compact(instruction=hint)
                except Exception as exc:
                    result = f"Compaction failed: {exc}"
                for chunk in _split_message(result):
                    msg.reply(chunk)
                return

            if text.lower() == "!clear_files":
                from .. import config as _cfg
                count = _cfg.clear_files()
                msg.reply(f"Cleared {count} file(s) from the downloads folder.")
                return

            # Build input (text or multimodal)
            chat_input = text or "What's in this image?"
            if has_image:
                chat_input = _build_wa_image_input(
                    client, msg, text or "What's in this image?"
                )

            lock = bot._get_lock(sid)
            if lock.locked():
                msg.reply("Processing previous message...")

            bot._register_file_sender(client, wa_id)

            try:
                with lock:
                    response = agent.chat(chat_input)
            except Exception as exc:
                logger.exception("[WhatsApp] Agent.chat() raised an exception")
                response = f"Sorry, something went wrong: {exc}"

            for chunk in _split_message(response or "(no response)"):
                msg.reply(chunk)

        logger.info("[WhatsApp] Webhook handler mounted on FastAPI app")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start_async(self) -> None:
        """No-op — PyWa is driven by incoming HTTP requests, not polling."""
        logger.info("[WhatsApp] Channel ready (webhook mode)")

    async def stop_async(self) -> None:
        """No-op — cleanup handled by FastAPI shutdown."""
        logger.info("[WhatsApp] Channel stopped")


# ── Utility ───────────────────────────────────────────────────────────────────

def _split_message(text: str, limit: int = 4096) -> list[str]:
    """Split a long string into WhatsApp-safe chunks."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


def _build_wa_image_input(client, msg, caption: str) -> list:
    """Download WhatsApp image and build multimodal content array."""
    try:
        image = msg.image
        data = image.download(in_memory=True)
        b64 = base64.b64encode(data).decode()
        media_type = getattr(image, "mime_type", "image/jpeg")
        return [
            {"type": "text", "text": caption},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64}",
                },
            },
        ]
    except Exception:
        logger.warning("[WhatsApp] Failed to download image")
        return caption


def _transcribe_wa_audio(client, msg) -> str | None:
    """Download WhatsApp voice/audio and transcribe via Deepgram."""
    from ..core.stt import no_key_message, transcribe_bytes

    media = getattr(msg, "voice", None) or getattr(msg, "audio", None)
    if media is None:
        return None

    try:
        data = media.download(in_memory=True)
    except Exception:
        logger.warning("[WhatsApp] Failed to download audio")
        return None

    mime = getattr(media, "mime_type", "audio/ogg")
    try:
        transcript = transcribe_bytes(data, mime)
    except Exception as exc:
        logger.warning("[WhatsApp] Deepgram failed: %s", exc)
        msg.reply(f"Voice transcription failed: {exc}")
        return None

    if transcript is None:
        msg.reply(no_key_message())
        return None
    if not transcript.strip():
        msg.reply("Could not recognise any speech in the audio.")
        return None

    logger.info("[WhatsApp] Audio transcribed: %s", transcript[:80])
    return transcript


def create_bot(session_manager: "SessionManager") -> WhatsAppBot:
    """Create a WhatsAppBot from pythonclaw.json / env vars."""
    phone_id = config.get_str(
        "channels", "whatsapp", "phoneNumberId",
        env="WHATSAPP_PHONE_NUMBER_ID",
    )
    if not phone_id:
        raise ValueError(
            "WhatsApp Phone Number ID not set "
            "(env WHATSAPP_PHONE_NUMBER_ID or channels.whatsapp.phoneNumberId)"
        )

    token = config.get_str(
        "channels", "whatsapp", "token",
        env="WHATSAPP_TOKEN",
    )
    if not token:
        raise ValueError(
            "WhatsApp access token not set "
            "(env WHATSAPP_TOKEN or channels.whatsapp.token)"
        )

    verify_token = config.get_str(
        "channels", "whatsapp", "verifyToken",
        env="WHATSAPP_VERIFY_TOKEN",
        default="pythonclaw_verify",
    )

    callback_url = config.get_str(
        "channels", "whatsapp", "callbackUrl",
        env="WHATSAPP_CALLBACK_URL",
    ) or None

    allowed_numbers = config.get_list(
        "channels", "whatsapp", "allowedNumbers",
        env="WHATSAPP_ALLOWED_NUMBERS",
    )

    require_mention = config.get_bool(
        "channels", "whatsapp", "requireMention", default=False,
    )

    return WhatsAppBot(
        session_manager=session_manager,
        phone_id=phone_id,
        token=token,
        verify_token=verify_token,
        callback_url=callback_url,
        allowed_numbers=allowed_numbers or None,
        require_mention=require_mention,
    )


create_bot_from_env = create_bot
