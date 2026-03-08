"""
Discord channel for PythonClaw.

Session IDs: "discord:dm:{user_id}" (DMs) or "discord:{channel_id}" (guilds)

Commands
--------
  !reset          — discard and recreate the current session
  !status         — show session info
  !compact [hint] — compact conversation history
  <text>          — forwarded to Agent.chat(), reply sent back
  <image>         — image attachments sent to LLM for analysis

The bot responds to:
  - Direct messages (always)
  - Channel mentions (@bot message) in guilds (when requireMention=true)
  - All messages in whitelisted channels (when requireMention=false)

Access control
--------------
Set DISCORD_ALLOWED_USERS to a comma-separated list of Discord user IDs.
Set DISCORD_ALLOWED_CHANNELS to restrict which guild channels the bot listens in.
Leave empty to allow all.

Group behaviour
---------------
Set ``channels.discord.requireMention`` to ``true`` to require @bot mention
in guild channels. Default is ``false`` (respond when mentioned OR in
whitelisted channels).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

import discord

from .. import config

if TYPE_CHECKING:
    from ..session_manager import SessionManager

logger = logging.getLogger(__name__)

MAX_MSG_LEN = 2000  # Discord message character limit


class DiscordBot:
    """
    Discord channel — pure I/O layer.

    Routes messages to the appropriate Agent via the shared SessionManager.
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        token: str,
        allowed_users: list[int] | None = None,
        allowed_channels: list[int] | None = None,
        require_mention: bool = False,
    ) -> None:
        self._sm = session_manager
        self._token = token
        self._allowed_users: set[int] = set(allowed_users) if allowed_users else set()
        self._allowed_channels: set[int] = set(allowed_channels) if allowed_channels else set()
        self._require_mention = require_mention

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._setup_handlers()

    # ── Session ID convention ─────────────────────────────────────────────────

    @staticmethod
    def _session_id(source_id: int, is_dm: bool = False) -> str:
        prefix = "discord:dm" if is_dm else "discord"
        return f"{prefix}:{source_id}"

    # ── Access control ────────────────────────────────────────────────────────

    def _is_allowed_user(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _is_allowed_channel(self, channel_id: int) -> bool:
        if not self._allowed_channels:
            return True
        return channel_id in self._allowed_channels

    # ── Message splitting ─────────────────────────────────────────────────────

    @staticmethod
    def _split_message(text: str, limit: int = MAX_MSG_LEN) -> list[str]:
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            chunks.append(text[:limit])
            text = text[limit:]
        return chunks

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _setup_handlers(self) -> None:
        client = self._client

        @client.event
        async def on_ready():
            logger.info("[Discord] Logged in as %s (id=%s)", client.user.name, client.user.id)

        @client.event
        async def on_message(message: discord.Message):
            if message.author == client.user:
                return
            if message.author.bot:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = client.user in message.mentions if not is_dm else False

            if not is_dm:
                if self._require_mention and not is_mentioned:
                    return
                if not self._require_mention and not is_mentioned:
                    if not self._is_allowed_channel(message.channel.id):
                        return

            if not self._is_allowed_user(message.author.id):
                await message.reply("Sorry, you are not authorised to use this bot.")
                return

            content = message.content.strip()
            if is_mentioned and client.user:
                content = content.replace(f"<@{client.user.id}>", "").strip()

            has_image = any(
                a.content_type and a.content_type.startswith("image/")
                for a in message.attachments
            )
            has_audio = any(
                a.content_type and a.content_type.startswith("audio/")
                for a in message.attachments
            )

            if has_audio and not content:
                transcript = await self._transcribe_audio(message)
                if transcript is None:
                    return
                content = transcript

            if not content and not has_image:
                return

            # Command dispatch
            if content.startswith("!reset"):
                await self._cmd_reset(message, is_dm)
                return
            if content.startswith("!status"):
                await self._cmd_status(message, is_dm)
                return
            if content.startswith("!compact"):
                hint = content[len("!compact"):].strip() or None
                await self._cmd_compact(message, is_dm, hint)
                return
            if content.startswith("!clear_files"):
                await self._cmd_clear_files(message)
                return

            chat_input = content or ""
            if has_image:
                chat_input = await self._build_image_input(
                    message, content or "What's in this image?"
                )

            await self._handle_chat(message, chat_input, is_dm)

    # ── Image handling ────────────────────────────────────────────────────────

    @staticmethod
    async def _build_image_input(message: discord.Message, caption: str) -> list:
        """Download image attachments and build multimodal content array."""
        parts: list[dict] = [{"type": "text", "text": caption}]
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                try:
                    data = await att.read()
                    b64 = base64.b64encode(data).decode()
                    media_type = att.content_type.split(";")[0]
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                        },
                    })
                except Exception:
                    logger.warning("[Discord] Failed to download attachment %s", att.filename)
        return parts

    # ── Voice / audio handling ─────────────────────────────────────────────────

    @staticmethod
    async def _transcribe_audio(message: discord.Message) -> str | None:
        """Download the first audio attachment and transcribe via Deepgram."""
        from ..core.stt import no_key_message, transcribe_bytes_async

        for att in message.attachments:
            if att.content_type and att.content_type.startswith("audio/"):
                try:
                    data = await att.read()
                except Exception:
                    logger.warning("[Discord] Failed to download audio %s", att.filename)
                    return None

                mime = att.content_type.split(";")[0]
                try:
                    transcript = await transcribe_bytes_async(data, mime)
                except Exception as exc:
                    logger.warning("[Discord] Deepgram failed: %s", exc)
                    await message.reply(f"Voice transcription failed: {exc}")
                    return None

                if transcript is None:
                    await message.reply(no_key_message())
                    return None
                if not transcript.strip():
                    await message.reply("Could not recognise any speech in the audio.")
                    return None

                logger.info("[Discord] Audio transcribed: %s", transcript[:80])
                return transcript
        return None

    # ── Command implementations ───────────────────────────────────────────────

    async def _cmd_reset(self, message: discord.Message, is_dm: bool) -> None:
        sid = self._session_id(message.author.id if is_dm else message.channel.id, is_dm)
        self._sm.reset(sid)
        await message.reply("Session reset. Starting fresh!")

    async def _cmd_status(self, message: discord.Message, is_dm: bool) -> None:
        sid = self._session_id(message.author.id if is_dm else message.channel.id, is_dm)
        agent = self._sm.get_or_create(sid)
        from ..core.compaction import estimate_tokens
        status = (
            f"**Session Status**\n"
            f"```\n"
            f"Session ID   : {sid}\n"
            f"Provider     : {type(agent.provider).__name__}\n"
            f"Skills       : {len(agent.loaded_skill_names)} loaded\n"
            f"Memories     : {len(agent.memory.list_all())} entries\n"
            f"History      : {len(agent.messages)} messages\n"
            f"Est. tokens  : ~{estimate_tokens(agent.messages)}\n"
            f"Compactions  : {agent.compaction_count}\n"
            f"```"
        )
        await message.reply(status)

    async def _cmd_clear_files(self, message: discord.Message) -> None:
        from .. import config as _cfg
        count = _cfg.clear_files()
        await message.reply(f"Cleared {count} file(s) from the downloads folder.")

    async def _cmd_compact(self, message: discord.Message, is_dm: bool, hint: str | None) -> None:
        sid = self._session_id(message.author.id if is_dm else message.channel.id, is_dm)
        agent = self._sm.get_or_create(sid)
        await message.reply("Compacting conversation history...")
        try:
            result = agent.compact(instruction=hint)
        except Exception as exc:
            logger.exception("[Discord] compact() raised an exception")
            result = f"Compaction failed: {exc}"
        for chunk in self._split_message(result or "(no result)"):
            await message.reply(chunk)

    async def _handle_chat(
        self,
        message: discord.Message,
        content: str | list,
        is_dm: bool,
    ) -> None:
        sid = self._session_id(message.author.id if is_dm else message.channel.id, is_dm)
        agent = self._sm.get_or_create(sid)

        if self._sm.is_locked(sid):
            await message.reply("Processing previous message\u2026")

        async with message.channel.typing():
            try:
                async with self._sm.acquire(sid):
                    loop = asyncio.get_event_loop()
                    self._register_file_sender(loop, message.channel)
                    response = await loop.run_in_executor(None, agent.chat, content)
            except Exception as exc:
                logger.exception("[Discord] Agent.chat() raised an exception")
                response = f"Sorry, something went wrong: {exc}"
        for chunk in self._split_message(response or "(no response)"):
            await message.reply(chunk)

    # ── File sending ──────────────────────────────────────────────────────────

    def _register_file_sender(
        self,
        loop: asyncio.AbstractEventLoop,
        channel: discord.abc.Messageable,
    ) -> None:
        """Register a sync callback so the Agent can send files via Discord."""
        from ..core.tools import set_file_sender

        def _sender(path: str, caption: str = "") -> None:
            async def _do_send():
                try:
                    await channel.send(
                        content=caption[:2000] if caption else None,
                        file=discord.File(path),
                    )
                except Exception as exc:
                    logger.warning("[Discord] send_file failed: %s", exc)

            future = asyncio.run_coroutine_threadsafe(_do_send(), loop)
            future.result(timeout=60)

        set_file_sender(_sender)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start_async(self) -> None:
        """Non-blocking start — for use inside an existing asyncio event loop."""
        logger.info("[Discord] Starting bot (async mode)...")
        await self._client.start(self._token)

    async def stop_async(self) -> None:
        logger.info("[Discord] Stopping bot...")
        await self._client.close()


# ── Utility ───────────────────────────────────────────────────────────────────

def create_bot(session_manager: "SessionManager") -> "DiscordBot":
    """Create a DiscordBot from pythonclaw.json / env vars."""
    token = config.get_str(
        "channels", "discord", "token", env="DISCORD_BOT_TOKEN",
    )
    if not token:
        raise ValueError("Discord token not set (env DISCORD_BOT_TOKEN or channels.discord.token)")
    allowed_users = config.get_int_list(
        "channels", "discord", "allowedUsers", env="DISCORD_ALLOWED_USERS",
    )
    allowed_channels = config.get_int_list(
        "channels", "discord", "allowedChannels", env="DISCORD_ALLOWED_CHANNELS",
    )
    require_mention = config.get_bool(
        "channels", "discord", "requireMention", default=False,
    )
    return DiscordBot(
        session_manager=session_manager,
        token=token,
        allowed_users=allowed_users or None,
        allowed_channels=allowed_channels or None,
        require_mention=require_mention,
    )


# Backward-compatible alias
create_bot_from_env = create_bot
