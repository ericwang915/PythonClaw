"""
Speech-to-text — cloud (Deepgram) or fully local (Whisper).

Provides both sync and async helpers so every channel can call
``transcribe_bytes`` without worrying about event-loop differences.

Backend is chosen by ``stt.provider`` in pythonclaw.json:
  - ``"deepgram"`` (default) — Deepgram API, needs ``deepgram.apiKey``
  - ``"whisper"``            — 100% local via ``faster-whisper`` (no API key);
                               pairs with the local Ollama setup

Returns the transcript string on success, or ``None`` when the selected
backend is unavailable (no Deepgram key, or ``faster-whisper`` not installed).

Language is configurable via ``deepgram.language`` in pythonclaw.json:
  - ``"auto"`` (default) — auto-detect, with fallback retries for short clips
  - ``"zh"``/``"en"``/``"ja"``/… — force a specific language
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEEPGRAM_BASE = "https://api.deepgram.com/v1/listen"


def _get_provider() -> str:
    from .. import config
    return (config.get_str("stt", "provider") or "deepgram").lower()

# Retried in order when auto-detect returns empty (short clips). Capped so a
# silent clip can't burn minutes of sequential API calls; override the pool
# via ``deepgram.fallbackLanguages`` in pythonclaw.json.
_FALLBACK_LANGUAGES = ("zh", "en", "ja")


def _get_fallback_languages() -> tuple[str, ...]:
    from .. import config
    langs = config.get_list("deepgram", "fallbackLanguages")
    return tuple(langs) if langs else _FALLBACK_LANGUAGES

_NO_KEY_MSG = (
    "Voice messages are not enabled yet.\n\n"
    "To unlock voice input, you need a Deepgram API key:\n"
    "1. Go to https://console.deepgram.com/signup and create a free account\n"
    "2. After signing in, go to API Keys (left sidebar)\n"
    "3. Click \"Create a New API Key\", give it a name, and copy the key\n"
    "4. Set it in Config -> deepgram -> apiKey (or set the DEEPGRAM_API_KEY env var)\n\n"
    "Deepgram offers $200 free credits on signup — no credit card required."
)


def _get_key() -> str | None:
    from .. import config
    return config.get("deepgram", "apiKey", env="DEEPGRAM_API_KEY") or None


def _get_config_language() -> str:
    from .. import config
    return config.get_str("deepgram", "language") or "auto"


def _get_model() -> str:
    from .. import config
    return config.get_str("deepgram", "model") or "nova-2"


# ── Local Whisper backend (optional: `pip install faster-whisper`) ────────────

_NO_WHISPER_MSG = (
    "Local voice transcription needs faster-whisper:\n\n"
    "  pip install faster-whisper\n\n"
    'Then set `stt.provider` to "whisper" in pythonclaw.json. '
    "It runs 100% locally — no API key, nothing leaves your machine."
)

_whisper_model = None
_whisper_key: tuple | None = None


def _get_whisper_model():
    """Return a cached faster-whisper model, or None if the package is missing."""
    global _whisper_model, _whisper_key
    from .. import config

    size = config.get_str("whisper", "model") or "base"
    device = config.get_str("whisper", "device") or "auto"
    compute = config.get_str("whisper", "computeType") or "default"
    key = (size, device, compute)

    if _whisper_model is not None and _whisper_key == key:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    _whisper_model = WhisperModel(size, device=device, compute_type=compute)
    _whisper_key = key
    return _whisper_model


def _transcribe_whisper(audio: bytes) -> str | None:
    """Transcribe locally. Returns ``None`` if faster-whisper isn't installed."""
    import io

    model = _get_whisper_model()
    if model is None:
        return None

    lang = _get_config_language()
    wlang = None if lang in ("auto", "multi", "") else lang
    segments, _info = model.transcribe(io.BytesIO(audio), language=wlang)
    return "".join(seg.text for seg in segments).strip()


def _build_url(language: str | None = None) -> str:
    """Build the Deepgram API URL.

    If *language* is given, use it directly (e.g. ``"zh"``).
    If ``None``, read from config (default ``"auto"`` → detect_language).
    """
    model = _get_model()
    lang = language or _get_config_language()

    params = [f"model={model}", "smart_format=true", "punctuate=true"]

    if lang == "auto":
        params.append("detect_language=true")
    else:
        params.append(f"language={lang}")

    return f"{_DEEPGRAM_BASE}?{'&'.join(params)}"


def _headers(key: str, content_type: str) -> dict[str, str]:
    return {
        "Authorization": f"Token {key}",
        "Content-Type": content_type,
    }


# ── Sync ──────────────────────────────────────────────────────────────────────

def transcribe_bytes(audio: bytes, content_type: str = "audio/ogg") -> str | None:
    """Blocking transcription with automatic fallback for short clips.

    Returns the transcript text, or ``None`` if the backend is unavailable.
    """
    if _get_provider() == "whisper":
        return _transcribe_whisper(audio)

    key = _get_key()
    if not key:
        return None

    cfg_lang = _get_config_language()

    if cfg_lang != "auto":
        return _call_sync(key, audio, content_type, language=cfg_lang)

    transcript = _call_sync(key, audio, content_type, language=None)
    if transcript:
        return transcript

    # One rejected language (e.g. a model/language combo returning 400) must
    # not abort the whole chain — catch per attempt and move on.
    for lang in _get_fallback_languages():
        try:
            transcript = _call_sync(key, audio, content_type, language=lang)
        except Exception as exc:
            logger.debug("[STT] Fallback language=%s failed: %s", lang, exc)
            continue
        if transcript:
            logger.info("[STT] Fallback to language=%s succeeded", lang)
            return transcript

    logger.warning("[STT] All fallback languages returned empty (bytes=%d)", len(audio))
    return ""


def _call_sync(
    key: str, audio: bytes, content_type: str, language: str | None
) -> str:
    import httpx

    url = _build_url(language=language or "auto")
    resp = httpx.post(
        url, content=audio,
        headers=_headers(key, content_type),
        timeout=30.0,
    )
    resp.raise_for_status()
    return _extract_transcript(resp.json())


# ── Async ─────────────────────────────────────────────────────────────────────

async def transcribe_bytes_async(
    audio: bytes, content_type: str = "audio/ogg"
) -> str | None:
    """Non-blocking transcription with automatic fallback for short clips.

    Returns the transcript text, or ``None`` if the backend is unavailable.
    """
    if _get_provider() == "whisper":
        # faster-whisper is blocking/CPU-bound — never run it on the event loop.
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, _transcribe_whisper, audio
        )

    key = _get_key()
    if not key:
        return None

    cfg_lang = _get_config_language()

    if cfg_lang != "auto":
        return await _call_async(key, audio, content_type, language=cfg_lang)

    transcript = await _call_async(key, audio, content_type, language=None)
    if transcript:
        return transcript

    for lang in _get_fallback_languages():
        try:
            transcript = await _call_async(key, audio, content_type, language=lang)
        except Exception as exc:
            logger.debug("[STT] Fallback language=%s failed: %s", lang, exc)
            continue
        if transcript:
            logger.info("[STT] Fallback to language=%s succeeded", lang)
            return transcript

    logger.warning("[STT] All fallback languages returned empty (bytes=%d)", len(audio))
    return ""


async def _call_async(
    key: str, audio: bytes, content_type: str, language: str | None
) -> str:
    import httpx

    url = _build_url(language=language or "auto")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url, content=audio,
            headers=_headers(key, content_type),
        )
        resp.raise_for_status()
        return _extract_transcript(resp.json())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_transcript(data: dict) -> str:
    try:
        return (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
    except (IndexError, KeyError):
        return ""


def no_key_message() -> str:
    """User-facing message when the selected STT backend is unavailable."""
    if _get_provider() == "whisper":
        return _NO_WHISPER_MSG
    return _NO_KEY_MSG
