"""
Speech-to-text via Deepgram Nova-2.

Provides both sync and async helpers so every channel can call
``transcribe_audio`` without worrying about event-loop differences.

Returns the transcript string on success, or ``None`` when the Deepgram
API key is not configured.

Language is configurable via ``deepgram.language`` in pythonclaw.json:
  - ``"multi"`` (default) — multilingual mode, works for any language
  - ``"zh"``/``"en"``/``"ja"``/… — force a specific language
  - ``"auto"`` — auto-detect (needs ~5 s+ of audio to be reliable)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEEPGRAM_BASE = "https://api.deepgram.com/v1/listen"

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


def _build_url() -> str:
    """Build the Deepgram API URL with language/model parameters."""
    from .. import config

    lang = config.get_str("deepgram", "language") or "multi"
    model = config.get_str("deepgram", "model") or "nova-2"

    params = [
        f"model={model}",
        "smart_format=true",
        "punctuate=true",
    ]

    if lang == "auto":
        params.append("detect_language=true")
    else:
        params.append(f"language={lang}")

    return f"{_DEEPGRAM_BASE}?{'&'.join(params)}"


def transcribe_bytes(audio: bytes, content_type: str = "audio/ogg") -> str | None:
    """Blocking transcription — safe to call from a thread / executor.

    Returns the transcript text, or ``None`` if no Deepgram key is set.
    Raises on network / API errors.
    """
    key = _get_key()
    if not key:
        return None

    import httpx

    url = _build_url()
    resp = httpx.post(
        url,
        content=audio,
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": content_type,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    transcript = _extract_transcript(data)
    if not transcript:
        detected = _extract_language(data)
        logger.warning(
            "[STT] Empty transcript (detected_lang=%s, bytes=%d, mime=%s)",
            detected, len(audio), content_type,
        )
    return transcript


async def transcribe_bytes_async(
    audio: bytes, content_type: str = "audio/ogg"
) -> str | None:
    """Non-blocking transcription for async contexts.

    Returns the transcript text, or ``None`` if no Deepgram key is set.
    """
    key = _get_key()
    if not key:
        return None

    import httpx

    url = _build_url()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            content=audio,
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": content_type,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    transcript = _extract_transcript(data)
    if not transcript:
        detected = _extract_language(data)
        logger.warning(
            "[STT] Empty transcript (detected_lang=%s, bytes=%d, mime=%s)",
            detected, len(audio), content_type,
        )
    return transcript


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


def _extract_language(data: dict) -> str:
    """Extract the detected language code from the Deepgram response."""
    try:
        return (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("detected_language", "unknown")
        )
    except (IndexError, KeyError):
        return "unknown"


def no_key_message() -> str:
    """User-facing message when Deepgram key is missing."""
    return _NO_KEY_MSG
