"""
FastAPI application for the PythonClaw Web Dashboard.

Provides REST endpoints for config/skills/status inspection, a config
save endpoint for editing settings from the browser, and a WebSocket
endpoint for real-time chat with the agent.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..core.agent import Agent
from ..core.llm.base import LLMProvider
from ..core.persistent_agent import PersistentAgent
from ..core.session_store import SessionStore
from ..core.skill_loader import SkillRegistry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_agent: Agent | None = None
_provider: LLMProvider | None = None
_store: SessionStore | None = None
_start_time: float = 0.0
_build_provider_fn = None
_active_bots: list = []
_chat_lock: asyncio.Lock | None = None
_fastapi_app: FastAPI | None = None

WEB_SESSION_ID = "web:dashboard"


def _get_chat_lock() -> asyncio.Lock:
    """Lazily create the web chat lock (must be done inside the event loop)."""
    global _chat_lock
    if _chat_lock is None:
        _chat_lock = asyncio.Lock()
    return _chat_lock


def create_app(provider: LLMProvider | None, *, build_provider_fn=None) -> FastAPI:
    """Build and return the FastAPI app.

    Parameters
    ----------
    provider          : LLM provider (may be None if not yet configured)
    build_provider_fn : callable that rebuilds the provider from config
                        (used after config save to hot-reload the provider)
    """
    global _provider, _store, _start_time, _build_provider_fn, _fastapi_app
    _provider = provider
    _store = SessionStore()
    _start_time = time.time()
    _build_provider_fn = build_provider_fn

    app = FastAPI(title="PythonClaw Dashboard", docs_url=None, redoc_url=None)
    _fastapi_app = app

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.add_api_route("/", _serve_index, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/config", _api_config_get, methods=["GET"])
    app.add_api_route("/api/config", _api_config_save, methods=["POST"])
    app.add_api_route("/api/skills", _api_skills, methods=["GET"])
    app.add_api_route("/api/status", _api_status, methods=["GET"])
    app.add_api_route("/api/memories", _api_memories, methods=["GET"])
    app.add_api_route("/api/identity", _api_identity, methods=["GET"])
    app.add_api_route("/api/identity/soul", _api_save_soul, methods=["POST"])
    app.add_api_route("/api/identity/persona", _api_save_persona, methods=["POST"])
    app.add_api_route("/api/identity/tools", _api_get_tools_notes, methods=["GET"])
    app.add_api_route("/api/identity/tools", _api_save_tools_notes, methods=["POST"])
    app.add_api_route("/api/memory/index", _api_get_index, methods=["GET"])
    app.add_api_route("/api/memory/index", _api_save_index, methods=["POST"])
    app.add_api_route("/api/transcribe", _api_transcribe, methods=["POST"])
    app.add_api_route("/api/marketplace/search", _api_marketplace_search, methods=["POST"])
    app.add_api_route("/api/marketplace/browse", _api_marketplace_browse, methods=["GET"])
    app.add_api_route("/api/marketplace/install", _api_marketplace_install, methods=["POST"])
    app.add_api_route("/api/marketplace/stats", _api_marketplace_stats, methods=["GET"])
    # Legacy aliases
    app.add_api_route("/api/skillhub/search", _api_marketplace_search, methods=["POST"])
    app.add_api_route("/api/skillhub/browse", _api_marketplace_browse, methods=["GET"])
    app.add_api_route("/api/skillhub/install", _api_marketplace_install, methods=["POST"])
    app.add_api_route("/api/channels", _api_channels_status, methods=["GET"])
    app.add_api_route("/api/channels/restart", _api_channels_restart, methods=["POST"])
    app.add_api_route("/api/files/clear", _api_clear_files, methods=["POST"])
    app.add_api_route("/api/files", _api_list_files, methods=["GET"])
    app.add_websocket_route("/ws/chat", _ws_chat)

    return app


def _get_agent() -> Agent | None:
    """Lazy-init the shared web agent with persistent sessions."""
    global _agent
    if _agent is not None:
        return _agent
    if _provider is None:
        return None
    try:
        verbose = config.get("agent", "verbose", default=False)
        _agent = PersistentAgent(
            provider=_provider,
            verbose=bool(verbose),
            store=_store,
            session_id=WEB_SESSION_ID,
        )
    except Exception as exc:
        logger.warning("[Web] Agent init failed: %s", exc)
        return None
    return _agent


def _reset_agent() -> None:
    """Discard the current agent so the next call rebuilds it."""
    global _agent
    _agent = None


# ── HTML ──────────────────────────────────────────────────────────────────────

async def _serve_index():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ── REST API ──────────────────────────────────────────────────────────────────

def _mask_secrets(obj: Any, _parent_key: str = "") -> Any:
    """Recursively mask values whose key contains 'apikey' or 'token'."""
    if isinstance(obj, dict):
        return {k: _mask_secrets(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_secrets(v) for v in obj]
    if isinstance(obj, str) and obj:
        key_lower = _parent_key.lower()
        if any(s in key_lower for s in ("apikey", "token", "secret", "password")):
            if len(obj) > 8:
                return obj[:4] + "*" * (len(obj) - 8) + obj[-4:]
            return "****"
    return obj


def _secret_keys_present(obj: Any, _parent_key: str = "") -> dict[str, str]:
    """Walk config and return a flat map of dotted-key → value for secret fields."""
    result: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{_parent_key}.{k}" if _parent_key else k
            if isinstance(v, (dict, list)):
                result.update(_secret_keys_present(v, full))
            elif isinstance(v, str) and v:
                if any(s in k.lower() for s in ("apikey", "token", "secret", "password")):
                    result[full] = v
    return result


_MASKED_PLACEHOLDER = "••••••••"


async def _api_config_get():
    raw = config.as_dict()
    masked = _mask_secrets(copy.deepcopy(raw))
    cfg_path = config.config_path()

    # Build a list of which secret fields have a value set (without revealing them)
    secrets_set = {k: True for k in _secret_keys_present(raw)}

    return {
        "config": masked,
        "configPath": str(cfg_path) if cfg_path else None,
        "providerReady": _provider is not None,
        "secretsSet": secrets_set,
    }


def _deep_set(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _deep_get_raw(d: dict, keys: list[str]) -> Any:
    """Get a value from a nested dict using a list of keys."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


async def _api_config_save(request: Request):
    """Save new configuration to pythonclaw.json and hot-reload the provider.

    Secret fields that arrive as the masked placeholder or empty string
    are preserved from the existing config (not overwritten).
    """
    global _provider

    try:
        body = await request.json()
        new_config = body.get("config")
        if not isinstance(new_config, dict):
            return JSONResponse({"ok": False, "error": "Invalid config object."}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    # Merge: for any secret field that is still the placeholder or empty,
    # keep the original value from the current config.
    existing = config.as_dict()
    existing_secrets = _secret_keys_present(existing)
    for dotted_key, original_value in existing_secrets.items():
        keys = dotted_key.split(".")
        incoming = _deep_get_raw(new_config, keys)
        if incoming is None or incoming == "" or incoming == _MASKED_PLACEHOLDER or "****" in str(incoming):
            _deep_set(new_config, keys, original_value)

    cfg_path = config.config_path()
    if cfg_path is None:
        cfg_path = config.PYTHONCLAW_HOME / "pythonclaw.json"

    try:
        json_text = json.dumps(new_config, indent=2, ensure_ascii=False)
        cfg_path.write_text(json_text + "\n", encoding="utf-8")
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Write failed: {exc}"}, status_code=500)

    config.load(str(cfg_path), force=True)
    logger.info("[Web] Config saved to %s", cfg_path)

    _reset_agent()
    if _build_provider_fn:
        try:
            _provider = _build_provider_fn()
            logger.info("[Web] Provider rebuilt successfully.")
        except Exception as exc:
            logger.warning("[Web] Provider rebuild failed: %s", exc)
            _provider = None

    channels_started = await _maybe_start_channels()

    return {
        "ok": True,
        "configPath": str(cfg_path),
        "providerReady": _provider is not None,
        "channelsStarted": channels_started,
    }


async def _api_skills():
    agent = _get_agent()
    if agent is None:
        try:
            pkg_templates = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "templates", "skills",
            )
            skills_dirs = [pkg_templates, os.path.join(str(config.PYTHONCLAW_HOME), "context", "skills")]
            skills_dirs = [d for d in skills_dirs if os.path.isdir(d)]
            registry = SkillRegistry(skills_dirs=skills_dirs)
            skills_meta = registry.discover()
        except Exception:
            return {"total": 0, "categories": {}}
    else:
        registry = agent._registry
        skills_meta = registry.discover()

    categories: dict[str, list] = {}
    for sm in skills_meta:
        cat = sm.category or "uncategorised"
        categories.setdefault(cat, []).append({
            "name": sm.name,
            "description": sm.description,
            "category": cat,
            "path": sm.path,
            "emoji": sm.emoji,
        })

    cat_meta = {}
    for cat_key, cat_obj in registry.categories.items():
        cat_meta[cat_key] = {
            "name": cat_obj.name,
            "description": cat_obj.description,
            "emoji": cat_obj.emoji,
        }

    return {"total": len(skills_meta), "categories": categories, "categoryMeta": cat_meta}


async def _api_status():
    uptime = int(time.time() - _start_time)
    provider_name = config.get_str("llm", "provider", env="LLM_PROVIDER", default="deepseek")

    agent = _get_agent()
    if agent is None:
        return {
            "provider": "Not configured",
            "providerName": provider_name,
            "providerReady": False,
            "skillsLoaded": 0,
            "skillsTotal": 0,
            "memoryCount": 0,
            "historyLength": 0,
            "compactionCount": 0,
            "uptimeSeconds": uptime,
            "webSearchEnabled": False,
        }

    session_file = _store._path(WEB_SESSION_ID) if _store else None
    return {
        "provider": type(agent.provider).__name__,
        "providerName": provider_name,
        "providerReady": True,
        "skillsLoaded": len(agent.loaded_skill_names),
        "skillsTotal": len(agent._registry.discover()),
        "memoryCount": len(agent.memory.list_all()),
        "historyLength": len(agent.messages),
        "compactionCount": agent.compaction_count,
        "uptimeSeconds": uptime,
        "webSearchEnabled": agent._web_search_enabled,
        "sessionFile": session_file,
        "sessionPersistent": True,
    }


async def _api_memories():
    agent = _get_agent()
    if agent is None:
        return {"total": 0, "memories": []}
    memories = agent.memory.list_all()
    return {"total": len(memories), "memories": memories}


async def _api_identity():
    """Return soul, persona content, and the full tool list."""
    from ..core.tools import (
        CRON_TOOLS,
        KNOWLEDGE_TOOL,
        MEMORY_TOOLS,
        META_SKILL_TOOLS,
        PRIMITIVE_TOOLS,
        SKILL_TOOLS,
        WEB_SEARCH_TOOL,
    )

    def _read_md(directory: str) -> str | None:
        p = Path(directory)
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix in (".md", ".txt") and f.is_file():
                    return f.read_text(encoding="utf-8").strip()
        return None

    home = config.PYTHONCLAW_HOME
    soul = _read_md(str(home / "context" / "soul"))
    persona = _read_md(str(home / "context" / "persona"))
    tools_notes = _read_md(str(home / "context" / "tools"))
    index_file = home / "context" / "memory" / "INDEX.md"
    index_content = None
    if index_file.is_file():
        try:
            index_content = index_file.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    def _tool_info(schema: dict) -> dict:
        fn = schema.get("function", {})
        return {"name": fn.get("name", ""), "description": fn.get("description", "")}

    tools = []
    tool_groups = [
        ("Primitive", PRIMITIVE_TOOLS),
        ("Skills", SKILL_TOOLS),
        ("Meta", META_SKILL_TOOLS),
        ("Memory", MEMORY_TOOLS),
        ("Cron", CRON_TOOLS),
    ]
    for group, schemas in tool_groups:
        for s in schemas:
            info = _tool_info(s)
            info["group"] = group
            tools.append(info)

    tools.append({**_tool_info(WEB_SEARCH_TOOL), "group": "Search"})
    tools.append({**_tool_info(KNOWLEDGE_TOOL), "group": "Knowledge"})

    return {
        "soul": soul,
        "persona": persona,
        "toolsNotes": tools_notes,
        "indexContent": index_content,
        "soulConfigured": soul is not None,
        "personaConfigured": persona is not None,
        "toolsNotesConfigured": tools_notes is not None,
        "indexConfigured": index_content is not None,
        "tools": tools,
    }


async def _api_save_soul(request: Request):
    """Save soul content to context/soul/SOUL.md and reload agent identity."""
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"ok": False, "error": "Content cannot be empty."}, status_code=400)

        soul_dir = config.PYTHONCLAW_HOME / "context" / "soul"
        soul_dir.mkdir(parents=True, exist_ok=True)
        soul_file = soul_dir / "SOUL.md"
        soul_file.write_text(content + "\n", encoding="utf-8")
        logger.info("[Web] Soul saved to %s", soul_file)

        _reload_agent_identity()
        return {"ok": True, "path": str(soul_file)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_save_persona(request: Request):
    """Save persona content to context/persona/persona.md and reload agent identity."""
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"ok": False, "error": "Content cannot be empty."}, status_code=400)

        persona_dir = config.PYTHONCLAW_HOME / "context" / "persona"
        persona_dir.mkdir(parents=True, exist_ok=True)
        persona_file = persona_dir / "persona.md"
        persona_file.write_text(content + "\n", encoding="utf-8")
        logger.info("[Web] Persona saved to %s", persona_file)

        _reload_agent_identity()
        return {"ok": True, "path": str(persona_file)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_get_tools_notes():
    """Return the current TOOLS.md content."""
    tools_dir = config.PYTHONCLAW_HOME / "context" / "tools"
    content = None
    if tools_dir.is_dir():
        for f in sorted(tools_dir.iterdir()):
            if f.suffix in (".md", ".txt") and f.is_file():
                content = f.read_text(encoding="utf-8").strip()
                break
    elif tools_dir.is_file():
        content = tools_dir.read_text(encoding="utf-8").strip()
    return {"ok": True, "content": content}


async def _api_save_tools_notes(request: Request):
    """Save TOOLS.md content and reload agent identity."""
    try:
        body = await request.json()
        content = body.get("content", "").strip()

        tools_dir = config.PYTHONCLAW_HOME / "context" / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        tools_file = tools_dir / "TOOLS.md"
        tools_file.write_text(content + "\n", encoding="utf-8")
        logger.info("[Web] TOOLS.md saved to %s", tools_file)

        _reload_agent_identity()
        return {"ok": True, "path": str(tools_file)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_get_index():
    """Return the INDEX.md curated system info content."""
    index_path = config.PYTHONCLAW_HOME / "context" / "memory" / "INDEX.md"
    content = ""
    if index_path.is_file():
        try:
            content = index_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return {"content": content, "path": str(index_path)}


async def _api_save_index(request: Request):
    """Save INDEX.md content and refresh agent memory."""
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        index_dir = config.PYTHONCLAW_HOME / "context" / "memory"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / "INDEX.md"
        index_file.write_text(content + "\n", encoding="utf-8")
        logger.info("[Web] INDEX.md saved to %s", index_file)

        agent = _get_agent()
        if agent is not None:
            agent.memory.storage._load()
            agent._init_system_prompt()

        return {"ok": True, "path": str(index_file)}
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc)}, status_code=500
        )


async def _api_transcribe(request: Request):
    """Proxy audio to Deepgram STT and return transcript."""
    from ..core.stt import no_key_message, transcribe_bytes_async

    content_type = request.headers.get("content-type", "audio/webm")
    body = await request.body()
    if not body:
        return JSONResponse({"ok": False, "error": "No audio data received."}, status_code=400)

    try:
        transcript = await transcribe_bytes_async(body, content_type)
    except Exception as exc:
        logger.warning("[Web] Deepgram error: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)

    if transcript is None:
        return JSONResponse({"ok": False, "error": no_key_message()}, status_code=400)

    return {"ok": True, "transcript": transcript}


async def _api_marketplace_search(request: Request):
    """Search ClawHub marketplace."""
    from ..core import skillhub

    try:
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse({"ok": False, "error": "Query is required."}, status_code=400)
        limit = int(body.get("limit", 10))
        results = await skillhub.search_async(query, limit=limit)
        return {"ok": True, "results": results}
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_marketplace_browse(request: Request):
    """Browse ClawHub catalog."""
    from ..core import skillhub

    try:
        limit = int(request.query_params.get("limit", 20))
        sort = request.query_params.get("sort", "score")
        results = await skillhub.browse_async(limit=limit, sort=sort)
        return {"ok": True, "results": results}
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_marketplace_install(request: Request):
    """Install a skill from ClawHub and hot-reload into the running agent."""
    from ..core import skillhub

    try:
        body = await request.json()
        skill_id = body.get("skill_id", "").strip()
        if not skill_id:
            return JSONResponse({"ok": False, "error": "skill_id is required."}, status_code=400)

        path = await skillhub.install_skill_async(skill_id)

        agent = _get_agent()
        skill_count = 0
        installed_name = ""
        if agent is not None:
            agent._refresh_skill_registry()
            skill_count = len(agent._registry.discover())
            for sm in agent._registry.discover():
                if sm.path == path:
                    installed_name = sm.name
                    break

        if not installed_name:
            import re as _re
            md_path = os.path.join(path, "SKILL.md")
            try:
                md_text = open(md_path, encoding="utf-8").read()
                m = _re.search(r"^name:\s*(.+)$", md_text, _re.MULTILINE)
                installed_name = m.group(1).strip() if m else skill_id
            except OSError:
                installed_name = skill_id

        return {
            "ok": True,
            "path": path,
            "skill_name": installed_name,
            "skill_count": skill_count,
            "message": f"Skill '{installed_name}' installed and ready to use.",
        }
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_marketplace_stats(request: Request):
    """Get ClawHub marketplace statistics."""
    from ..core import skillhub

    try:
        result = await skillhub.verify_api_async()
        return result
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _maybe_start_channels() -> list[str]:
    """Start channels whose tokens are now configured but not yet running."""
    global _active_bots
    if _provider is None:
        return []

    wanted = []
    tg_token = config.get_str("channels", "telegram", "token", default="")
    if tg_token:
        wanted.append("telegram")
    dc_token = config.get_str("channels", "discord", "token", default="")
    if dc_token:
        wanted.append("discord")
    wa_phone = config.get_str("channels", "whatsapp", "phoneNumberId", default="")
    wa_token = config.get_str("channels", "whatsapp", "token", default="")
    if wa_phone and wa_token:
        wanted.append("whatsapp")

    if not wanted:
        return []

    running_types = set()
    for bot in _active_bots:
        cls_name = type(bot).__name__.lower()
        if "telegram" in cls_name:
            running_types.add("telegram")
        elif "discord" in cls_name:
            running_types.add("discord")
        elif "whatsapp" in cls_name:
            running_types.add("whatsapp")

    to_start = [ch for ch in wanted if ch not in running_types]
    if not to_start:
        return list(running_types)

    try:
        from ..server import start_channels
        new_bots = await start_channels(_provider, to_start, fastapi_app=_fastapi_app)
        _active_bots.extend(new_bots)
        return [ch for ch in wanted if ch in running_types or ch in to_start]
    except Exception as exc:
        logger.warning("[Web] Channel start failed: %s", exc)
        return list(running_types)


async def _api_channels_status():
    """Return status of messaging channels."""
    channels = []
    for bot in _active_bots:
        cls_name = type(bot).__name__
        if "Telegram" in cls_name:
            ch_type = "telegram"
        elif "Discord" in cls_name:
            ch_type = "discord"
        elif "WhatsApp" in cls_name:
            ch_type = "whatsapp"
        else:
            ch_type = cls_name
        channels.append({"type": ch_type, "running": True})

    running_types = {c["type"] for c in channels}

    tg_token = config.get_str("channels", "telegram", "token", default="")
    dc_token = config.get_str("channels", "discord", "token", default="")
    wa_phone = config.get_str("channels", "whatsapp", "phoneNumberId", default="")
    wa_token = config.get_str("channels", "whatsapp", "token", default="")

    if tg_token and "telegram" not in running_types:
        channels.append({"type": "telegram", "running": False, "tokenSet": True})
    if dc_token and "discord" not in running_types:
        channels.append({"type": "discord", "running": False, "tokenSet": True})
    if wa_phone and wa_token and "whatsapp" not in running_types:
        channels.append({"type": "whatsapp", "running": False, "tokenSet": True})

    return {"channels": channels}


async def _api_channels_restart(request: Request):
    """Stop and restart all configured channels."""
    global _active_bots

    for bot in _active_bots:
        if hasattr(bot, "stop_async"):
            try:
                await bot.stop_async()
            except Exception:
                pass
    _active_bots = []

    started = await _maybe_start_channels()
    return {"ok": True, "channels": started}


def _reload_agent_identity() -> None:
    """Reload the agent's soul/persona/tools from disk without full reset."""
    global _agent
    if _agent is None:
        return
    from ..core.agent import _load_text_dir_or_file
    home = config.PYTHONCLAW_HOME
    _agent.soul_instruction = _load_text_dir_or_file(
        str(home / "context" / "soul"), label="Soul"
    )
    _agent.persona_instruction = _load_text_dir_or_file(
        str(home / "context" / "persona"), label="Persona"
    )
    _agent.tools_notes = _load_text_dir_or_file(
        str(home / "context" / "tools"), label="Tools"
    )
    _agent._needs_onboarding = False
    _agent._init_system_prompt()


# ── Files management ──────────────────────────────────────────────────────────

async def _api_clear_files(request: Request):
    """Delete all downloaded/generated files."""
    count = config.clear_files()
    return JSONResponse({"ok": True, "cleared": count})


async def _api_list_files(request: Request):
    """List files in the shared files directory."""
    d = config.files_dir()
    files = []
    for entry in sorted(d.iterdir()):
        if entry.is_file():
            files.append({
                "name": entry.name,
                "size": entry.stat().st_size,
                "modified": entry.stat().st_mtime,
            })
    return JSONResponse({"files": files, "dir": str(d)})


# ── Web file sender ───────────────────────────────────────────────────────────

def _register_web_file_sender(loop: asyncio.AbstractEventLoop, ws: WebSocket) -> None:
    """Register a sync callback so the Agent can push file-download links to the web UI."""
    from ..core.tools import set_file_sender

    def _sender(path: str, caption: str = "") -> None:
        import base64 as _b64

        name = os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            data = _b64.b64encode(fh.read()).decode()

        async def _push():
            try:
                await ws.send_json({
                    "type": "file",
                    "filename": name,
                    "size": size,
                    "caption": caption,
                    "data": data,
                })
            except Exception as exc:
                logger.warning("[Web] send_file via WS failed: %s", exc)

        future = asyncio.run_coroutine_threadsafe(_push(), loop)
        future.result(timeout=60)

    set_file_sender(_sender)


# ── WebSocket Chat ────────────────────────────────────────────────────────────

async def _ws_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("[Web] WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                message = payload.get("message", "").strip()
                image_data = payload.get("image")  # data:image/...;base64,...
            except (json.JSONDecodeError, AttributeError):
                message = data.strip()
                image_data = None

            if not message and not image_data:
                continue

            agent = _get_agent()
            if agent is None:
                await websocket.send_json({
                    "type": "error",
                    "content": "LLM provider is not configured yet. Go to the Config tab and set your API key, then save.",
                })
                continue

            if message.startswith("/compact"):
                hint = message[len("/compact"):].strip() or None
                result = agent.compact(instruction=hint)
                await websocket.send_json({"type": "response", "content": result})
                continue

            if message == "/status":
                status = await _api_status()
                await websocket.send_json({"type": "response", "content": json.dumps(status, indent=2)})
                continue

            if message == "/clear":
                if _store:
                    _store.delete(WEB_SESSION_ID)
                if agent is not None:
                    agent.clear_history()
                await websocket.send_json({"type": "response", "content": "Chat history cleared. Agent is still active with all skills and memory intact."})
                continue

            lock = _get_chat_lock()
            if lock.locked():
                await websocket.send_json({"type": "thinking", "content": "Processing previous message\u2026"})
            else:
                await websocket.send_json({"type": "thinking", "content": ""})

            loop = asyncio.get_event_loop()

            _register_web_file_sender(loop, websocket)

            try:
                token_queue: asyncio.Queue[str | None] = asyncio.Queue()

                def _on_token(text: str) -> None:
                    loop.call_soon_threadsafe(token_queue.put_nowait, text)

                async def _stream_tokens() -> None:
                    while True:
                        tok = await token_queue.get()
                        if tok is None:
                            break
                        try:
                            await websocket.send_json(
                                {"type": "stream", "content": tok}
                            )
                        except Exception:
                            break

                # Build multimodal input if image is attached
                chat_input: str | list = message or ""
                if image_data:
                    chat_input = [
                        {"type": "text", "text": message or "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ]

                async with lock:
                    stream_task = asyncio.create_task(_stream_tokens())
                    try:
                        response = await loop.run_in_executor(
                            None, agent.chat_stream, chat_input, _on_token
                        )
                    finally:
                        loop.call_soon_threadsafe(
                            token_queue.put_nowait, None
                        )
                        await stream_task
                await websocket.send_json(
                    {"type": "response", "content": response}
                )
            except Exception as exc:
                logger.exception("[Web] Chat error")
                await websocket.send_json({"type": "error", "content": str(exc)})

    except WebSocketDisconnect:
        logger.info("[Web] WebSocket client disconnected")
    except Exception:
        logger.exception("[Web] WebSocket error")
