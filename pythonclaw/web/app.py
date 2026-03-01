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
from ..core.persistent_agent import PersistentAgent
from ..core.session_store import SessionStore
from ..core.llm.base import LLMProvider
from ..core.skill_loader import SkillRegistry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_agent: Agent | None = None
_provider: LLMProvider | None = None
_store: SessionStore | None = None
_start_time: float = 0.0
_build_provider_fn = None

WEB_SESSION_ID = "web:dashboard"


def create_app(provider: LLMProvider | None, *, build_provider_fn=None) -> FastAPI:
    """Build and return the FastAPI app.

    Parameters
    ----------
    provider          : LLM provider (may be None if not yet configured)
    build_provider_fn : callable that rebuilds the provider from config
                        (used after config save to hot-reload the provider)
    """
    global _provider, _store, _start_time, _build_provider_fn
    _provider = provider
    _store = SessionStore()
    _start_time = time.time()
    _build_provider_fn = build_provider_fn

    app = FastAPI(title="PythonClaw Dashboard", docs_url=None, redoc_url=None)

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
    app.add_api_route("/api/transcribe", _api_transcribe, methods=["POST"])
    app.add_api_route("/api/skillhub/search", _api_skillhub_search, methods=["POST"])
    app.add_api_route("/api/skillhub/browse", _api_skillhub_browse, methods=["GET"])
    app.add_api_route("/api/skillhub/install", _api_skillhub_install, methods=["POST"])
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
        cfg_path = Path.cwd() / "pythonclaw.json"

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

    return {"ok": True, "configPath": str(cfg_path), "providerReady": _provider is not None}


async def _api_skills():
    agent = _get_agent()
    if agent is None:
        try:
            pkg_templates = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "templates", "skills",
            )
            skills_dirs = [pkg_templates, os.path.join("context", "skills")]
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
        })

    return {"total": len(skills_meta), "categories": categories}


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
        PRIMITIVE_TOOLS, SKILL_TOOLS, META_SKILL_TOOLS,
        MEMORY_TOOLS, WEB_SEARCH_TOOL, KNOWLEDGE_TOOL, CRON_TOOLS,
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

    cwd = Path.cwd()
    soul = _read_md(str(cwd / "context" / "soul"))
    persona = _read_md(str(cwd / "context" / "persona"))

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
        "soulConfigured": soul is not None,
        "personaConfigured": persona is not None,
        "tools": tools,
    }


async def _api_save_soul(request: Request):
    """Save soul content to context/soul/SOUL.md and reload agent identity."""
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"ok": False, "error": "Content cannot be empty."}, status_code=400)

        soul_dir = Path.cwd() / "context" / "soul"
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

        persona_dir = Path.cwd() / "context" / "persona"
        persona_dir.mkdir(parents=True, exist_ok=True)
        persona_file = persona_dir / "persona.md"
        persona_file.write_text(content + "\n", encoding="utf-8")
        logger.info("[Web] Persona saved to %s", persona_file)

        _reload_agent_identity()
        return {"ok": True, "path": str(persona_file)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_transcribe(request: Request):
    """Proxy audio to Deepgram STT and return transcript."""
    import urllib.request
    import urllib.error

    dg_key = config.get("deepgram", "apiKey", env="DEEPGRAM_API_KEY")
    if not dg_key:
        return JSONResponse(
            {"ok": False, "error": "Deepgram API key not configured. Set it in Config."},
            status_code=400,
        )

    content_type = request.headers.get("content-type", "audio/webm")
    body = await request.body()
    if not body:
        return JSONResponse({"ok": False, "error": "No audio data received."}, status_code=400)

    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&detect_language=true"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Token {dg_key}",
            "Content-Type": content_type,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        logger.warning("[Web] Deepgram error %s: %s", exc.code, err_body)
        return JSONResponse(
            {"ok": False, "error": f"Deepgram API error ({exc.code})"},
            status_code=502,
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)

    try:
        transcript = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
    except (IndexError, KeyError):
        transcript = ""

    return {"ok": True, "transcript": transcript}


async def _api_skillhub_search(request: Request):
    """Search SkillHub marketplace."""
    from ..core import skillhub

    try:
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse({"ok": False, "error": "Query is required."}, status_code=400)
        limit = int(body.get("limit", 10))
        category = body.get("category")
        results = skillhub.search(query, limit=limit, category=category)
        return {"ok": True, "results": results}
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_skillhub_browse(request: Request):
    """Browse SkillHub catalog."""
    from ..core import skillhub

    try:
        limit = int(request.query_params.get("limit", 20))
        sort = request.query_params.get("sort", "score")
        category = request.query_params.get("category")
        results = skillhub.browse(limit=limit, sort=sort, category=category or None)
        return {"ok": True, "results": results}
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def _api_skillhub_install(request: Request):
    """Install a skill from SkillHub."""
    from ..core import skillhub

    try:
        body = await request.json()
        skill_id = body.get("skill_id", "").strip()
        if not skill_id:
            return JSONResponse({"ok": False, "error": "skill_id is required."}, status_code=400)

        path = skillhub.install_skill(skill_id)

        agent = _get_agent()
        if agent is not None:
            agent._refresh_skill_registry()

        return {"ok": True, "path": path, "message": f"Skill installed to {path}"}
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def _reload_agent_identity() -> None:
    """Reload the agent's soul/persona from disk without full reset."""
    global _agent
    if _agent is None:
        return
    from ..core.agent import _load_text_dir_or_file
    cwd = Path.cwd()
    _agent.soul_instruction = _load_text_dir_or_file(
        str(cwd / "context" / "soul"), label="Soul"
    )
    _agent.persona_instruction = _load_text_dir_or_file(
        str(cwd / "context" / "persona"), label="Persona"
    )
    _agent._needs_onboarding = False
    _agent._init_system_prompt()


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
            except (json.JSONDecodeError, AttributeError):
                message = data.strip()

            if not message:
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

            await websocket.send_json({"type": "thinking", "content": ""})

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, agent.chat, message)
                await websocket.send_json({"type": "response", "content": response})
            except Exception as exc:
                logger.exception("[Web] Chat error")
                await websocket.send_json({"type": "error", "content": str(exc)})

    except WebSocketDisconnect:
        logger.info("[Web] WebSocket client disconnected")
    except Exception:
        logger.exception("[Web] WebSocket error")
