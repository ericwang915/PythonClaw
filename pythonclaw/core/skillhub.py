"""
Skill marketplace client for PythonClaw.

Primary source: ClawHub (https://topclawhubskills.com/api) — the OpenClaw
public skills registry.  Free, no API key required, 13K+ skills.

All endpoints are unauthenticated and return JSON directly.

Available ClawHub endpoints
---------------------------
  GET /api/search?q=TERM   — free-text search
  GET /api/top-downloads   — most downloaded skills
  GET /api/top-stars       — most starred skills
  GET /api/newest          — recently published skills
  GET /api/certified       — security-verified skills
  GET /api/stats           — platform statistics
  GET /api/health          — API status

Skill download (full ZIP with SKILL.md + assets):
  GET https://wry-manatee-359.convex.site/api/v1/download?slug=SLUG
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request
import zipfile
from typing import Any

logger = logging.getLogger(__name__)

CLAWHUB_API = "https://topclawhubskills.com/api"
CLAWHUB_WEB = "https://clawhub.com"
CLAWHUB_DOWNLOAD = "https://wry-manatee-359.convex.site/api/v1/download"


def _get_ssl_ctx() -> ssl.SSLContext:
    """Build an SSL context; use unverified fallback for macOS cert issues."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict:
    """Make a GET request to the ClawHub API (no auth required)."""
    url = f"{CLAWHUB_API}{path}"
    if params:
        qs = "&".join(
            f"{k}={urllib.request.quote(str(v))}"
            for k, v in params.items() if v is not None
        )
        if qs:
            url = f"{url}?{qs}"

    headers = {"User-Agent": "PythonClaw/1.0", "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15, context=_get_ssl_ctx()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        logger.warning("ClawHub API error %s: %s", exc.code, err_body)
        raise RuntimeError(f"ClawHub API error ({exc.code}): {err_body}") from exc
    except Exception as exc:
        raise RuntimeError(f"ClawHub request failed: {exc}") from exc


def _normalize(skills: list[dict]) -> list[dict]:
    """Normalize ClawHub skill records to a consistent format."""
    results: list[dict] = []
    for s in skills:
        results.append({
            "id": s.get("slug", ""),
            "name": s.get("display_name", s.get("slug", "")),
            "description": s.get("summary", "")[:160],
            "author": s.get("owner_handle", ""),
            "downloads": s.get("downloads", 0),
            "stars": s.get("stars", 0),
            "certified": s.get("is_certified", False),
            "source_url": f"{CLAWHUB_WEB}/skills/{s.get('slug', '')}",
        })
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def search(query: str, *, limit: int = 10, **_kwargs: Any) -> list[dict]:
    """Search ClawHub for skills matching a query."""
    result = _api_get("/search", params={"q": query})
    data = result.get("data", [])
    return _normalize(data[:limit])


def browse(
    *,
    limit: int = 20,
    sort: str = "score",
    **_kwargs: Any,
) -> list[dict]:
    """Browse the ClawHub catalog.

    *sort* maps to ClawHub endpoints:
      - "score" / "downloads" → /top-downloads
      - "stars"               → /top-stars
      - "recent" / "newest"   → /newest
      - "certified"           → /certified
    """
    endpoint_map = {
        "score": "/top-downloads",
        "downloads": "/top-downloads",
        "composite": "/top-downloads",
        "stars": "/top-stars",
        "recent": "/newest",
        "newest": "/newest",
        "certified": "/certified",
    }
    endpoint = endpoint_map.get(sort, "/top-downloads")
    result = _api_get(endpoint)
    data = result.get("data", [])
    return _normalize(data[:limit])


def get_skill_detail(skill_id: str) -> dict | None:
    """Fetch metadata for a skill from ClawHub search API."""
    try:
        result = _api_get("/search", params={"q": skill_id})
        data = result.get("data", [])
        for s in data:
            if s.get("slug") == skill_id:
                return _normalize([s])[0]

        if data:
            return _normalize([data[0]])[0]
    except Exception as exc:
        logger.warning("ClawHub detail fetch failed for '%s': %s", skill_id, exc)

    return None


def stats() -> dict:
    """Get ClawHub platform statistics."""
    result = _api_get("/stats")
    return result.get("data", result)


def verify_api() -> dict:
    """Verify ClawHub API is reachable (no key needed).

    Returns ``{"ok": True, ...}`` on success.
    """
    try:
        result = _api_get("/health")
        if result.get("ok"):
            count = result.get("skill_count", "?")
            return {
                "ok": True,
                "message": f"ClawHub API is online ({count} skills available).",
            }
        return {"ok": False, "error": "Unexpected response from ClawHub API."}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


# ── Install ───────────────────────────────────────────────────────────────────

def _download_skill_zip(slug: str) -> bytes:
    """Download the full skill ZIP from ClawHub's Convex CDN."""
    url = f"{CLAWHUB_DOWNLOAD}?slug={urllib.request.quote(slug)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "PythonClaw/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_get_ssl_ctx()) as resp:
            return resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download skill '{slug}' from ClawHub: {exc}"
        ) from exc


def install_skill(
    skill_id: str,
    *,
    target_dir: str | None = None,
    skill_md_override: str | None = None,
) -> str:
    """Download and install a skill from ClawHub into the local skills directory.

    Downloads the full ZIP archive from ClawHub (contains SKILL.md plus
    any assets, scripts, references, etc.) and extracts it.

    Returns the path to the installed skill directory.
    """
    if target_dir is None:
        from .. import config as _cfg
        target_dir = os.path.join(str(_cfg.PYTHONCLAW_HOME), "context", "skills")

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", skill_id).strip("_")
    if not safe_name:
        safe_name = "imported_skill"

    category = "clawhub"
    skill_dir = os.path.join(target_dir, category, safe_name)
    os.makedirs(skill_dir, exist_ok=True)

    if skill_md_override:
        md_path = os.path.join(skill_dir, "SKILL.md")
        md = skill_md_override
        if not md.startswith("---"):
            md = f"---\nname: {safe_name}\ndescription: Imported from ClawHub ({skill_id})\n---\n\n{md}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    else:
        raw_zip = _download_skill_zip(skill_id)
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw_zip))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"ClawHub returned invalid ZIP for '{skill_id}'."
            ) from exc

        for member in zf.namelist():
            if member.startswith("__MACOSX") or member.startswith("."):
                continue
            dest = os.path.join(skill_dir, member)
            if member.endswith("/"):
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(zf.read(member))

        if not os.path.exists(os.path.join(skill_dir, "SKILL.md")):
            logger.warning("No SKILL.md found in ZIP for '%s'", skill_id)

    source_url = f"{CLAWHUB_WEB}/skills/{skill_id}"
    meta_path = os.path.join(skill_dir, ".clawhub.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"id": skill_id, "source": source_url, "installed_by": "pythonclaw"},
            f, indent=2,
        )

    return skill_dir


# ── Async variants (httpx) ────────────────────────────────────────────────────

async def _api_get_async(path: str, params: dict[str, Any] | None = None) -> dict:
    """Non-blocking GET request to the ClawHub API using httpx."""
    import httpx

    url = f"{CLAWHUB_API}{path}"
    if params:
        qs = "&".join(
            f"{k}={urllib.request.quote(str(v))}"
            for k, v in params.items() if v is not None
        )
        if qs:
            url = f"{url}?{qs}"

    try:
        async with httpx.AsyncClient(
            verify=False, timeout=15.0,
        ) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "PythonClaw/1.0", "Accept": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300]
        logger.warning("ClawHub API error %s: %s", exc.response.status_code, body)
        raise RuntimeError(
            f"ClawHub API error ({exc.response.status_code}): {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"ClawHub request failed: {exc}") from exc


async def _download_skill_zip_async(slug: str) -> bytes:
    """Non-blocking download of the full skill ZIP from ClawHub's Convex CDN."""
    import httpx

    url = f"{CLAWHUB_DOWNLOAD}?slug={urllib.request.quote(slug)}"
    try:
        async with httpx.AsyncClient(
            verify=False, timeout=30.0,
        ) as client:
            resp = await client.get(
                url, headers={"User-Agent": "PythonClaw/1.0"},
            )
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download skill '{slug}' from ClawHub: {exc}"
        ) from exc


async def search_async(query: str, *, limit: int = 10) -> list[dict]:
    """Async version of :func:`search`."""
    result = await _api_get_async("/search", params={"q": query})
    data = result.get("data", [])
    return _normalize(data[:limit])


async def browse_async(*, limit: int = 20, sort: str = "score") -> list[dict]:
    """Async version of :func:`browse`."""
    endpoint_map = {
        "score": "/top-downloads",
        "downloads": "/top-downloads",
        "composite": "/top-downloads",
        "stars": "/top-stars",
        "recent": "/newest",
        "newest": "/newest",
        "certified": "/certified",
    }
    endpoint = endpoint_map.get(sort, "/top-downloads")
    result = await _api_get_async(endpoint)
    data = result.get("data", [])
    return _normalize(data[:limit])


async def verify_api_async() -> dict:
    """Async version of :func:`verify_api`."""
    try:
        result = await _api_get_async("/health")
        if result.get("ok"):
            count = result.get("skill_count", "?")
            return {
                "ok": True,
                "message": f"ClawHub API is online ({count} skills available).",
            }
        return {"ok": False, "error": "Unexpected response from ClawHub API."}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


async def install_skill_async(
    skill_id: str,
    *,
    target_dir: str | None = None,
) -> str:
    """Async version of :func:`install_skill`.

    Uses httpx for the HTTP download; file extraction is fast local I/O.
    """
    if target_dir is None:
        from .. import config as _cfg
        target_dir = os.path.join(str(_cfg.PYTHONCLAW_HOME), "context", "skills")

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", skill_id).strip("_") or "imported_skill"

    category = "clawhub"
    skill_dir = os.path.join(target_dir, category, safe_name)
    os.makedirs(skill_dir, exist_ok=True)

    raw_zip = await _download_skill_zip_async(skill_id)
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_zip))
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"ClawHub returned invalid ZIP for '{skill_id}'."
        ) from exc

    for member in zf.namelist():
        if member.startswith("__MACOSX") or member.startswith("."):
            continue
        dest = os.path.join(skill_dir, member)
        if member.endswith("/"):
            os.makedirs(dest, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(zf.read(member))

    if not os.path.exists(os.path.join(skill_dir, "SKILL.md")):
        logger.warning("No SKILL.md found in ZIP for '%s'", skill_id)

    source_url = f"{CLAWHUB_WEB}/skills/{skill_id}"
    meta_path = os.path.join(skill_dir, ".clawhub.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"id": skill_id, "source": source_url, "installed_by": "pythonclaw"},
            f, indent=2,
        )

    return skill_dir


# ── Formatting ────────────────────────────────────────────────────────────────

def format_search_results(results: list[dict]) -> str:
    """Format search results for CLI display."""
    if not results:
        return "No skills found."

    lines = []
    for i, r in enumerate(results, 1):
        name = r.get("name", r.get("title", "???"))
        desc = r.get("description", "")[:80]
        sid = r.get("id", r.get("slug", ""))
        downloads = r.get("downloads", "")
        stars = r.get("stars", "")
        certified = r.get("certified", False)

        header = f"  {i}. {name}"
        if downloads:
            header += f"  ↓{downloads:,}" if isinstance(downloads, int) else f"  ↓{downloads}"
        if stars:
            header += f"  ★{stars}"
        if certified:
            header += "  ✓certified"

        lines.append(header)
        if desc:
            lines.append(f"     {desc}")
        if sid:
            lines.append(f"     ID: {sid}")
        lines.append("")

    return "\n".join(lines)
