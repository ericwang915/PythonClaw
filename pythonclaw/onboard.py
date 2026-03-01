"""
Interactive onboarding wizard for PythonClaw.

Guides a new user through LLM provider selection, API key entry,
and optional service key configuration.  Writes pythonclaw.json.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path

from . import config

# ── ANSI helpers (no external deps) ──────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


# ── Provider definitions ─────────────────────────────────────────────────────

PROVIDERS = [
    {
        "key": "deepseek",
        "name": "DeepSeek",
        "default_model": "deepseek-chat",
        "default_base": "https://api.deepseek.com/v1",
        "env": "DEEPSEEK_API_KEY",
    },
    {
        "key": "grok",
        "name": "Grok (xAI)",
        "default_model": "grok-3",
        "default_base": "https://api.x.ai/v1",
        "env": "GROK_API_KEY",
    },
    {
        "key": "claude",
        "name": "Claude (Anthropic)",
        "default_model": "claude-sonnet-4-20250514",
        "default_base": None,
        "env": "ANTHROPIC_API_KEY",
    },
    {
        "key": "gemini",
        "name": "Gemini (Google)",
        "default_model": "gemini-2.0-flash",
        "default_base": None,
        "env": "GEMINI_API_KEY",
    },
    {
        "key": "kimi",
        "name": "Kimi (Moonshot)",
        "default_model": "moonshot-v1-128k",
        "default_base": "https://api.moonshot.cn/v1",
        "env": "KIMI_API_KEY",
    },
    {
        "key": "glm",
        "name": "GLM (Zhipu / ChatGLM)",
        "default_model": "glm-4-flash",
        "default_base": "https://open.bigmodel.cn/api/paas/v4/",
        "env": "GLM_API_KEY",
    },
]


# ── Core logic ───────────────────────────────────────────────────────────────

def run_onboard(config_path: str | None = None) -> Path:
    """Run the interactive onboarding wizard.  Returns path to saved config."""
    print()
    print(_c("  ╔══════════════════════════════════════╗", _CYAN))
    print(_c("  ║       PythonClaw — Setup Wizard      ║", _CYAN))
    print(_c("  ╚══════════════════════════════════════╝", _CYAN))
    print()

    # Load existing config if present
    cfg = _load_existing(config_path)

    # 1. Choose LLM provider
    provider = _choose_provider(cfg)

    # 2. Enter API key
    api_key = _get_api_key(provider, cfg)

    # 3. Update config
    prov = provider["key"]
    cfg.setdefault("llm", {})
    cfg["llm"]["provider"] = prov
    cfg["llm"].setdefault(prov, {})
    cfg["llm"][prov]["apiKey"] = api_key
    cfg["llm"][prov].setdefault("model", provider["default_model"])
    if provider["default_base"]:
        cfg["llm"][prov].setdefault("baseUrl", provider["default_base"])

    # 4. Optional keys
    _optional_keys(cfg)

    # 5. Validate
    _validate_key(cfg, provider)

    # 6. Save
    out_path = _save_config(cfg, config_path)

    print()
    print(_c("  ✔ Setup complete!", _GREEN))
    print(f"    Config saved to: {_c(str(out_path), _BOLD)}")
    print()
    return out_path


def _load_existing(config_path: str | None) -> dict:
    """Load existing config or return empty dict."""
    try:
        config.load(config_path)
        return config.as_dict()
    except Exception:
        return {}


def _choose_provider(cfg: dict) -> dict:
    current = cfg.get("llm", {}).get("provider", "")
    print(_c("  Choose your LLM provider:", _BOLD))
    print()
    for i, p in enumerate(PROVIDERS, 1):
        marker = _c(" (current)", _DIM) if p["key"] == current else ""
        print(f"    {_c(str(i), _CYAN)}. {p['name']}{marker}")
    print()

    while True:
        default_hint = ""
        if current:
            idx = next((i for i, p in enumerate(PROVIDERS) if p["key"] == current), None)
            if idx is not None:
                default_hint = f" [{idx + 1}]"

        choice = input(f"  Enter number (1-{len(PROVIDERS)}){default_hint}: ").strip()
        if not choice and current:
            return next(p for p in PROVIDERS if p["key"] == current)
        try:
            n = int(choice)
            if 1 <= n <= len(PROVIDERS):
                selected = PROVIDERS[n - 1]
                print(f"  → {_c(selected['name'], _GREEN)}")
                print()
                return selected
        except ValueError:
            pass
        print(_c("  Invalid choice, try again.", _RED))


def _get_api_key(provider: dict, cfg: dict) -> str:
    existing = cfg.get("llm", {}).get(provider["key"], {}).get("apiKey", "")
    has_existing = bool(existing) and existing != ""

    hint = ""
    if has_existing:
        masked = existing[:4] + "****" + existing[-4:] if len(existing) > 8 else "****"
        hint = f" (current: {masked}, press Enter to keep)"

    print(f"  {provider['name']} API Key{hint}")
    key = getpass.getpass("  API Key: ").strip()

    if not key and has_existing:
        print("  → Keeping existing key")
        return existing
    if not key:
        print(_c("  API key is required.", _RED))
        return _get_api_key(provider, cfg)

    print(f"  → Key set ({key[:4]}****)")
    print()
    return key


def _optional_keys(cfg: dict) -> None:
    print(_c("  Optional services (press Enter to skip):", _DIM))
    print()

    # Tavily
    tavily_existing = cfg.get("tavily", {}).get("apiKey", "")
    if not tavily_existing:
        tavily = input("  Tavily API Key (web search): ").strip()
        if tavily:
            cfg.setdefault("tavily", {})["apiKey"] = tavily
            print("  → Tavily key set")

    # Deepgram
    dg_existing = cfg.get("deepgram", {}).get("apiKey", "")
    if not dg_existing:
        dg = input("  Deepgram API Key (voice input): ").strip()
        if dg:
            cfg.setdefault("deepgram", {})["apiKey"] = dg
            print("  → Deepgram key set")

    # SkillHub
    sh_existing = cfg.get("skillhub", {}).get("apiKey", "")
    if not sh_existing:
        sh = input("  SkillHub API Key (marketplace): ").strip()
        if sh:
            cfg.setdefault("skillhub", {})["apiKey"] = sh
            print("  → SkillHub key set")

    print()
    _channel_keys(cfg)


def _channel_keys(cfg: dict) -> None:
    print(_c("  Channels (press Enter to skip):", _DIM))
    print()

    channels = cfg.setdefault("channels", {})

    # Telegram
    tg = channels.setdefault("telegram", {"token": "", "allowedUsers": []})
    tg_existing = tg.get("token", "")
    if tg_existing:
        masked = tg_existing[:6] + "****" + tg_existing[-4:] if len(tg_existing) > 10 else "****"
        print(f"  Telegram Bot Token (current: {masked}, press Enter to keep)")
    token = input("  Telegram Bot Token: ").strip()
    if token:
        tg["token"] = token
        print("  → Telegram token set")
    elif tg_existing:
        print("  → Keeping existing Telegram token")

    allowed = input("  Telegram Allowed User IDs (comma-separated, or Enter to allow all): ").strip()
    if allowed:
        tg["allowedUsers"] = [uid.strip() for uid in allowed.split(",") if uid.strip()]
        print(f"  → {len(tg['allowedUsers'])} user(s) whitelisted")

    print()

    # Discord
    dc = channels.setdefault("discord", {"token": "", "allowedUsers": [], "allowedChannels": []})
    dc_existing = dc.get("token", "")
    if dc_existing:
        masked = dc_existing[:6] + "****" + dc_existing[-4:] if len(dc_existing) > 10 else "****"
        print(f"  Discord Bot Token (current: {masked}, press Enter to keep)")
    dc_token = input("  Discord Bot Token: ").strip()
    if dc_token:
        dc["token"] = dc_token
        print("  → Discord token set")
    elif dc_existing:
        print("  → Keeping existing Discord token")

    dc_channels = input("  Discord Allowed Channel IDs (comma-separated, or Enter to allow all): ").strip()
    if dc_channels:
        dc["allowedChannels"] = [ch.strip() for ch in dc_channels.split(",") if ch.strip()]
        print(f"  → {len(dc['allowedChannels'])} channel(s) whitelisted")

    print()


def _validate_key(cfg: dict, provider: dict) -> None:
    """Make a quick test call to validate the API key."""
    print(f"  Validating {provider['name']} API key...", end=" ", flush=True)

    prov_key = provider["key"]
    api_key = cfg["llm"][prov_key]["apiKey"]

    try:
        if prov_key in ("deepseek", "grok", "kimi", "glm"):
            from .core.llm.openai_compatible import OpenAICompatibleProvider
            base_url = cfg["llm"][prov_key].get("baseUrl", provider["default_base"])
            model = cfg["llm"][prov_key].get("model", provider["default_model"])
            p = OpenAICompatibleProvider(api_key=api_key, base_url=base_url, model_name=model)
            p.chat([{"role": "user", "content": "hi"}], max_tokens=5)
        elif prov_key == "claude":
            from .core.llm.anthropic_client import AnthropicProvider
            p = AnthropicProvider(api_key=api_key)
            p.chat([{"role": "user", "content": "hi"}], max_tokens=5)
        elif prov_key == "gemini":
            from .core.llm.gemini_client import GeminiProvider
            p = GeminiProvider(api_key=api_key)
            p.chat([{"role": "user", "content": "hi"}], max_tokens=5)
        else:
            print(_c("skipped (unknown provider type)", _YELLOW))
            return

        print(_c("✔ Valid!", _GREEN))
    except Exception as exc:
        err_str = str(exc)
        if len(err_str) > 100:
            err_str = err_str[:100] + "..."
        print(_c(f"✘ {err_str}", _RED))
        print(_c("  You can fix this later in pythonclaw.json or the web dashboard.", _DIM))


def _save_config(cfg: dict, config_path: str | None) -> Path:
    """Write config to disk."""
    if config_path:
        out = Path(config_path)
    else:
        out = Path.cwd() / "pythonclaw.json"

    # Ensure default sections exist
    cfg.setdefault("channels", {"telegram": {"token": "", "allowedUsers": []}, "discord": {"token": "", "allowedUsers": [], "allowedChannels": []}})
    cfg.setdefault("tavily", {}).setdefault("apiKey", "")
    cfg.setdefault("deepgram", {}).setdefault("apiKey", "")
    cfg.setdefault("skillhub", {}).setdefault("apiKey", "")
    cfg.setdefault("heartbeat", {"intervalSec": 60, "alertChatId": None})
    cfg.setdefault("memory", {"dir": None})
    cfg.setdefault("web", {"host": "0.0.0.0", "port": 7788})
    cfg.setdefault("skills", {})
    cfg.setdefault("agent", {"autoCompactThreshold": 0, "verbose": True})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config.load(str(out), force=True)
    return out


def needs_onboard(config_path: str | None = None) -> bool:
    """Check if onboarding is needed (no config or no API key)."""
    try:
        config.load(config_path)
    except Exception:
        return True

    provider = config.get_str("llm", "provider", default="")
    if not provider:
        return True

    api_key = config.get_str("llm", provider, "apiKey", default="")
    return not api_key
