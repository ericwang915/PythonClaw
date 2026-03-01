"""
PythonClaw CLI — entry point.

Subcommands
-----------
  onboard   Interactive first-time setup wizard
  start     Start the agent daemon (web dashboard + optional channels)
  stop      Stop the running daemon
  status    Show daemon status
  chat      Interactive CLI chat (foreground)
  skill     SkillHub marketplace (search / browse / install / info)
"""

import argparse
import asyncio
import logging

from . import config
from .core.persistent_agent import PersistentAgent
from .core.session_store import SessionStore


# ── Provider builder ─────────────────────────────────────────────────────────

def _build_provider():
    """Instantiate the LLM provider selected by config."""
    provider_name = config.get_str(
        "llm", "provider", env="LLM_PROVIDER", default="deepseek"
    ).lower()

    if provider_name == "deepseek":
        from .core.llm.openai_compatible import OpenAICompatibleProvider
        api_key = config.get_str("llm", "deepseek", "apiKey", env="DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set (env or pythonclaw.json)")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=config.get_str(
                "llm", "deepseek", "baseUrl", default="https://api.deepseek.com/v1",
            ),
            model_name=config.get_str(
                "llm", "deepseek", "model", default="deepseek-chat",
            ),
        )

    if provider_name == "grok":
        from .core.llm.openai_compatible import OpenAICompatibleProvider
        api_key = config.get_str("llm", "grok", "apiKey", env="GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY not set (env or pythonclaw.json)")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=config.get_str(
                "llm", "grok", "baseUrl", default="https://api.x.ai/v1",
            ),
            model_name=config.get_str(
                "llm", "grok", "model", default="grok-3",
            ),
        )

    if provider_name in ("claude", "anthropic"):
        from .core.llm.anthropic_client import AnthropicProvider
        api_key = config.get_str("llm", "claude", "apiKey", env="ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set (env or pythonclaw.json)")
        return AnthropicProvider(api_key=api_key)

    if provider_name == "gemini":
        from .core.llm.gemini_client import GeminiProvider
        api_key = config.get_str("llm", "gemini", "apiKey", env="GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set (env or pythonclaw.json)")
        return GeminiProvider(api_key=api_key)

    if provider_name in ("kimi", "moonshot"):
        from .core.llm.openai_compatible import OpenAICompatibleProvider
        api_key = config.get_str("llm", "kimi", "apiKey", env="KIMI_API_KEY")
        if not api_key:
            raise ValueError("KIMI_API_KEY not set (env or pythonclaw.json)")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=config.get_str(
                "llm", "kimi", "baseUrl", default="https://api.moonshot.cn/v1",
            ),
            model_name=config.get_str(
                "llm", "kimi", "model", env="KIMI_MODEL", default="moonshot-v1-128k",
            ),
        )

    if provider_name in ("glm", "zhipu", "chatglm"):
        from .core.llm.openai_compatible import OpenAICompatibleProvider
        api_key = config.get_str("llm", "glm", "apiKey", env="GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY not set (env or pythonclaw.json)")
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=config.get_str(
                "llm", "glm", "baseUrl",
                default="https://open.bigmodel.cn/api/paas/v4/",
            ),
            model_name=config.get_str(
                "llm", "glm", "model", env="GLM_MODEL", default="glm-4-flash",
            ),
        )

    raise ValueError(f"Unknown LLM_PROVIDER: '{provider_name}'")


# ── Ensure config is ready (auto-onboard if needed) ─────────────────────────

def _ensure_configured(config_path: str | None = None) -> None:
    """If no API key is configured, run the onboard wizard first."""
    from .onboard import needs_onboard, run_onboard

    if needs_onboard(config_path):
        print("[PythonClaw] No LLM provider configured. Starting setup wizard...\n")
        run_onboard(config_path)


# ── Subcommand handlers ─────────────────────────────────────────────────────

def _cmd_onboard(args) -> None:
    from .onboard import run_onboard
    run_onboard(args.config)


def _cmd_start(args) -> None:
    _ensure_configured(args.config)

    if args.foreground:
        _run_foreground(args)
    else:
        from .daemon import start_daemon
        start_daemon(channels=args.channels, config_path=args.config)


def _run_foreground(args) -> None:
    """Run the web server (+ optional channels) in the foreground."""
    provider = None
    try:
        provider = _build_provider()
    except Exception as exc:
        print(f"[PythonClaw] Warning: LLM provider not configured ({exc})")

    channels = args.channels or []

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if channels:
        # Run web + channels together
        from .server import run_server
        print(f"[PythonClaw] Starting (web + {', '.join(channels)})...")
        asyncio.run(run_server(provider, channels=channels))
    else:
        # Web-only
        try:
            import uvicorn
        except ImportError:
            print("Error: Web mode requires 'fastapi' and 'uvicorn'.")
            print("Install with: pip install pythonclaw[web]")
            return

        from .web.app import create_app

        host = config.get_str("web", "host", default="0.0.0.0")
        port = config.get_int("web", "port", default=7788)

        app = create_app(provider, build_provider_fn=_build_provider)
        print(f"[PythonClaw] Web dashboard: http://localhost:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_stop(args) -> None:
    from .daemon import stop_daemon
    stop_daemon()


def _cmd_status(args) -> None:
    from .daemon import print_status
    print_status()


def _cmd_chat(args) -> None:
    _ensure_configured(args.config)

    try:
        provider = _build_provider()
    except Exception as exc:
        print(f"Error: {exc}")
        return

    provider_name = config.get_str("llm", "provider", env="LLM_PROVIDER", default="deepseek")
    verbose = config.get("agent", "verbose", default=True)

    store = SessionStore()
    session_id = "cli"

    print(f"Initializing Agent with Provider: {provider_name.upper()}...")
    agent = PersistentAgent(
        provider=provider,
        verbose=bool(verbose),
        store=store,
        session_id=session_id,
    )
    print(f"Loaded {len(agent.loaded_skill_names)} active skills.")

    restored = len(agent.messages) - 1
    if restored > 0:
        print(f"Restored {restored} messages from previous session.")

    cfg_path = config.config_path()
    cfg_source = f" (config: {cfg_path})" if cfg_path else ""
    print("\n--- PythonClaw Agent ---")
    print(f"Provider: {provider_name}{cfg_source}")
    print(f"Session: {store._path(session_id)}")
    print("Commands: 'exit' to quit | '/compact [hint]' | '/status' | '/clear'")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.startswith("/compact"):
                hint = user_input[len("/compact"):].strip() or None
                result = agent.compact(instruction=hint)
                print(f"Bot: {result}")
                continue

            if user_input == "/status":
                memory_count = len(agent.memory.list_all())
                print(
                    f"Bot: Session Status\n"
                    f"  Provider     : {type(agent.provider).__name__}\n"
                    f"  Skills       : {len(agent.loaded_skill_names)} loaded\n"
                    f"  Memories     : {memory_count} entries\n"
                    f"  History      : {len(agent.messages)} messages\n"
                    f"  Compactions  : {agent.compaction_count}\n"
                    f"  Session File : {store._path(session_id)}"
                )
                continue

            if user_input == "/clear":
                store.delete(session_id)
                agent.clear_history()
                print("Bot: Chat history cleared. Agent is still active with all skills and memory intact.")
                continue

            response = agent.chat(user_input)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def _cmd_skill(args) -> None:
    from .core import skillhub

    action = args.skill_action
    if not action:
        print("Usage: pythonclaw skill {search,browse,install,info}")
        return

    if action == "search":
        query = " ".join(args.query)
        if not query:
            print("Usage: pythonclaw skill search <query>")
            return
        print(f"Searching SkillHub for: {query} ...")
        try:
            results = skillhub.search(query, limit=args.limit or 10)
            print(skillhub.format_search_results(results))
        except RuntimeError as exc:
            print(f"Error: {exc}")

    elif action == "browse":
        print("Browsing SkillHub catalog ...")
        try:
            results = skillhub.browse(limit=args.limit or 20, sort=args.sort or "score")
            print(skillhub.format_search_results(results))
        except RuntimeError as exc:
            print(f"Error: {exc}")

    elif action == "install":
        skill_id = args.skill_id
        if not skill_id:
            print("Usage: pythonclaw skill install <skill-id>")
            return
        print(f"Installing skill: {skill_id} ...")
        try:
            path = skillhub.install_skill(skill_id)
            print(f"Installed to: {path}")
            print("The skill will be available next time the agent starts.")
        except RuntimeError as exc:
            print(f"Error: {exc}")

    elif action == "info":
        skill_id = args.skill_id
        if not skill_id:
            print("Usage: pythonclaw skill info <skill-id>")
            return
        print(f"Fetching skill detail: {skill_id} ...")
        try:
            detail = skillhub.get_skill_detail(skill_id)
            if not detail:
                print("Skill not found.")
                return
            print(f"\n  Name: {detail.get('name', '?')}")
            print(f"  ID:   {detail.get('id', skill_id)}")
            if detail.get("description"):
                print(f"  Desc: {detail['description']}")
            if detail.get("source_url"):
                print(f"  URL:  {detail['source_url']}")
            if detail.get("skill_md"):
                print(f"\n--- SKILL.md Preview ---\n{detail['skill_md'][:500]}")
        except RuntimeError as exc:
            print(f"Error: {exc}")


# ── Argument parser ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythonclaw",
        description="PythonClaw — Autonomous AI Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  pythonclaw onboard       Set up your LLM provider\n"
            "  pythonclaw start         Start the agent daemon\n"
            "  pythonclaw chat          Interactive CLI chat\n"
            "\n"
            "Docs: https://github.com/ericwang915/PythonClaw"
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to pythonclaw.json config file.",
    )
    # Hidden --mode for backward compat
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--channels", nargs="+", default=None, help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command")

    # onboard
    sub.add_parser("onboard", help="Interactive first-time setup wizard")

    # start
    sp_start = sub.add_parser("start", help="Start the agent daemon")
    sp_start.add_argument(
        "--foreground", "-f", action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    sp_start.add_argument(
        "--channels", nargs="+",
        choices=["telegram", "discord"],
        help="Also start messaging channels",
    )

    # stop
    sub.add_parser("stop", help="Stop the running daemon")

    # status
    sub.add_parser("status", help="Show daemon status")

    # chat
    sub.add_parser("chat", help="Interactive CLI chat (foreground)")

    # skill
    skill_parser = sub.add_parser("skill", help="SkillHub marketplace commands")
    skill_sub = skill_parser.add_subparsers(dest="skill_action")

    sp_search = skill_sub.add_parser("search", help="Search skills on SkillHub")
    sp_search.add_argument("query", nargs="+", help="Search query")
    sp_search.add_argument("--limit", type=int, default=10, help="Max results")

    sp_browse = skill_sub.add_parser("browse", help="Browse SkillHub catalog")
    sp_browse.add_argument("--limit", type=int, default=20, help="Max results")
    sp_browse.add_argument("--sort", default="score",
                           choices=["score", "stars", "recent", "composite"])

    sp_install = skill_sub.add_parser("install", help="Install a skill from SkillHub")
    sp_install.add_argument("skill_id", help="Skill ID (from search results)")

    sp_info = skill_sub.add_parser("info", help="Show details for a SkillHub skill")
    sp_info.add_argument("skill_id", help="Skill ID")

    return parser


# ── Backward-compat --mode handler ───────────────────────────────────────────

def _handle_legacy_mode(args) -> None:
    """Support the old ``--mode cli|web|telegram|discord`` flags."""
    mode = args.mode
    channels_arg = getattr(args, "channels", None)

    if channels_arg:
        try:
            provider = _build_provider()
        except Exception as exc:
            print(f"Error: {exc}")
            return
        from .server import run_server
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        asyncio.run(run_server(provider, channels=channels_arg))
        return

    if mode == "web":
        provider = None
        try:
            provider = _build_provider()
        except Exception as exc:
            print(f"[PythonClaw] Warning: LLM provider not configured ({exc})")
        try:
            import uvicorn
        except ImportError:
            print("Error: pip install pythonclaw[web]")
            return
        from .web.app import create_app
        host = config.get_str("web", "host", default="0.0.0.0")
        port = config.get_int("web", "port", default=7788)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        app = create_app(provider, build_provider_fn=_build_provider)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return

    if mode in ("telegram", "discord"):
        try:
            provider = _build_provider()
        except Exception as exc:
            print(f"Error: {exc}")
            return
        from .server import run_server
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        asyncio.run(run_server(provider, channels=[mode]))
        return

    # Default: cli
    try:
        provider = _build_provider()
    except Exception as exc:
        print(f"Error: {exc}")
        return
    _cmd_chat(args)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    config.load()

    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        config.load(args.config, force=True)

    # Handle legacy --mode flag
    if args.mode and not args.command:
        _handle_legacy_mode(args)
        return

    dispatch = {
        "onboard": _cmd_onboard,
        "start": _cmd_start,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "chat": _cmd_chat,
        "skill": _cmd_skill,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
