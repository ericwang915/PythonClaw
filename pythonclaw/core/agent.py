"""
Agent — the core reasoning loop for PythonClaw.

Responsibilities
----------------
  - Maintain conversation history (messages list)
  - Build the per-session tool set and dispatch tool calls
  - Three-tier progressive skill loading (catalog → instructions → resources)
  - Trigger context compaction (auto or manual)
  - Interface with memory (MemoryManager) and knowledge (KnowledgeRAG)

What this class is NOT responsible for
---------------------------------------
  - Session lifecycle (→ SessionManager)
  - Persistence across restarts (→ PersistentAgent subclass)
  - I/O channels like Telegram (→ channels/)
  - Scheduling (→ scheduler/)
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime

from .. import config
from .compaction import (
    DEFAULT_AUTO_THRESHOLD_TOKENS,
    DEFAULT_RECENT_KEEP,
    estimate_tokens,
)
from .compaction import (
    compact as _do_compact,
)
from .knowledge.rag import KnowledgeRAG
from .llm.base import LLMProvider
from .memory.manager import MemoryManager
from .skill_loader import SkillRegistry
from .tools import (
    AVAILABLE_TOOLS,
    CRON_TOOLS,
    KNOWLEDGE_TOOL,
    MEMORY_TOOLS,
    META_SKILL_TOOLS,
    PRIMITIVE_TOOLS,
    SKILL_TOOLS,
    WEB_SEARCH_TOOL,
    configure_venv,
    set_sandbox,
)

logger = logging.getLogger(__name__)


def _load_text_dir_or_file(path: str | None, label: str = "File") -> str:
    """
    Load text from a single file or from all .md/.txt files in a directory.
    Returns an empty string if *path* is None or does not exist.
    """
    if not path or not os.path.exists(path):
        return ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if os.path.isdir(path):
        parts = []
        for filename in sorted(os.listdir(path)):
            if filename.lower().endswith((".md", ".txt")):
                with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                    parts.append(f"\n\n--- {label}: {filename} ---\n" + f.read())
        return "".join(parts)
    return ""


def _detail_log_dir() -> str:
    from .. import config as _cfg
    return os.path.join(str(_cfg.PYTHONCLAW_HOME), "context", "logs")


def _detail_log_file() -> str:
    return os.path.join(_detail_log_dir(), "history_detail.jsonl")


def _log_detail(entry: dict) -> None:
    """Append a JSON line to the detailed interaction log."""
    try:
        log_dir = _detail_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        entry["ts"] = datetime.now().isoformat(timespec="milliseconds")
        with open(_detail_log_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


class Agent:
    """
    Stateful LLM agent with tool use, three-tier skill loading, memory,
    and compaction.

    Parameters
    ----------
    provider           : LLM backend (DeepSeek, Grok, Claude, Gemini, …)
    session_id         : session identifier (enables per-group context isolation)
    memory_dir         : path to memory directory (auto-detected if None)
    skills_dirs        : list of skill directory paths
    knowledge_path     : path to knowledge directory for RAG
    persona_path       : path to persona .md file or directory
    soul_path          : path to SOUL.md file or directory
    verbose            : print debug info to stdout
    show_full_context  : print the full context window before each LLM call
    max_chat_history   : max non-system messages kept in the sliding window
    auto_compaction    : trigger compaction when token estimate exceeds threshold
    compaction_threshold : token threshold for auto-compaction
    compaction_recent_keep : number of recent messages kept verbatim after compaction
    cron_manager       : CronScheduler instance (enables cron_add/remove/list tools)
    """

    MAX_TOOL_ROUNDS = 12
    MAX_PARALLEL_SKILLS = 5
    TOOL_TIMEOUT = 300

    def __init__(
        self,
        provider: LLMProvider,
        session_id: str | None = None,
        memory_dir: str | None = None,
        skills_dirs: list[str] | None = None,
        knowledge_path: str | None = None,
        persona_path: str | None = None,
        soul_path: str | None = None,
        tools_path: str | None = None,
        verbose: bool = False,
        show_full_context: bool = False,
        max_chat_history: int = 10,
        auto_compaction: bool = True,
        compaction_threshold: int = DEFAULT_AUTO_THRESHOLD_TOKENS,
        compaction_recent_keep: int = DEFAULT_RECENT_KEEP,
        cron_manager=None,
    ) -> None:
        if memory_dir is None and skills_dirs is None and knowledge_path is None and persona_path is None:
            from .. import config as _cfg
            home = str(_cfg.PYTHONCLAW_HOME)
            context_dir = os.path.join(home, "context")
            if not os.path.exists(context_dir):
                if verbose:
                    print(f"[Agent] Context not found. Initialising default context in {context_dir}...")
                try:
                    from ..init import init
                    init(home)
                except ImportError:
                    try:
                        from pythonclaw.init import init
                        init(home)
                    except ImportError:
                        print("[Agent] Warning: Could not auto-initialise context.")
            if verbose:
                print(f"[Agent] Using default context at {context_dir}")

            # Per-group isolation: each session gets its own memory directory
            if session_id and _cfg.per_group_isolation():
                group_dir = str(_cfg.group_context_dir(session_id))
                os.makedirs(os.path.join(group_dir, "memory"), exist_ok=True)
                memory_dir = os.path.join(group_dir, "memory")
                if verbose:
                    print(f"[Agent] Per-group memory: {memory_dir}")
            else:
                memory_dir = os.path.join(context_dir, "memory")

            knowledge_path = os.path.join(context_dir, "knowledge")
            skills_dirs = [os.path.join(context_dir, "skills")]
            persona_path = os.path.join(context_dir, "persona")
            if soul_path is None:
                soul_path = os.path.join(context_dir, "soul")
            if tools_path is None:
                tools_path = os.path.join(context_dir, "tools")

        # Sandbox: restrict file-write tools to the home directory
        sandbox_root = str(config.PYTHONCLAW_HOME)
        set_sandbox([sandbox_root, os.path.expanduser("~")])
        if verbose:
            print(f"[Agent] Sandbox root: {sandbox_root}")

        # Venv: ensure all subprocesses use the project's virtual environment
        venv_path = configure_venv()
        if verbose and venv_path:
            print(f"[Agent] Virtual env: {venv_path}")

        self.provider = provider
        self.session_id = session_id
        self.messages: list[dict] = []
        self.verbose = verbose
        self.show_full_context = show_full_context
        self.max_chat_history = max_chat_history
        self.auto_compaction = auto_compaction
        self.compaction_threshold = compaction_threshold
        self.compaction_recent_keep = compaction_recent_keep
        self.compaction_count: int = 0
        self._cron_manager = cron_manager

        self.loaded_skill_names: set[str] = set()
        self.pending_injections: list[str] = []
        self.MAX_PARALLEL_SKILLS = config.get_int(
            "agent", "maxParallelSkills", default=5,
        )
        self._bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-bg")

        # Memory — with optional global fallback for per-group isolation
        mem_dir = memory_dir or config.get("memory", "dir", env="PYTHONCLAW_MEMORY_DIR")
        global_mem_dir: str | None = None
        if session_id and config.per_group_isolation():
            global_mem_dir = os.path.join(str(config.PYTHONCLAW_HOME), "context", "memory")
        self.memory = MemoryManager(mem_dir, global_memory_dir=global_mem_dir)

        # Knowledge RAG (hybrid retrieval)
        self.rag: KnowledgeRAG | None = None
        if knowledge_path and os.path.exists(knowledge_path):
            self.rag = KnowledgeRAG(
                knowledge_dir=knowledge_path,
                provider=provider,
                use_reranker=True,
            )
            if verbose:
                print(f"[Agent] KnowledgeRAG: '{knowledge_path}' ({len(self.rag)} chunks)")

        # Web search (Tavily)
        self._web_search_enabled = bool(
            config.get("tavily", "apiKey", env="TAVILY_API_KEY")
        )
        if verbose and self._web_search_enabled:
            print("[Agent] Web search enabled (Tavily)")

        # Skills — always include the built-in templates + user context/skills
        self.skills_dirs: list[str] = []
        pkg_templates = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "templates", "skills",
        )
        if os.path.isdir(pkg_templates):
            self.skills_dirs.append(pkg_templates)
        if skills_dirs:
            for d in ([skills_dirs] if isinstance(skills_dirs, str) else skills_dirs):
                if d not in self.skills_dirs:
                    self.skills_dirs.append(d)

        # Identity layers
        self.soul_instruction = _load_text_dir_or_file(soul_path, label="Soul")
        self.persona_instruction = _load_text_dir_or_file(persona_path, label="Persona")
        self.tools_notes = _load_text_dir_or_file(tools_path, label="Tools")

        # Detect if the user has set up their own soul/persona (not template defaults)
        self._needs_onboarding = not self._has_user_identity(soul_path, persona_path)

        if verbose and self.soul_instruction:
            print(f"[Agent] Soul loaded ({len(self.soul_instruction)} chars)")
        if verbose and self.persona_instruction:
            print(f"[Agent] Persona loaded ({len(self.persona_instruction)} chars)")
        if verbose and self.tools_notes:
            print(f"[Agent] TOOLS.md loaded ({len(self.tools_notes)} chars)")
        if verbose and self._needs_onboarding:
            print("[Agent] No user identity found — onboarding will be triggered")

        self._init_system_prompt()

    @staticmethod
    def _has_user_identity(soul_path: str | None, persona_path: str | None) -> bool:
        """Return True if the user has customized soul or persona files."""
        for p in (soul_path, persona_path):
            if p is None:
                continue
            if os.path.isdir(p):
                for fname in os.listdir(p):
                    fpath = os.path.join(p, fname)
                    if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                        return True
            elif os.path.isfile(p) and os.path.getsize(p) > 0:
                return True
        return False

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_system_prompt(self) -> None:
        """
        Build the initial system message with three-tier skill loading.

        Level 1 (Metadata) is injected here — the full skill catalog
        (name + description for every installed skill).  This lets the
        LLM decide when to activate a skill without any discovery calls.
        """
        self._registry = SkillRegistry(skills_dirs=self.skills_dirs)
        skill_catalog = self._registry.build_catalog()

        soul_section = f"\n\n## Core Identity (Soul)\n{self.soul_instruction}" if self.soul_instruction else ""
        persona_section = f"\n\n## Role & Persona\n{self.persona_instruction}" if self.persona_instruction else ""
        tools_section = f"\n\n## Local Notes (TOOLS.md)\n{self.tools_notes}" if self.tools_notes else ""

        web_search_section = ""
        if self._web_search_enabled:
            web_search_section = """
3. **Web Search**: `web_search` (powered by Tavily)
   Search the web for real-time information when you need up-to-date data,
   current events, facts you're unsure about, or technical documentation.
   Supports topic filters (general/news/finance) and time range filters."""

        bot_name = ""
        try:
            if hasattr(self, "memory"):
                bn = self.memory.list_all().get("bot_name", "")
                if bn and bn != "PythonClaw":
                    bot_name = f' Your name is "{bn}".'
        except Exception:
            pass

        system_msg = f"""You are a PythonClaw agent — an autonomous AI assistant.{bot_name}{soul_section}{persona_section}{tools_section}

### Tools
- **Primitives**: `run_command`, `read_file`, `write_file`, `list_files`
- **Skills** — call `use_skill(name)` to activate. Catalog:
{skill_catalog}
- **Memory**: `remember(key,val)`, `recall(query)`, `memory_get(path)`, `memory_list_files()`, `forget(key)`, `update_index(content)`
- **Skill creation**: `create_skill` — create generic reusable skills when none fit{web_search_section}

### Task Execution Modes
Choose your approach based on task complexity:

**ReAct** (simple tasks, 1-2 steps): Act directly — call tools and respond immediately.

**Plan & Execute** (complex tasks, 3+ steps, research, multi-source analysis):
1. Output a short numbered plan (3-6 steps) as your first response
2. Execute each step using tools — call multiple tools in parallel when steps are independent (up to {self.MAX_PARALLEL_SKILLS} parallel skills)
3. After each step, briefly summarize what you found before moving on
4. After all steps, synthesize a concise final answer

You decide which mode fits. Don't announce the mode name.

### Rules
- Batch independent tool calls in one response (parallel execution).
- Minimize search rounds (1-3 max). Combine queries. Don't repeat.
- Proactively `remember` user preferences, decisions, key facts.
- Use `recall` when user references past context.
- Memory auto-loaded at session start. INDEX.md = curated system info.
- All downloaded/generated files go in the shared files directory (`~/.pythonclaw/context/files/`). The `run_command` tool uses this as its working directory.
- NEVER output tool calls as XML or text. Always use the function calling API.

### Response Guidelines
- **Language matching**: ALWAYS reply in the SAME language the user used in their message. If the user writes in Chinese, reply in Chinese. If in English, reply in English. Mirror the user's language exactly.
- Answer the user's question directly and concisely.
- Keep responses focused — under 300 words when possible. Break long answers into short paragraphs.
- Do NOT mention what skills or tools you have available, unless explicitly asked.
- Do NOT list other things you can do at the end of your response.
"""
        # ── Auto-inject memory context ────────────────────────────────────
        boot_mem = self.memory.boot_context(max_chars=3000)
        if boot_mem:
            system_msg += f"\n\n## Loaded Memory (auto-injected at session start)\n{boot_mem}\n"

        if getattr(self, "_needs_onboarding", False):
            system_msg += """
### First-Time Onboarding
**IMPORTANT**: No user identity (soul/persona) has been configured yet.
On the VERY FIRST user message, start a friendly onboarding conversation.

**Language rule**: Always conduct onboarding in **English** by default.
If the user replies in another language, switch to that language for
the rest of the onboarding (and set that as their language preference).

1. Greet the user warmly and introduce yourself as PythonClaw
2. Ask: "What would you like to name me?" (let the user give you a custom name)
3. Ask: "What should I call you?" (wait for response)
4. Ask: "What kind of personality would you like me to have? (e.g. professional, friendly, humorous, encouraging)"
5. Ask: "What area would you like me to focus on? (e.g. software development, finance, research, daily assistant)"

After collecting ALL answers, use the `onboarding` skill to write the
soul.md and persona.md files. Detect the user's language from their
replies (default to English if they replied in English) and pass it as
the `--language` argument. Then use `remember` to save:
- `bot_name`: the custom name the user gave you
- `user_name`: the user's name
- user preferences to long-term memory

Ask the questions ONE AT A TIME, waiting for each answer before asking the next.
If the user's first message already contains task content (not just "hi"),
still start onboarding but keep it brief — you can help with their task after.
"""
        elif getattr(self, "memory", None):
            try:
                all_mem = self.memory.list_all()
                if "bot_name" not in all_mem:
                    system_msg += """
### Bot Naming
The user hasn't given you a custom name yet. On the first message,
briefly ask: "By the way, would you like to give me a name? You can
call me anything you like!" If they give a name, `remember("bot_name", name)`.
If they say no or skip, `remember("bot_name", "PythonClaw")` and move on.
Don't repeat this if `bot_name` already exists in memory.
"""
            except Exception:
                pass

        self.messages.append({"role": "system", "content": system_msg})
        if self.verbose:
            logger.debug("System prompt built. Skill catalog: %d skills.", len(self._registry.discover()))

    # ── Tool management ───────────────────────────────────────────────────────

    def _normalize_input(self, user_input: str | list) -> str | list:
        """If provider doesn't support images, extract text from multimodal input."""
        if isinstance(user_input, str):
            return user_input
        if getattr(self.provider, "supports_images", False):
            return user_input
        text_parts = []
        for part in user_input:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "image_url":
                text_parts.append("[image attached — your LLM provider does not support image input]")
        return "\n".join(text_parts) if text_parts else str(user_input)

    def _cap_parallel_skills(self, tool_calls: list) -> list:
        """Enforce MAX_PARALLEL_SKILLS — cap skill activations per round.

        Non-skill tool calls (run_command, remember, etc.) are not limited.
        Excess skill calls get stub responses appended to messages.
        """
        skill_names = {"use_skill"}
        skill_calls = [tc for tc in tool_calls if tc.function.name in skill_names]
        if len(skill_calls) <= self.MAX_PARALLEL_SKILLS:
            return tool_calls

        keep = set(id(tc) for tc in skill_calls[:self.MAX_PARALLEL_SKILLS])
        kept: list = []
        for tc in tool_calls:
            if tc.function.name in skill_names and id(tc) not in keep:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": (
                        f"(skipped — max {self.MAX_PARALLEL_SKILLS} "
                        "parallel skills per round)"
                    ),
                })
            else:
                kept.append(tc)
        logger.info(
            "Capped parallel skills: %d → %d",
            len(skill_calls), self.MAX_PARALLEL_SKILLS,
        )
        return kept

    def _build_tools(self) -> list[dict]:
        """Assemble the full tool schema list for the current session."""
        tools = PRIMITIVE_TOOLS + SKILL_TOOLS + META_SKILL_TOOLS + MEMORY_TOOLS
        if self._web_search_enabled:
            tools = tools + [WEB_SEARCH_TOOL]
        if self.rag:
            tools = tools + [KNOWLEDGE_TOOL]
        if self._cron_manager:
            tools = tools + CRON_TOOLS
        return tools

    def _execute_tool_call(self, tool_call) -> str:
        """Dispatch a single tool call and return the string result."""
        func_name: str = tool_call.function.name
        try:
            args: dict = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return f"Error: could not parse tool arguments: {exc}"

        if self.verbose:
            logger.debug("Tool: %s  args=%s", func_name, args)

        try:
            if func_name == "use_skill":
                result = self._use_skill(args.get("skill_name"))
            elif func_name == "list_skill_resources":
                resources = self._registry.list_resources(args.get("skill_name", ""))
                if resources:
                    result = "Resources:\n" + "\n".join(f"  - {r}" for r in resources)
                else:
                    result = "No bundled resources found (or skill not found)."
            elif func_name == "remember":
                result = self.memory.remember(args.get("content"), args.get("key"))
            elif func_name == "recall":
                result = self.memory.recall(args.get("query", "*"))
            elif func_name == "memory_get":
                result = self.memory.memory_get(args.get("path", "MEMORY.md"))
                if not result:
                    result = "(file not found or empty)"
            elif func_name == "memory_list_files":
                files = self.memory.list_files()
                result = "\n".join(files) if files else "(no memory files)"
            elif func_name == "forget":
                result = self.memory.forget(args.get("key", ""))
            elif func_name == "update_index":
                path = self.memory.write_index(args.get("content", ""))
                result = f"INDEX.md updated at {path}"
            elif func_name == "consult_knowledge_base" and self.rag:
                hits = self.rag.retrieve(args.get("query"), top_k=5)
                if hits:
                    result = "Found relevant info:\n" + "\n".join(
                        f"- [{h['source']}]: {h['content']}" for h in hits
                    )
                else:
                    result = "No relevant information found in the knowledge base."
            elif func_name == "cron_add" and self._cron_manager:
                result = self._cron_manager.add_dynamic_job(
                    job_id=args.get("job_id"),
                    cron_expr=args.get("cron"),
                    prompt=args.get("prompt"),
                    deliver_to="telegram" if args.get("deliver_to_chat_id") else None,
                    chat_id=args.get("deliver_to_chat_id"),
                )
            elif func_name == "cron_remove" and self._cron_manager:
                result = self._cron_manager.remove_dynamic_job(args.get("job_id"))
            elif func_name == "cron_list" and self._cron_manager:
                result = self._cron_manager.list_jobs()
            elif func_name == "create_skill":
                result = AVAILABLE_TOOLS["create_skill"](**args)
                self._refresh_skill_registry()
            elif func_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[func_name](**args)
            else:
                result = f"Error: unknown tool '{func_name}'."
        except Exception as exc:
            result = f"Error executing '{func_name}': {exc}"

        if self.verbose:
            preview = str(result)[:200] + ("..." if len(str(result)) > 200 else "")
            logger.debug("Result: %s", preview)

        return str(result)

    # ── Skill registry refresh (after create_skill) ────────────────────────

    def _refresh_skill_registry(self) -> None:
        """Invalidate the registry cache so newly created skills are discovered."""
        self._registry.invalidate()
        new_catalog = self._registry.build_catalog()
        self.messages.append({
            "role": "system",
            "content": (
                "[Skill Registry Updated]\n"
                "A new skill has been created. Updated skill catalog:\n\n"
                f"{new_catalog}"
            ),
        })
        if self.verbose:
            count = len(self._registry.discover())
            logger.debug("Skill registry refreshed — %d skills now available.", count)

    # ── Skill loading (Level 2) ───────────────────────────────────────────────

    @staticmethod
    def _check_dependencies(deps: list[str]) -> list[str]:
        """Return the subset of *deps* (pip package names) that are NOT installed."""
        from importlib.metadata import PackageNotFoundError, distribution

        missing: list[str] = []
        for pkg in deps:
            try:
                distribution(pkg)
            except PackageNotFoundError:
                missing.append(pkg)
        return missing

    def _use_skill(self, skill_name: str) -> str:
        """
        Level 2: Load a skill's full instructions into context.

        Called when the LLM triggers ``use_skill``.  The SKILL.md body
        is injected as a system message so subsequent turns can follow
        the instructions.

        If the skill directory contains a ``check_setup.sh`` script, it
        is executed automatically before activation.  When the check fails
        (non-zero exit), the skill is still loaded but a prominent warning
        with the script output is included so the LLM can guide the user
        through the fix.
        """
        if skill_name in self.loaded_skill_names:
            return f"Skill '{skill_name}' is already active."

        skill = self._registry.load_skill(skill_name)
        if not skill:
            return f"Error: skill '{skill_name}' not found in catalog."

        # ── Dependency check ─────────────────────────────────────────────────
        dep_warning = ""
        if skill.metadata.dependencies:
            missing = self._check_dependencies(skill.metadata.dependencies)
            if missing:
                pip_cmd = f"pip install {' '.join(missing)}"
                dep_warning = (
                    f"\n\n⚠️ **MISSING DEPENDENCIES**: {', '.join(missing)}\n"
                    f"This skill requires packages that are not installed.\n"
                    f"Ask the user: \"This skill needs **{', '.join(missing)}**. "
                    f"Would you like me to install {'them' if len(missing) > 1 else 'it'}?\"\n"
                    f"If the user agrees, run: `{pip_cmd}`\n"
                    f"Do NOT proceed with skill commands until dependencies are installed.\n"
                )
                if self.verbose:
                    logger.debug("Skill '%s' missing deps: %s", skill_name, missing)

        # ── Pre-activation environment check ─────────────────────────────────
        setup_warning = ""
        check_script = os.path.join(skill.metadata.path, "check_setup.sh")
        if os.path.isfile(check_script):
            import subprocess

            from .tools import _venv_env
            try:
                proc = subprocess.run(
                    ["bash", check_script],
                    capture_output=True, text=True, timeout=15,
                    env=_venv_env(),
                )
                if proc.returncode != 0:
                    output = (proc.stdout + proc.stderr).strip()
                    setup_warning = (
                        f"\n\n⚠️ **SETUP CHECK FAILED** (exit code {proc.returncode}):\n"
                        f"```\n{output}\n```\n"
                        f"Please tell the user what went wrong and how to fix it "
                        f"before attempting to use this skill's commands.\n"
                    )
                    if self.verbose:
                        logger.debug("Skill '%s' setup check FAILED: %s", skill_name, output)
                else:
                    setup_info = proc.stdout.strip()
                    setup_warning = f"\n\n✅ Setup check passed:\n```\n{setup_info}\n```\n"
                    if self.verbose:
                        logger.debug("Skill '%s' setup check passed.", skill_name)
            except Exception as exc:
                setup_warning = f"\n\n⚠️ Setup check could not run: {exc}\n"

        resources = self._registry.list_resources(skill_name)
        resource_hint = ""
        if resources:
            resource_hint = (
                "\n\n**Bundled resources** (use `read_file` / `run_command` to access):\n"
                + "\n".join(f"  - `{skill.metadata.path}/{r}`" for r in resources)
            )

        injection = (
            f"\n[SKILL ACTIVATED: {skill.name}]\n"
            f"Path: {skill.metadata.path}\n\n"
            f"{skill.instructions}{resource_hint}{dep_warning}{setup_warning}\n"
        )
        self.pending_injections.append(injection)
        self.loaded_skill_names.add(skill_name)
        if self.verbose:
            logger.debug("Skill activated: %s (Level 2 loaded)", skill_name)

        status = "activated"
        if dep_warning:
            status = "activated but MISSING DEPENDENCIES — ask user to install"
        elif "FAILED" in setup_warning:
            status = "activated with setup warnings — tell the user how to fix"
        return (
            f"Skill '{skill_name}' {status}. "
            f"Instructions loaded into context. "
            f"Bundled resources: {resources or 'none'}."
        )

    # ── History management ────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
        """Ensure every assistant message with ``tool_calls`` is immediately
        followed by matching ``tool`` messages, and every ``tool`` message has
        a preceding assistant message with a matching ``tool_calls`` entry.

        Broken pairs (caused by pruning, failed restores, or errors) are
        removed so the LLM API never receives an invalid sequence.
        """
        result: list[dict] = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            tool_calls = msg.get("tool_calls")

            if msg.get("role") == "assistant" and tool_calls:
                expected_ids: set[str] = set()
                for tc in tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        expected_ids.add(tc_id)

                # Collect subsequent tool responses that belong to this batch
                j = i + 1
                collected_tool_msgs: list[dict] = []
                while j < n and messages[j].get("role") in ("tool", "system"):
                    if messages[j].get("role") == "tool":
                        collected_tool_msgs.append(messages[j])
                    else:
                        break  # system injection sits between tool batch and next turn
                    j += 1

                found_ids = {
                    m.get("tool_call_id")
                    for m in collected_tool_msgs
                    if m.get("tool_call_id")
                }

                if expected_ids and expected_ids <= found_ids:
                    # Valid pair — keep assistant + matching tool messages
                    result.append(msg)
                    result.extend(collected_tool_msgs)
                    i = j
                else:
                    # Broken pair — skip the assistant message and any
                    # orphaned tool responses
                    logger.debug(
                        "Dropping broken tool-call sequence: expected %s, got %s",
                        expected_ids, found_ids,
                    )
                    i = j
            elif msg.get("role") == "tool":
                # Orphaned tool message (no preceding assistant with tool_calls) — skip
                i += 1
            else:
                result.append(msg)
                i += 1
        return result

    def _get_pruned_messages(self) -> list[dict]:
        """
        Build a context window for the API call:
          - All system messages (system prompt + skill injections + compaction summaries)
          - The most recent `max_chat_history` non-system messages

        Ensures the window contains only valid tool-call/response pairs.
        """
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        chat_msgs   = [m for m in self.messages if m.get("role") != "system"]

        if len(chat_msgs) > self.max_chat_history:
            chat_msgs = chat_msgs[-self.max_chat_history:]

        chat_msgs = self._sanitize_tool_pairs(chat_msgs)
        return system_msgs + chat_msgs

    # ── Compaction ────────────────────────────────────────────────────────────

    def compact(self, instruction: str | None = None) -> str:
        """
        Manually compact conversation history.

        Summarises older messages into a single [Compaction Summary] system
        entry, flushes important facts to long-term memory, and persists the
        summary to context/compaction/history.jsonl.

        Parameters
        ----------
        instruction : optional focus hint, e.g. "focus on open tasks"
        """
        chat_msgs = [m for m in self.messages if m.get("role") != "system"]
        if len(chat_msgs) <= self.compaction_recent_keep:
            return (
                f"Nothing to compact yet — only {len(chat_msgs)} message(s) in history "
                f"(threshold: {self.compaction_recent_keep})."
            )
        try:
            new_messages, summary = _do_compact(
                messages=self.messages,
                provider=self.provider,
                memory=self.memory,
                recent_keep=self.compaction_recent_keep,
                instruction=instruction,
            )
        except Exception as exc:
            return f"Compaction failed: {exc}"

        self.messages = new_messages
        self.compaction_count += 1

        lines = summary.splitlines()
        preview = "\n".join(lines[:5])
        if len(lines) > 5:
            preview += f"\n... ({len(lines) - 5} more lines)"
        return f"Compaction #{self.compaction_count} complete.\n\nSummary:\n{preview}"

    _memory_flushed_this_cycle: bool = False

    def _maybe_auto_compact(self) -> bool:
        """Auto-compact if the estimated token count exceeds the threshold.

        Before compacting, a proactive memory flush runs when the token
        count crosses a soft threshold (80% of the compaction threshold).
        This ensures durable facts are saved even if compaction itself fails.
        """
        if not self.auto_compaction:
            return False

        tokens = estimate_tokens(self.messages)
        soft_threshold = int(self.compaction_threshold * 0.8)

        if not self._memory_flushed_this_cycle and tokens >= soft_threshold:
            self._bg_executor.submit(self._proactive_memory_flush)
            self._memory_flushed_this_cycle = True

        if tokens < self.compaction_threshold:
            return False

        if self.verbose:
            logger.debug("Auto-compaction triggered.")
        try:
            new_messages, _ = _do_compact(
                messages=self.messages,
                provider=self.provider,
                memory=self.memory,
                recent_keep=self.compaction_recent_keep,
            )
            self.messages = new_messages
            self.compaction_count += 1
            self._memory_flushed_this_cycle = False
            return True
        except Exception as exc:
            if self.verbose:
                logger.debug("Auto-compaction failed (non-fatal): %s", exc)
            return False

    def _proactive_memory_flush(self) -> None:
        """Silently flush key facts to memory before compaction threshold.

        Runs once per compaction cycle when tokens cross 80% of the
        threshold. This way, important facts are persisted even if
        compaction is delayed or fails.
        """
        from .compaction import memory_flush

        chat_msgs = [m for m in self.messages if m.get("role") != "system"]
        if len(chat_msgs) < 4:
            return
        try:
            saved = memory_flush(chat_msgs, self.provider, self.memory)
            if self.verbose and saved:
                logger.debug("Proactive memory flush saved %d fact(s).", saved)
        except Exception as exc:
            logger.debug("Proactive memory flush failed (non-fatal): %s", exc)

    # ── Session management ─────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Clear conversation history but keep the agent intact.

        Preserves loaded skills, memory, RAG, provider, and all config.
        Only resets messages to a fresh system prompt and clears
        conversation-specific state.
        """
        self.messages.clear()
        self.loaded_skill_names.clear()
        self.compaction_count = 0
        self._init_system_prompt()

    # ── Main chat loop ────────────────────────────────────────────────────────

    def chat(self, user_input: str | list, **kwargs) -> str:
        """Send *user_input* to the LLM and return the final text response.

        *user_input* can be a plain string or a content-array for
        multimodal input (e.g. ``[{"type":"text","text":"..."}, {"type":"image_url",...}]``).
        """
        user_input = self._normalize_input(user_input)
        self.messages.append({"role": "user", "content": user_input})

        _log_detail({
            "event": "user_input",
            "content": user_input if isinstance(user_input, str) else "(multimodal)",
        })

        current_tools = self._build_tools()
        tool_rounds = 0
        chat_start = time.monotonic()

        while True:
            try:
                self._maybe_auto_compact()
                messages_to_send = self._get_pruned_messages()

                if self.show_full_context:
                    logger.debug(
                        "Context window (%d messages):\n%s",
                        len(messages_to_send),
                        json.dumps(messages_to_send, indent=2, ensure_ascii=False),
                    )

                response = self.provider.chat(
                    messages=messages_to_send,
                    tools=current_tools,
                    tool_choice="auto",
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    self.messages.append(message.model_dump())
                    _log_detail({
                        "event": "response",
                        "tool_rounds": tool_rounds,
                        "elapsed_ms": int((time.monotonic() - chat_start) * 1000),
                        "response_len": len(message.content or ""),
                    })
                    return message.content

                tool_rounds += 1
                if tool_rounds > self.MAX_TOOL_ROUNDS:
                    msg_dump = message.model_dump()
                    self.messages.append(msg_dump)

                    # Provide stub responses for every tool_call so the
                    # history stays valid for the API (each tool_call_id
                    # MUST have a matching tool-role message).
                    for tc in message.tool_calls:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(skipped — tool-call limit reached)",
                        })

                    limit_msg = (
                        f"Reached the maximum of {self.MAX_TOOL_ROUNDS} tool-call rounds. "
                        "Please provide a final answer with the information gathered so far."
                    )
                    self.messages.append({"role": "system", "content": limit_msg})
                    if self.verbose:
                        logger.debug("Tool round limit (%d) reached, forcing text reply.", self.MAX_TOOL_ROUNDS)
                    try:
                        final = self.provider.chat(
                            messages=self._get_pruned_messages(),
                            tools=current_tools,
                            tool_choice="none",
                        )
                        final_msg = final.choices[0].message
                        self.messages.append(final_msg.model_dump())
                        return final_msg.content
                    except Exception as exc:
                        return f"Error (after hitting tool limit): {exc}"

                self.messages.append(message.model_dump())
                self.pending_injections = []

                tool_calls = message.tool_calls
                tool_calls = self._cap_parallel_skills(tool_calls)

                _log_detail({
                    "event": "tool_calls",
                    "round": tool_rounds,
                    "calls": [
                        {"name": tc.function.name, "args": tc.function.arguments}
                        for tc in tool_calls
                    ],
                })

                t0 = time.monotonic()
                results: dict[str, str] = {}
                with ThreadPoolExecutor(max_workers=min(len(tool_calls), 16)) as pool:
                    futures = {
                        pool.submit(self._execute_tool_call, tc): tc
                        for tc in tool_calls
                    }
                    for future in as_completed(futures, timeout=self.TOOL_TIMEOUT):
                        tc = futures[future]
                        try:
                            results[tc.id] = future.result()
                        except Exception as exc:
                            results[tc.id] = f"Error: {exc}"
                for tc in tool_calls:
                    if tc.id not in results:
                        results[tc.id] = (
                            f"Error: tool '{tc.function.name}' timed out "
                            f"after {self.TOOL_TIMEOUT}s"
                        )
                _log_detail({
                    "event": "tool_results",
                    "round": tool_rounds,
                    "count": len(tool_calls),
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    "tools": [tc.function.name for tc in tool_calls],
                })
                for tc in tool_calls:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": results[tc.id],
                    })

                for injection in self.pending_injections:
                    self.messages.append({"role": "system", "content": injection})
                self.pending_injections = []

            except FuturesTimeout:
                logger.warning("Tool execution timed out at round %d", tool_rounds)
                for tc in tool_calls:
                    if tc.id not in results:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: timed out after {self.TOOL_TIMEOUT}s",
                        })
                continue
            except Exception as exc:
                logger.exception("Critical error in Agent.chat()")
                return f"Error: {exc}"

    def chat_stream(
        self,
        user_input: str | list,
        on_token: object = None,
    ) -> str:
        """Streaming variant of ``chat()``.

        *user_input* can be a plain string or a multimodal content array.
        *on_token* is called with each text chunk as it arrives.
        Returns the full final text, same as ``chat()``.
        """
        user_input = self._normalize_input(user_input)
        self.messages.append({"role": "user", "content": user_input})
        _log_detail({
            "event": "user_input",
            "content": user_input if isinstance(user_input, str) else "(multimodal)",
        })

        current_tools = self._build_tools()
        tool_rounds = 0
        chat_start = time.monotonic()

        while True:
            try:
                self._maybe_auto_compact()
                messages_to_send = self._get_pruned_messages()

                gen = self.provider.chat_stream(
                    messages=messages_to_send,
                    tools=current_tools,
                    tool_choice="auto",
                )
                response = None
                while True:
                    try:
                        chunk = next(gen)
                        if chunk.get("type") == "text_delta" and on_token:
                            on_token(chunk["text"])
                    except StopIteration as si:
                        response = si.value
                        break

                if response is None:
                    return ""

                message = response.choices[0].message

                if not message.tool_calls:
                    self.messages.append(message.model_dump())
                    _log_detail({
                        "event": "response",
                        "tool_rounds": tool_rounds,
                        "elapsed_ms": int(
                            (time.monotonic() - chat_start) * 1000
                        ),
                        "response_len": len(message.content or ""),
                    })
                    return message.content or ""

                tool_rounds += 1
                if tool_rounds > self.MAX_TOOL_ROUNDS:
                    msg_dump = message.model_dump()
                    self.messages.append(msg_dump)
                    for tc in message.tool_calls:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(skipped — tool-call limit reached)",
                        })
                    limit_msg = (
                        f"Reached the maximum of {self.MAX_TOOL_ROUNDS} "
                        "tool-call rounds. Provide a final answer."
                    )
                    self.messages.append(
                        {"role": "system", "content": limit_msg}
                    )
                    final = self.provider.chat(
                        messages=self._get_pruned_messages(),
                        tools=current_tools,
                        tool_choice="none",
                    )
                    final_msg = final.choices[0].message
                    self.messages.append(final_msg.model_dump())
                    return final_msg.content or ""

                self.messages.append(message.model_dump())
                self.pending_injections = []

                tool_calls = message.tool_calls
                tool_calls = self._cap_parallel_skills(tool_calls)

                if on_token:
                    names = ", ".join(tc.function.name for tc in tool_calls)
                    on_token(f"\n\n`[calling: {names}]`\n\n")

                results: dict[str, str] = {}
                with ThreadPoolExecutor(
                    max_workers=min(len(tool_calls), 16),
                ) as pool:
                    futures = {
                        pool.submit(self._execute_tool_call, tc): tc
                        for tc in tool_calls
                    }
                    for future in as_completed(
                        futures, timeout=self.TOOL_TIMEOUT
                    ):
                        tc = futures[future]
                        try:
                            results[tc.id] = future.result()
                        except Exception as exc:
                            results[tc.id] = f"Error: {exc}"
                for tc in tool_calls:
                    if tc.id not in results:
                        results[tc.id] = (
                            f"Error: tool '{tc.function.name}' timed out "
                            f"after {self.TOOL_TIMEOUT}s"
                        )
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": results[tc.id],
                    })

                for injection in self.pending_injections:
                    self.messages.append(
                        {"role": "system", "content": injection}
                    )
                self.pending_injections = []

            except FuturesTimeout:
                logger.warning("Tool execution timed out in stream round %d", tool_rounds)
                for tc in tool_calls:
                    if tc.id not in results:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: timed out after {self.TOOL_TIMEOUT}s",
                        })
                continue
            except Exception as exc:
                logger.exception("Critical error in Agent.chat_stream()")
                return f"Error: {exc}"
