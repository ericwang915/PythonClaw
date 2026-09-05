<p align="center">
  <img src="assets/logo-300.png" alt="PythonClaw" width="150">
</p>

<h1 align="center">PythonClaw</h1>

<p align="center">
  <strong>A personal AI agent you own — in pure Python.</strong><br>
  <code>pip install</code> and talk to it from your terminal, a web dashboard, or Telegram /
  Discord / WhatsApp.<br>It remembers what matters, learns new skills on demand, and runs tasks on a schedule.
</p>

<p align="center">
  <a href="https://github.com/ericwang915/PythonClaw/actions/workflows/ci.yml">
    <img src="https://github.com/ericwang915/PythonClaw/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/pythonclaw/">
    <img src="https://img.shields.io/pypi/v/pythonclaw?color=blue" alt="PyPI">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/pythonclaw" alt="Python">
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/ericwang915/PythonClaw" alt="MIT License">
  </a>
  <a href="https://github.com/ericwang915/PythonClaw/stargazers">
    <img src="https://img.shields.io/github/stars/ericwang915/PythonClaw?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <em>The Python reimagining of <a href="https://github.com/openclaw/openclaw">OpenClaw</a> — no Node.js, no Rust, no C extensions. Just Python.</em>
</p>

<!--
  ┌─ TOP PRIORITY: add a demo GIF here ─────────────────────────────────────────┐
  │ A 30–60s screen capture is the single biggest lever for turning repo        │
  │ visitors into stars. Suggested shot: terminal running `pip install          │
  │ pythonclaw && pythonclaw onboard` on the left, a Telegram chat on the right  │
  │ where you send a voice note and the agent installs a skill and does a task.  │
  │ Save it as assets/demo.gif, then uncomment:                                  │
  │   <p align="center"><img src="assets/demo.gif" alt="PythonClaw demo" width="760"></p>
  └─────────────────────────────────────────────────────────────────────────────┘
-->

<p align="center">
  <a href="#quick-start">Quick&nbsp;Start</a> ·
  <a href="#run-100-local-with-ollama">Local&nbsp;with&nbsp;Ollama</a> ·
  <a href="#docker">Docker</a> ·
  <a href="#supported-llm-providers">Providers</a> ·
  <a href="#skills">Skills</a> ·
  <a href="#configuration">Config</a> ·
  <a href="#use-as-a-library">Library</a>
</p>

---

## Why PythonClaw

- 🐍 **Pure Python, zero build step** — one `pip install`, no Node/Rust/C toolchain. Import it as a library, not just a CLI.
- 🔌 **Any model, or none** — DeepSeek, Claude, Gemini, Kimi, GLM… or **[100% local with Ollama](#run-100-local-with-ollama)**: no API key, nothing leaves your machine.
- 📡 **One agent, everywhere** — the same brain answers from your CLI, a browser dashboard, and group chats, each with isolated memory.
- 🧩 **It grows itself** — pulls from a 13K-skill [marketplace](#clawhub-marketplace), and can write and install a brand-new skill at runtime when none fits.

## Quick Start

```bash
pip install pythonclaw

pythonclaw onboard    # pick a provider, paste an API key (or choose Ollama — no key)
pythonclaw start      # daemon + web dashboard at http://localhost:7788
pythonclaw chat       # or just chat in the terminal
```

Prefer to stay fully offline? One line, no key:

```bash
LLM_PROVIDER=ollama pythonclaw chat      # needs a local Ollama running
```

<details>
<summary><b>Install from source</b></summary>

```bash
git clone https://github.com/ericwang915/PythonClaw.git
cd PythonClaw
pip install -e .
pythonclaw onboard
```
</details>

---

## What's inside

| | Feature | Details |
|---|---------|---------|
| 🧠 | **Provider-agnostic** | DeepSeek, Grok, Claude, Gemini, Kimi, GLM, **Ollama (100% local)** — or any OpenAI-compatible API |
| 🛠️ | **Self-extending skills** | Three-tier progressive loading (metadata → instructions → resources) + a 13K-skill [ClawHub](https://clawhub.com) marketplace, and the agent can author its own |
| 💾 | **Persistent memory** | Plain-Markdown long-term memory with daily logs and semantic recall — grep-able, backup-able, no database |
| 🔍 | **Hybrid RAG** | BM25 + dense embeddings + RRF fusion + LLM re-ranking over your own docs |
| 🌐 | **Web dashboard** | Browser UI for chat, config, skill catalog, identity editing, and marketplace |
| 🎙️ | **Voice input** | Speech-to-text via Deepgram, or **fully local with Whisper** (`stt.provider: "whisper"`) |
| ⏰ | **Cron jobs** | Schedule tasks in config, or let the agent schedule its own and message you |
| 📡 | **Multi-channel** | CLI, Web, Telegram, Discord, WhatsApp — one agent behind every front-end |

<details>
<summary><b>Under the hood</b> — reliability &amp; context engineering</summary>

<br>

| | | |
|---|---|---|
| 🔄 | **Daemon mode** | PID-managed background process with `start` / `stop` / `status` |
| 🧬 | **Soul + Persona** | Separate the agent's core identity from its swappable role presentation |
| 🔧 | **TOOLS.md** | Your local environment notes (SSH hosts, paths, defaults) — kept apart from shareable skills |
| 🔒 | **Per-group isolation** | Each chat session can get its own memory, persona, and soul |
| 📐 | **Context engineering** | Tool-output truncation with paged spill-over files, turn-integrity pruning, pair-safe compaction with pre-flush to memory |
| ⚡ | **Parallel by default** | Batched parallel tool calls, `multi_search` fan-out, non-blocking tool timeouts |
| 🛑 | **Loop breaker** | Repeated identical tool calls are short-circuited so the agent can't burn rounds retrying itself |
| 🔁 | **Concurrency control** | Per-session locks + a global semaphore prevent history interleaving |
| 🛡️ | **Hardened** | Zip-slip-safe skill installs, TLS-verified downloads, path-traversal guards, atomic persistence, loopback-only dashboard by default |

</details>

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pythonclaw onboard` | Interactive setup wizard — choose LLM provider, enter API key |
| `pythonclaw start` | Start the agent as a background daemon |
| `pythonclaw start -f` | Start in foreground (no daemonize) |
| `pythonclaw start --channels telegram discord whatsapp` | Start with messaging channels |
| `pythonclaw stop` | Stop the running daemon |
| `pythonclaw status` | Show daemon status (PID, uptime, port) |
| `pythonclaw chat` | Interactive CLI chat (foreground REPL) |
| `pythonclaw skill search <query>` | Search skills on [ClawHub](https://clawhub.com) |
| `pythonclaw skill browse` | Browse top-rated skills |
| `pythonclaw skill install <id>` | Install a community skill |
| `pythonclaw skill info <id>` | View skill details |

### First Run

```
$ pythonclaw start

  ╔══════════════════════════════════════╗
  ║       PythonClaw — Setup Wizard      ║
  ╚══════════════════════════════════════╝

  Choose your LLM provider:

    1. DeepSeek
    2. Grok (xAI)
    3. Claude (Anthropic)
    4. Gemini (Google)
    5. Kimi (Moonshot)
    6. GLM (Zhipu / ChatGLM)
    7. Ollama (100% local — no API key)

  Enter number (1-7): 1
  → DeepSeek

  API Key: ********
  → Key set (sk-****)

  Validating... ✔ Valid!
  ✔ Setup complete!

[PythonClaw] Daemon started (PID 12345).
[PythonClaw] Dashboard: http://localhost:7788
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         PythonClaw                            │
├──────────┬────────────┬───────────┬──────────────────────────┤
│ CLI      │ Daemon     │ Sessions  │      Core                │
│          │            │           │                          │
│ onboard  │ start /    │ Store(MD) │ Agent                    │
│ chat     │ stop /     │ Manager   │ ├─ Memory (Markdown)     │
│ skill …  │ status     │ Locks +   │ ├─ RAG (Hybrid)          │
│          │            │ Semaphore │ ├─ Skills (3-tier)        │
│ Web UI ◄─┤ Channels   │           │ ├─ Compaction            │
│ Voice In │ Telegram   │ Per-group │ ├─ Soul + Persona        │
│          │ Discord    │ Isolation │ ├─ Group Context          │
│          │ WhatsApp   │           │ └─ Tool Execution        │
├──────────┴────────────┴───────────┴──────────────────────────┤
│               LLM Provider Abstraction Layer                 │
│ DeepSeek │ Grok │ Claude │ Gemini │ Kimi │ GLM │ Ollama │ …  │
├──────────────────────────────────────────────────────────────┤
│              ClawHub Marketplace (clawhub.com)               │
└──────────────────────────────────────────────────────────────┘
```

---

## Web Dashboard

Start with `pythonclaw start` and open **http://localhost:7788**.

- **Dashboard** — agent status, soul/persona preview, tool list
- **Chat** — real-time chat with voice input (Deepgram)
- **Skill Catalog** — browse installed skills by category
- **Marketplace** — search and install skills from [ClawHub](https://clawhub.com)
- **Configuration** — edit LLM provider, API keys, and settings in-browser

---

## Configuration

All configuration lives in `pythonclaw.json` (auto-created by `pythonclaw onboard`).
See [`pythonclaw.example.json`](pythonclaw.example.json) for the full template.

```jsonc
{
  "llm": {
    "provider": "grok",
    "grok": { "apiKey": "xai-...", "model": "grok-3" }
  },
  "tavily":   { "apiKey": "" },
  "deepgram": { "apiKey": "" },
  "web":      { "host": "127.0.0.1", "port": 7788 },
  "channels": {
    "telegram": { "token": "" },
    "discord":  { "token": "" },
    "whatsapp": { "phoneNumberId": "", "token": "", "verifyToken": "pythonclaw_verify" }
  },
  "isolation":   { "perGroup": false },
  "concurrency": { "maxAgents": 4 }
}
```

Environment variables (e.g. `DEEPSEEK_API_KEY`, `TAVILY_API_KEY`, `LLM_PROVIDER`) override JSON values.

> **Security:** the dashboard has full agent access, so `web.host` defaults to `127.0.0.1` (loopback only). Expose it deliberately — set `web.host` to `0.0.0.0` (or `PYTHONCLAW_WEB_HOST=0.0.0.0` in containers) only behind your own auth/tunnel.

---

## Supported LLM Providers

| Provider | `llm.provider` | Default Model | API key |
|----------|----------------|---------------|---------|
| **DeepSeek** | `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| **Grok (xAI)** | `grok` | `grok-3` | `GROK_API_KEY` |
| **Claude (Anthropic)** | `claude` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` (or `claude setup-token`) |
| **Gemini (Google)** | `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| **Kimi (Moonshot)** | `kimi` | `moonshot-v1-128k` | `KIMI_API_KEY` |
| **GLM (Zhipu)** | `glm` | `glm-4-flash` | `GLM_API_KEY` |
| **Ollama (local)** 🆕 | `ollama` | `llama3.1` | none — 100% local |
| **Any OpenAI-compatible** 🆕 | `custom` | `gpt-4o-mini` | `OPENAI_API_KEY` |

`custom` works with OpenAI, OpenRouter, LM Studio, vLLM, llama.cpp server — anything speaking the chat-completions protocol (set `llm.custom.baseUrl`).

### Run 100% local with Ollama

No API key, no cloud, your data never leaves the machine:

```bash
ollama pull llama3.1        # or qwen3, mistral, …
pip install pythonclaw

LLM_PROVIDER=ollama pythonclaw chat        # CLI
LLM_PROVIDER=ollama pythonclaw start       # daemon + web dashboard
```

Pick a different model with `OLLAMA_MODEL=qwen3`, or point at a remote Ollama with `OLLAMA_BASE_URL=http://gpu-box:11434/v1`.

---

## Docker

```bash
docker run -p 7788:7788 \
  -e LLM_PROVIDER=deepseek -e DEEPSEEK_API_KEY=sk-... \
  -v pythonclaw-data:/root/.pythonclaw \
  $(docker build -q .)
```

Or with compose (edit the environment block / use a `.env` file):

```bash
git clone https://github.com/ericwang915/PythonClaw.git && cd PythonClaw
docker compose up -d
```

Fully local stack: run Ollama on the host and set `LLM_PROVIDER=ollama` — the compose file already routes `host.docker.internal` for you.

---

## Skills

### Three-Tier Progressive Loading

| Level | Loaded When | Content |
|-------|-------------|---------|
| **L1 — Metadata** | Always (startup) | `name` + `description` from YAML frontmatter |
| **L2 — Instructions** | Agent activates skill | Full SKILL.md body |
| **L3 — Resources** | As needed | Bundled scripts, schemas, data files |

```yaml
---
name: code_runner
description: Execute Python code safely in an isolated subprocess.
---
# Code Runner

## Instructions
Run `python {skill_path}/run_code.py "expression"`
```

### ClawHub Marketplace

Browse and install 13,000+ community skills from [ClawHub](https://clawhub.com) — free, no API key required:

```bash
pythonclaw skill search "database backup"
pythonclaw skill install <skill-id>
```

Also accessible from the web dashboard **Marketplace** tab.

---

## Memory & RAG

### Markdown Memory

```
~/.pythonclaw/context/memory/
├── MEMORY.md           # Curated long-term memory
└── 2026-02-23.md       # Daily append-only log
```

When **per-group isolation** is enabled (`"isolation": { "perGroup": true }` in config),
each session (Telegram chat, Discord channel, etc.) gets its own `memory/`, `persona/`,
and `soul/` under `~/.pythonclaw/context/groups/<session-id>/`, while global memories
remain accessible via read-through fallback.

### TOOLS.md — Local Notes

```
~/.pythonclaw/context/tools/
└── TOOLS.md              # Your environment-specific cheat sheet
```

Skills define *how* tools work. `TOOLS.md` stores *your* specifics — SSH hosts, device
nicknames, project paths, preferred defaults, API endpoints. Keeping them apart means
you can update skills without losing your notes, and share skills without leaking your
infrastructure. Editable from the web dashboard.

### Hybrid RAG Pipeline

```
Query → BM25 (sparse) + Embeddings (dense) → RRF Fusion → LLM Re-ranker → Top-K
```

---

## Use as a Library

```python
from pythonclaw import Agent
from pythonclaw.core.llm.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    model_name="deepseek-chat",
)

agent = Agent(provider=provider)
print(agent.chat("What is the capital of France?"))
```

---

## Project Structure

```
PythonClaw/
├── pythonclaw/
│   ├── main.py                # CLI entry (onboard/start/stop/status/chat/skill)
│   ├── onboard.py             # Interactive setup wizard
│   ├── daemon.py              # PID-based daemon lifecycle
│   ├── server.py              # Multi-channel daemon server
│   ├── core/
│   │   ├── agent.py           # Core reasoning loop
│   │   ├── tools.py           # Tool schemas and execution
│   │   ├── skill_loader.py    # Three-tier skill system
│   │   ├── skillhub.py        # ClawHub marketplace client
│   │   ├── persistent_agent.py
│   │   ├── compaction.py      # Context compaction
│   │   ├── llm/               # Provider adapters
│   │   ├── memory/            # Markdown memory
│   │   ├── knowledge/         # Knowledge-base RAG
│   │   └── retrieval/         # BM25 + dense + fusion + reranker
│   ├── channels/              # Telegram, Discord, WhatsApp
│   ├── scheduler/             # Cron jobs, heartbeat
│   ├── web/                   # FastAPI dashboard + static assets
│   └── templates/             # Built-in skill templates
├── context/                   # Runtime data (gitignored)
├── pyproject.toml
├── pythonclaw.example.json    # Configuration template
└── LICENSE
```

---

## Development

```bash
git clone https://github.com/ericwang915/PythonClaw.git
cd PythonClaw
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/ -v
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Comparison with OpenClaw

| Feature | OpenClaw | PythonClaw |
|---------|----------|------------|
| Language | TypeScript / Node.js | **Python** |
| Install | `npm i -g openclaw` | `pip install pythonclaw` |
| CLI | `openclaw start/stop` | `pythonclaw start/stop/status` |
| Dashboard | Web UI | Web UI (localhost:7788) |
| Memory | Markdown | Markdown (long-term + daily) |
| Skills | Plugin system | Three-tier + ClawHub marketplace |
| Channels | Discord, Telegram, WhatsApp | CLI, Web, Telegram, Discord, WhatsApp |
| Voice | — | Deepgram STT + local Whisper |
| LLM Providers | OpenAI, Anthropic, Gemini | DeepSeek, Grok, Claude, Gemini, Kimi, GLM + any OpenAI-compatible |
| Run fully local | — | **Yes — Ollama, no API key** |
| Deploy | npm | pip · Docker · docker-compose |
| Daemon | Background process | PID-managed (`start`/`stop`/`status`) |

---

## License

[MIT](LICENSE)

---

<p align="center">
  <sub>If PythonClaw helps you, consider giving it a ⭐</sub>
</p>
