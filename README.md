<p align="center">
  <img src="assets/logo-300.png" alt="PythonClaw" width="160">
</p>

<h1 align="center">PythonClaw</h1>

<p align="center">
  <strong>OpenClaw, reimagined in pure Python — purely Pythonic design.</strong><br>
  Memory · RAG · Skills · Web Dashboard · Voice · Daemon · Multi-Channel
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
  <em>The Python port of <a href="https://github.com/openclaw/openclaw">OpenClaw</a> — no Node.js, no Rust, no C extensions. Just Python.</em>
</p>

---

## Highlights

| | Feature | Details |
|---|---------|---------|
| 🧠 | **Provider-agnostic** | DeepSeek, Grok, Claude, Gemini, Kimi, GLM — or any OpenAI-compatible API |
| 🛠️ | **Three-tier skills** | Progressive loading: metadata → instructions → resources. Community marketplace via [SkillHub](https://www.skillhub.club) |
| 💾 | **Persistent memory** | Markdown-based long-term memory with daily logs and semantic recall |
| 🔍 | **Hybrid RAG** | BM25 + dense embeddings + RRF fusion + LLM re-ranking |
| 🌐 | **Web dashboard** | Browser UI for chat, config, skill catalog, identity editing, and marketplace |
| 🎙️ | **Voice input** | Deepgram speech-to-text in the web chat |
| ⏰ | **Cron jobs** | Schedule tasks via YAML or let the agent create its own |
| 📡 | **Multi-channel** | CLI, Web, Telegram, Discord — same agent, different interfaces |
| 🔄 | **Daemon mode** | PID-managed background process with `start` / `stop` / `status` |
| 🧬 | **Soul + Persona** | Separate core identity from swappable role presentation |

---

## Quick Start

```bash
pip install pythonclaw

# First-time setup — choose your LLM provider and enter API key
pythonclaw onboard

# Start the agent daemon (web dashboard at http://localhost:7788)
pythonclaw start

# Interactive CLI chat
pythonclaw chat

# Stop the daemon
pythonclaw stop
```

**From source:**

```bash
git clone https://github.com/ericwang915/PythonClaw.git
cd PythonClaw
pip install -e ".[all]"
pythonclaw onboard
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pythonclaw onboard` | Interactive setup wizard — choose LLM provider, enter API key |
| `pythonclaw start` | Start the agent as a background daemon |
| `pythonclaw start -f` | Start in foreground (no daemonize) |
| `pythonclaw start --channels telegram discord` | Start with messaging channels |
| `pythonclaw stop` | Stop the running daemon |
| `pythonclaw status` | Show daemon status (PID, uptime, port) |
| `pythonclaw chat` | Interactive CLI chat (foreground REPL) |
| `pythonclaw skill search <query>` | Search skills on [SkillHub](https://www.skillhub.club) |
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

  Enter number (1-6): 2
  → Grok (xAI)

  API Key: ********
  → Key set (xai-****)

  Validating... ✔ Valid!
  ✔ Setup complete!

[PythonClaw] Daemon started (PID 12345).
[PythonClaw] Dashboard: http://localhost:7788
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       PythonClaw                         │
├──────────┬───────────┬───────────┬───────────────────────┤
│ CLI      │ Daemon    │ Sessions  │      Core             │
│          │           │           │                       │
│ onboard  │ start /   │ Store(MD) │ Agent                 │
│ chat     │ stop /    │ Manager   │ ├─ Memory (Markdown)  │
│ skill …  │ status    │           │ ├─ RAG (Hybrid)       │
│          │           │           │ ├─ Skills (3-tier)    │
│ Web UI ◄─┤ Channels  │           │ ├─ Compaction         │
│ Voice In │ Telegram  │           │ ├─ Soul + Persona     │
│          │ Discord   │           │ └─ Tool Execution     │
├──────────┴───────────┴───────────┴───────────────────────┤
│             LLM Provider Abstraction Layer               │
│ DeepSeek │ Grok │ Claude │ Gemini │ Kimi │ GLM          │
├──────────────────────────────────────────────────────────┤
│            SkillHub Marketplace (skillhub.club)          │
└──────────────────────────────────────────────────────────┘
```

---

## Web Dashboard

Start with `pythonclaw start` and open **http://localhost:7788**.

- **Dashboard** — agent status, soul/persona preview, tool list
- **Chat** — real-time chat with voice input (Deepgram)
- **Skill Catalog** — browse installed skills by category
- **Marketplace** — search and install skills from [SkillHub](https://www.skillhub.club)
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
  "web":      { "host": "0.0.0.0", "port": 7788 },
  "channels": {
    "telegram": { "token": "" },
    "discord":  { "token": "" }
  }
}
```

Environment variables (e.g. `DEEPSEEK_API_KEY`, `TAVILY_API_KEY`) override JSON values.

---

## Supported LLM Providers

| Provider | Default Model | Install Extra |
|----------|---------------|---------------|
| **DeepSeek** | `deepseek-chat` | — |
| **Grok (xAI)** | `grok-3` | — |
| **Claude (Anthropic)** | `claude-sonnet-4-20250514` | `pip install pythonclaw[anthropic]` |
| **Gemini (Google)** | `gemini-2.0-flash` | `pip install pythonclaw[gemini]` |
| **Kimi (Moonshot)** | `moonshot-v1-128k` | — |
| **GLM (Zhipu)** | `glm-4-flash` | — |
| Any OpenAI-compatible | Custom | — |

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

### SkillHub Marketplace

Browse and install 22,000+ community skills from [skillhub.club](https://www.skillhub.club):

```bash
pythonclaw skill search "database backup"
pythonclaw skill install <skill-id>
```

Also accessible from the web dashboard **Marketplace** tab.

---

## Memory & RAG

### Markdown Memory

```
context/memory/
├── MEMORY.md           # Curated long-term memory
└── 2026-02-23.md       # Daily append-only log
```

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
│   │   ├── skillhub.py        # SkillHub marketplace client
│   │   ├── persistent_agent.py
│   │   ├── compaction.py      # Context compaction
│   │   ├── llm/               # Provider adapters
│   │   ├── memory/            # Markdown memory
│   │   ├── knowledge/         # Knowledge-base RAG
│   │   └── retrieval/         # BM25 + dense + fusion + reranker
│   ├── channels/              # Telegram, Discord
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
pip install -e ".[all]"
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
| Skills | Plugin system | Three-tier + SkillHub marketplace |
| Channels | Discord, Telegram, WhatsApp | CLI, Web, Telegram, Discord |
| Voice | — | Deepgram STT |
| LLM Providers | OpenAI, Anthropic, Gemini | DeepSeek, Grok, Claude, Gemini, Kimi, GLM |
| Daemon | Background process | PID-managed (`start`/`stop`/`status`) |

---

## License

[MIT](LICENSE)

---

<p align="center">
  <sub>If PythonClaw helps you, consider giving it a ⭐</sub>
</p>
