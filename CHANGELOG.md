# Changelog

All notable changes to PythonClaw are documented here.

## [0.7.2] — 2026-09-05

### Fixed
- **Gemini function calling crashed on any tool with a `default` value** — the Gemini `Schema` proto rejects `default`/`additionalProperties`; tool schemas are now sanitized to only the fields Gemini accepts ([#2](https://github.com/ericwang915/PythonClaw/issues/2))

### Added
- **`pythonclaw --version` / `-V`** flag ([#5](https://github.com/ericwang915/PythonClaw/issues/5))
- **Local Whisper STT backend** — set `stt.provider: "whisper"` to transcribe voice 100% locally via `faster-whisper`, no API key ([#6](https://github.com/ericwang915/PythonClaw/issues/6))

## [0.7.1] — 2026-08-08

### Added
- **Ollama provider** — run 100% local, no API key (`LLM_PROVIDER=ollama`), now also selectable in the `onboard` wizard
- **`custom` provider** — any OpenAI-compatible endpoint (OpenAI, OpenRouter, LM Studio, vLLM, llama.cpp server)
- **Docker support** — `Dockerfile` + `docker-compose.yml`, env-first configuration (`PYTHONCLAW_WEB_HOST`, `PYTHONCLAW_WEB_PORT`)
- Onboard detection now honours env-var configuration, so containers never hang on the interactive wizard
- Reworked README: sharper value prop, "Why PythonClaw", quick-start with the local one-liner, table of contents, corrected security defaults

## [0.7.0] — 2026-08-08

Security release — please upgrade. Full notes: [v0.7.0 release](https://github.com/ericwang915/PythonClaw/releases/tag/v0.7.0).

### Security
- Fixed zip-slip arbitrary file write in ClawHub skill installs (sync + async)
- Enabled TLS verification on async hub downloads
- Web dashboard binds to `127.0.0.1` by default (was `0.0.0.0` with no auth)
- Path-traversal guards for `memory_get` and skill resource lookups
- File-send callback is per-agent — no more cross-session file delivery

### Agent loop
- Tool-output truncation with paged spill-over files; `read_file` offset/limit; `run_command` timeout arg
- Turn-integrity pruning — the current turn can't slide out of the context window
- Loop breaker for repeated identical tool calls
- Non-blocking tool timeouts
- Pair-safe compaction, summary roll-off, tool-args-aware token estimation
- `multi_search` — parallel web-search fan-out

### Fixed
- Gemini tool use (referenced non-existent SDK symbols), parallel tool-result batching
- Streamed conversations now persist across restarts
- Anthropic: `tool_choice="none"`, truncated tool-JSON hardening, setup-token OAuth, copy-on-merge
- Session store: atomic writes, multimodal messages, `###`/`---` round-trip, `_ts` stripped from API payloads
- Memory store: atomic writes, `## ` heading values, daily-log cache invalidation
- Cron weekday remapping (`0 9 * * 1-5` now fires Mon–Fri)
- Config: string-safe JSON5, resilient int parsing, malformed config no longer bricks the CLI
- Telegram/WhatsApp/Discord channel hardening (locks, flood control, webhook fast-ack, allowlists)

## [0.6.6] — 2026-08-05
- Voice input (Deepgram), PDF skills, `send_file`, files management

## [0.6.5]
- Download real skill content from ClawHub

## [0.6.4]
- Cleaner Telegram streaming, generous timeouts

## [0.6.3]
- Plan & Execute mode, cleaner Telegram output
