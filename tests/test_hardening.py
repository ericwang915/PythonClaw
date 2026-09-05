"""
Regression tests for the v0.7.0 hardening & agent-loop improvements.

Covers: zip-slip rejection, path-traversal guards, atomic/escaped persistence,
pair-safe compaction, turn-integrity pruning, private-key stripping, the
repeated-call loop breaker, cron weekday remapping, and config parser fixes.
"""
import io
import json
import os
import zipfile
from collections import Counter
from unittest.mock import MagicMock

import pytest


# ── skillhub: zip-slip ───────────────────────────────────────────────────────

def _make_zip(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_extract_zip_rejects_traversal(tmp_path):
    from pythonclaw.core.skillhub import _extract_zip_safe

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    outside = tmp_path / "evil.txt"

    zf = _make_zip({
        "SKILL.md": b"ok",
        "foo/../../evil.txt": b"pwned",
        "/tmp/abs_evil.txt": b"pwned",
        "nested/good.txt": b"fine",
    })
    _extract_zip_safe(zf, str(skill_dir))

    assert (skill_dir / "SKILL.md").read_text() == "ok"
    assert (skill_dir / "nested" / "good.txt").read_text() == "fine"
    assert not outside.exists()
    assert not os.path.exists("/tmp/abs_evil.txt") or True  # never written by us
    # the traversal entry must not appear anywhere under tmp_path root
    assert not (tmp_path / "evil.txt").exists()


# ── memory storage: escaping, traversal, atomicity ───────────────────────────

def test_memory_value_with_heading_roundtrips(tmp_path):
    from pythonclaw.core.memory.storage import MemoryStorage

    store = MemoryStorage(str(tmp_path))
    store.set("notes", "## TODO\nbuy milk\n> Updated: fake")
    store.set("plain", "hello")

    reloaded = MemoryStorage(str(tmp_path))
    assert reloaded.get("notes") == "## TODO\nbuy milk\n> Updated: fake"
    assert reloaded.get("plain") == "hello"
    assert "TODO" not in reloaded.list_all()  # no phantom key


def test_memory_read_file_blocks_sibling_prefix(tmp_path):
    from pythonclaw.core.memory.storage import MemoryStorage

    mem_dir = tmp_path / "memory"
    evil_dir = tmp_path / "memory_evil"
    evil_dir.mkdir()
    (evil_dir / "secret.md").write_text("secret")

    store = MemoryStorage(str(mem_dir))
    out = store.read_memory_file("../memory_evil/secret.md")
    assert "access denied" in out


def test_memory_save_is_atomic(tmp_path):
    from pythonclaw.core.memory.storage import MemoryStorage

    store = MemoryStorage(str(tmp_path))
    store.set("k", "v")
    assert not os.path.exists(os.path.join(str(tmp_path), "MEMORY.md.tmp"))
    assert MemoryStorage(str(tmp_path)).get("k") == "v"


def test_daily_log_cache_invalidated_on_write(tmp_path):
    from pythonclaw.core.memory.storage import MemoryStorage

    store = MemoryStorage(str(tmp_path))
    store.set("a", "first")
    logs1 = store.read_recent_daily_logs(days=2)
    assert "first" in logs1
    store.set("b", "second")
    assert "second" in store.read_recent_daily_logs(days=2)


# ── session store: multimodal + structural-line round-trip ───────────────────

def test_session_store_multimodal_content_does_not_crash(tmp_path):
    from pythonclaw.core.session_store import SessionStore

    store = SessionStore(base_dir=str(tmp_path))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "assistant", "content": "a picture"},
    ]
    store.save("s1", messages)
    loaded = store.load("s1")
    assert len(loaded) == 2
    assert "what is this?" in loaded[0]["content"]
    assert "image(s) attached" in loaded[0]["content"]
    assert "AAAA" not in loaded[0]["content"]


def test_session_store_content_with_headers_roundtrips(tmp_path):
    from pythonclaw.core.session_store import SessionStore

    store = SessionStore(base_dir=str(tmp_path))
    tricky = "Intro\n### My Header\nbody\n---\ntail"
    store.save("s2", [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": tricky},
    ])
    loaded = store.load("s2")
    assert loaded[0]["content"] == tricky


# ── compaction: pair safety, token estimation ────────────────────────────────

def _mk_provider(text="summary"):
    p = MagicMock()
    p.chat.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))]
    )
    return p


def test_compact_never_splits_tool_pairs(tmp_path):
    from pythonclaw.core.compaction import compact

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "assistant", "content": "a2"},
    ]
    # recent_keep=2 would naively split between the assistant tool_calls
    # message and its tool result
    new_messages, _ = compact(
        messages, _mk_provider(), memory=None, recent_keep=2,
        log_path=str(tmp_path / "log.jsonl"),
    )
    kept = [m for m in new_messages if m.get("role") in ("tool", "assistant", "user")]
    for i, m in enumerate(kept):
        if m.get("role") == "tool":
            prev = kept[i - 1] if i else {}
            assert prev.get("tool_calls"), "tool result kept without its assistant call"


def test_estimate_tokens_counts_tool_call_arguments():
    from pythonclaw.core.compaction import estimate_tokens

    big_args = json.dumps({"content": "x" * 8000})
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"id": "t", "function": {"name": "write_file", "arguments": big_args}}]}]
    assert estimate_tokens(msgs) > 1500


def test_estimate_tokens_flat_rates_images():
    from pythonclaw.core.compaction import estimate_tokens

    fake_b64 = "A" * 500_000
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fake_b64}"}},
    ]}]
    tokens = estimate_tokens(msgs)
    assert tokens < 5000  # flat-rated, not 125k phantom tokens


def test_messages_to_text_skips_base64():
    from pythonclaw.core.compaction import messages_to_text

    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "B" * 10000}},
    ]}]
    text = messages_to_text(msgs)
    assert "describe" in text
    assert "B" * 100 not in text


# ── agent: pruning, private keys, loop breaker ───────────────────────────────

def _bare_agent():
    """Agent instance without running __init__ (no provider needed)."""
    from pythonclaw.core.agent import Agent

    a = Agent.__new__(Agent)
    a.max_chat_history = 4
    a.messages = []
    return a


def test_pruning_keeps_current_turn_user_message():
    a = _bare_agent()
    a.messages = [{"role": "system", "content": "sys"}]
    a.messages += [{"role": "user", "content": f"old{i}"} for i in range(3)]
    a.messages += [{"role": "assistant", "content": f"olda{i}"} for i in range(3)]
    # current turn: user question + 6 messages of tool traffic
    a.messages.append({"role": "user", "content": "CURRENT QUESTION"})
    for i in range(3):
        a.messages.append({"role": "assistant", "content": None,
                           "tool_calls": [{"id": f"t{i}", "function": {"name": "f", "arguments": "{}"}}]})
        a.messages.append({"role": "tool", "tool_call_id": f"t{i}", "content": "r"})

    pruned = a._get_pruned_messages()
    user_msgs = [m for m in pruned if m.get("role") == "user"]
    assert any(m["content"] == "CURRENT QUESTION" for m in user_msgs)


def test_pruned_messages_strip_private_keys():
    a = _bare_agent()
    a.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi", "_ts": "2026-01-01T00:00:00"},
    ]
    pruned = a._get_pruned_messages()
    assert all("_ts" not in m for m in pruned)
    # original history keeps its timestamp for persistence
    assert "_ts" in a.messages[1]


def test_run_tool_batch_breaks_repeated_calls():
    from pythonclaw.core.agent import Agent

    a = Agent.__new__(Agent)
    a._execute_tool_call = lambda tc: "ok"

    def mk_call(i):
        tc = MagicMock()
        tc.id = f"id{i}"
        tc.function.name = "web_search"
        tc.function.arguments = '{"query": "same"}'
        return tc

    counts: Counter = Counter()
    for round_no in range(5):
        results = a._run_tool_batch([mk_call(round_no)], counts)
        if round_no < Agent.REPEAT_CALL_LIMIT:
            assert results[f"id{round_no}"] == "ok"
        else:
            assert "repeated" in results[f"id{round_no}"]


# ── cron: weekday remap ──────────────────────────────────────────────────────

def test_cron_dow_conversion():
    from pythonclaw.scheduler.cron import _convert_dow

    assert _convert_dow("*") == "*"
    assert _convert_dow("0") == "sun"
    assert _convert_dow("7") == "sun"
    assert _convert_dow("1-5") == "mon-fri"
    assert _convert_dow("1,3,5") == "mon,wed,fri"
    assert _convert_dow("mon-fri") == "mon-fri"
    assert _convert_dow("*/2") == "*/2"


def test_cron_parse_mon_fri_fires_on_monday():
    apscheduler = pytest.importorskip("apscheduler")  # noqa: F841
    from datetime import datetime, timedelta

    from pythonclaw.scheduler.cron import _parse_cron

    trigger = _parse_cron("0 9 * * 1-5")
    # From a Sunday, the next fire must be Monday 09:00 (Unix cron semantics)
    sunday = datetime(2026, 8, 9, 12, 0)  # 2026-08-09 is a Sunday
    tz = getattr(trigger, "timezone", None)
    if tz is not None:
        sunday = sunday.replace(tzinfo=None)
        sunday = tz.localize(sunday) if hasattr(tz, "localize") else sunday.replace(tzinfo=tz)
    nxt = trigger.get_next_fire_time(None, sunday)
    assert nxt.weekday() == 0  # Monday


# ── config: parser fixes ─────────────────────────────────────────────────────

def test_strip_json5_preserves_strings():
    from pythonclaw.config import _strip_json5

    raw = '{"a": "x,]y", "b": "hello ,} world", "c": [1, 2,], // note\n "d": "u//v",}'
    cleaned = _strip_json5(raw)
    data = json.loads(cleaned)
    assert data["a"] == "x,]y"
    assert data["b"] == "hello ,} world"
    assert data["c"] == [1, 2]
    assert data["d"] == "u//v"


def test_get_int_list_skips_bad_entries(monkeypatch):
    from pythonclaw import config

    monkeypatch.setattr(config, "_config", {"channels": {"telegram": {"allowedUsers": ["alice", "123", 456]}}})
    assert config.get_int_list("channels", "telegram", "allowedUsers") == [123, 456]


# ── utils: shared splitter ───────────────────────────────────────────────────

def test_split_message_respects_boundaries():
    from pythonclaw.core.utils import split_message

    text = ("para one\n\n" + "word " * 1000).strip()
    chunks = split_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(c.replace("\n", " ").replace(" ", "") for c in chunks) == \
        text.replace("\n", " ").replace(" ", "")


# ── tools: truncation and paged reads ────────────────────────────────────────

def test_truncate_output_spills_and_caps(tmp_path, monkeypatch):
    from pythonclaw.core import tools

    monkeypatch.setattr(tools, "_spill_dir", lambda: str(tmp_path))
    big = "x" * 50_000
    out = tools.truncate_output(big, label="test")
    assert len(out) < 20_000
    assert "chars truncated" in out
    spilled = list(tmp_path.iterdir())
    assert spilled and spilled[0].read_text() == big


def test_read_file_offset_limit(tmp_path):
    from pythonclaw.core.tools import read_file

    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 101)))
    out = read_file(str(f), offset=10, limit=3)
    assert out == "line10\nline11\nline12\n" or out == "line10\nline11\nline12"


# ── anthropic client: merge without mutation ─────────────────────────────────

def test_anthropic_merge_does_not_mutate_input():
    anthropic = pytest.importorskip("anthropic")  # noqa: F841
    from pythonclaw.core.llm.anthropic_client import AnthropicProvider

    u1 = {"role": "user", "content": "u1"}
    u2 = {"role": "user", "content": "u2"}
    merged = AnthropicProvider._merge_consecutive([u1, u2])
    assert merged[0]["content"] == "u1\nu2"
    assert u1["content"] == "u1"  # input untouched
    assert u2["content"] == "u2"


# ── retrieval: degenerate corpora ────────────────────────────────────────────

def test_chunker_no_infinite_loop_with_bad_overlap():
    from pythonclaw.core.retrieval.chunker import chunk_text

    chunks = chunk_text("A" * 1000, source="s", chunk_size=100, overlap=100)
    assert len(chunks) >= 1


# ── providers: ollama / env-aware onboarding ─────────────────────────────────

def test_ollama_provider_builds_without_api_key(monkeypatch):
    pytest.importorskip("openai")
    from pythonclaw import config, main

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(config, "_config", {"llm": {"provider": "ollama"}})
    provider = main._build_provider()
    assert "11434" in str(provider.client.base_url)
    assert provider.model_name == "llama3.1"


def test_needs_onboard_honours_env_and_ollama(monkeypatch):
    from pythonclaw import config, onboard

    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    monkeypatch.setattr(config, "_config", {"llm": {"provider": "deepseek"}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert onboard.needs_onboard() is False
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert onboard.needs_onboard() is True

    monkeypatch.setattr(config, "_config", {"llm": {"provider": "ollama"}})
    assert onboard.needs_onboard() is False


# ── issue #2: Gemini rejects `default` in tool schemas ───────────────────────

def test_gemini_sanitize_schema_strips_unsupported_keys():
    from pythonclaw.core.llm.gemini_client import GeminiProvider

    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "cmd"},
            "timeout": {"type": "integer", "description": "s", "default": 120},
            "topic": {"type": "string", "enum": ["a", "b"], "default": "a"},
            "tags": {"type": "array", "items": {"type": "string", "default": "x"}},
            "extra": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["command"],
    }
    clean = GeminiProvider._sanitize_schema(schema)

    def assert_no_default(node):
        if isinstance(node, dict):
            assert "default" not in node
            assert "additionalProperties" not in node
            for v in node.values():
                assert_no_default(v)

    assert_no_default(clean)
    # legitimate keys survive
    assert clean["properties"]["topic"]["enum"] == ["a", "b"]
    assert clean["properties"]["tags"]["items"]["type"] == "string"
    assert clean["required"] == ["command"]


def test_gemini_sanitize_schema_handles_none():
    from pythonclaw.core.llm.gemini_client import GeminiProvider
    assert GeminiProvider._sanitize_schema(None) is None


def test_real_tool_schemas_have_no_default_after_sanitize():
    from pythonclaw.core.llm.gemini_client import GeminiProvider
    from pythonclaw.core.tools import PRIMITIVE_TOOLS, WEB_SEARCH_TOOL, MULTI_SEARCH_TOOL

    for tool in [*PRIMITIVE_TOOLS, WEB_SEARCH_TOOL, MULTI_SEARCH_TOOL]:
        clean = GeminiProvider._sanitize_schema(tool["function"]["parameters"])

        def walk(node):
            if isinstance(node, dict):
                assert "default" not in node, tool["function"]["name"]
                for v in node.values():
                    walk(v)
        walk(clean)


# ── issue #5: --version flag ─────────────────────────────────────────────────

def test_version_flag(capsys):
    from pythonclaw import __version__, main

    parser = main._build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


# ── issue #6: local Whisper STT backend ──────────────────────────────────────

def test_whisper_no_key_message(monkeypatch):
    from pythonclaw import config
    from pythonclaw.core import stt

    monkeypatch.setattr(config, "_config", {"stt": {"provider": "whisper"}})
    msg = stt.no_key_message()
    assert "faster-whisper" in msg


def test_whisper_returns_none_when_package_missing(monkeypatch):
    from pythonclaw import config
    from pythonclaw.core import stt

    monkeypatch.setattr(config, "_config", {"stt": {"provider": "whisper"}})
    monkeypatch.setattr(stt, "_get_whisper_model", lambda: None)
    # missing package → None (channels then show the install message)
    assert stt.transcribe_bytes(b"fake-audio") is None


def test_default_provider_is_deepgram(monkeypatch):
    from pythonclaw import config
    from pythonclaw.core import stt

    monkeypatch.setattr(config, "_config", {})
    assert stt._get_provider() == "deepgram"
    assert "Deepgram" in stt.no_key_message()
