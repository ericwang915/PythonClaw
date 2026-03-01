"""
Daemon process lifecycle manager for PythonClaw.

Handles starting the agent as a background process, writing a PID file,
stopping via SIGTERM, and querying status.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config

# ── Paths ────────────────────────────────────────────────────────────────────

PYTHONCLAW_DIR = Path.home() / ".pythonclaw"
PID_FILE = PYTHONCLAW_DIR / "pythonclaw.pid"
LOG_FILE = PYTHONCLAW_DIR / "daemon.log"
META_FILE = PYTHONCLAW_DIR / "daemon.meta.json"


def _ensure_dir() -> None:
    PYTHONCLAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ───────────────────────────────────────────────────────────────

def start_daemon(
    channels: list[str] | None = None,
    config_path: str | None = None,
) -> None:
    """Start PythonClaw as a background daemon.

    Spawns a detached subprocess running ``pythonclaw start --foreground``
    and writes its PID to ``~/.pythonclaw/pythonclaw.pid``.
    """
    pid = read_pid()
    if pid and _is_alive(pid):
        port = _read_meta().get("port", 7788)
        print(f"[PythonClaw] Already running (PID {pid}).")
        print(f"[PythonClaw] Dashboard: http://localhost:{port}")
        return

    _ensure_dir()

    cmd = [sys.executable, "-m", "pythonclaw", "start", "--foreground"]
    if config_path:
        cmd += ["--config", config_path]
    if channels:
        cmd += ["--channels"] + channels

    log_handle = open(LOG_FILE, "a", encoding="utf-8")
    log_handle.write(f"\n{'='*60}\n")
    log_handle.write(f"Starting PythonClaw daemon at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_handle.write(f"Command: {' '.join(cmd)}\n")
    log_handle.write(f"{'='*60}\n")
    log_handle.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=os.getcwd(),
    )

    _write_pid(proc.pid)

    port = config.get_int("web", "port", default=7788)
    _write_meta({
        "pid": proc.pid,
        "port": port,
        "cwd": os.getcwd(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channels": channels or [],
        "config_path": config_path,
    })

    # Brief pause to check the process didn't immediately crash
    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"[PythonClaw] Daemon failed to start (exit code {proc.returncode}).")
        print(f"[PythonClaw] Check logs: {LOG_FILE}")
        _cleanup_pid()
        return

    print(f"[PythonClaw] Daemon started (PID {proc.pid}).")
    print(f"[PythonClaw] Dashboard: http://localhost:{port}")
    print(f"[PythonClaw] Logs: {LOG_FILE}")


def stop_daemon() -> None:
    """Stop the running PythonClaw daemon."""
    pid = read_pid()
    if not pid:
        print("[PythonClaw] No daemon is running.")
        return

    if not _is_alive(pid):
        print(f"[PythonClaw] Stale PID file (process {pid} not found). Cleaning up.")
        _cleanup_pid()
        return

    print(f"[PythonClaw] Stopping daemon (PID {pid})...", end=" ", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("already stopped.")
        _cleanup_pid()
        return

    # Wait up to 10 seconds for graceful shutdown
    for _ in range(20):
        time.sleep(0.5)
        if not _is_alive(pid):
            break

    if _is_alive(pid):
        print("forcing...", end=" ", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.5)

    _cleanup_pid()
    print("stopped.")


def daemon_status() -> dict | None:
    """Check daemon status.  Returns metadata dict or None."""
    pid = read_pid()
    if not pid:
        return None

    if not _is_alive(pid):
        _cleanup_pid()
        return None

    meta = _read_meta()
    meta["pid"] = pid
    meta["alive"] = True

    started = meta.get("started_at", "")
    if started:
        try:
            from datetime import datetime
            start_dt = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
            uptime_sec = int((datetime.now() - start_dt).total_seconds())
            hours, remainder = divmod(uptime_sec, 3600)
            minutes, seconds = divmod(remainder, 60)
            meta["uptime"] = f"{hours}h {minutes}m {seconds}s"
        except Exception:
            meta["uptime"] = "unknown"

    return meta


def print_status() -> None:
    """Print daemon status to stdout."""
    status = daemon_status()
    if not status:
        print("[PythonClaw] No daemon is running.")
        return

    print("[PythonClaw] Daemon Status")
    print(f"  PID      : {status['pid']}")
    print(f"  Uptime   : {status.get('uptime', 'unknown')}")
    print(f"  Port     : {status.get('port', '?')}")
    print(f"  CWD      : {status.get('cwd', '?')}")
    print(f"  Started  : {status.get('started_at', '?')}")
    channels = status.get("channels", [])
    if channels:
        print(f"  Channels : {', '.join(channels)}")
    print(f"  Dashboard: http://localhost:{status.get('port', 7788)}")
    print(f"  Logs     : {LOG_FILE}")


# ── Internal helpers ─────────────────────────────────────────────────────────

def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    _ensure_dir()
    PID_FILE.write_text(str(pid) + "\n")


def _cleanup_pid() -> None:
    PID_FILE.unlink(missing_ok=True)
    META_FILE.unlink(missing_ok=True)


def _write_meta(meta: dict) -> None:
    _ensure_dir()
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _read_meta() -> dict:
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
