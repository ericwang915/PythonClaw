"""
Regression tests for issue #3: a script executed via run_command that calls
``subprocess.Popen(cmd, timeout=N)`` crashes with

    TypeError: Popen.__init__() got an unexpected keyword argument 'timeout'

The only salient word in that message is "timeout", which users read as a
network/API timeout ("PythonClaw keeps giving me timeout errors even after
I've bound the API").  run_command must attach a hint that names the real
cause so the agent can fix the script and the user is not misled.
"""
import sys

import pytest

from pythonclaw.core import tools
from pythonclaw.core.tools import _script_error_hint, run_command


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_files_dir", lambda: str(tmp_path))
    monkeypatch.setattr(tools, "_spill_dir", lambda: str(tmp_path))


# ── _script_error_hint unit tests ────────────────────────────────────────────

def test_hint_for_popen_timeout_typeerror():
    err = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "TypeError: Popen.__init__() got an unexpected keyword argument 'timeout'"
    )
    hint = _script_error_hint(err)
    assert "NOT a network/API timeout" in hint
    assert "subprocess.run(cmd, timeout=N)" in hint
    assert "communicate(timeout=N)" in hint


def test_hint_matches_bare_exception_message():
    # Scripts with a catch-all `except Exception as exc: print(exc)` surface
    # the message without the "TypeError:" prefix — the hint must still fire.
    err = "execution failed: Popen.__init__() got an unexpected keyword argument 'timeout'"
    assert "NOT a network/API timeout" in _script_error_hint(err)


def test_hint_for_other_popen_kwarg_is_generic():
    err = "TypeError: Popen.__init__() got an unexpected keyword argument 'capture_output'"
    hint = _script_error_hint(err)
    assert "`capture_output`" in hint
    assert "NOT a network/API timeout" not in hint


def test_no_hint_for_unrelated_errors():
    assert _script_error_hint("Error (exit 1):\nNameError: name 'x' is not defined") == ""
    assert _script_error_hint("") == ""


# ── run_command integration ──────────────────────────────────────────────────

def test_run_command_appends_hint_on_popen_timeout_crash():
    code = "import subprocess; subprocess.Popen(['echo', 'hi'], timeout=5)"
    out = run_command(f'{sys.executable} -c "{code}"')
    assert "unexpected keyword argument 'timeout'" in out
    assert "[hint]" in out
    assert "NOT a network/API timeout" in out


def test_run_command_appends_hint_when_script_swallows_error():
    # Script catches the TypeError, prints it, and exits 0.
    code = (
        "import subprocess\n"
        "try:\n"
        "    subprocess.Popen(['echo', 'hi'], timeout=5)\n"
        "except Exception as exc:\n"
        "    print(f'execution failed: {exc}')\n"
    )
    out = run_command(f'{sys.executable} -c "{code}"')
    assert "[hint]" in out
    assert "NOT a network/API timeout" in out


def test_run_command_no_hint_on_clean_output():
    out = run_command(f'{sys.executable} -c "print(42)"')
    assert "42" in out
    assert "[hint]" not in out
