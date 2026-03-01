"""
PersistentAgent — an Agent subclass that automatically saves its message
history to a SessionStore after every chat() or compact() call.

On construction it restores the previous conversation from the store so that
sessions survive server restarts.

Restoration strategy
--------------------
  messages[0]   — always rebuilt fresh by Agent.__init__ (soul + persona + skills)
  messages[1:]  — restored from the Markdown session store

This means soul/persona/skill changes take effect on the next restart while
the full conversation history (including compaction summaries and skill
injection messages) is preserved.

Timestamps
----------
Each message carries a ``_ts`` field (ISO 8601) that records when it was
created.  This enables time-based truncation in the SessionStore.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from .agent import Agent

if TYPE_CHECKING:
    from .session_store import SessionStore

logger = logging.getLogger(__name__)


class PersistentAgent(Agent):
    """Agent that auto-saves to and restores from a Markdown SessionStore."""

    def __init__(
        self,
        *args,
        store: "SessionStore",
        session_id: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._store = store
        self._session_id = session_id
        self._restore()

    # ── Restore ──────────────────────────────────────────────────────────────

    def _restore(self) -> None:
        """Load saved messages and merge with the freshly built system prompt."""
        saved = self._store.load(self._session_id)
        if not saved:
            return

        initial_system = self.messages[0]   # freshly built system prompt
        self.messages = [initial_system] + saved

        # Re-infer which skills were loaded so _use_skill doesn't double-inject
        for msg in saved:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # Support both old ("Skill Enabled:") and new ("SKILL ACTIVATED:") formats
                m = re.search(r"(?:Skill Enabled|SKILL ACTIVATED):\s*(.+)", content)
                if m:
                    self.loaded_skill_names.add(m.group(1).strip().rstrip("]"))

        logger.info(
            "[PersistentAgent] Restored session '%s': %d messages, %d skills",
            self._session_id, len(saved), len(self.loaded_skill_names),
        )

    # ── Timestamp injection ──────────────────────────────────────────────────

    @staticmethod
    def _ensure_ts(msg: dict) -> dict:
        """Add a ``_ts`` field to a message if it doesn't have one."""
        if "_ts" not in msg:
            msg["_ts"] = datetime.now().isoformat(timespec="seconds")
        return msg

    # ── Auto-save ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        # Ensure every message has a timestamp before saving
        for msg in self.messages[1:]:
            self._ensure_ts(msg)
        self._store.save(self._session_id, self.messages)

    def chat(self, user_input: str) -> str:
        response = super().chat(user_input)
        self._save()
        return response

    def compact(self, instruction: str | None = None) -> str:
        result = super().compact(instruction)
        self._save()
        return result
