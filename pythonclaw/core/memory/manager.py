"""
MemoryManager — long-term key-value memory with hybrid RAG recall.

Storage
-------
Memories are stored as Markdown files:
  - MEMORY.md        — curated long-term memory (latest value per key)
  - YYYY-MM-DD.md    — daily append-only log

When writing, both MEMORY.md and today's daily log are updated.
When reading, MEMORY.md is the source of truth (holds latest per key).
Conflict resolution: if the same key is written multiple times, the most
recent write wins (MEMORY.md is always overwritten with the latest value).

Recall
------
When a specific query is given, the manager converts every memory entry into a
short "chunk"  ("{key}: {value}")  and runs hybrid sparse + dense retrieval to
return the most relevant ones.  When the query is empty or "*", ALL memories
are returned (full-dump mode, used by compaction and legacy callers).
"""

from __future__ import annotations

import logging

from .storage import MemoryStorage
from ..retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_DUMP_TRIGGERS = {"", "*", "all", "everything"}


class MemoryManager:
    """
    Manages long-term memories stored as Markdown files.

    Parameters
    ----------
    memory_dir : path to the memory directory (contains MEMORY.md + daily logs).
    use_dense  : include embedding retrieval for recall (False by default for
                 memory — BM25 alone is fast and sufficient for small corpora).
    """

    def __init__(
        self,
        memory_dir: str | None = None,
        use_dense: bool = False,
    ) -> None:
        import os

        if memory_dir is None:
            home = os.path.expanduser("~")
            memory_dir = os.path.join(home, ".ada", "memory")

        self.storage = MemoryStorage(memory_dir)
        self._use_dense = use_dense

    # ── Core operations ──────────────────────────────────────────────────────

    def remember(self, content: str, key: str | None = None) -> str:
        """Store *content* under *key* in long-term memory."""
        if not key:
            raise ValueError("Key is required for memory storage.")
        self.storage.set(key, content)
        return f"Memory stored: [{key}] = {content}"

    def recall(self, query: str, top_k: int = 10) -> str:
        """
        Retrieve memories relevant to *query*.

        - If query is empty / "*" / "all" → returns ALL memories (full dump).
        - Otherwise → runs hybrid BM25 (+ optional dense) retrieval and
          returns the top *top_k* most relevant entries.
        """
        all_memories = self.storage.list_all()
        if not all_memories:
            return "No memories found."

        # Full-dump mode
        if query.strip().lower() in _DUMP_TRIGGERS:
            lines = [f"- {k}: {v}" for k, v in all_memories.items()]
            return "\n".join(lines)

        # Smart retrieval
        corpus = [
            {"source": k, "content": f"{k}: {v}"}
            for k, v in all_memories.items()
        ]

        retriever = HybridRetriever(
            provider=None,          # no LLM re-ranker for memory
            use_sparse=True,
            use_dense=self._use_dense,
            use_reranker=False,
        )
        retriever.fit(corpus)
        hits = retriever.retrieve(query, top_k=top_k)

        if not hits:
            logger.debug("[MemoryManager] No RAG hits for '%s', returning all.", query)
            lines = [f"- {k}: {v}" for k, v in all_memories.items()]
            return "(No close match found; showing all memories)\n" + "\n".join(lines)

        lines = [f"- {h['source']}: {h['content'].split(': ', 1)[-1]}" for h in hits]
        return "\n".join(lines)

    def forget(self, key: str) -> str:
        """Remove a memory entry by key."""
        if self.storage.get(key) is not None:
            self.storage.delete(key)
            return f"Forgot: {key}"
        return f"Nothing found for: {key}"

    # ── Helpers used by compaction ───────────────────────────────────────────

    def list_all(self) -> dict:
        """Return the raw {key: value} dict (used by compaction.memory_flush)."""
        return self.storage.list_all()
