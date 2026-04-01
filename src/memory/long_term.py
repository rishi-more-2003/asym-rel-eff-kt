"""M3: Long-Term Memory -- episodic store with BM25 retrieval and compression.

After each episode, a compression step (one 235B sample call) produces a
2-3 sentence structured entry. Retrieval selects top-3 entries by BM25
against the new task description. Gated by salience > tau.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.bm25 import bm25_score

COMPRESS_SYSTEM = (
    "You are a memory consolidation module. Given an episode transcript, "
    "produce a concise 2-3 sentence summary capturing: (1) the task and its "
    "category, (2) what the child struggled with or did not know, "
    "(3) key explanations or corrections from the caregiver, and "
    "(4) whether the task succeeded. Output ONLY the summary text."
)


@dataclass
class EpisodicEntry:
    episode_id: int
    task_id: str
    category: str
    summary: str
    reward: float
    salience: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EpisodicEntry:
        return cls(**d)


class LongTermMemory:
    """Episodic store with BM25 retrieval and 235B-driven compression."""

    def __init__(self):
        self.entries: list[EpisodicEntry] = []

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k most relevant past summaries by BM25 similarity."""
        if not self.entries:
            return []
        scored = [
            (bm25_score(query, e.summary), e) for e in self.entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e.summary for _, e in scored[:top_k]]

    def add(self, entry: EpisodicEntry) -> None:
        self.entries.append(entry)

    @property
    def size(self) -> int:
        return len(self.entries)

    def build_compress_prompt(self, transcript: list[dict[str, str]]) -> tuple[str, str]:
        """Build the (system, user) prompt pair for episode compression."""
        lines = []
        for turn in transcript:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")
        user_text = "\n".join(lines)
        return COMPRESS_SYSTEM, user_text

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [EpisodicEntry.from_dict(d) for d in data]
