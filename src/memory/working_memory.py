"""M2: Working Memory -- sliding window of the last k dialogue turns.

Primed at episode start with the task description and top-3 BM25-retrieved
long-term memories. Rendered via tokenizer.apply_chat_template() for each
sample() call.
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class WorkingMemory:
    """Sliding window of the last k dialogue turns."""

    def __init__(self, max_turns: int = config.WORKING_MEMORY_SIZE):
        self.max_turns = max_turns
        self._turns: list[dict[str, str]] = []
        self._primer_count = 0

    def prime(self, task_description: str, ltm_entries: list[str] | None = None) -> None:
        """Set up the working memory at episode start."""
        self._turns = []
        parts = [f"Task: {task_description}"]
        if ltm_entries:
            parts.append("Relevant past experiences:")
            for i, entry in enumerate(ltm_entries, 1):
                parts.append(f"  {i}. {entry}")
        primer = "\n".join(parts)
        self._turns.append({"role": "user", "content": primer})
        self._primer_count = 1

    def add_turn(self, role: str, content: str) -> None:
        """Append a dialogue turn, evicting oldest non-primer turns if needed."""
        self._turns.append({"role": role, "content": content})
        while len(self._turns) - self._primer_count > self.max_turns:
            self._turns.pop(self._primer_count)

    def get_messages(self) -> list[dict[str, str]]:
        """Return the current message list for chat template rendering."""
        return list(self._turns)

    def get_transcript(self) -> list[dict[str, str]]:
        """Return all turns (including primer) for logging."""
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns) - self._primer_count

    def clear(self) -> None:
        self._turns = []
        self._primer_count = 0
