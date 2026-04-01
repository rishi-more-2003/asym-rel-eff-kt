"""Adaptive scaffolding with competence tracker and 3 hint levels.

Tracks per-category competence as a rolling average of partial credit
over the last N episodes. Three scaffolding levels control the
caregiver's hint style, implementing Vygotsky's ZPD.
"""

from __future__ import annotations

import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

FULL_SUPPORT = (
    "The child is a beginner at this type of task. Give specific, "
    "step-by-step guidance. Name the exact objects and actions needed. "
    "Break down complex steps into smaller ones."
)

GUIDED_DISCOVERY = (
    "The child has some experience with this type of task. Ask guiding "
    "questions rather than giving direct answers. Hint at the next step "
    "without naming it explicitly. Encourage the child to reason about "
    "what comes next."
)

MINIMAL_SUPPORT = (
    "The child is becoming skilled at this type of task. Only help if "
    "directly asked. Offer encouragement rather than instructions. "
    "Let the child work through challenges independently."
)

LEVEL_NAMES = {
    "full_support": FULL_SUPPORT,
    "guided_discovery": GUIDED_DISCOVERY,
    "minimal_support": MINIMAL_SUPPORT,
}


class CompetenceTracker:
    """Tracks per-category competence as a rolling average of partial credit."""

    def __init__(
        self,
        window: int = config.SCAFFOLDING_WINDOW,
        thresholds: tuple[float, float] = config.SCAFFOLDING_THRESHOLDS,
    ):
        self.window = window
        self.low_threshold, self.high_threshold = thresholds
        self._history: dict[str, list[float]] = defaultdict(list)

    def update(self, category: str, partial_credit: float) -> None:
        history = self._history[category]
        history.append(partial_credit)
        if len(history) > self.window:
            self._history[category] = history[-self.window:]

    def competence(self, category: str) -> float:
        history = self._history.get(category, [])
        if not history:
            return 0.0
        return sum(history) / len(history)

    def scaffolding_level(self, category: str) -> str:
        c = self.competence(category)
        if c < self.low_threshold:
            return "full_support"
        elif c < self.high_threshold:
            return "guided_discovery"
        return "minimal_support"

    def scaffolding_prompt(self, category: str) -> str:
        return LEVEL_NAMES[self.scaffolding_level(category)]

    def to_dict(self) -> dict:
        return dict(self._history)

    def load_dict(self, d: dict) -> None:
        self._history = defaultdict(list, {k: list(v) for k, v in d.items()})
