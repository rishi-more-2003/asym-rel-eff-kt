"""Semantic action judge using the 235B model.

Instead of fragile string matching, the caregiver model judges whether
the child's action attempt achieves the intended step. Returns structured
verdicts with partial credit and explanations.
"""

from __future__ import annotations

import sys, os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.tinker_utils import (
    build_chat_prompt, fire_sample, await_future, parse_json,
)

JUDGE_SYSTEM = (
    "You are an impartial evaluator of a child's household task attempt. "
    "Given the intended action step and the child's actual utterance, "
    "determine whether the child's action accomplishes the intended step. "
    "Be reasonable: if the child's phrasing is different but semantically "
    "achieves the same outcome, count it as correct. "
    "Respond with ONLY a JSON object, no other text."
)

JUDGE_USER_TEMPLATE = """\
Intended action: {ground_truth}
Target object: {target_object}
Preconditions that must hold: {preconditions}
Expected effects: {effects}

The child said: "{child_utterance}"

Does the child's action accomplish the intended step?
Respond JSON: {{"correct": true/false, "partial_credit": 0.0-1.0, "explanation": "brief reason"}}"""


@dataclass
class JudgeResult:
    correct: bool
    partial_credit: float
    explanation: str


class SemanticJudge:
    """Uses the 235B model to evaluate child action attempts."""

    def __init__(self, caregiver_client, tokenizer):
        self._client = caregiver_client
        self._tokenizer = tokenizer

    def _build_prompt(self, child_utterance: str, step: dict) -> str:
        return JUDGE_USER_TEMPLATE.format(
            ground_truth=step.get("action", ""),
            target_object=step.get("target_object", ""),
            preconditions=", ".join(step.get("preconditions", [])),
            effects=", ".join(step.get("effects", [])),
            child_utterance=child_utterance,
        )

    def fire_judge(self, child_utterance: str, step: dict):
        """Fire-and-forget: submit the judge request, return the future."""
        user_text = self._build_prompt(child_utterance, step)
        prompt = build_chat_prompt(JUDGE_SYSTEM, user_text)
        return fire_sample(
            self._client, prompt,
            max_tokens=256, temperature=0.1,
        )

    async def evaluate_async(self, child_utterance: str, step: dict) -> JudgeResult:
        """Submit judge request and await result."""
        future = self.fire_judge(child_utterance, step)
        raw = await await_future(future, tokenizer=self._tokenizer)
        return self._parse_result(raw)

    def _parse_result(self, raw: str) -> JudgeResult:
        parsed = parse_json(raw)
        if parsed is None:
            lower = raw.lower()
            correct = "true" in lower and "correct" in lower
            return JudgeResult(correct=correct, partial_credit=0.5, explanation=raw[:200])
        return JudgeResult(
            correct=bool(parsed.get("correct", False)),
            partial_credit=float(parsed.get("partial_credit", 0.0)),
            explanation=str(parsed.get("explanation", "")),
        )
