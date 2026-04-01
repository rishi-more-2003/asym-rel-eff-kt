"""Caregiver agent: 235B frozen expert.

Shared across all runs. Provides responses, semantic judging (via judge.py),
and LTM compression. Supports three execution modes:
  - combined:    judge + caregiver in a single 235B call (fastest)
  - speculative: fire judge + 2 caregiver variants in parallel
  - sequential:  judge, then caregiver (cheapest)
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.tinker_utils import (
    build_chat_prompt, build_chat_prompt_multi, fire_sample, await_future,
    parse_json,
)
from src.judge import JudgeResult

CONFIRM_ADDENDUM = (
    "\n\nThe child just performed a correct action. Acknowledge their "
    "success briefly and encourage them to continue with the next step."
)

CORRECT_ADDENDUM = (
    "\n\nThe child just made a mistake. The judge's assessment: "
    '"{explanation}"\n'
    "Help the child understand what went wrong and guide them toward "
    "the correct action, using your current scaffolding level."
)

COMBINED_SYSTEM = (
    "You are a knowledgeable parent who also evaluates the child's actions. "
    "First, judge whether the child's action accomplishes the intended step. "
    "Then, write your response to the child as a caring parent. "
    "Respond with ONLY a JSON object, no other text."
)

COMBINED_USER_TEMPLATE = """\
Intended action: {ground_truth}
Target object: {target_object}
Expected effects: {effects}

The child said: "{child_utterance}"

Your current teaching approach: {scaffolding}
{child_model_section}
Respond JSON:
{{
  "correct": true/false,
  "partial_credit": 0.0-1.0,
  "explanation": "brief internal assessment",
  "caregiver_response": "your spoken response to the child (personalize based on what you know about this child)"
}}"""


class CaregiverAgent:
    """Frozen 235B caregiver. Shared SamplingClient across all runs."""

    def __init__(self, sampling_client, tokenizer):
        self._client = sampling_client
        self._tokenizer = tokenizer

    # ── Combined mode (1 call = judge + response) ────────────────────

    def fire_combined(
        self,
        child_utterance: str,
        step: dict,
        scaffolding_desc: str,
        child_model_profile: str = "",
    ):
        """Fire a single call that both judges and responds (non-blocking)."""
        if child_model_profile:
            model_section = f"\n{child_model_profile}\n"
        else:
            model_section = ""
        user_text = COMBINED_USER_TEMPLATE.format(
            ground_truth=step.get("action", ""),
            target_object=step.get("target_object", ""),
            effects=", ".join(step.get("effects", [])),
            child_utterance=child_utterance,
            scaffolding=scaffolding_desc or "Give helpful guidance.",
            child_model_section=model_section,
        )
        prompt = build_chat_prompt(COMBINED_SYSTEM, user_text)
        return fire_sample(self._client, prompt, max_tokens=512, temperature=0.3)

    def parse_combined(self, raw: str) -> tuple[JudgeResult, str]:
        """Parse the combined judge+caregiver JSON response."""
        parsed = parse_json(raw)
        if parsed is None:
            return (
                JudgeResult(correct=False, partial_credit=0.3, explanation="parse error"),
                raw[:300],
            )
        judge = JudgeResult(
            correct=bool(parsed.get("correct", False)),
            partial_credit=float(parsed.get("partial_credit", 0.0)),
            explanation=str(parsed.get("explanation", "")),
        )
        response = str(parsed.get("caregiver_response", parsed.get("explanation", "")))
        return judge, response

    # ── Speculative mode (3 parallel calls) ──────────────────────────

    def fire_confirmation(
        self, system_prompt: str, messages: list[dict[str, str]],
    ):
        """Fire speculative confirmation response (non-blocking)."""
        confirm_messages = list(messages)
        if confirm_messages:
            last = confirm_messages[-1].copy()
            last["content"] = last["content"] + CONFIRM_ADDENDUM
            confirm_messages[-1] = last
        prompt = build_chat_prompt_multi(
            system_prompt, confirm_messages, tokenizer=self._tokenizer,
        )
        return fire_sample(self._client, prompt, max_tokens=512, temperature=0.7)

    def fire_correction(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        explanation: str,
    ):
        """Fire speculative correction response (non-blocking)."""
        correction_messages = list(messages)
        addendum = CORRECT_ADDENDUM.format(explanation=explanation)
        if correction_messages:
            last = correction_messages[-1].copy()
            last["content"] = last["content"] + addendum
            correction_messages[-1] = last
        prompt = build_chat_prompt_multi(
            system_prompt, correction_messages, tokenizer=self._tokenizer,
        )
        return fire_sample(self._client, prompt, max_tokens=512, temperature=0.7)

    # ── Sequential/fallback mode ─────────────────────────────────────

    async def respond_async(
        self, system_prompt: str, messages: list[dict[str, str]],
    ) -> str:
        """Non-speculative single response (sequential/fallback path)."""
        prompt = build_chat_prompt_multi(
            system_prompt, messages, tokenizer=self._tokenizer,
        )
        future = fire_sample(self._client, prompt, max_tokens=512, temperature=0.7)
        return await await_future(future, tokenizer=self._tokenizer)
