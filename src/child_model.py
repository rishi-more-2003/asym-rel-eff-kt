"""Caregiver's evolving model of the child's knowledge state.

This is what makes the Relational condition genuinely relational.
Rather than being just a salience weight, the relationship becomes
bidirectional: the caregiver maintains a persistent mental model of
THIS specific child and adapts its teaching accordingly.

Feature A: The caregiver prompt includes a personalized child profile
           (strengths, weaknesses, common mistakes, recent improvements).
Feature B: After each episode, the caregiver reflects on what it learned
           about the child, updating a structured "theory of child" that
           persists across episodes and gets injected into future context.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.tinker_utils import (
    build_chat_prompt, fire_sample, await_future, parse_json,
    sample_chat_async,
)

REFLECT_SYSTEM = (
    "You are a parent reflecting on a teaching episode with your child. "
    "Based on the episode, update your understanding of the child. "
    "Respond with ONLY a JSON object, no other text."
)

REFLECT_USER_TEMPLATE = """\
Episode summary:
- Task: {task_goal} (category: {category}, difficulty: {difficulty})
- Steps completed: {steps_completed}/{total_steps}
- Reward: {reward:.2f}

Child's actions during the episode:
{child_actions}

Your current understanding of the child:
{current_profile}

Based on this episode, update your model of the child.
Respond JSON:
{{
  "new_knowledge_gaps": ["specific things the child doesn't understand yet"],
  "new_mistakes": ["specific mistake patterns you noticed"],
  "improvements": ["things the child did better than before"],
  "teaching_notes": "one sentence about how to adjust your teaching for this child"
}}"""


@dataclass
class ChildModel:
    """The caregiver's theory of the child -- updated after every episode."""

    category_scores: dict[str, float] = field(default_factory=dict)
    knowledge_gaps: list[str] = field(default_factory=list)
    common_mistakes: list[str] = field(default_factory=list)
    recent_improvements: list[str] = field(default_factory=list)
    teaching_notes: list[str] = field(default_factory=list)
    episodes_observed: int = 0
    episode_history: list[dict] = field(default_factory=list)

    _MAX_GAPS = 10
    _MAX_MISTAKES = 8
    _MAX_IMPROVEMENTS = 5
    _MAX_NOTES = 5
    _MAX_HISTORY = 20
    _REFLECT_EVERY = 5

    def update_scores(self, category: str, reward: float) -> None:
        """Update per-category competence (exponential moving average)."""
        alpha = 0.3
        prev = self.category_scores.get(category, 0.0)
        self.category_scores[category] = alpha * reward + (1 - alpha) * prev

    @property
    def overall_level(self) -> str:
        if not self.category_scores:
            return "beginner"
        avg = sum(self.category_scores.values()) / len(self.category_scores)
        if avg < 0.3:
            return "beginner"
        elif avg < 0.7:
            return "developing"
        return "proficient"

    @property
    def strengths(self) -> list[tuple[str, float]]:
        return sorted(
            ((c, s) for c, s in self.category_scores.items() if s >= 0.5),
            key=lambda x: x[1], reverse=True,
        )[:5]

    @property
    def weaknesses(self) -> list[tuple[str, float]]:
        return sorted(
            ((c, s) for c, s in self.category_scores.items() if s < 0.5),
            key=lambda x: x[1],
        )[:5]

    def generate_profile(self) -> str:
        """Generate a natural-language child profile for the caregiver prompt."""
        if self.episodes_observed == 0:
            return (
                "This is a new child you haven't worked with before. "
                "You have no prior knowledge of their abilities."
            )

        lines = [
            f"Your understanding of this child "
            f"(based on {self.episodes_observed} episodes together):",
            f"Overall level: {self.overall_level}",
        ]

        if self.strengths:
            s_strs = [f"{c} ({s:.0%})" for c, s in self.strengths]
            lines.append(f"Strengths: {', '.join(s_strs)}")

        if self.weaknesses:
            w_strs = [f"{c} ({s:.0%})" for c, s in self.weaknesses]
            lines.append(f"Areas needing help: {', '.join(w_strs)}")

        if self.knowledge_gaps:
            lines.append(f"Knowledge gaps you've noticed:")
            for gap in self.knowledge_gaps[-5:]:
                lines.append(f"  - {gap}")

        if self.common_mistakes:
            lines.append(f"Mistake patterns:")
            for m in self.common_mistakes[-4:]:
                lines.append(f"  - {m}")

        if self.recent_improvements:
            lines.append(f"Recent improvements:")
            for imp in self.recent_improvements[-3:]:
                lines.append(f"  - {imp}")

        if self.teaching_notes:
            lines.append(f"Your teaching insight: {self.teaching_notes[-1]}")

        return "\n".join(lines)

    def record_episode(
        self,
        task_goal: str,
        category: str,
        difficulty: int,
        steps_completed: int,
        total_steps: int,
        reward: float,
        child_utterances: list[str],
    ) -> None:
        """Record lightweight per-episode data (no API call)."""
        self.episodes_observed += 1
        self.update_scores(category, reward)

        self.episode_history.append({
            "task_goal": task_goal,
            "category": category,
            "difficulty": difficulty,
            "steps_completed": steps_completed,
            "total_steps": total_steps,
            "reward": reward,
        })
        if len(self.episode_history) > self._MAX_HISTORY:
            self.episode_history = self.episode_history[-self._MAX_HISTORY:]

    def should_reflect(self) -> bool:
        return self.episodes_observed % self._REFLECT_EVERY == 0

    def build_reflect_prompt(
        self,
        task_goal: str,
        category: str,
        difficulty: int,
        steps_completed: int,
        total_steps: int,
        reward: float,
        child_utterances: list[str],
    ) -> tuple[str, str]:
        """Build the reflection prompt for the 235B model."""
        child_actions = "\n".join(
            f"  Turn {i+1}: {u}" for i, u in enumerate(child_utterances[-8:])
        ) or "  (no actions recorded)"

        user = REFLECT_USER_TEMPLATE.format(
            task_goal=task_goal,
            category=category,
            difficulty=difficulty,
            steps_completed=steps_completed,
            total_steps=total_steps,
            reward=reward,
            child_actions=child_actions,
            current_profile=self.generate_profile(),
        )
        return REFLECT_SYSTEM, user

    def apply_reflection(self, parsed: dict) -> None:
        """Apply the 235B model's reflection to update the child model."""
        new_gaps = parsed.get("new_knowledge_gaps", [])
        if isinstance(new_gaps, list):
            for gap in new_gaps:
                if gap and gap not in self.knowledge_gaps:
                    self.knowledge_gaps.append(str(gap))
            self.knowledge_gaps = self.knowledge_gaps[-self._MAX_GAPS:]

        new_mistakes = parsed.get("new_mistakes", [])
        if isinstance(new_mistakes, list):
            for m in new_mistakes:
                if m and m not in self.common_mistakes:
                    self.common_mistakes.append(str(m))
            self.common_mistakes = self.common_mistakes[-self._MAX_MISTAKES:]

        improvements = parsed.get("improvements", [])
        if isinstance(improvements, list):
            self.recent_improvements = [
                str(imp) for imp in improvements if imp
            ][-self._MAX_IMPROVEMENTS:]

        notes = parsed.get("teaching_notes", "")
        if notes:
            self.teaching_notes.append(str(notes))
            self.teaching_notes = self.teaching_notes[-self._MAX_NOTES:]

    async def reflect_async(
        self,
        task_goal: str,
        category: str,
        difficulty: int,
        steps_completed: int,
        total_steps: int,
        reward: float,
        child_utterances: list[str],
        caregiver_client,
        tokenizer,
    ) -> None:
        """Use the 235B model to deeply reflect on the child's progress."""
        system, user = self.build_reflect_prompt(
            task_goal, category, difficulty,
            steps_completed, total_steps, reward,
            child_utterances,
        )
        raw = await sample_chat_async(
            caregiver_client, system,
            [{"role": "user", "content": user}],
            max_tokens=512, temperature=0.3,
            tokenizer=tokenizer,
        )
        parsed = parse_json(raw)
        if parsed:
            self.apply_reflection(parsed)

    def save(self, path: str) -> None:
        data = {
            "category_scores": self.category_scores,
            "knowledge_gaps": self.knowledge_gaps,
            "common_mistakes": self.common_mistakes,
            "recent_improvements": self.recent_improvements,
            "teaching_notes": self.teaching_notes,
            "episodes_observed": self.episodes_observed,
            "episode_history": self.episode_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> ChildModel:
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model.category_scores = data.get("category_scores", {})
        model.knowledge_gaps = data.get("knowledge_gaps", [])
        model.common_mistakes = data.get("common_mistakes", [])
        model.recent_improvements = data.get("recent_improvements", [])
        model.teaching_notes = data.get("teaching_notes", [])
        model.episodes_observed = data.get("episodes_observed", 0)
        model.episode_history = data.get("episode_history", [])
        return model
