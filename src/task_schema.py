from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ActionStep:
    """A single step in a household task with causal structure."""

    step_number: int
    action: str
    target_object: str
    preconditions: list[str]
    effects: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActionStep:
        return cls(**d)


@dataclass
class HouseholdTask:
    """A multi-step procedural household scenario with enriched structure.

    Follows Section 3.1 of the proposal. Each scenario has a goal description,
    available objects, an affordance map, and a causally-structured action
    sequence. The caregiver sees the full solution; the child does not.

    The enriched fields (common_mistakes, caregiver_hints, initial_state,
    goal_state) directly support the mother-child interaction protocol
    in Section 3.3.
    """

    task_id: str
    category: str
    goal_description: str
    difficulty: int  # number of steps (1-8)
    available_objects: list[str]
    distractor_objects: list[str]
    affordance_map: dict[str, list[str]]  # object -> possible actions
    action_sequence: list[ActionStep]
    initial_state: dict[str, str]  # object -> starting state
    goal_state: dict[str, str]  # object -> required final state
    common_mistakes: list[str]
    caregiver_hints: list[str]  # ordered easy -> specific
    split: str = "train"
    verification_attempts: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HouseholdTask:
        d = d.copy()
        if d.get("action_sequence") and isinstance(d["action_sequence"][0], dict):
            d["action_sequence"] = [
                ActionStep.from_dict(s) for s in d["action_sequence"]
            ]
        return cls(**d)

    def child_view(self) -> dict:
        """What the child agent sees: goal, objects, no solution or hints."""
        return {
            "task_id": self.task_id,
            "category": self.category,
            "goal_description": self.goal_description,
            "difficulty": self.difficulty,
            "available_objects": self.available_objects + self.distractor_objects,
            "initial_state": self.initial_state,
        }

    def caregiver_view(self) -> dict:
        """What the caregiver agent sees: full information including solution."""
        return self.to_dict()


@dataclass
class TaskSkeleton:
    """Lightweight task outline produced in Stage 1 before full expansion."""

    skeleton_id: str
    category: str
    difficulty: int
    goal_description: str
    key_objects: list[str]
    rooms: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TaskSkeleton:
        return cls(**d)


@dataclass
class TaskDatabase:
    """Container for the full set of generated scenarios."""

    tasks: list[HouseholdTask] = field(default_factory=list)

    @property
    def train_tasks(self) -> list[HouseholdTask]:
        return [t for t in self.tasks if t.split == "train"]

    @property
    def eval_tasks(self) -> list[HouseholdTask]:
        return [t for t in self.tasks if t.split == "eval"]

    def add(self, task: HouseholdTask) -> None:
        self.tasks.append(task)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total": len(self.tasks),
                    "train_count": len(self.train_tasks),
                    "eval_count": len(self.eval_tasks),
                    "tasks": [t.to_dict() for t in self.tasks],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> TaskDatabase:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = cls()
        for td in data["tasks"]:
            db.add(HouseholdTask.from_dict(td))
        return db

    def summary(self) -> str:
        lines = [
            f"TaskDatabase: {len(self.tasks)} total "
            f"({len(self.train_tasks)} train, {len(self.eval_tasks)} eval)",
        ]
        by_diff: dict[int, list[HouseholdTask]] = {}
        for t in self.tasks:
            by_diff.setdefault(t.difficulty, []).append(t)
        for d in sorted(by_diff):
            lines.append(f"  difficulty {d}: {len(by_diff[d])} tasks")
        by_cat: dict[str, int] = {}
        for t in self.tasks:
            by_cat[t.category] = by_cat.get(t.category, 0) + 1
        for c in sorted(by_cat):
            lines.append(f"  {c}: {by_cat[c]} tasks")
        return "\n".join(lines)
