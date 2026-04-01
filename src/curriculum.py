"""Difficulty-graded training curriculum.

Orders training tasks from easiest (difficulty=1) to hardest (difficulty=8),
with shuffling within each difficulty level. This ensures the child agent
encounters simpler tasks first, building competence before tackling complex
multi-step scenarios.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.task_schema import HouseholdTask


def build_curriculum(
    tasks: list[HouseholdTask],
    num_episodes: int,
    seed: int = 42,
) -> list[HouseholdTask]:
    """Build an ordered curriculum of tasks for training.

    Tasks are grouped by difficulty, shuffled within each group, then
    concatenated in ascending difficulty order. If num_episodes exceeds
    the number of tasks, the curriculum wraps around (the hardest tasks
    repeat first since those benefit most from repetition).
    """
    rng = random.Random(seed)

    by_difficulty: dict[int, list[HouseholdTask]] = defaultdict(list)
    for task in tasks:
        by_difficulty[task.difficulty].append(task)

    ordered: list[list[HouseholdTask]] = []
    for diff in sorted(by_difficulty.keys()):
        group = list(by_difficulty[diff])
        rng.shuffle(group)
        ordered.append(group)

    flat = [task for group in ordered for task in group]

    curriculum: list[HouseholdTask] = []
    while len(curriculum) < num_episodes:
        remaining = num_episodes - len(curriculum)
        curriculum.extend(flat[:remaining])

    return curriculum
