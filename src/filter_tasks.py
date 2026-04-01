"""Stage 4: Deduplication, distribution balancing, and train/eval split.

Uses BM25 pairwise similarity to reject near-duplicate goals (keeping the
one that needed fewer verification retries), then trims or balances the
distribution across difficulty levels and categories. Finally assigns the
160/40 train/eval split stratified by difficulty.
"""

from __future__ import annotations

import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import HouseholdTask, TaskDatabase
from src.bm25 import bm25_score


def deduplicate(
    tasks: list[HouseholdTask],
    threshold: float = config.BM25_DUPLICATE_THRESHOLD,
) -> list[HouseholdTask]:
    """Remove near-duplicate tasks based on goal description similarity."""
    print(f"Stage 4a: Deduplicating {len(tasks)} tasks (threshold={threshold})...")

    keep = list(range(len(tasks)))
    removed: set[int] = set()

    for i in range(len(tasks)):
        if i in removed:
            continue
        for j in range(i + 1, len(tasks)):
            if j in removed:
                continue
            sim = bm25_score(
                tasks[i].goal_description, tasks[j].goal_description
            )
            if sim > threshold:
                # Keep the one that verified more easily
                drop = j if tasks[i].verification_attempts <= tasks[j].verification_attempts else i
                removed.add(drop)
                print(f"    [dedup] Dropping {tasks[drop].task_id} "
                      f"(sim={sim:.2f} with {tasks[i if drop == j else j].task_id})")
                if drop == i:
                    break

    result = [tasks[k] for k in range(len(tasks)) if k not in removed]
    print(f"  Removed {len(removed)} duplicates, {len(result)} remain")
    return result


def balance_distribution(
    tasks: list[HouseholdTask],
    target_per_difficulty: int = config.TASKS_PER_DIFFICULTY,
) -> list[HouseholdTask]:
    """Trim over-represented difficulty levels to the target count."""
    print(f"Stage 4b: Balancing distribution (target {target_per_difficulty}/level)...")

    by_diff: dict[int, list[HouseholdTask]] = {}
    for t in tasks:
        by_diff.setdefault(t.difficulty, []).append(t)

    balanced: list[HouseholdTask] = []
    for diff in sorted(by_diff):
        level_tasks = by_diff[diff]
        random.shuffle(level_tasks)
        taken = level_tasks[:target_per_difficulty]
        balanced.extend(taken)
        if len(level_tasks) > target_per_difficulty:
            print(f"    difficulty {diff}: trimmed {len(level_tasks)} -> {len(taken)}")
        elif len(level_tasks) < target_per_difficulty:
            print(f"    difficulty {diff}: under target "
                  f"({len(level_tasks)}/{target_per_difficulty})")
        else:
            print(f"    difficulty {diff}: {len(taken)} (on target)")

    print(f"  Balanced total: {len(balanced)}")
    return balanced


def assign_splits(
    tasks: list[HouseholdTask],
    eval_count: int = config.NUM_EVAL_TASKS,
) -> list[HouseholdTask]:
    """Assign train/eval splits, stratified by difficulty."""
    print(f"Stage 4c: Assigning train/eval split ({eval_count} eval)...")

    by_diff: dict[int, list[HouseholdTask]] = {}
    for t in tasks:
        by_diff.setdefault(t.difficulty, []).append(t)

    eval_per_level = eval_count // len(config.DIFFICULTY_LEVELS)

    for diff, level_tasks in by_diff.items():
        random.shuffle(level_tasks)
        for j, t in enumerate(level_tasks):
            t.split = "eval" if j < eval_per_level else "train"

    train = sum(1 for t in tasks if t.split == "train")
    eval_ = sum(1 for t in tasks if t.split == "eval")
    print(f"  Split: {train} train, {eval_} eval")
    return tasks


def reassign_task_ids(tasks: list[HouseholdTask]) -> list[HouseholdTask]:
    """Give tasks clean sequential IDs after filtering."""
    for i, t in enumerate(tasks, 1):
        t.task_id = f"task_{i:04d}"
    return tasks


def filter_and_split(tasks: list[HouseholdTask]) -> TaskDatabase:
    """Run the full Stage 4 pipeline: dedup -> balance -> split -> package."""
    tasks = deduplicate(tasks)
    tasks = balance_distribution(tasks)
    tasks = assign_splits(tasks)
    tasks = reassign_task_ids(tasks)

    db = TaskDatabase()
    for t in tasks:
        db.add(t)

    print(f"\n{db.summary()}")
    return db
