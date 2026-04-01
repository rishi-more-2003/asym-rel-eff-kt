"""Multi-stage pipeline orchestrator for household task database generation.

Runs all stages in sequence:
  Stage 0: Generate/load household object ontology
  Stage 1: Generate diverse task skeletons (few-shot, diversity-aware)
  Stage 2: Expand skeletons into full structured tasks (ontology-grounded)
  Stage 3: Self-verify tasks and retry failures
  Stage 4: Deduplicate, balance distribution, assign train/eval splits

Uses Qwen3-235B-A22B-Instruct via Tinker for all generation/verification.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.ontology import generate_ontology, save_ontology, load_ontology
from src.generate_skeletons import generate_skeletons, save_skeletons, load_skeletons
from src.expand_tasks import expand_skeletons
from src.verify_tasks import verify_and_retry
from src.filter_tasks import filter_and_split
from src.tinker_utils import ensure_api_key


def run_pipeline(seed: int = 42) -> None:
    random.seed(seed)
    start = time.time()

    print("=" * 60)
    print("Household Task Database Generation Pipeline")
    print(f"  Model: {config.GENERATION_MODEL}")
    print(f"  Target: {config.TOTAL_TASKS} tasks "
          f"({config.NUM_TRAINING_TASKS} train / {config.NUM_EVAL_TASKS} eval)")
    print("=" * 60)

    ensure_api_key()

    # ── Stage 0: Object Ontology ──────────────────────────────
    if os.path.exists(config.ONTOLOGY_PATH):
        print(f"\nStage 0: Loading existing ontology from {config.ONTOLOGY_PATH}")
        ontology = load_ontology()
        total_objects = sum(len(objs) for objs in ontology.values())
        print(f"  Loaded: {len(ontology)} rooms, {total_objects} objects")
    else:
        print()
        ontology = generate_ontology()
        save_ontology(ontology)

    # ── Stage 1: Task Skeletons ───────────────────────────────
    if os.path.exists(config.SKELETONS_PATH):
        print(f"\nStage 1: Loading existing skeletons from {config.SKELETONS_PATH}")
        skeletons = load_skeletons()
        print(f"  Loaded: {len(skeletons)} skeletons")
    else:
        print()
        skeletons = generate_skeletons()
        save_skeletons(skeletons)

    # ── Stage 2: Full Task Expansion ──────────────────────────
    print()
    tasks = expand_skeletons(skeletons, ontology)

    # ── Stage 3: Self-Verification with Retries ───────────────
    print()
    print("Stage 3: Self-verification with retry loop...")
    tasks = verify_and_retry(tasks, skeletons, ontology, max_retries=1)

    # ── Stage 4: Dedup, Balance, Split ────────────────────────
    print()
    db = filter_and_split(tasks)

    # ── Save Results ──────────────────────────────────────────
    os.makedirs(config.TASKS_DIR, exist_ok=True)
    db.save(config.TASK_DB_PATH)
    print(f"\nSaved task database -> {config.TASK_DB_PATH}")

    for task in db.tasks:
        path = os.path.join(config.TASKS_DIR, f"{task.task_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved {len(db.tasks)} individual task files -> {config.TASKS_DIR}/")

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print("=" * 60)


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
