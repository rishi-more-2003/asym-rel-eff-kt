"""Async training loop: asyncio.gather over 12 concurrent runs with checkpointing.

Each run = one (condition, seed) pair with its own child TrainingClient,
memory stores, and scaffolding tracker. All runs share one frozen 235B
caregiver SamplingClient.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import TaskDatabase
from src.memory.instinct import InstinctBuffer
from src.memory.working_memory import WorkingMemory
from src.memory.long_term import LongTermMemory
from src.judge import SemanticJudge
from src.scaffolding import CompetenceTracker
from src.trainer import SalienceWeightedTrainer
from src.agents.caregiver import CaregiverAgent
from src.agents.child import ChildAgent
from src.conditions import ConditionConfig, ALL_CONDITIONS, CONDITIONS_BY_NAME
from src.curriculum import build_curriculum
from src.metrics_logger import MetricsLogger
from src.episode import run_episode
from src.child_model import ChildModel
from src.tinker_utils import get_service_client, get_tokenizer


async def train_single_run(
    condition: ConditionConfig,
    seed: int,
    child_agent: ChildAgent,
    caregiver_agent: CaregiverAgent | None,
    judge: SemanticJudge,
    task_db: TaskDatabase,
    tokenizer,
) -> str:
    """Run a full training curriculum for one (condition, seed) pair."""

    run_name = f"{condition.name}_seed{seed}"
    run_dir = os.path.join(config.RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"  [{run_name}] Starting training ({config.NUM_EPISODES} episodes)")

    child_instinct = InstinctBuffer(
        "child" if condition.use_instinct else "peer"
    )
    caregiver_instinct = (
        InstinctBuffer("caregiver") if condition.has_caregiver else None
    )

    ltm = LongTermMemory()
    ltm_path = os.path.join(run_dir, "ltm.json")
    if os.path.exists(ltm_path):
        ltm.load(ltm_path)

    scaffolding = CompetenceTracker() if condition.use_scaffolding else None
    scaffold_path = os.path.join(run_dir, "scaffolding.json")
    if scaffolding and os.path.exists(scaffold_path):
        with open(scaffold_path, "r") as f:
            scaffolding.load_dict(json.load(f))

    child_model_obj = None
    child_model_path = os.path.join(run_dir, "child_model.json")
    if condition.use_child_model:
        child_model_obj = ChildModel.load(child_model_path)

    trainer = SalienceWeightedTrainer(child_agent.training_client, tokenizer)

    logger = MetricsLogger(run_dir)

    existing_metrics = logger.load_metrics()
    start_episode = len(existing_metrics)
    if start_episode > 0:
        print(f"  [{run_name}] Resuming from episode {start_episode}")

    curriculum = build_curriculum(
        task_db.train_tasks, config.NUM_EPISODES, seed=seed,
    )

    for ep in range(start_episode, config.NUM_EPISODES):
        task = curriculum[ep]

        child_wm = WorkingMemory()
        caregiver_wm = WorkingMemory() if condition.has_caregiver else None

        result = await run_episode(
            task=task,
            episode_num=ep,
            child_agent=child_agent,
            caregiver_agent=caregiver_agent,
            judge=judge,
            child_instinct=child_instinct,
            caregiver_instinct=caregiver_instinct,
            child_wm=child_wm,
            caregiver_wm=caregiver_wm,
            ltm=ltm,
            scaffolding=scaffolding,
            trainer=trainer,
            logger=logger,
            condition=condition,
            seed=seed,
            child_model=child_model_obj,
        )

        if (ep + 1) % 10 == 0:
            print(
                f"  [{run_name}] Episode {ep + 1}/{config.NUM_EPISODES} | "
                f"reward={result.reward_info.reward:.2f} | "
                f"salience={result.salience:.2f} | "
                f"steps={result.reward_info.steps_completed}/{result.reward_info.total_steps} | "
                f"lora_updates={trainer.update_count}"
            )

        if (ep + 1) % config.CHECKPOINT_EVERY == 0:
            ltm.save(ltm_path)
            if scaffolding:
                with open(scaffold_path, "w") as f:
                    json.dump(scaffolding.to_dict(), f)
            if child_model_obj:
                child_model_obj.save(child_model_path)

    await trainer.flush_remaining()
    if trainer.sampling_client is not None:
        child_agent.set_sampling_client(trainer.sampling_client)

    ltm.save(ltm_path)
    if scaffolding:
        with open(scaffold_path, "w") as f:
            json.dump(scaffolding.to_dict(), f)
    if child_model_obj:
        child_model_obj.save(child_model_path)

    print(
        f"  [{run_name}] Complete | "
        f"LoRA updates: {trainer.update_count} | "
        f"LTM entries: {ltm.size}"
    )
    return run_name


async def run_all_training(
    conditions: list[ConditionConfig] | None = None,
    seeds: list[int] | None = None,
) -> None:
    """Launch all training runs concurrently via asyncio.gather."""

    if conditions is None:
        conditions = ALL_CONDITIONS
    if seeds is None:
        seeds = list(range(1, config.NUM_SEEDS + 1))

    task_db = TaskDatabase.load(config.TASK_DB_PATH)
    print(f"Loaded task database: {task_db.summary()}")

    service = get_service_client()
    tokenizer = get_tokenizer()

    print(f"Creating shared caregiver client ({config.CAREGIVER_MODEL})...")
    caregiver_sampling = await service.create_sampling_client_async(
        base_model=config.CAREGIVER_MODEL,
    )
    caregiver_agent = CaregiverAgent(caregiver_sampling, tokenizer)
    judge = SemanticJudge(caregiver_sampling, tokenizer)

    print(f"Creating {len(conditions) * len(seeds)} child training clients...")

    runs = []
    for cond in conditions:
        for seed in seeds:
            print(f"  Creating child client for {cond.name}_seed{seed}...")
            child_training = await service.create_lora_training_client_async(
                base_model=config.CHILD_MODEL,
                rank=config.LORA_RANK,
            )
            child_agent = ChildAgent(child_training, tokenizer)

            cg = caregiver_agent if cond.has_caregiver else None

            runs.append(train_single_run(
                condition=cond,
                seed=seed,
                child_agent=child_agent,
                caregiver_agent=cg,
                judge=judge,
                task_db=task_db,
                tokenizer=tokenizer,
            ))

    print(f"\nLaunching {len(runs)} concurrent training runs...")
    start = time.time()
    results = await asyncio.gather(*runs, return_exceptions=True)
    elapsed = time.time() - start

    for r in results:
        if isinstance(r, Exception):
            print(f"  [ERROR] Run failed: {r}")
        else:
            print(f"  [OK] {r}")

    print(f"\nAll training complete in {elapsed:.1f}s")
