"""Evaluation suite: H1-H3, learning curves, teaching efficiency,
generalization distance, and per-category heatmaps.

All evaluation tasks run in parallel via asyncio.gather for maximum
Tinker throughput.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import TaskDatabase, HouseholdTask
from src.memory.instinct import InstinctBuffer
from src.memory.working_memory import WorkingMemory
from src.judge import SemanticJudge, JudgeResult
from src.bm25 import bm25_score
from src.conditions import ALL_CONDITIONS, ConditionConfig
from src.metrics_logger import MetricsLogger
from src.tinker_utils import (
    get_service_client, get_tokenizer,
    build_chat_prompt_multi, fire_sample, await_future,
)


@dataclass
class EvalResult:
    task_id: str
    category: str
    difficulty: int
    steps_completed: int
    total_steps: int
    reward: float
    partial_credits: list[float]


async def eval_single_task(
    task: HouseholdTask,
    child_sampling_client,
    judge: SemanticJudge,
    tokenizer,
) -> EvalResult:
    """Evaluate a trained child on a single held-out task (no caregiver, no training)."""

    instinct = InstinctBuffer("child")
    wm = WorkingMemory()
    task_desc = json.dumps(task.child_view(), indent=2)
    wm.prime(task_desc)

    action_steps = [s.to_dict() if hasattr(s, "to_dict") else s
                    for s in task.action_sequence]
    step_idx = 0
    total_steps = len(action_steps)
    partial_credits: list[float] = []

    for turn in range(config.MAX_EPISODE_TURNS):
        if step_idx >= total_steps:
            break

        prompt = build_chat_prompt_multi(
            instinct.prompt, wm.get_messages(), tokenizer=tokenizer,
        )
        future = fire_sample(child_sampling_client, prompt, max_tokens=512)
        child_output = await await_future(future, tokenizer=tokenizer)

        wm.add_turn("assistant", child_output)

        current_step = action_steps[step_idx]
        result = await judge.evaluate_async(child_output, current_step)
        partial_credits.append(result.partial_credit)

        if result.partial_credit >= config.ACTION_MATCH_THRESHOLD:
            step_idx += 1

        feedback = f"Step evaluation: {'correct' if result.correct else 'incorrect'}."
        wm.add_turn("user", feedback)

    reward = step_idx / total_steps if total_steps > 0 else 0.0

    return EvalResult(
        task_id=task.task_id,
        category=task.category,
        difficulty=task.difficulty,
        steps_completed=step_idx,
        total_steps=total_steps,
        reward=reward,
        partial_credits=partial_credits,
    )


async def evaluate_h1_transfer(
    child_sampling_client,
    judge: SemanticJudge,
    eval_tasks: list[HouseholdTask],
    tokenizer,
) -> dict:
    """H1: Transfer accuracy on held-out tasks."""
    futures = [
        eval_single_task(task, child_sampling_client, judge, tokenizer)
        for task in eval_tasks
    ]
    results = await asyncio.gather(*futures)

    total_reward = sum(r.reward for r in results)
    avg_reward = total_reward / len(results) if results else 0.0

    by_difficulty: dict[int, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_difficulty[r.difficulty].append(r.reward)
        by_category[r.category].append(r.reward)

    return {
        "avg_completion_rate": avg_reward,
        "num_tasks": len(results),
        "by_difficulty": {
            d: sum(v) / len(v) for d, v in sorted(by_difficulty.items())
        },
        "by_category": {
            c: sum(v) / len(v) for c, v in sorted(by_category.items())
        },
        "per_task": [
            {"task_id": r.task_id, "reward": r.reward, "steps": f"{r.steps_completed}/{r.total_steps}"}
            for r in results
        ],
    }


def compute_h2_habit_acceleration(metrics: list[dict]) -> dict:
    """H2: Habit acceleration from training logs.

    Measures task completion rate at episode 40/80/120/160 checkpoints.
    """
    checkpoints = [40, 80, 120, 160]
    completion_at = {}
    for cp in checkpoints:
        episodes = [m for m in metrics if m["episode"] < cp]
        if episodes:
            completion_at[cp] = sum(m["reward"] for m in episodes) / len(episodes)
        else:
            completion_at[cp] = 0.0

    lora_episodes = [m["episode"] for m in metrics if m.get("lora_updated")]
    cumulative_updates = []
    count = 0
    for m in sorted(metrics, key=lambda x: x["episode"]):
        if m.get("lora_updated"):
            count += 1
        cumulative_updates.append({"episode": m["episode"], "cumulative_lora": count})

    return {
        "completion_at_checkpoints": completion_at,
        "total_lora_updates": count,
        "cumulative_updates": cumulative_updates,
    }


async def evaluate_h3_tom(
    child_sampling_client,
    relational_run_dir: str,
    tokenizer,
    num_samples: int = 50,
) -> dict:
    """H3: ToM proxy -- can the child predict caregiver responses?

    Samples (context, caregiver_response) pairs from Relational transcripts.
    Asks the child to predict the caregiver's response and scores with BM25.
    """
    transcript_dir = os.path.join(relational_run_dir, "transcripts")
    if not os.path.exists(transcript_dir):
        return {"avg_similarity": 0.0, "num_samples": 0}

    transcript_files = sorted([
        f for f in os.listdir(transcript_dir) if f.endswith(".json")
    ])

    pairs: list[tuple[list[dict], str]] = []
    for tf in transcript_files:
        with open(os.path.join(transcript_dir, tf), "r") as f:
            transcript = json.load(f)
        for i, turn in enumerate(transcript):
            if turn.get("role") == "user" and i > 0:
                context = transcript[:i]
                caregiver_response = turn["content"]
                pairs.append((context, caregiver_response))
        if len(pairs) >= num_samples * 2:
            break

    import random
    rng = random.Random(42)
    rng.shuffle(pairs)
    pairs = pairs[:num_samples]

    if not pairs:
        return {"avg_similarity": 0.0, "num_samples": 0}

    system = (
        "You are a child who has been learning household tasks with a caregiver. "
        "Based on the conversation so far, predict what the caregiver would say next. "
        "Write only the caregiver's response, nothing else."
    )

    async def predict_single(context, actual):
        prompt = build_chat_prompt_multi(system, context, tokenizer=tokenizer)
        future = fire_sample(child_sampling_client, prompt, max_tokens=256)
        prediction = await await_future(future, tokenizer=tokenizer)
        similarity = bm25_score(prediction, actual)
        return similarity

    futures = [predict_single(ctx, actual) for ctx, actual in pairs]
    similarities = await asyncio.gather(*futures)

    return {
        "avg_similarity": sum(similarities) / len(similarities),
        "num_samples": len(similarities),
        "similarities": list(similarities),
    }


def compute_learning_curves(metrics: list[dict], window: int = 10) -> list[dict]:
    """Rolling-average completion rate over episodes."""
    sorted_m = sorted(metrics, key=lambda x: x["episode"])
    curves = []
    for i in range(len(sorted_m)):
        start = max(0, i - window + 1)
        window_metrics = sorted_m[start:i + 1]
        avg_reward = sum(m["reward"] for m in window_metrics) / len(window_metrics)
        curves.append({
            "episode": sorted_m[i]["episode"],
            "rolling_avg_reward": avg_reward,
        })
    return curves


def compute_teaching_efficiency(metrics: list[dict]) -> dict:
    """Average turns-to-completion per condition."""
    successful = [m for m in metrics if m["reward"] > 0]
    if not successful:
        return {"avg_turns": 0.0, "successful_episodes": 0}
    avg_turns = sum(m["turns"] for m in successful) / len(successful)
    return {
        "avg_turns": avg_turns,
        "successful_episodes": len(successful),
        "total_episodes": len(metrics),
    }


def compute_generalization_distance(
    eval_results: list[dict],
    train_tasks: list[HouseholdTask],
) -> list[dict]:
    """For each eval task, compute BM25 distance to nearest training task."""
    train_goals = [t.goal_description for t in train_tasks]
    gen_distances = []
    for er in eval_results:
        task_goal = er.get("task_id", "")
        matching_task = None
        for t in train_tasks:
            if t.task_id == er.get("task_id"):
                matching_task = t
                break
        if matching_task is None:
            continue

        max_sim = max(
            (bm25_score(matching_task.goal_description, tg) for tg in train_goals),
            default=0.0,
        )
        gen_distances.append({
            "task_id": er["task_id"],
            "reward": er["reward"],
            "nearest_train_similarity": max_sim,
            "distance": 1.0 - max_sim,
        })

    return gen_distances


def compute_category_heatmap(
    all_eval_results: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Completion rate matrix: condition x category."""
    heatmap: dict[str, dict[str, float]] = {}
    for condition_name, results in all_eval_results.items():
        by_cat: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_cat[r["category"]].append(r["reward"])
        heatmap[condition_name] = {
            cat: sum(v) / len(v) for cat, v in sorted(by_cat.items())
        }
    return heatmap


async def run_full_evaluation(
    conditions: list[ConditionConfig] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Run the full evaluation suite across all conditions and seeds."""

    if conditions is None:
        conditions = ALL_CONDITIONS
    if seeds is None:
        seeds = list(range(1, config.NUM_SEEDS + 1))

    task_db = TaskDatabase.load(config.TASK_DB_PATH)
    eval_tasks = task_db.eval_tasks
    print(f"Evaluation: {len(eval_tasks)} held-out tasks")

    service = get_service_client()
    tokenizer = get_tokenizer()

    caregiver_sampling = await service.create_sampling_client_async(
        base_model=config.CAREGIVER_MODEL,
    )
    judge = SemanticJudge(caregiver_sampling, tokenizer)

    all_results: dict[str, dict] = {}

    for cond in conditions:
        cond_results: dict[str, list] = {"h1": [], "h2": [], "curves": [], "efficiency": []}

        for seed in seeds:
            run_name = f"{cond.name}_seed{seed}"
            run_dir = os.path.join(config.RUNS_DIR, run_name)

            logger = MetricsLogger(run_dir)
            metrics = logger.load_metrics()

            if not metrics:
                print(f"  [{run_name}] No metrics found, skipping")
                continue

            h2 = compute_h2_habit_acceleration(metrics)
            cond_results["h2"].append({"seed": seed, **h2})

            curves = compute_learning_curves(metrics)
            cond_results["curves"].append({"seed": seed, "curves": curves})

            efficiency = compute_teaching_efficiency(metrics)
            cond_results["efficiency"].append({"seed": seed, **efficiency})

            print(f"  [{run_name}] Running H1 transfer evaluation...")
            child_training = await service.create_lora_training_client_async(
                base_model=config.CHILD_MODEL,
                rank=config.LORA_RANK,
            )
            child_sampling = child_training.save_weights_and_get_sampling_client()

            h1 = await evaluate_h1_transfer(
                child_sampling, judge, eval_tasks, tokenizer,
            )
            cond_results["h1"].append({"seed": seed, **h1})
            print(f"  [{run_name}] H1 avg completion: {h1['avg_completion_rate']:.3f}")

            if cond.name == "relational":
                print(f"  [{run_name}] Running H3 ToM evaluation...")
                h3 = await evaluate_h3_tom(
                    child_sampling, run_dir, tokenizer,
                )
                cond_results["h3"] = cond_results.get("h3", [])
                cond_results["h3"].append({"seed": seed, **h3})
                print(f"  [{run_name}] H3 avg similarity: {h3['avg_similarity']:.3f}")

        all_results[cond.name] = cond_results

    eval_output_path = os.path.join(config.RUNS_DIR, "evaluation_results.json")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {eval_output_path}")

    return all_results
