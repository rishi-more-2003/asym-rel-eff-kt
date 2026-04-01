"""Async episode runner with speculative parallel execution.

Runs a single episode: child attempts a task, judge evaluates,
caregiver responds (with speculative parallelism), and post-episode
processing (LTM compression, LoRA update, logging) runs in parallel.
"""

from __future__ import annotations

import asyncio
import json
import sys, os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import HouseholdTask
from src.memory.instinct import InstinctBuffer
from src.memory.working_memory import WorkingMemory
from src.memory.long_term import LongTermMemory, EpisodicEntry
from src.judge import SemanticJudge, JudgeResult
from src.scaffolding import CompetenceTracker
from src.reward import compute_reward, EpisodeReward
from collections import Counter
from src.salience import compute_salience
from src.trainer import SalienceWeightedTrainer
from src.metrics_logger import MetricsLogger, EpisodeMetrics
from src.conditions import ConditionConfig
from src.tinker_utils import (
    build_chat_prompt, fire_sample, await_future,
    sample_chat_async,
)


@dataclass
class EpisodeResult:
    task: HouseholdTask
    reward_info: EpisodeReward
    salience: float
    turns_used: int
    transcript: list[dict[str, str]]
    lora_updated: bool
    child_utterances: list[str] = field(default_factory=list)
    caregiver_utterances: list[str] = field(default_factory=list)


async def run_episode(
    task: HouseholdTask,
    episode_num: int,
    child_agent,
    caregiver_agent,
    judge: SemanticJudge,
    child_instinct: InstinctBuffer,
    caregiver_instinct: InstinctBuffer | None,
    child_wm: WorkingMemory,
    caregiver_wm: WorkingMemory | None,
    ltm: LongTermMemory,
    scaffolding: CompetenceTracker | None,
    trainer: SalienceWeightedTrainer,
    logger: MetricsLogger,
    condition: ConditionConfig,
    seed: int,
    child_model=None,
) -> EpisodeResult:
    """Run a single episode with speculative parallel execution."""

    task_desc = json.dumps(task.child_view(), indent=2)
    ltm_entries = ltm.retrieve(task.goal_description, top_k=3)
    child_wm.prime(task_desc, ltm_entries)

    if caregiver_wm is not None and caregiver_instinct is not None:
        if scaffolding and condition.use_scaffolding:
            scaffold_prompt = scaffolding.scaffolding_prompt(task.category)
            caregiver_instinct.set_scaffolding(scaffold_prompt)
        if child_model is not None and condition.use_child_model:
            caregiver_instinct.set_child_model(child_model.generate_profile())
        caregiver_task_desc = json.dumps(task.caregiver_view(), indent=2)
        caregiver_wm.prime(caregiver_task_desc)

    action_steps = [s.to_dict() if hasattr(s, "to_dict") else s
                    for s in task.action_sequence]
    step_idx = 0
    total_steps = len(action_steps)
    partial_credits: list[float] = []
    child_utterances: list[str] = []
    caregiver_utterances: list[str] = []
    turns_used = 0

    if config.ADAPTIVE_TURN_LIMITS:
        max_turns = config.TURN_LIMIT_BY_DIFFICULTY.get(
            task.difficulty, config.MAX_EPISODE_TURNS,
        )
    else:
        max_turns = config.MAX_EPISODE_TURNS

    scaffold_desc = ""
    if scaffolding and condition.use_scaffolding:
        scaffold_desc = scaffolding.scaffolding_prompt(task.category)

    for turn in range(max_turns):
        if step_idx >= total_steps:
            break
        turns_used = turn + 1

        child_output = await child_agent.sample_async(
            child_instinct.prompt, child_wm.get_messages(),
        )
        child_utterances.append(child_output)
        child_wm.add_turn("assistant", child_output)

        current_step = action_steps[step_idx]

        if condition.has_caregiver and caregiver_agent is not None:
            child_profile = ""
            if child_model is not None and condition.use_child_model:
                child_profile = child_model.generate_profile()

            caregiver_output, judge_pc = await _caregiver_turn(
                child_output, current_step, judge,
                caregiver_agent, caregiver_instinct,
                caregiver_wm, child_wm,
                partial_credits, scaffold_desc,
                child_model_profile=child_profile,
            )
            if caregiver_output is not None:
                caregiver_utterances.append(caregiver_output)
                child_wm.add_turn("user", caregiver_output)
                if caregiver_wm is not None:
                    caregiver_wm.add_turn("assistant", caregiver_output)
        else:
            result = await judge.evaluate_async(child_output, current_step)
            partial_credits.append(result.partial_credit)
            feedback = f"Result: {'correct' if result.correct else 'incorrect'}."
            child_wm.add_turn("user", feedback)

        if partial_credits and partial_credits[-1] >= config.ACTION_MATCH_THRESHOLD:
            step_idx += 1

    reward_info = compute_reward(step_idx, total_steps, partial_credits)

    category_counts = Counter(e.category for e in ltm.entries)
    salience = compute_salience(
        task_category=task.category,
        goal_description=task.goal_description,
        category_counts=category_counts,
        ltm_entries=ltm.entries,
        reward=reward_info.reward,
        partial_credits=partial_credits,
        turns_used=turns_used,
        max_turns=max_turns,
        caregiver_present=condition.has_caregiver,
        gamma=condition.salience_gamma,
    )

    transcript = child_wm.get_transcript()

    post_tasks = []

    if salience > config.SALIENCE_TAU:
        post_tasks.append(_compress_and_store(
            ltm, transcript, episode_num, task, reward_info.reward, salience,
            judge._client, judge._tokenizer,
        ))

    lora_updated = False
    if reward_info.reward > 0:
        post_tasks.append(trainer.update_async(
            child_instinct.prompt, transcript, "assistant",
            salience, reward_info.reward,
        ))

    if child_model is not None and condition.use_child_model:
        child_model.record_episode(
            task.goal_description, task.category, task.difficulty,
            step_idx, total_steps, reward_info.reward, child_utterances,
        )
        if child_model.should_reflect():
            post_tasks.append(child_model.reflect_async(
                task.goal_description, task.category, task.difficulty,
                step_idx, total_steps, reward_info.reward, child_utterances,
                judge._client, judge._tokenizer,
            ))

    post_tasks.append(logger.log_episode(
        EpisodeMetrics(
            episode=episode_num,
            task_id=task.task_id,
            category=task.category,
            difficulty=task.difficulty,
            turns=turns_used,
            steps_completed=step_idx,
            total_steps=total_steps,
            reward=reward_info.reward,
            salience=salience,
            lora_updated=reward_info.reward > 0,
            competence_level=(
                scaffolding.scaffolding_level(task.category)
                if scaffolding else "none"
            ),
            ltm_size=ltm.size,
            condition=condition.name,
            seed=seed,
        ),
        transcript=transcript,
    ))

    results = await asyncio.gather(*post_tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, bool) and r:
            lora_updated = True
        if isinstance(r, Exception):
            print(f"    [warn] Post-episode task failed: {r}")

    if trainer.sampling_client is not None:
        child_agent.set_sampling_client(trainer.sampling_client)

    if scaffolding and condition.use_scaffolding:
        avg_pc = (
            sum(partial_credits) / len(partial_credits)
            if partial_credits else 0.0
        )
        scaffolding.update(task.category, avg_pc)

    return EpisodeResult(
        task=task,
        reward_info=reward_info,
        salience=salience,
        turns_used=turns_used,
        transcript=transcript,
        lora_updated=lora_updated,
        child_utterances=child_utterances,
        caregiver_utterances=caregiver_utterances,
    )


async def _caregiver_turn(
    child_output: str,
    current_step: dict,
    judge: SemanticJudge,
    caregiver_agent,
    caregiver_instinct: InstinctBuffer,
    caregiver_wm: WorkingMemory | None,
    child_wm: WorkingMemory,
    partial_credits: list[float],
    scaffold_desc: str,
    child_model_profile: str = "",
) -> tuple[str | None, float]:
    """Execute a caregiver turn using the configured parallelism mode.

    Returns (caregiver_output, partial_credit).
    Modes:
      combined:    1 call (judge + response fused) -- fastest
      speculative: 3 parallel calls (judge + confirm + correct)
      sequential:  judge first, then caregiver
    """
    mode = config.CAREGIVER_MODE

    if mode == "combined":
        future = caregiver_agent.fire_combined(
            child_output, current_step, scaffold_desc,
            child_model_profile=child_model_profile,
        )
        raw = await await_future(future, tokenizer=caregiver_agent._tokenizer)
        judge_result, caregiver_output = caregiver_agent.parse_combined(raw)
        partial_credits.append(judge_result.partial_credit)
        return caregiver_output, judge_result.partial_credit

    caregiver_msgs = caregiver_wm.get_messages() if caregiver_wm else child_wm.get_messages()

    if mode == "speculative":
        judge_future = judge.fire_judge(child_output, current_step)
        confirm_future = caregiver_agent.fire_confirmation(
            caregiver_instinct.prompt, caregiver_msgs,
        )
        correct_future = caregiver_agent.fire_correction(
            caregiver_instinct.prompt, caregiver_msgs,
            explanation="The child's action may not match the expected step.",
        )

        judge_raw = await await_future(judge_future, tokenizer=judge._tokenizer)
        judge_result = judge._parse_result(judge_raw)
        partial_credits.append(judge_result.partial_credit)

        if judge_result.correct:
            caregiver_output = await await_future(
                confirm_future, tokenizer=caregiver_agent._tokenizer,
            )
        else:
            caregiver_output = await await_future(
                correct_future, tokenizer=caregiver_agent._tokenizer,
            )
        return caregiver_output, judge_result.partial_credit

    # sequential mode
    judge_result = await judge.evaluate_async(child_output, current_step)
    partial_credits.append(judge_result.partial_credit)
    caregiver_output = await caregiver_agent.respond_async(
        caregiver_instinct.prompt, caregiver_msgs,
    )
    return caregiver_output, judge_result.partial_credit


async def _compress_and_store(
    ltm: LongTermMemory,
    transcript: list[dict[str, str]],
    episode_num: int,
    task: HouseholdTask,
    reward: float,
    salience: float,
    caregiver_client,
    tokenizer,
) -> None:
    """Compress episode transcript via 235B and store in LTM."""
    system, user = ltm.build_compress_prompt(transcript)
    summary = await sample_chat_async(
        caregiver_client, system,
        [{"role": "user", "content": user}],
        max_tokens=256, temperature=0.3,
        tokenizer=tokenizer,
    )

    ltm.add(EpisodicEntry(
        episode_id=episode_num,
        task_id=task.task_id,
        category=task.category,
        summary=summary.strip(),
        reward=reward,
        salience=salience,
    ))
