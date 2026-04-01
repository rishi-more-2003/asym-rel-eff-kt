"""Reward computation with semantic judge integration.

Reward is the fraction of steps completed: steps_completed / total_steps.
Supports partial credit from the semantic judge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeReward:
    steps_completed: int
    total_steps: int
    partial_credits: list[float]
    reward: float


def compute_reward(
    steps_completed: int,
    total_steps: int,
    partial_credits: list[float],
) -> EpisodeReward:
    """Compute episode reward as fraction of steps completed.

    Also averages partial credits across all attempted steps for a
    finer-grained signal.
    """
    if total_steps == 0:
        reward = 0.0
    else:
        reward = steps_completed / total_steps
    return EpisodeReward(
        steps_completed=steps_completed,
        total_steps=total_steps,
        partial_credits=partial_credits,
        reward=reward,
    )
