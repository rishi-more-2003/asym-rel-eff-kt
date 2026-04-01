"""Salience score: s = alpha * novelty + beta * prediction_error + gamma * teaching.

Three graded, theory-driven components that produce a meaningful [0, 1]
distribution from actual episode outcomes:

  1. Novelty  (alpha)  -- information-theoretic: category decay + BM25 distance
  2. Prediction Error (beta) -- Rescorla-Wagner: outcome surprise + ZPD match
  3. Teaching Signal  (gamma) -- Vygotsky: correction-improvement patterns

Controls what gets written to long-term memory (gated by tau) and the
weight of LoRA training updates.
"""

from __future__ import annotations

import math
import sys, os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.bm25 import bm25_score


# ── Component 1: Novelty ────────────────────────────────────────────

def _category_novelty(task_category: str, category_counts: Counter) -> float:
    """Graded decay: 1/(1+count). Never-seen=1.0, seen once=0.5, etc."""
    count = category_counts.get(task_category, 0)
    return 1.0 / (1.0 + count)


def _task_novelty(goal_description: str, ltm_entries) -> float:
    """BM25 distance to most similar LTM entry.

    Returns 1.0 when the task is completely unlike anything in memory,
    0.0 when there's a near-duplicate already stored.
    """
    if not ltm_entries:
        return 1.0
    max_sim = max(
        bm25_score(goal_description, e.summary) for e in ltm_entries
    )
    return 1.0 - min(max_sim, 1.0)


def compute_novelty(
    task_category: str,
    goal_description: str,
    category_counts: Counter,
    ltm_entries,
) -> float:
    """Combined novelty: category frequency decay + task-level BM25 distance."""
    cat = _category_novelty(task_category, category_counts)
    task = _task_novelty(goal_description, ltm_entries)
    return 0.5 * cat + 0.5 * task


# ── Component 2: Prediction Error ───────────────────────────────────

def _expected_reward(task_category: str, ltm_entries) -> float:
    """Running average reward for this category from LTM history.

    Returns 0.5 (maximum uncertainty) when the category has never been
    seen, so the first encounter always has moderate prediction error
    regardless of outcome.
    """
    rewards = [e.reward for e in ltm_entries if e.category == task_category]
    if not rewards:
        return 0.5
    return sum(rewards) / len(rewards)


def _outcome_surprise(reward: float, expected: float) -> float:
    """|reward - expected|. Both positive and negative surprises are salient."""
    return abs(reward - expected)


def _zpd_match(reward: float) -> float:
    """Gaussian centered on partial completion (reward ~ 0.5).

    Episodes at the boundary of the child's ability are most salient.
    Perfect success and complete failure are less informative.
    """
    return math.exp(-((reward - 0.5) / 0.35) ** 2)


def compute_prediction_error(
    reward: float,
    task_category: str,
    ltm_entries,
) -> float:
    """Combined prediction error: outcome surprise + ZPD match."""
    expected = _expected_reward(task_category, ltm_entries)
    surprise = _outcome_surprise(reward, expected)
    zpd = _zpd_match(reward)
    return 0.6 * surprise + 0.4 * zpd


# ── Component 3: Teaching Signal ────────────────────────────────────

def _productive_struggle(
    partial_credits: list[float], threshold: float,
) -> float:
    """Ratio of correction-then-improvement transitions in partial credits.

    Detects moments where the child failed (credit < threshold) and then
    succeeded on the next attempt (credit >= threshold), indicating the
    caregiver's feedback actually helped.
    """
    if len(partial_credits) < 2:
        return 0.0
    corrections = 0
    for i in range(len(partial_credits) - 1):
        if partial_credits[i] < threshold <= partial_credits[i + 1]:
            corrections += 1
    return corrections / (len(partial_credits) - 1)


def _effort(turns_used: int, max_turns: int) -> float:
    """Fraction of turn budget consumed. More turns = more interaction."""
    if max_turns <= 0:
        return 0.0
    return min(turns_used / max_turns, 1.0)


def compute_teaching(
    partial_credits: list[float],
    turns_used: int,
    max_turns: int,
    caregiver_present: bool,
    threshold: float = config.ACTION_MATCH_THRESHOLD,
) -> float:
    """Combined teaching signal: productive struggle + effort.

    Returns 0.0 when no caregiver is present (Solo, Peer conditions).
    """
    if not caregiver_present:
        return 0.0
    struggle = _productive_struggle(partial_credits, threshold)
    eff = _effort(turns_used, max_turns)
    return 0.6 * struggle + 0.4 * eff


# ── Top-level salience ──────────────────────────────────────────────

def compute_salience(
    task_category: str,
    goal_description: str,
    category_counts: Counter,
    ltm_entries,
    reward: float,
    partial_credits: list[float],
    turns_used: int,
    max_turns: int,
    caregiver_present: bool,
    alpha: float = config.SALIENCE_ALPHA,
    beta: float = config.SALIENCE_BETA,
    gamma: float = config.SALIENCE_GAMMA,
) -> float:
    """Compute salience score for an episode.

    s = alpha * novelty + beta * prediction_error + gamma * teaching

    Each component produces a graded [0, 1] signal from actual episode
    outcomes. Gamma is set per-condition (0.0 for Solo/Peer/RoleLabeled,
    0.3 for Relational).
    """
    novelty = compute_novelty(
        task_category, goal_description, category_counts, ltm_entries,
    )
    pred_error = compute_prediction_error(reward, task_category, ltm_entries)
    teaching = compute_teaching(
        partial_credits, turns_used, max_turns, caregiver_present,
    )
    return alpha * novelty + beta * pred_error + gamma * teaching
