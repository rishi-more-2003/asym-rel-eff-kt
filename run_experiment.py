"""CLI entry point for the experiment pipeline.

Usage:
    python run_experiment.py train                                   # all conditions, all seeds
    python run_experiment.py train --conditions solo relational      # subset
    python run_experiment.py train --seeds 1 2                       # specific seeds
    python run_experiment.py train --mode-caregiver combined         # fastest (default)
    python run_experiment.py train --mode-caregiver speculative      # 3 parallel 235B calls
    python run_experiment.py train --mode-caregiver sequential       # cheapest
    python run_experiment.py eval                                    # evaluate all
    python run_experiment.py all                                     # train + eval
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.tinker_utils import ensure_api_key
from src.conditions import CONDITIONS_BY_NAME, ALL_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MSI experiment pipeline",
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval", "all"],
        help="train: run training, eval: run evaluation, all: both",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions to run (solo, peer, role_labeled, relational). Default: all",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=f"Seeds to run. Default: 1..{config.NUM_SEEDS}",
    )
    parser.add_argument(
        "--mode-caregiver",
        choices=["combined", "speculative", "sequential"],
        default=None,
        help="Caregiver execution mode (default: combined = fastest)",
    )
    parser.add_argument(
        "--no-adaptive-turns",
        action="store_true",
        help="Disable adaptive turn limits (use fixed MAX_EPISODE_TURNS for all tasks)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    ensure_api_key()

    if args.mode_caregiver:
        config.CAREGIVER_MODE = args.mode_caregiver
    if args.no_adaptive_turns:
        config.ADAPTIVE_TURN_LIMITS = False

    conditions = None
    if args.conditions:
        conditions = []
        for name in args.conditions:
            if name not in CONDITIONS_BY_NAME:
                print(f"Unknown condition: {name}")
                print(f"Valid conditions: {list(CONDITIONS_BY_NAME.keys())}")
                sys.exit(1)
            conditions.append(CONDITIONS_BY_NAME[name])

    seeds = args.seeds

    os.makedirs(config.RUNS_DIR, exist_ok=True)

    conds = conditions or ALL_CONDITIONS
    sds = seeds or list(range(1, config.NUM_SEEDS + 1))
    print(f"Caregiver mode: {config.CAREGIVER_MODE}")
    print(f"Adaptive turns: {config.ADAPTIVE_TURN_LIMITS}")
    print(f"Conditions: {[c.name for c in conds]}")
    print(f"Seeds: {sds}")
    print(f"Concurrent runs: {len(conds) * len(sds)}")
    print()

    if args.mode in ("train", "all"):
        from src.training import run_all_training

        print("=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)
        await run_all_training(conditions=conditions, seeds=seeds)

    if args.mode in ("eval", "all"):
        from src.evaluation import run_full_evaluation

        print("=" * 60)
        print("EVALUATION PHASE")
        print("=" * 60)
        await run_full_evaluation(conditions=conditions, seeds=seeds)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
