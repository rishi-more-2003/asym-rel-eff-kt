"""Stage 3: Self-verification of generated tasks.

Feeds each expanded task back to the model with a structured verification
prompt that checks causal consistency, goal achievement, object usage,
and difficulty accuracy. Uses a soft threshold: tasks that pass the
critical checks (causal_chain, goal_achieved, step_count) are accepted.
Failed tasks are re-expanded with feedback (1 retry).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import HouseholdTask, TaskSkeleton
from src.tinker_utils import sample_all, parse_json

CRITICAL_CHECKS = {"causal_chain", "goal_achieved", "step_count"}

VERIFY_SYSTEM = """\
You are reviewing a household task for a cognitive science benchmark. \
Check it for basic logical consistency. Be practical, not pedantic — \
minor imperfections are acceptable as long as the task is logically sound \
and usable.

Respond with valid JSON only — no markdown fences, no commentary."""


def _build_verify_prompt(task: HouseholdTask) -> str:
    task_json = json.dumps(task.to_dict(), indent=2)
    return f"""\
Review this household task for logical consistency.

Task:
{task_json}

Check these criteria (pass/fail each):

1. CAUSAL_CHAIN: Do the steps follow a logical causal order? Can each step
   reasonably be performed given what happened before it?

2. GOAL_ACHIEVED: Would completing all steps achieve the stated goal?

3. STEP_COUNT: Does the number of action steps match the stated difficulty?

4. OBJECT_USAGE: Are the key objects used sensibly in the steps?

5. MISTAKES_PLAUSIBLE: Are the common_mistakes reasonable errors a beginner
   might make?

6. HINTS_PROGRESSIVE: Do caregiver_hints go from general to specific?

Respond with JSON:
{{
  "checks": {{
    "causal_chain": {{"pass": true/false, "issue": "brief note or empty"}},
    "goal_achieved": {{"pass": true/false, "issue": "brief note or empty"}},
    "step_count": {{"pass": true/false, "issue": "brief note or empty"}},
    "object_usage": {{"pass": true/false, "issue": "brief note or empty"}},
    "mistakes_plausible": {{"pass": true/false, "issue": "brief note or empty"}},
    "hints_progressive": {{"pass": true/false, "issue": "brief note or empty"}}
  }}
}}

JSON only:"""


def _compute_verdict(checks: dict) -> tuple[str, list[str]]:
    """Derive pass/fail from individual check results.

    A task passes if all critical checks (causal_chain, goal_achieved,
    step_count) pass. Non-critical failures are noted but tolerated.
    """
    failures = []
    for name, result in checks.items():
        if isinstance(result, dict) and not result.get("pass", True):
            issue = result.get("issue", result.get("reason", ""))
            failures.append(f"{name}: {issue}")

    critical_failed = any(
        name in CRITICAL_CHECKS
        for name, result in checks.items()
        if isinstance(result, dict) and not result.get("pass", True)
    )
    verdict = "fail" if critical_failed else "pass"
    return verdict, failures


def verify_tasks(tasks: list[HouseholdTask]) -> list[dict]:
    """Fire ALL verification requests at once for maximum parallelism."""
    print(f"Stage 3: Verifying {len(tasks)} tasks (all-at-once)...")

    prompts = [(VERIFY_SYSTEM, _build_verify_prompt(t)) for t in tasks]
    raw_texts = sample_all(
        prompts, temperature=0.2, max_tokens=512,
        progress_label="verification",
    )

    results: list[dict] = []
    for task, raw in zip(tasks, raw_texts):
        parsed = parse_json(raw)
        if parsed is None:
            results.append({
                "task": task,
                "verdict": "pass",
                "failure_reasons": [],
                "checks": {},
            })
            continue

        checks = parsed.get("checks", {})
        verdict, failure_reasons = _compute_verdict(checks)
        results.append({
            "task": task,
            "verdict": verdict,
            "failure_reasons": failure_reasons,
            "checks": checks,
        })

    passed = sum(1 for r in results if r["verdict"] == "pass")
    failed = len(results) - passed
    print(f"  Verification complete: {passed} passed, {failed} failed")
    return results


def verify_and_retry(
    tasks: list[HouseholdTask],
    skeletons: list[TaskSkeleton],
    ontology: dict[str, dict[str, list[str]]],
    max_retries: int = 1,
    **kwargs,
) -> list[HouseholdTask]:
    """Verify tasks, re-expand failures with feedback, return final set."""
    from src.expand_tasks import expand_skeletons_with_feedback

    skeleton_map = {s.skeleton_id.replace("skel_", "task_"): s for s in skeletons}

    verified_tasks: list[HouseholdTask] = []
    pending = list(tasks)

    for attempt in range(max_retries + 1):
        if not pending:
            break

        label = "Initial verification" if attempt == 0 else f"Retry {attempt}"
        print(f"\n  {label}: {len(pending)} tasks to verify")

        results = verify_tasks(pending)

        failed_items: list[tuple[TaskSkeleton, str]] = []
        for r in results:
            task: HouseholdTask = r["task"]
            if r["verdict"] == "pass":
                task.verification_attempts = attempt
                verified_tasks.append(task)
            elif attempt < max_retries:
                skel = skeleton_map.get(task.task_id)
                if skel is None:
                    task.verification_attempts = attempt
                    verified_tasks.append(task)
                    continue
                feedback = "; ".join(r.get("failure_reasons", []))
                failed_items.append((skel, feedback))
            else:
                task.verification_attempts = attempt
                verified_tasks.append(task)

        if failed_items and attempt < max_retries:
            print(f"    Re-expanding {len(failed_items)} failed tasks...")
            retried = expand_skeletons_with_feedback(failed_items, ontology)
            for t in retried:
                t.verification_attempts = attempt + 1
            pending = retried
        else:
            pending = []

    passed = sum(1 for t in verified_tasks if t.verification_attempts == 0)
    retried = sum(1 for t in verified_tasks if t.verification_attempts > 0)
    print(f"\n  Final: {len(verified_tasks)} tasks "
          f"({passed} passed first try, {retried} required retries)")
    return verified_tasks
