"""Stage 2: Expand task skeletons into fully structured HouseholdTask objects.

Given each skeleton and the object ontology, generates the complete enriched
task: causally-structured ActionSteps with preconditions/effects, initial and
goal states, distractor objects, common mistakes, and progressive caregiver
hints. Uses few-shot examples and ontology grounding.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.task_schema import ActionStep, HouseholdTask, TaskSkeleton
from src.ontology import get_objects_for_category
from src.tinker_utils import sample_all, parse_json


EXPAND_SYSTEM = """\
You are a cognitive science researcher designing a household task benchmark. \
You create detailed, causally-structured procedural tasks that will be used \
to study how an AI child agent learns from an AI caregiver agent.

Each action step must have explicit preconditions (what must be true before \
the step) and effects (what changes after the step). This causal structure \
lets us verify logical consistency.

Respond with valid JSON only — no markdown fences, no commentary."""

# Two fully worked examples embedded directly in the prompt
FEW_SHOT_EXPANSION = r"""
EXAMPLE 1:
Skeleton: {"goal_description": "Make a peanut butter and jelly sandwich", "category": "meal_preparation", "difficulty": 4, "key_objects": ["bread", "peanut_butter", "jelly", "knife", "plate"]}
Result:
{
  "available_objects": ["bread", "peanut_butter", "jelly", "knife", "plate"],
  "distractor_objects": ["fork", "toaster"],
  "affordance_map": {
    "bread": ["slice", "place_on", "hold"],
    "peanut_butter": ["open_jar", "scoop", "spread"],
    "jelly": ["open_jar", "scoop", "spread"],
    "knife": ["spread", "cut"],
    "plate": ["place_on", "hold"],
    "fork": ["stab", "stir"],
    "toaster": ["toast", "plug_in"]
  },
  "action_sequence": [
    {
      "step_number": 1,
      "action": "Take two slices of bread and place them on the plate",
      "target_object": "bread",
      "preconditions": ["bread is in bread bag", "plate is on counter"],
      "effects": ["two bread slices are on the plate"]
    },
    {
      "step_number": 2,
      "action": "Open the peanut butter jar and spread peanut butter on one slice using the knife",
      "target_object": "peanut_butter",
      "preconditions": ["bread slices are on the plate", "peanut butter jar is closed"],
      "effects": ["one bread slice has peanut butter on it", "peanut butter jar is open"]
    },
    {
      "step_number": 3,
      "action": "Open the jelly jar and spread jelly on the other slice using the knife",
      "target_object": "jelly",
      "preconditions": ["second bread slice is plain on the plate"],
      "effects": ["second bread slice has jelly on it"]
    },
    {
      "step_number": 4,
      "action": "Press the two slices together with toppings facing inward",
      "target_object": "bread",
      "preconditions": ["one slice has peanut butter", "other slice has jelly"],
      "effects": ["sandwich is assembled on the plate"]
    }
  ],
  "initial_state": {
    "bread": "in bread bag on counter",
    "peanut_butter": "jar closed in pantry",
    "jelly": "jar closed in refrigerator",
    "knife": "in utensil drawer",
    "plate": "in cabinet"
  },
  "goal_state": {
    "sandwich": "assembled on plate",
    "bread": "two slices pressed together with fillings inside"
  },
  "common_mistakes": [
    "Spreading both peanut butter and jelly on the same slice, leaving the other slice plain",
    "Forgetting to use the knife and trying to spread with fingers",
    "Placing toppings on the outside of the bread instead of facing inward",
    "Not opening the jars before trying to scoop"
  ],
  "caregiver_hints": [
    "Think about what you need to put on the bread first",
    "You will need two slices, one for each spread",
    "Use the knife to spread the peanut butter on one slice",
    "Make sure the toppings face each other when you put the slices together"
  ]
}

EXAMPLE 2:
Skeleton: {"goal_description": "Water a potted basil plant on the kitchen windowsill", "category": "gardening", "difficulty": 1, "key_objects": ["watering_can", "basil_plant"]}
Result:
{
  "available_objects": ["watering_can", "basil_plant"],
  "distractor_objects": ["pruning_shears"],
  "affordance_map": {
    "watering_can": ["fill", "pour", "carry"],
    "basil_plant": ["water", "inspect", "move"],
    "pruning_shears": ["cut", "trim"]
  },
  "action_sequence": [
    {
      "step_number": 1,
      "action": "Fill the watering can at the kitchen sink and pour water gently into the basil plant pot",
      "target_object": "watering_can",
      "preconditions": ["watering can is empty", "basil plant is on windowsill"],
      "effects": ["basil plant soil is moist", "watering can is partially empty"]
    }
  ],
  "initial_state": {
    "watering_can": "empty, stored under sink",
    "basil_plant": "dry soil, on kitchen windowsill"
  },
  "goal_state": {
    "basil_plant": "soil is evenly moist"
  },
  "common_mistakes": [
    "Over-watering until water pools on the surface and overflows the saucer"
  ],
  "caregiver_hints": [
    "The plant looks thirsty — what could you use to give it water?"
  ]
}
"""


def _build_expansion_prompt(
    skeleton: TaskSkeleton,
    ontology: dict[str, dict[str, list[str]]],
) -> str:
    available_objects = get_objects_for_category(ontology, skeleton.category)
    object_list = ", ".join(sorted(available_objects.keys()))

    skel_json = json.dumps(skeleton.to_dict(), indent=2)

    return f"""\
{FEW_SHOT_EXPANSION}

Now expand this skeleton into a full task using the same JSON format as the examples above.

Skeleton:
{skel_json}

Objects available in the relevant rooms (pick from these, add others only if truly necessary):
{object_list}

Requirements:
- available_objects: list all objects the task genuinely needs
- distractor_objects: 1-3 plausible but unnecessary objects from the same room
- affordance_map: cover ALL objects (available + distractor) with 2-5 concrete verb affordances
- action_sequence: exactly {skeleton.difficulty} steps, each with step_number, action, target_object, preconditions, effects
- Preconditions for step 1 should reference the initial_state
- Each subsequent step's preconditions must be satisfied by effects of prior steps
- initial_state: starting condition of every relevant object
- goal_state: what must be true when the task is done
- common_mistakes: {max(1, skeleton.difficulty // 2)}-{skeleton.difficulty} plausible errors a novice would make
- caregiver_hints: {max(1, skeleton.difficulty // 2)}-{skeleton.difficulty} hints ordered from vague to specific

JSON only:"""


def expand_skeletons(
    skeletons: list[TaskSkeleton],
    ontology: dict[str, dict[str, list[str]]],
) -> list[HouseholdTask]:
    """Expand all skeletons into full HouseholdTask objects.

    Submits ALL requests to Tinker at once for maximum parallelism.
    """
    print(f"Stage 2: Expanding {len(skeletons)} skeletons into full tasks...")

    prompts = [
        (EXPAND_SYSTEM, _build_expansion_prompt(skel, ontology))
        for skel in skeletons
    ]
    raw_texts = sample_all(prompts, temperature=0.5, progress_label="expansion")

    tasks: list[HouseholdTask] = []
    failed = 0
    for skel, raw in zip(skeletons, raw_texts):
        parsed = parse_json(raw)
        if parsed is None:
            print(f"    [fail] JSON parse failed for {skel.skeleton_id}: "
                  f"{skel.goal_description[:50]}")
            failed += 1
            continue

        errors = _validate_expansion(parsed, skel.difficulty)
        if errors:
            print(f"    [fail] Validation errors for {skel.skeleton_id}: "
                  f"{'; '.join(errors)}")
            failed += 1
            continue

        task = _build_task(skel, parsed)
        tasks.append(task)

    print(f"  Expansion complete: {len(tasks)} tasks, {failed} failures")
    return tasks


def expand_skeletons_with_feedback(
    items: list[tuple[TaskSkeleton, str]],
    ontology: dict[str, dict[str, list[str]]],
) -> list[HouseholdTask]:
    """Re-expand skeletons with verification feedback. All requests fired at once."""
    prompts = []
    for skel, feedback in items:
        prompt_text = _build_expansion_prompt(skel, ontology)
        if feedback:
            prompt_text += (
                "\n\nPREVIOUS ATTEMPT HAD ISSUES. Please fix:\n" + feedback
            )
        prompts.append((EXPAND_SYSTEM, prompt_text))

    raw_texts = sample_all(prompts, temperature=0.5, progress_label="re-expansion")

    tasks: list[HouseholdTask] = []
    for (skel, _fb), raw in zip(items, raw_texts):
        parsed = parse_json(raw)
        if parsed is None:
            continue
        errors = _validate_expansion(parsed, skel.difficulty)
        if errors:
            continue
        tasks.append(_build_task(skel, parsed))

    return tasks


def _validate_expansion(data: dict, expected_steps: int) -> list[str]:
    errors = []
    required = [
        "available_objects", "distractor_objects", "affordance_map",
        "action_sequence", "initial_state", "goal_state",
        "common_mistakes", "caregiver_hints",
    ]
    for key in required:
        if key not in data:
            errors.append(f"missing field: {key}")

    seq = data.get("action_sequence", [])
    if isinstance(seq, list):
        if len(seq) != expected_steps:
            errors.append(f"expected {expected_steps} steps, got {len(seq)}")
        for j, step in enumerate(seq):
            if not isinstance(step, dict):
                errors.append(f"step {j} is not a dict")
                continue
            for field in ["action", "target_object", "preconditions", "effects"]:
                if field not in step:
                    errors.append(f"step {j} missing '{field}'")

    return errors


def _build_task(skeleton: TaskSkeleton, data: dict) -> HouseholdTask:
    steps = []
    for j, s in enumerate(data["action_sequence"]):
        steps.append(ActionStep(
            step_number=s.get("step_number", j + 1),
            action=s["action"],
            target_object=s["target_object"],
            preconditions=s.get("preconditions", []),
            effects=s.get("effects", []),
        ))

    return HouseholdTask(
        task_id=skeleton.skeleton_id.replace("skel_", "task_"),
        category=skeleton.category,
        goal_description=skeleton.goal_description,
        difficulty=skeleton.difficulty,
        available_objects=data.get("available_objects", skeleton.key_objects),
        distractor_objects=data.get("distractor_objects", []),
        affordance_map=data.get("affordance_map", {}),
        action_sequence=steps,
        initial_state=data.get("initial_state", {}),
        goal_state=data.get("goal_state", {}),
        common_mistakes=data.get("common_mistakes", []),
        caregiver_hints=data.get("caregiver_hints", []),
    )
