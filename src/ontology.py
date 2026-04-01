"""Stage 0: Generate and manage the household object ontology.

Produces a structured mapping of rooms -> objects -> canonical affordances.
This grounds all task generation in a consistent world model so that
objects and actions are reused coherently across scenarios.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.tinker_utils import sample_text, parse_json


ONTOLOGY_SYSTEM = """\
You are an expert in household environments and cognitive science. \
You design detailed object ontologies for procedural task benchmarks. \
Respond with valid JSON only — no markdown fences, no commentary."""

ONTOLOGY_PROMPT = """\
Create a comprehensive household object ontology organized by room.

For each room, list 15-25 common objects found there. For each object, \
list 2-5 actions (affordances) that can be performed with or on that object.

Use this exact JSON format:
{{
  "kitchen": {{
    "knife": ["cut", "spread", "chop", "slice"],
    "stove": ["heat", "boil", "simmer", "fry"],
    "cutting_board": ["place_on", "support"],
    "refrigerator": ["open", "close", "store_in", "retrieve_from"],
    ...
  }},
  "bathroom": {{
    "toothbrush": ["brush_teeth", "scrub"],
    ...
  }},
  ...
}}

Required rooms (include ALL of these):
- kitchen
- bathroom
- bedroom
- living_room
- laundry_room
- garden
- garage

Requirements:
- Each room must have 15-25 objects
- Each object must have 2-5 affordances
- Affordances should be concrete verbs (e.g., "cut" not "use")
- Include both large items (stove, washer) and small items (sponge, nail)
- Objects should be realistic for a typical middle-class household

JSON only:"""


def generate_ontology() -> dict[str, dict[str, list[str]]]:
    """Sample the generation model to produce the object ontology."""
    print("Stage 0: Generating household object ontology...")
    raw = sample_text(
        ONTOLOGY_SYSTEM, ONTOLOGY_PROMPT,
        max_tokens=4096, temperature=0.4,
    )
    ontology = parse_json(raw)
    if ontology is None:
        raise RuntimeError(f"Failed to parse ontology JSON. Raw output:\n{raw[:500]}")
    _validate_ontology(ontology)
    return ontology


def _validate_ontology(ontology: dict) -> None:
    required_rooms = [
        "kitchen", "bathroom", "bedroom", "living_room",
        "laundry_room", "garden", "garage",
    ]
    missing = [r for r in required_rooms if r not in ontology]
    if missing:
        raise ValueError(f"Ontology missing rooms: {missing}")
    for room, objects in ontology.items():
        if not isinstance(objects, dict) or len(objects) < 5:
            raise ValueError(
                f"Room '{room}' has too few objects ({len(objects) if isinstance(objects, dict) else 'invalid'})"
            )
        for obj, affordances in objects.items():
            if not isinstance(affordances, list) or len(affordances) < 1:
                raise ValueError(
                    f"Object '{obj}' in '{room}' has invalid affordances"
                )


def save_ontology(ontology: dict, path: str | None = None) -> str:
    path = path or config.ONTOLOGY_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)
    total_objects = sum(len(objs) for objs in ontology.values())
    print(f"  Saved ontology: {len(ontology)} rooms, {total_objects} objects -> {path}")
    return path


def load_ontology(path: str | None = None) -> dict[str, dict[str, list[str]]]:
    path = path or config.ONTOLOGY_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_objects_for_category(
    ontology: dict[str, dict[str, list[str]]], category: str
) -> dict[str, list[str]]:
    """Return a merged object->affordance map for rooms relevant to a category."""
    rooms = config.ROOM_CATEGORY_MAP.get(category, list(ontology.keys()))
    merged: dict[str, list[str]] = {}
    for room in rooms:
        if room in ontology:
            for obj, affs in ontology[room].items():
                if obj not in merged:
                    merged[obj] = list(affs)
                else:
                    merged[obj] = list(set(merged[obj] + affs))
    return merged


if __name__ == "__main__":
    ont = generate_ontology()
    save_ontology(ont)
