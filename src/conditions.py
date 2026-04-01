"""Experimental condition configurations.

Four conditions isolate the effect of relational structure:
Solo, Peer, RoleLabeled, Relational.

The key differentiator between Role-Labeled and Relational is now
genuinely mechanistic:
  - Role-Labeled: generic teacher with scaffolding but no personalized
    model of the child (caregiver treats every child the same).
  - Relational: caregiver maintains a persistent mental model of THIS
    child (strengths, weaknesses, knowledge gaps, mistake patterns)
    and adapts teaching accordingly. This plus teaching salience makes
    the relationship bidirectional.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    has_caregiver: bool
    has_peer: bool
    use_instinct: bool
    use_scaffolding: bool
    use_child_model: bool
    salience_gamma: float
    description: str


SOLO = ConditionConfig(
    name="solo",
    has_caregiver=False,
    has_peer=False,
    use_instinct=False,
    use_scaffolding=False,
    use_child_model=False,
    salience_gamma=0.0,
    description="Single agent, no teacher. Memory augmented by task success only.",
)

PEER = ConditionConfig(
    name="peer",
    has_caregiver=False,
    has_peer=True,
    use_instinct=False,
    use_scaffolding=False,
    use_child_model=False,
    salience_gamma=0.0,
    description="Two identical agents, no role asymmetry. Both get LoRA updates.",
)

ROLE_LABELED = ConditionConfig(
    name="role_labeled",
    has_caregiver=True,
    has_peer=False,
    use_instinct=True,
    use_scaffolding=True,
    use_child_model=False,
    salience_gamma=0.0,
    description=(
        "Asymmetric prompts + scaffolding, but caregiver has no personalized "
        "model of the child and teaching is excluded from salience."
    ),
)

RELATIONAL = ConditionConfig(
    name="relational",
    has_caregiver=True,
    has_peer=False,
    use_instinct=True,
    use_scaffolding=True,
    use_child_model=True,
    salience_gamma=0.3,
    description=(
        "Full relational: caregiver maintains a persistent mental model of "
        "the child (knowledge gaps, mistake patterns, improvements) and "
        "adapts teaching. Teaching episodes weighted in salience."
    ),
)

ALL_CONDITIONS = [SOLO, PEER, ROLE_LABELED, RELATIONAL]

CONDITIONS_BY_NAME = {c.name: c for c in ALL_CONDITIONS}
