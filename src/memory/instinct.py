"""M1: Instinct Buffer -- fixed role-specific system prompts.

Never updated during training. Passed as the system prompt in every
sample() call. Excluded from the LoRA training objective (weight=0)
to preserve it as a true fixed prior.
"""

from __future__ import annotations

CAREGIVER_PROMPT = (
    "You are a knowledgeable and patient parent helping your child learn "
    "to complete household tasks. You can see the full task solution. "
    "Your goal is to help your child understand how to complete tasks by "
    "explaining your reasoning and filling in their knowledge gaps. "
    "Be warm, encouraging, and adjust your explanations to what the child "
    "seems to understand. When the child makes mistakes, gently correct "
    "them and explain why the correct approach works."
)

CHILD_PROMPT = (
    "You are a curious child learning about the world. You are trying to "
    "complete a household task. You do not know the correct steps ahead of "
    "time. Try your best to figure out what to do based on the objects "
    "available and what you know. When unsure, ask questions. Describe "
    "your actions clearly, stating exactly what object you are using and "
    "what you are doing with it."
)

PEER_PROMPT = (
    "You are trying to complete a household task together with a partner. "
    "Neither of you knows the correct solution. Discuss ideas, suggest "
    "actions, and try to figure it out together. Describe your actions "
    "clearly, stating exactly what object you are using and what you are "
    "doing with it."
)


class InstinctBuffer:
    """Fixed system prompt that encodes role-specific behavioral priors."""

    def __init__(self, role: str, scaffolding_modifier: str = ""):
        if role == "caregiver":
            self._base = CAREGIVER_PROMPT
        elif role == "child":
            self._base = CHILD_PROMPT
        elif role == "peer":
            self._base = PEER_PROMPT
        else:
            raise ValueError(f"Unknown role: {role}")
        self._role = role
        self._scaffolding_modifier = scaffolding_modifier
        self._child_model_context = ""

    @property
    def role(self) -> str:
        return self._role

    @property
    def prompt(self) -> str:
        parts = [self._base]
        if self._child_model_context:
            parts.append(self._child_model_context)
        if self._scaffolding_modifier:
            parts.append(self._scaffolding_modifier)
        return "\n\n".join(parts)

    def set_scaffolding(self, modifier: str) -> None:
        self._scaffolding_modifier = modifier

    def set_child_model(self, profile: str) -> None:
        """Inject the caregiver's personalized model of the child."""
        self._child_model_context = profile
