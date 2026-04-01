"""Child agent: 8B learner with LoRA adapter.

Each run gets its own TrainingClient (independent LoRA weights).
The SamplingClient is refreshed after each optim_step via the trainer.
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.tinker_utils import (
    build_chat_prompt_multi, fire_sample, await_future,
)


class ChildAgent:
    """8B learner agent with LoRA adapter managed by SalienceWeightedTrainer."""

    def __init__(self, training_client, tokenizer):
        self._training_client = training_client
        self._tokenizer = tokenizer
        self._sampling_client = None
        self._init_sampling_client()

    def _init_sampling_client(self):
        """Get initial sampling client from training client."""
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client()

    @property
    def sampling_client(self):
        return self._sampling_client

    @property
    def training_client(self):
        return self._training_client

    def set_sampling_client(self, client) -> None:
        """Update sampling client after a LoRA weight update."""
        self._sampling_client = client

    async def sample_async(
        self, system_prompt: str, messages: list[dict[str, str]],
    ) -> str:
        """Sample a child action/utterance asynchronously."""
        prompt = build_chat_prompt_multi(
            system_prompt, messages, tokenizer=self._tokenizer,
        )
        future = fire_sample(
            self._sampling_client, prompt,
            max_tokens=512, temperature=0.8,
        )
        return await await_future(future, tokenizer=self._tokenizer)
