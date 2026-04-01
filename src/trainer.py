"""Salience-weighted SFT trainer wrapping Tinker's forward_backward + optim_step.

On every successful episode (reward > 0), builds a supervised training datum
from the child's successful action trajectory. Loss weight is scaled by
salience * reward so high-salience episodes get stronger gradient signal.

Accumulates a batch of MIN_BATCH_SIZE data before running a single
forward_backward + optim_step to avoid overfitting on individual examples.
"""

from __future__ import annotations

import asyncio
import sys, os

from tinker import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

MIN_BATCH_SIZE = config.LORA_BATCH_SIZE


class SalienceWeightedTrainer:
    """Wraps Tinker TrainingClient for salience-weighted LoRA SFT updates."""

    def __init__(self, training_client, tokenizer):
        self._client = training_client
        self._tokenizer = tokenizer
        self._sampling_client = None
        self._update_count = 0
        self._pending_data: list[types.Datum] = []

    @property
    def sampling_client(self):
        return self._sampling_client

    @property
    def update_count(self) -> int:
        return self._update_count

    def _build_training_datum(
        self,
        system_prompt: str,
        transcript: list[dict[str, str]],
        child_role: str,
        weight_multiplier: float,
    ) -> tuple[list[int], list[float]]:
        """Build token ids and per-token weights for SFT.

        System prompt and task context get weight=0 (frozen prior).
        Child's correct actions get weight=weight_multiplier.
        Caregiver turns get weight=0 (not training on teacher output).
        """
        full_messages = [{"role": "system", "content": system_prompt}] + transcript
        text = self._tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)

        weights = [0.0] * len(token_ids)

        for turn_idx, turn in enumerate(transcript):
            msgs_up_to = [{"role": "system", "content": system_prompt}]
            msgs_up_to += transcript[:turn_idx + 1]
            up_to_text = self._tokenizer.apply_chat_template(
                msgs_up_to, tokenize=False, add_generation_prompt=False,
            )
            end_pos = len(self._tokenizer.encode(up_to_text, add_special_tokens=False))

            msgs_before = [{"role": "system", "content": system_prompt}]
            msgs_before += transcript[:turn_idx]
            before_text = self._tokenizer.apply_chat_template(
                msgs_before, tokenize=False, add_generation_prompt=False,
            )
            start_pos = len(self._tokenizer.encode(before_text, add_special_tokens=False))

            if turn["role"] == child_role:
                for i in range(start_pos, min(end_pos, len(weights))):
                    weights[i] = weight_multiplier

        return token_ids, weights

    async def update_async(
        self,
        system_prompt: str,
        transcript: list[dict[str, str]],
        child_role: str,
        salience: float,
        reward: float,
    ) -> bool:
        """Accumulate a training datum; flush a batch when we have enough."""
        weight_multiplier = salience * reward
        if weight_multiplier <= 0:
            return False

        token_ids, weights = self._build_training_datum(
            system_prompt, transcript, child_role, weight_multiplier,
        )

        model_input = types.ModelInput.from_ints(token_ids)
        datum = types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=token_ids, dtype="int64", shape=[len(token_ids)],
                ),
                "weights": types.TensorData(
                    data=weights, dtype="float32", shape=[len(weights)],
                ),
            },
        )
        self._pending_data.append(datum)

        if len(self._pending_data) >= MIN_BATCH_SIZE:
            await self._flush()
            return True
        return False

    async def flush_remaining(self) -> bool:
        """Force-flush any remaining pending data (call at end of training)."""
        if self._pending_data:
            await self._flush()
            return True
        return False

    async def _flush(self) -> None:
        """Run forward_backward + optim_step on the accumulated batch."""
        batch = self._pending_data
        self._pending_data = []

        loop = asyncio.get_event_loop()

        def _do_training():
            fwd_future = self._client.forward_backward(batch, "cross_entropy")
            fwd_future.result()
            adam_params = types.AdamParams(learning_rate=config.LORA_LR)
            opt_future = self._client.optim_step(adam_params)
            opt_future.result()
            return self._client.save_weights_and_get_sampling_client()

        self._sampling_client = await loop.run_in_executor(None, _do_training)
        self._update_count += 1
