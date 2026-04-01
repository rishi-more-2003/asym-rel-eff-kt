"""Shared Tinker client setup and prompting utilities used across all stages.

Provides both synchronous (for dataset generation) and async (for experiment
pipeline) interfaces. The async wrappers use concurrent.futures to bridge
Tinker's future-based API into asyncio.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Optional

import tinker
from tinker import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


_service_client: Optional[tinker.ServiceClient] = None
_sampling_client = None
_tokenizer = None


def ensure_api_key() -> None:
    if not config.TINKER_API_KEY:
        print("ERROR: TINKER_API_KEY is not set in .env")
        print("Please add your Tinker API key to the .env file and try again.")
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = config.TINKER_API_KEY


def get_service_client() -> tinker.ServiceClient:
    global _service_client
    if _service_client is None:
        ensure_api_key()
        _service_client = tinker.ServiceClient()
    return _service_client


def get_sampling_client():
    global _sampling_client
    if _sampling_client is None:
        sc = get_service_client()
        _sampling_client = sc.create_sampling_client(
            base_model=config.GENERATION_MODEL
        )
    return _sampling_client


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_sampling_client().get_tokenizer()
    return _tokenizer


# ── Prompt building ──────────────────────────────────────────────────

def build_chat_prompt(
    system: str, user: str, temperature: Optional[float] = None
) -> types.ModelInput:
    """Build a Tinker ModelInput from system + user messages."""
    tokenizer = get_tokenizer()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return types.ModelInput.from_ints(token_ids)


def build_chat_prompt_multi(
    system: str,
    messages: list[dict[str, str]],
    tokenizer=None,
) -> types.ModelInput:
    """Build a Tinker ModelInput from a system prompt and multi-turn messages."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    full_messages = [{"role": "system", "content": system}] + messages
    text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return types.ModelInput.from_ints(token_ids)


# ── Synchronous sampling (for dataset generation) ────────────────────

def sample_text(
    system: str,
    user: str,
    max_tokens: int = config.SAMPLING_MAX_TOKENS,
    temperature: float = config.SAMPLING_TEMPERATURE,
) -> str:
    """One-shot: build prompt, sample, decode, return raw text."""
    prompt = build_chat_prompt(system, user)
    client = get_sampling_client()
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n\n"],
    )
    result = client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    tokenizer = get_tokenizer()
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


def sample_all(
    prompts: list[tuple[str, str]],
    max_tokens: int = config.SAMPLING_MAX_TOKENS,
    temperature: float = config.SAMPLING_TEMPERATURE,
    progress_label: str = "",
) -> list[str]:
    """Submit ALL prompts at once, collect results with progress reporting."""
    if not prompts:
        return []

    client = get_sampling_client()
    tokenizer = get_tokenizer()
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n\n"],
    )

    futures = []
    for system, user in prompts:
        prompt = build_chat_prompt(system, user)
        future = client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        futures.append(future)

    if progress_label:
        print(f"    Submitted {len(futures)} requests for {progress_label}")

    results: list[str] = []
    for idx, future in enumerate(futures):
        result = future.result()
        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        results.append(text)

        if progress_label and (idx + 1) % 20 == 0:
            print(f"    {progress_label}: {idx + 1}/{len(futures)} collected")

    if progress_label:
        print(f"    {progress_label}: {len(futures)}/{len(futures)} done")

    return results


sample_text_batch = sample_all


# ── Async sampling (for experiment pipeline) ─────────────────────────

async def sample_async(
    client,
    prompt: types.ModelInput,
    max_tokens: int = 1024,
    temperature: float = config.SAMPLING_TEMPERATURE,
    tokenizer=None,
) -> str:
    """Submit a single sample request and await the result in asyncio."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n\n"],
    )
    future = client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, future.result)
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


async def sample_chat_async(
    client,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = config.SAMPLING_TEMPERATURE,
    tokenizer=None,
) -> str:
    """Build a multi-turn chat prompt and sample asynchronously."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    prompt = build_chat_prompt_multi(system, messages, tokenizer=tokenizer)
    return await sample_async(
        client, prompt, max_tokens=max_tokens,
        temperature=temperature, tokenizer=tokenizer,
    )


def fire_sample(
    client,
    prompt: types.ModelInput,
    max_tokens: int = 1024,
    temperature: float = config.SAMPLING_TEMPERATURE,
):
    """Fire a sample request, returning the Tinker future (non-blocking)."""
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n\n"],
    )
    return client.sample(prompt=prompt, sampling_params=params, num_samples=1)


async def await_future(future, tokenizer=None) -> str:
    """Await a Tinker future inside asyncio."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, future.result)
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


# ── JSON parsing ─────────────────────────────────────────────────────

def parse_json(raw_text: str) -> Optional[dict]:
    """Extract and parse a JSON object from model output."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_json_lenient(raw_text: str) -> Optional[dict | list]:
    """Parse JSON object or array, tolerating markdown fences and trailing text."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None
