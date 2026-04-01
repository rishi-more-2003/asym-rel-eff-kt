"""Async per-episode JSONL metrics logger and transcript saver.

Writes one JSON line per episode to data/runs/{condition}_{seed}/metrics.jsonl
and full transcripts to data/runs/{condition}_{seed}/transcripts/episode_{N}.json.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EpisodeMetrics:
    episode: int
    task_id: str
    category: str
    difficulty: int
    turns: int
    steps_completed: int
    total_steps: int
    reward: float
    salience: float
    lora_updated: bool
    competence_level: str
    ltm_size: int
    condition: str
    seed: int


class MetricsLogger:
    """Async JSONL metrics logger and transcript saver."""

    def __init__(self, run_dir: str):
        self._run_dir = run_dir
        self._transcript_dir = os.path.join(run_dir, "transcripts")
        self._metrics_path = os.path.join(run_dir, "metrics.jsonl")
        os.makedirs(self._transcript_dir, exist_ok=True)

    async def log_episode(
        self,
        metrics: EpisodeMetrics,
        transcript: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Write metrics line and optional transcript file."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_metrics, metrics)
        if transcript is not None:
            await loop.run_in_executor(
                None, self._write_transcript, metrics.episode, transcript,
            )

    def _write_metrics(self, metrics: EpisodeMetrics) -> None:
        line = json.dumps(asdict(metrics), ensure_ascii=False)
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _write_transcript(
        self, episode_num: int, transcript: list[dict[str, str]],
    ) -> None:
        path = os.path.join(
            self._transcript_dir, f"episode_{episode_num:04d}.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

    def load_metrics(self) -> list[dict]:
        """Load all logged metrics for evaluation."""
        if not os.path.exists(self._metrics_path):
            return []
        metrics = []
        with open(self._metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
        return metrics
