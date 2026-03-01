"""
CostTracker — per-episode and per-session LLM API cost logging and aggregation.

When using a real LLM agent (LLMAgent), the token cost of memory context is measurable
in actual dollars. CostTracker:
  1. Aggregates per-episode costs from LLMAgent.episode_stats.
  2. Computes derived metrics: cost_per_door_opened, tokens_per_step, cost_efficiency.
  3. Produces a cost breakdown table for thesis figures.
  4. Exports to JSON for persistent logging.

The key insight for the thesis:
  - FlatWindow stores everything → large context → high token cost → high API cost.
  - EpisodicSemantic stores selectively → small context → lower cost → better efficiency.
  - CostTracker quantifies this: efficiency = task_reward / total_cost_usd.

Metrics tracked per episode:
  - prompt_tokens: tokens in the system+user messages (dominated by memory context).
  - completion_tokens: tokens in the model's response (always small, ~5 tokens).
  - memory_tokens: tokens from the memory context section specifically.
  - total_cost_usd: (prompt_tokens * input_price + completion_tokens * output_price) / 1M.
  - doors_opened: partial score metric.
  - cost_per_door: total_cost / max(1, doors_opened).
  - efficiency: task_reward / (1 + total_cost_usd * 1000).

Usage:
    tracker = CostTracker(memory_system_name="EpisodicSemantic")
    for episode in range(n_episodes):
        ... run episode ...
        ep_stats = agent.reset_episode()
        tracker.record(ep_stats, reward=..., memory_size=...)

    tracker.print_summary()
    tracker.to_json("results/cost_tracking.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EpisodeCostRecord:
    episode: int
    prompt_tokens: int
    completion_tokens: int
    memory_tokens: int          # tokens from memory context (estimated)
    total_cost_usd: float
    reward: float
    memory_size: int
    cost_per_reward_unit: float  # cost / max(1e-9, reward)
    efficiency: float            # reward / (1 + cost_usd * 1000)
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Records and aggregates LLM API costs across episodes for a given memory system.
    """

    def __init__(self, memory_system_name: str = "Unknown", model: str = "gpt-4o-mini") -> None:
        self._name = memory_system_name
        self._model = model
        self._records: list[EpisodeCostRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        episode_stats: Any,    # LLMAgent.EpisodeStats or compatible dict
        reward: float,
        memory_size: int = 0,
        memory_tokens: int | None = None,
    ) -> EpisodeCostRecord:
        """
        Record a completed episode.

        Parameters
        ----------
        episode_stats : EpisodeStats or dict with keys prompt_tokens, completion_tokens, total_cost_usd
        reward : float
            Episode reward (partial score or binary).
        memory_size : int
            Number of events in memory at end of episode.
        memory_tokens : int, optional
            If known, the exact token count of memory context. Otherwise estimated as
            80% of prompt_tokens (system prompt is ~20% of typical prompt).
        """
        if hasattr(episode_stats, "to_dict"):
            stats = episode_stats.to_dict()
        elif isinstance(episode_stats, dict):
            stats = episode_stats
        else:
            stats = {
                "total_prompt_tokens": getattr(episode_stats, "total_prompt_tokens", 0),
                "total_completion_tokens": getattr(episode_stats, "total_completion_tokens", 0),
                "total_cost_usd": getattr(episode_stats, "total_cost_usd", 0.0),
            }

        prompt_tokens = stats.get("total_prompt_tokens", 0)
        completion_tokens = stats.get("total_completion_tokens", 0)
        cost_usd = stats.get("total_cost_usd", 0.0)

        if memory_tokens is None:
            # Estimate: memory context is ~70% of prompt (rest is system prompt + observation)
            memory_tokens = int(prompt_tokens * 0.7)

        ep_num = len(self._records) + 1
        cost_per_reward = cost_usd / max(1e-9, reward) if reward > 0 else float("inf")
        efficiency = reward / (1.0 + cost_usd * 1000.0)

        record = EpisodeCostRecord(
            episode=ep_num,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            memory_tokens=memory_tokens,
            total_cost_usd=cost_usd,
            reward=reward,
            memory_size=memory_size,
            cost_per_reward_unit=cost_per_reward,
            efficiency=efficiency,
        )
        self._records.append(record)
        return record

    def record_proxy(
        self,
        retrieval_tokens: int,
        reward: float,
        memory_size: int = 0,
        model: str = "gpt-4o-mini",
    ) -> EpisodeCostRecord:
        """
        Record an episode using proxy token counts (no real API, but estimate cost).
        Useful for comparing cost across memory systems even without real LLM.
        """
        pricing = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50}.get(model, 0.15)
        # retrieval_tokens = number of events retrieved; each event ~ 15 tokens
        est_prompt_tokens = retrieval_tokens * 15 + 200   # 200 for system + obs
        est_cost = est_prompt_tokens * pricing / 1_000_000

        fake_stats = {
            "total_prompt_tokens": est_prompt_tokens,
            "total_completion_tokens": 5,
            "total_cost_usd": est_cost,
        }
        return self.record(fake_stats, reward, memory_size, memory_tokens=retrieval_tokens * 15)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if not self._records:
            return {"error": "no records"}
        import statistics
        costs = [r.total_cost_usd for r in self._records]
        rewards = [r.reward for r in self._records]
        efficiencies = [r.efficiency for r in self._records if r.reward > 0]
        return {
            "memory_system": self._name,
            "n_episodes": len(self._records),
            "total_cost_usd": sum(costs),
            "mean_cost_usd": statistics.mean(costs),
            "mean_prompt_tokens": statistics.mean(r.prompt_tokens for r in self._records),
            "mean_memory_tokens": statistics.mean(r.memory_tokens for r in self._records),
            "mean_reward": statistics.mean(rewards),
            "mean_efficiency": statistics.mean(efficiencies) if efficiencies else 0.0,
            "total_episodes": len(self._records),
            "success_rate": sum(1 for r in self._records if r.reward > 0) / len(self._records),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== Cost Tracking: {self._name} ===")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "memory_system": self._name,
            "model": self._model,
            "summary": self.summary(),
            "records": [asdict(r) for r in self._records],
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"[CostTracker] Saved to {path}")

    @classmethod
    def from_json(cls, path: str | Path) -> "CostTracker":
        data = json.loads(Path(path).read_text())
        tracker = cls(memory_system_name=data["memory_system"], model=data["model"])
        for r in data["records"]:
            tracker._records.append(EpisodeCostRecord(**r))
        return tracker

    @property
    def records(self) -> list[EpisodeCostRecord]:
        return self._records.copy()


def compare_costs(trackers: list[CostTracker]) -> list[dict]:
    """
    Compare cost-efficiency across multiple memory systems.
    Returns sorted list of summary dicts (best efficiency first).
    """
    summaries = [t.summary() for t in trackers]
    return sorted(summaries, key=lambda s: -s.get("mean_efficiency", 0))
