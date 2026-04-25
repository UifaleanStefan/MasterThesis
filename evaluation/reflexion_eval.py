"""
Reflexion ablation runner — multi-episode persistent-memory comparison.

Standard benchmarks instantiate fresh memory per episode, which is correct
for systems like V4 that don't accumulate verbal lessons across episodes.
V6 (and V7) deliberately persist lessons across episodes — the whole point
of Reflexion. This runner provides:

  * persistent_run_episodes(memory, env_factory, policy_factory, ...)
        Runs N episodes against ONE memory instance. The memory's clear()
        is called between episodes (V6's clear keeps lessons and wipes
        events; V4's clear wipes everything; both are correct per their
        contracts).

  * compare_variants(env_factory, n_episodes, ...)
        Headline 4-way comparison: V4-base, V6-no-lessons (control to
        isolate the lesson effect), V6-with-lessons, V6-with-lessons +
        ReflexionPolicy. Returns a dict of per-variant metrics with
        per-episode reward arrays for paired statistical tests.

The runner returns paired-by-episode-seed reward arrays so downstream
analysis can compute Cohen's d / t-tests via evaluation.statistics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.loop import run_episode_with_any_memory
from agent.policy import ExplorationPolicy
from agent.reflexion_policy import ReflexionPolicy
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from memory.graph_memory_v6 import GraphMemoryV6, MemoryParamsV6
from memory.lesson import make_generator


# ---------------------------------------------------------------------------
# Single-variant runner
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    name: str
    rewards: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    mem_sizes: list[float] = field(default_factory=list)
    lessons_recorded: list[bool] = field(default_factory=list)
    n_lessons_final: int = 0

    @property
    def mean_reward(self) -> float:
        return statistics.mean(self.rewards) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return statistics.stdev(self.rewards) if len(self.rewards) > 1 else 0.0

    @property
    def mean_precision(self) -> float | None:
        return statistics.mean(self.precisions) if self.precisions else None

    @property
    def mean_mem_size(self) -> float:
        return statistics.mean(self.mem_sizes) if self.mem_sizes else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_precision": self.mean_precision,
            "mean_memory_size": self.mean_mem_size,
            "rewards": list(self.rewards),
            "lessons_recorded_per_ep": list(self.lessons_recorded),
            "n_lessons_final": self.n_lessons_final,
            "n_episodes": len(self.rewards),
        }


def persistent_run_episodes(
    memory: Any,
    env_factory: Callable[[int], Any],
    policy_factory: Callable[[int, Any], Any],
    *,
    n_episodes: int,
    k: int = 8,
    seed_offset: int = 6000,
    lesson_generator: Callable | None = None,
    name: str = "variant",
    verbose: bool = False,
) -> VariantResult:
    """
    Loop ``n_episodes`` against a SINGLE memory instance.

    Args:
        memory             : already-instantiated memory (NOT a factory).
        env_factory        : callable(seed) -> fresh env.
        policy_factory     : callable(seed, memory) -> fresh policy. The
                             memory argument lets a ReflexionPolicy bind
                             to the persistent instance.
        n_episodes         : number of episodes to run.
        k                  : retrieval top-k.
        seed_offset        : episode_seed = seed_offset + ep_index.
        lesson_generator   : optional, plumbed through to the run loop.

    Memory.clear() is called between episodes; V6's clear() keeps lessons.
    """
    result = VariantResult(name=name)
    for ep in range(n_episodes):
        ep_seed = seed_offset + ep
        env = env_factory(ep_seed)
        env.reset()
        policy = policy_factory(ep_seed, memory)
        # NOTE: we deliberately do NOT call memory.clear() before the first
        # episode — V6 starts empty. We DO call it between episodes (the
        # run loop also calls it via its own initial memory.clear()).
        success, events, stats = run_episode_with_any_memory(
            env,
            policy,
            memory,
            k=k,
            episode_seed=ep_seed,
            lesson_generator=lesson_generator,
        )
        result.rewards.append(float(stats.get("reward", 0.0)))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            result.precisions.append(float(prec))
        result.mem_sizes.append(float(stats.get("memory_size", 0)))
        result.lessons_recorded.append(bool(stats.get("lesson_recorded", False)))
        if verbose:
            ls = "L" if stats.get("lesson_recorded") else "."
            print(f"    ep {ep:3d} seed={ep_seed} reward={result.rewards[-1]:.3f} {ls}")

    if hasattr(memory, "get_stats"):
        stats = memory.get_stats()
        result.n_lessons_final = int(stats.get("n_lessons", 0))
    return result


# ---------------------------------------------------------------------------
# 4-way comparison
# ---------------------------------------------------------------------------


def compare_variants(
    env_factory: Callable[[int], Any],
    *,
    v4_params: MemoryParamsV4,
    v6_params: MemoryParamsV6,
    n_episodes: int = 30,
    k: int = 8,
    seed_offset: int = 6000,
    policy_seed: int = 42,
    verbose: bool = False,
) -> dict[str, VariantResult]:
    """
    Run the four headline variants on the given env, with PAIRED episode
    seeds so the resulting reward arrays can be compared via paired t-test.

    Variants:
      V4-base       — fresh-memory-per-episode V4. The control.
      V6-no-lessons — V6 with w_lesson=0; should match V4-base by invariant.
      V6-w-lessons  — V6 with the supplied w_lesson; persistent across eps;
                      ExplorationPolicy is used directly (lessons influence
                      retrieval but the rule-based policy doesn't read them
                      explicitly because they're not in past_events).
      V6-Reflexion  — V6 with the supplied w_lesson AND ReflexionPolicy
                      wrapping ExplorationPolicy: lessons are injected as
                      synthetic past_events, so the policy's regex picks
                      them up.
    """
    results: dict[str, VariantResult] = {}

    def base_policy(seed: int, _mem: Any):
        return ExplorationPolicy(seed=seed)

    def reflexion_policy(seed: int, mem: Any):
        return ReflexionPolicy(
            base_policy=ExplorationPolicy(seed=seed),
            memory=mem,
            k_lessons=3,
        )

    # ---- V4-base (fresh memory per ep) ----
    if verbose:
        print("\n  [V4-base]")
    res = VariantResult(name="V4-base")
    for ep in range(n_episodes):
        ep_seed = seed_offset + ep
        env = env_factory(ep_seed)
        env.reset()
        mem = GraphMemoryV4(v4_params)
        policy = ExplorationPolicy(seed=policy_seed + ep)
        success, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=ep_seed,
        )
        res.rewards.append(float(stats.get("reward", 0.0)))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            res.precisions.append(float(prec))
        res.mem_sizes.append(float(stats.get("memory_size", 0)))
        if verbose:
            print(f"    ep {ep:3d} reward={res.rewards[-1]:.3f}")
    results["V4-base"] = res

    # ---- V6-no-lessons (fresh memory per ep, w_lesson=0) ----
    if verbose:
        print("\n  [V6-no-lessons]")
    no_lesson_params = MemoryParamsV6(
        **{k_: getattr(v6_params, k_) for k_ in vars(v6_params) if k_ != "w_lesson"},
        w_lesson=0.0,
    )
    res = VariantResult(name="V6-no-lessons")
    for ep in range(n_episodes):
        ep_seed = seed_offset + ep
        env = env_factory(ep_seed)
        env.reset()
        mem = GraphMemoryV6(no_lesson_params)
        policy = ExplorationPolicy(seed=policy_seed + ep)
        success, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=ep_seed,
        )
        res.rewards.append(float(stats.get("reward", 0.0)))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            res.precisions.append(float(prec))
        res.mem_sizes.append(float(stats.get("memory_size", 0)))
        if verbose:
            print(f"    ep {ep:3d} reward={res.rewards[-1]:.3f}")
    results["V6-no-lessons"] = res

    # ---- V6-w-lessons (persistent memory, no Reflexion wrapper) ----
    if verbose:
        print("\n  [V6-w-lessons]")
    mem_persistent = GraphMemoryV6(v6_params)
    gen = make_generator("heuristic", episode_seed=policy_seed)
    results["V6-w-lessons"] = persistent_run_episodes(
        memory=mem_persistent,
        env_factory=env_factory,
        policy_factory=base_policy,
        n_episodes=n_episodes,
        k=k,
        seed_offset=seed_offset,
        lesson_generator=gen,
        name="V6-w-lessons",
        verbose=verbose,
    )

    # ---- V6-Reflexion (persistent memory + ReflexionPolicy) ----
    if verbose:
        print("\n  [V6-Reflexion]")
    mem_persistent_2 = GraphMemoryV6(v6_params)
    results["V6-Reflexion"] = persistent_run_episodes(
        memory=mem_persistent_2,
        env_factory=env_factory,
        policy_factory=reflexion_policy,
        n_episodes=n_episodes,
        k=k,
        seed_offset=seed_offset,
        lesson_generator=gen,
        name="V6-Reflexion",
        verbose=verbose,
    )

    return results


# ---------------------------------------------------------------------------
# Same-env retry experiment (Reflexion's original setup)
# ---------------------------------------------------------------------------


@dataclass
class RetryResult:
    """Per (env_seed, try_idx) cell."""
    env_seed: int
    try_idx: int
    reward: float
    precision: float | None
    memory_size: float
    lesson_recorded: bool


def retry_run(
    memory_factory: Callable[[], Any],
    env_factory: Callable[[int], Any],
    policy_factory: Callable[[int, Any], Any],
    *,
    env_seeds: list[int],
    n_tries: int,
    k: int = 8,
    policy_seed: int = 42,
    lesson_generator_factory: Callable[[int], Callable] | None = None,
    persistent_memory: bool = True,
    name: str = "variant",
    verbose: bool = False,
) -> list[RetryResult]:
    """
    Same-env retry experiment.

    For each env_seed in env_seeds:
        construct ONE memory instance (if persistent_memory) or none yet
        for try_idx in 0..n_tries-1:
            if not persistent_memory: memory = memory_factory()
            policy = policy_factory(policy_seed + try_idx, memory)
            episode_seed = env_seed
            run one episode, optionally generating a lesson at the end
        memory is discarded (or reused next env_seed if not persistent)

    Returns a flat list of RetryResult (one per cell).

    Reflexion semantics:
      * persistent_memory=True, lesson_generator=heuristic:
            V6 accumulates lessons across tries on the same env layout.
            ReflexionPolicy (passed in via policy_factory) injects them.
      * persistent_memory=False:
            V4-base behavior: each try gets fresh memory, no learning.
    """
    results: list[RetryResult] = []
    for env_seed in env_seeds:
        memory = memory_factory() if persistent_memory else None
        gen = lesson_generator_factory(env_seed) if lesson_generator_factory else None

        for try_idx in range(n_tries):
            if not persistent_memory:
                memory = memory_factory()
            env = env_factory(env_seed)
            env.reset()
            policy = policy_factory(policy_seed + try_idx, memory)
            success, _, stats = run_episode_with_any_memory(
                env,
                policy,
                memory,
                k=k,
                episode_seed=env_seed,
                lesson_generator=gen,
            )
            results.append(
                RetryResult(
                    env_seed=env_seed,
                    try_idx=try_idx,
                    reward=float(stats.get("reward", 0.0)),
                    precision=stats.get("retrieval_precision"),
                    memory_size=float(stats.get("memory_size", 0)),
                    lesson_recorded=bool(stats.get("lesson_recorded", False)),
                )
            )
            if verbose:
                tag = "L" if stats.get("lesson_recorded") else "."
                print(
                    f"    [{name}] env_seed={env_seed:5d} try={try_idx:2d}  "
                    f"reward={results[-1].reward:.3f}  {tag}"
                )
    return results


def compare_retry_variants(
    env_factory: Callable[[int], Any],
    *,
    v4_params: MemoryParamsV4,
    v6_params: MemoryParamsV6,
    env_seeds: list[int],
    n_tries: int,
    k: int = 8,
    policy_seed: int = 42,
    verbose: bool = False,
) -> dict[str, list[RetryResult]]:
    """
    Three-way same-env retry comparison:
      * V4-base       — fresh memory each try; no learning across tries.
      * V6-w-lessons  — persistent V6, lessons accumulate, base policy.
      * V6-Reflexion  — persistent V6 + ReflexionPolicy reading lessons.

    Returns a dict mapping variant_name -> list[RetryResult] flattened
    over (env_seed, try_idx).
    """
    out: dict[str, list[RetryResult]] = {}

    def base_policy(seed, _m):
        return ExplorationPolicy(seed=seed)

    def reflex_policy(seed, mem):
        return ReflexionPolicy(
            base_policy=ExplorationPolicy(seed=seed),
            memory=mem,
            k_lessons=3,
        )

    out["V4-base"] = retry_run(
        memory_factory=lambda: GraphMemoryV4(v4_params),
        env_factory=env_factory,
        policy_factory=base_policy,
        env_seeds=env_seeds,
        n_tries=n_tries,
        k=k,
        policy_seed=policy_seed,
        persistent_memory=False,
        lesson_generator_factory=None,
        name="V4-base",
        verbose=verbose,
    )

    def v6_factory():
        return GraphMemoryV6(v6_params)

    def gen_factory(env_seed):
        return make_generator("heuristic", episode_seed=env_seed)

    out["V6-w-lessons"] = retry_run(
        memory_factory=v6_factory,
        env_factory=env_factory,
        policy_factory=base_policy,
        env_seeds=env_seeds,
        n_tries=n_tries,
        k=k,
        policy_seed=policy_seed,
        persistent_memory=True,
        lesson_generator_factory=gen_factory,
        name="V6-w-lessons",
        verbose=verbose,
    )

    out["V6-Reflexion"] = retry_run(
        memory_factory=v6_factory,
        env_factory=env_factory,
        policy_factory=reflex_policy,
        env_seeds=env_seeds,
        n_tries=n_tries,
        k=k,
        policy_seed=policy_seed,
        persistent_memory=True,
        lesson_generator_factory=gen_factory,
        name="V6-Reflexion",
        verbose=verbose,
    )

    return out


def summarize_by_try(
    results: list[RetryResult],
) -> dict[int, dict]:
    """
    Aggregate RetryResults by try_idx — compute mean reward, std, and a list
    of (env_seed, reward) pairs per try index.

    Used to plot reward-vs-try-number curves for each variant.
    """
    from collections import defaultdict
    bucket: dict[int, list[float]] = defaultdict(list)
    for r in results:
        bucket[r.try_idx].append(r.reward)
    out: dict[int, dict] = {}
    for tidx, vals in sorted(bucket.items()):
        out[tidx] = {
            "mean_reward": statistics.mean(vals),
            "std_reward": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "rewards": list(vals),
        }
    return out
