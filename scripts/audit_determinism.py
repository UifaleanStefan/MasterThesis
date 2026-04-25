"""
Determinism audit — for each memory system in the benchmark grid, run the same
event sequence twice with the same episode seed and assert the resulting
retrieval lists are bit-identical.

Why this matters: AGENTS.md mandates that all stochastic decisions in memory
construction use ``random.Random(hash((episode_seed, step, "memory")))`` so
runs are reproducible. Some systems (notably the neural controllers) use
``np.random.RandomState`` instead. This script makes silent drift visible.

Usage (PowerShell, from project root):
    python scripts/audit_determinism.py
    python scripts/audit_determinism.py --systems GraphMemoryV4 NeuralV2

Exit code: 0 if every system is deterministic, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

# Ensure project root is on sys.path when invoked as `python scripts/...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory import (  # noqa: E402
    AttentionMemory,
    CausalMemory,
    EpisodicSemanticMemory,
    Event,
    FlatMemory,
    GraphMemory,
    GraphMemoryV2,
    GraphMemoryV3,
    GraphMemoryV4,
    GraphMemoryV5,
    GraphMemoryV6,
    HierarchicalMemory,
    MemoryParams,
    MemoryParamsV2,
    MemoryParamsV3,
    MemoryParamsV4,
    MemoryParamsV6,
    SemanticMemory,
    SummaryMemory,
    WorkingMemory,
)
from memory.neural_controller import NeuralMemoryController
from memory.neural_controller_v2 import NeuralMemoryControllerV2
from memory.neural_controller_v2_small import NeuralMemoryControllerV2Small


SAMPLE_EVENTS = [
    Event(0, "you are in a room. you see a red key.", "pickup"),
    Event(1, "you are in a room. you see a sign: blue key opens north door.",
          "move_north", is_hint=True),
    Event(2, "you are in a room. you see a blue key.", "pickup"),
    Event(3, "you are in a room. you see a blue door requires blue key.",
          "use_door"),
    Event(4, "you are in a room. you see the goal.", "move_east"),
    Event(5, "you are in a room. nothing of interest.", "move_west"),
]
QUERY = "you see a blue door"


def _factories() -> dict[str, Callable[[], object]]:
    """Each factory returns a fresh memory instance with non-trivial parameters."""
    p_v4 = MemoryParamsV4(
        theta_store=0.3, theta_novel=0.9, theta_surprise=0.7,
        theta_temporal=0.5, w_recency=3.0, w_embed=1.0, mode="learnable",
    )
    return {
        "FlatMemory":                lambda: FlatMemory(window_size=20),
        "SemanticMemory":            lambda: SemanticMemory(max_capacity=20),
        "SummaryMemory":             lambda: SummaryMemory(raw_buffer_size=10, summarize_every=5),
        "EpisodicSemanticMemory":    lambda: EpisodicSemanticMemory(episodic_size=10),
        "HierarchicalMemory":        lambda: HierarchicalMemory(),
        "WorkingMemory(7)":          lambda: WorkingMemory(capacity=7),
        "CausalMemory":              lambda: CausalMemory(),
        "AttentionMemory":           lambda: AttentionMemory(temperature=0.5),
        "GraphMemoryV1":             lambda: GraphMemory(MemoryParams(0.5, 0.1, 0.8, "learnable")),
        "GraphMemoryV2":             lambda: GraphMemoryV2(MemoryParamsV2()),
        "GraphMemoryV3":             lambda: GraphMemoryV3(MemoryParamsV3()),
        "GraphMemoryV4":             lambda: GraphMemoryV4(p_v4),
        "GraphMemoryV5":             lambda: GraphMemoryV5(p_v4),
        "GraphMemoryV6":             lambda: GraphMemoryV6(MemoryParamsV6(
            theta_store=p_v4.theta_store, theta_novel=p_v4.theta_novel,
            theta_erich=p_v4.theta_erich, theta_surprise=p_v4.theta_surprise,
            theta_entity=p_v4.theta_entity, theta_temporal=p_v4.theta_temporal,
            theta_decay=p_v4.theta_decay, w_graph=p_v4.w_graph,
            w_embed=p_v4.w_embed, w_recency=p_v4.w_recency,
            w_lesson=0.0, theta_lesson_decay=0.0, mode="learnable",
        )),
        "NeuralController_V1":       lambda: NeuralMemoryController(seed=0),
        "NeuralControllerV2":        lambda: NeuralMemoryControllerV2(seed=0),
        "NeuralControllerV2Small":   lambda: NeuralMemoryControllerV2Small(seed=0),
    }


def _run_once(factory: Callable[[], object], episode_seed: int) -> tuple[list, dict]:
    mem = factory()
    for ev in SAMPLE_EVENTS:
        mem.add_event(ev, episode_seed=episode_seed)
    retrieved = mem.get_relevant_events(QUERY, current_step=10, k=4)
    summary = [(e.step, e.observation, e.action) for e in retrieved]
    stats = mem.get_stats() if hasattr(mem, "get_stats") else {}
    return summary, stats


def audit(systems: list[str] | None = None, episode_seeds: list[int] | None = None) -> int:
    factories = _factories()
    if systems:
        unknown = [s for s in systems if s not in factories]
        if unknown:
            print(f"[ERROR] Unknown systems: {unknown}. Available: {list(factories)}")
            return 2
        factories = {s: factories[s] for s in systems}
    if episode_seeds is None:
        episode_seeds = [0, 7, 42]

    print("=" * 76)
    print(f"  Determinism Audit  ({len(factories)} systems × {len(episode_seeds)} seeds)")
    print("=" * 76)
    print(f"  {'System':<28}  {'Seeds':<10}  {'Status'}")
    print("-" * 76)

    failures: list[tuple[str, int, str]] = []
    for name, factory in factories.items():
        ok = True
        detail = ""
        for seed in episode_seeds:
            try:
                a, _ = _run_once(factory, seed)
                b, _ = _run_once(factory, seed)
                if a != b:
                    ok = False
                    detail = f"seed={seed}: differing retrievals (run1!=run2)"
                    failures.append((name, seed, detail))
                    break
            except Exception as exc:
                ok = False
                detail = f"seed={seed}: {type(exc).__name__}: {exc}"
                failures.append((name, seed, detail))
                break
        seeds_label = ",".join(str(s) for s in episode_seeds)
        status = "OK" if ok else f"FAIL — {detail}"
        print(f"  {name:<28}  {seeds_label:<10}  {status}")

    print("-" * 76)
    if failures:
        print(f"\n[FAIL] {len(failures)} non-deterministic system(s) found.")
        for name, seed, detail in failures:
            print(f"  - {name} (seed {seed}): {detail}")
        return 1
    print("\n[OK] All audited systems are deterministic across repeated runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Subset of systems to audit (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Episode seeds to test (default: 0 7 42)")
    args = parser.parse_args(argv)
    try:
        return audit(systems=args.systems, episode_seeds=args.seeds)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
