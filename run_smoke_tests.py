"""
Smoke tests for the Learnable Memory thesis pipeline.

Runs a minimal set of episodes to verify that:
  - Grid-world benchmark path works (Key-Door, MultiHop-KeyDoor × FlatMemory, GraphMemoryV4, EpisodicSemantic)
  - DocumentQA memory eval works (reading phase + one QA, one system)
  - Runner DocumentQA+LLM branch and DB save can be exercised (optional, with LLM disabled or mock)

Usage (PowerShell):
    python run_smoke_tests.py
    python run_smoke_tests.py --no-runner   # skip runner.py branch

Exits with 0 only if all attempted runs succeed; prints which (env, system) failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_grid_episode(env_name: str, env, policy, memory_factory, k: int = 8) -> tuple[bool, str]:
    """Run one episode; return (success, error_message)."""
    from agent.loop import run_episode_with_any_memory
    try:
        mem = memory_factory()
        _, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=42
        )
        return True, ""
    except Exception as e:
        return False, str(e)


def _run_document_qa_memory_one_system() -> tuple[bool, str]:
    """Run DocumentQA memory eval for one system (EpisodicSemantic), one doc, minimal work."""
    from environment.document_qa import DocumentQA
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from evaluation.document_qa_memory import _run_reading_phase, _recall_at_k_for_qa, _get_doc
    try:
        env = DocumentQA(document_name="fantasy_lore", seed=0, question_shuffle=False)
        memory = EpisodicSemanticMemory(episodic_size=30)
        _run_reading_phase(env, memory, episode_seed=0)
        doc = _get_doc("fantasy_lore")
        qa_pairs = doc["qa_pairs"]
        n_paragraphs = len(doc["paragraphs"])
        if not qa_pairs:
            return True, ""
        first_qa = qa_pairs[0]
        question = first_qa["question"]
        relevant = first_qa.get("relevant_paragraphs", [])
        r = _recall_at_k_for_qa(memory, question, relevant, k=8, current_step=n_paragraphs)
        assert 0 <= r <= 1
        return True, ""
    except Exception as e:
        return False, str(e)


def _run_runner_docqa_branch() -> tuple[bool, str]:
    """Exercise DocumentQA+LLM path: one episode with LLMAgent (uses fallback if no API key)."""
    try:
        from environment.document_qa import DocumentQA
        from memory.episodic_semantic_memory import EpisodicSemanticMemory
        from agent.loop import run_document_qa_episode_with_llm
        from agent.llm_agent import LLMAgent
        from agent.context_formatter import FormatStyle
        env = DocumentQA(document_name="fantasy_lore", seed=0, question_shuffle=False)
        mem = EpisodicSemanticMemory(episodic_size=30)
        llm = LLMAgent(model="gpt-4o-mini", format_style=FormatStyle("flat"))
        _, score, cost_usd, stats = run_document_qa_episode_with_llm(
            env, mem, llm, k=8, episode_seed=0
        )
        assert isinstance(score, (int, float)), "score"
        assert isinstance(cost_usd, (int, float)), "cost_usd"
        assert "retrieval_tokens" in stats and "memory_size" in stats
        return True, ""
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for thesis pipeline")
    parser.add_argument("--no-runner", action="store_true", help="Skip runner DocumentQA branch test")
    args = parser.parse_args()

    failed: list[str] = []
    print("=" * 60)
    print("Smoke tests — Learnable Memory pipeline")
    print("=" * 60)

    # ── Grid: Key-Door + MultiHop × FlatMemory, GraphMemoryV4, EpisodicSemantic ──
    from environment import ToyEnvironment, MultiHopKeyDoor
    from agent import ExplorationPolicy
    from memory.flat_memory import FlatMemory
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4

    default_v4 = MemoryParamsV4(
        theta_store=0.293, theta_novel=0.908, theta_erich=0.198, theta_surprise=0.785,
        theta_entity=0.285, theta_temporal=0.278, theta_decay=0.668,
        w_graph=0.0, w_embed=1.079, w_recency=3.777, mode="learnable",
    )
    systems = {
        "FlatMemory": lambda: FlatMemory(window_size=50),
        "GraphMemoryV4": lambda: GraphMemoryV4(default_v4),
        "EpisodicSemantic": lambda: EpisodicSemanticMemory(episodic_size=30),
    }
    envs = {
        "Key-Door": (ToyEnvironment(seed=0), ExplorationPolicy(seed=0)),
        "MultiHop-KeyDoor": (MultiHopKeyDoor(seed=0), ExplorationPolicy(seed=0)),
    }
    for env_name, (env, policy) in envs.items():
        for sys_name, factory in systems.items():
            ok, err = _run_grid_episode(env_name, env, policy, factory)
            if ok:
                print(f"  [OK] {env_name} × {sys_name}")
            else:
                msg = f"{env_name} × {sys_name}: {err}"
                failed.append(msg)
                print(f"  [FAIL] {msg}")

    # ── DocumentQA memory eval (one system) ──
    ok, err = _run_document_qa_memory_one_system()
    if ok:
        print("  [OK] DocumentQA memory eval (EpisodicSemantic, one QA)")
    else:
        failed.append(f"DocumentQA memory eval: {err}")
        print(f"  [FAIL] DocumentQA memory eval: {err}")

    # ── Runner DocumentQA branch (optional) ──
    if not args.no_runner:
        ok, err = _run_runner_docqa_branch()
        if ok:
            print("  [OK] Runner DocumentQA branch (1 episode, LLM disabled)")
        else:
            failed.append(f"Runner DocumentQA: {err}")
            print(f"  [FAIL] Runner DocumentQA: {err}")
    else:
        print("  [SKIP] Runner DocumentQA branch (--no-runner)")

    print("=" * 60)
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
