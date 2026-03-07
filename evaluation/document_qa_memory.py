"""
DocumentQA memory-quality evaluation — recall@k without LLM.

Runs the DocumentQA reading phase with each memory system (paragraphs stored as events),
then for each QA pair retrieves top-k by question and computes whether any of the
relevant_paragraphs (from doc metadata) appear in the retrieved set.

This isolates memory quality: which systems retain and retrieve the right paragraphs
for multi-hop questions? No policy or LLM required.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from memory.event import Event

# Document structure for recall computation (paragraph index -> event.step)
_DOCS = {
    "fantasy_lore": None,  # loaded on first use
    "mystery_case": None,
}


def _get_doc(document_name: str) -> dict:
    from environment.document_qa import FANTASY_LORE, MYSTERY_CASE, _DOCUMENTS
    if document_name not in _DOCUMENTS:
        raise ValueError(f"Unknown document: {document_name}")
    return _DOCUMENTS[document_name]


def _run_reading_phase(env, memory: Any, episode_seed: int = 0) -> None:
    """Run DocumentQA reading phase: store each paragraph as an event. Step index = paragraph index."""
    memory.clear()
    obs = env.reset()
    # First observation is "[Document: Title]\nparagraph_0"
    memory.add_event(Event(step=0, observation=obs, action="next"), episode_seed=episode_seed)
    step_idx = 1
    while not env.done:
        obs, done, _ = env.step("next")
        if done:
            break
        if env.phase == "reading":
            memory.add_event(Event(step=step_idx, observation=obs, action="next"), episode_seed=episode_seed)
            step_idx += 1
        else:
            break


def _recall_at_k_for_qa(
    memory: Any,
    question: str,
    relevant_paragraph_indices: list[int],
    k: int,
    current_step: int,
) -> float:
    """Return 1.0 if any of the top-k retrieved events have step in relevant_paragraph_indices, else 0.0."""
    retrieved = memory.get_relevant_events(question, current_step=current_step, k=k)
    retrieved_steps = {e.step for e in retrieved}
    hit = any(pidx in retrieved_steps for pidx in relevant_paragraph_indices)
    return 1.0 if hit else 0.0


def run_document_qa_memory_eval(
    document_name: str = "fantasy_lore",
    k: int = 8,
    memory_factories: dict[str, Any] | None = None,
    seed: int = 0,
    skip_systems: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run all memory systems on DocumentQA reading + retrieval; report mean recall@k per system.

    Returns
    -------
    results : dict[str, dict]
        {system_name: {"mean_recall": float, "per_question_recalls": list[float], "n_questions": int}}
    """
    from environment.document_qa import DocumentQA

    doc = _get_doc(document_name)
    qa_pairs = doc["qa_pairs"]
    n_paragraphs = len(doc["paragraphs"])

    if memory_factories is None:
        memory_factories = _make_document_qa_memory_systems()

    env = DocumentQA(document_name=document_name, seed=seed, question_shuffle=False)
    results = {}

    for sys_name, factory in memory_factories.items():
        if skip_systems and sys_name in skip_systems:
            continue
        memory = factory()
        env = DocumentQA(document_name=document_name, seed=seed, question_shuffle=False)
        _run_reading_phase(env, memory, episode_seed=seed)

        recalls = []
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            relevant = qa.get("relevant_paragraphs", [])
            # current_step: after reading, we're in "QA phase"; use n_paragraphs + qa_idx
            current_step = n_paragraphs + qa_idx
            r = _recall_at_k_for_qa(memory, question, relevant, k=k, current_step=current_step)
            recalls.append(r)

        results[sys_name] = {
            "mean_recall": statistics.mean(recalls),
            "std_recall": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            "per_question_recalls": recalls,
            "n_questions": len(qa_pairs),
        }

    return results


def _make_document_qa_memory_systems(learned_thetas: dict | None = None) -> dict[str, Any]:
    """Same memory systems as benchmark; includes GraphMemoryV4 and V5."""
    from memory.flat_memory import FlatMemory
    from memory.semantic_memory import SemanticMemory
    from memory.summary_memory import SummaryMemory
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from memory.rag_memory import RAGMemory
    from memory.graph_memory import GraphMemory, MemoryParams
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
    from memory.graph_memory_v5 import GraphMemoryV5
    from memory.hierarchical_memory import HierarchicalMemory
    from memory.working_memory import WorkingMemory
    from memory.causal_memory import CausalMemory
    from memory.attention_memory import AttentionMemory

    default_theta = (0.956, 0.378, 1.000)
    if learned_thetas:
        default_theta = learned_thetas.get("MultiHop-KeyDoor", default_theta)
    default_v4_params = MemoryParamsV4(
        theta_store=0.293, theta_novel=0.908, theta_erich=0.198, theta_surprise=0.785,
        theta_entity=0.285, theta_temporal=0.278, theta_decay=0.668,
        w_graph=0.0, w_embed=1.079, w_recency=3.777, mode="learnable",
    )

    return {
        "FlatWindow(50)":       lambda: FlatMemory(window_size=50),
        "GraphMemory+Theta":    lambda: GraphMemory(MemoryParams(*default_theta, "learnable")),
        "GraphMemoryV4":        lambda: GraphMemoryV4(default_v4_params),
        "GraphMemoryV5":        lambda: GraphMemoryV5(default_v4_params),
        "SemanticMemory":       lambda: SemanticMemory(max_capacity=80),
        "SummaryMemory":        lambda: SummaryMemory(raw_buffer_size=30, summarize_every=25),
        "EpisodicSemantic":     lambda: EpisodicSemanticMemory(episodic_size=30),
        "RAGMemory":            lambda: RAGMemory(),
        "HierarchicalMemory":   lambda: HierarchicalMemory(),
        "WorkingMemory(7)":     lambda: WorkingMemory(capacity=7),
        "CausalMemory":         lambda: CausalMemory(),
        "AttentionMemory":      lambda: AttentionMemory(temperature=0.5),
    }


def print_document_qa_table(results: dict[str, dict]) -> None:
    """Print a simple table of mean recall@k per system."""
    print("\n" + "=" * 60)
    print("DocumentQA memory-quality — mean recall@k (no LLM)")
    print("=" * 60)
    print(f"{'System':<22}  {'Mean recall':>12}  {'Std':>8}")
    print("-" * 60)
    for name, data in sorted(results.items(), key=lambda x: -x[1]["mean_recall"]):
        r = data["mean_recall"]
        s = data["std_recall"]
        print(f"{name:<22}  {r:>12.4f}  {s:>8.4f}")
    print("=" * 60)


def save_document_qa_results(results: dict, path: str | Path) -> None:
    """Save results to JSON (exclude per_question_recalls if large)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out = {}
    for sys_name, data in results.items():
        out[sys_name] = {
            "mean_recall": data["mean_recall"],
            "std_recall": data["std_recall"],
            "n_questions": data["n_questions"],
        }
    path.write_text(json.dumps(out, indent=2))
