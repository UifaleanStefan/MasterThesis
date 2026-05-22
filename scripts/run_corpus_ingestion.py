"""
Corpus-cumulative ingestion tracer — Stage 3 Phase 2 "magnum opus" workstream.

UNIT OF ANALYSIS: the benchmark, not the document.

Builds ONE GraphMemoryV4 instance per (benchmark, config) and ingests every
paragraph of every document in the benchmark sequentially, persisting the
memory across documents. Captures graph-state snapshots at:

  * Every doc boundary (lightweight: counts + top entities + recent events)
  * Every 25 paragraphs within the first 20 docs (within-doc keyframes —
    where the graph is forming)
  * Every 10 docs after doc 100 (sparse late-stage)
  * Always: doc 0, 1, 5, 10, 50, 100, 500, last (anchor frames)
  * End-of-corpus: full graph snapshot (nodes + edges, capped to avoid
    multi-GB files on NarrativeQA-scale runs)

Why this matters for the thesis:
  The current Phase 4 numbers measure "doc-isolated V4" — the memory is
  cleared between documents. That measures per-document retrieval quality.
  The magnum opus framing instead measures CORPUS-CUMULATIVE behavior:
  by the time the agent reaches contract #237 of CUAD, it has seen 236
  prior contracts and the same entity (`termination_clause`,
  `governing_law`, etc.) has accumulated hundreds of mention edges. The
  graph encodes a learned representation of the DOMAIN, not just the
  document. This script produces the per-step evidence for that claim.

Output:
  results/stage3/corpus_traces/{benchmark}__{config}/
    snapshots.json        — timeline of snapshot dicts (≤1000 entries)
    meta.json             — manifest, total_docs, total_paragraphs, etc.
    final_graph.json      — full final state (nodes + edges, capped at 5000 nodes)
    ingestion.log         — stdout transcript

Usage:
    python scripts/run_corpus_ingestion.py --benchmark financebench --config v4-tuned
    python scripts/run_corpus_ingestion.py --benchmark cuad --config v4-canonical --limit-docs 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import ADAPTERS, get_adapter
from memory.event import Event
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from results.manifest import build_manifest
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC, vec_to_params

OUT_ROOT = ROOT / "results" / "stage3" / "corpus_traces"


# ----------------------------------------------------------------------
# Snapshot scheduling
# ----------------------------------------------------------------------

# Anchor frames: always snapshot at these doc indices (in addition to per-doc
# boundaries up to FREQUENT_DOCS, and within-doc keyframes for first
# KEYFRAME_DOCS).
ANCHOR_DOC_INDICES = {0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000}

# For docs 0..KEYFRAME_DOCS-1, snapshot every KEYFRAME_EVERY paragraphs.
KEYFRAME_DOCS = 20
KEYFRAME_EVERY = 25

# For docs 0..FREQUENT_DOCS-1, snapshot at every doc boundary.
FREQUENT_DOCS = 100

# For docs FREQUENT_DOCS..end, snapshot only every SPARSE_EVERY docs.
SPARSE_EVERY = 10

# Cap on number of snapshots overall (safety net for huge benchmarks).
MAX_SNAPSHOTS = 1500

# Cap on per-snapshot top-entity list length.
TOP_ENTITIES_PER_SNAPSHOT = 30

# Cap on per-snapshot recent-events list length.
RECENT_EVENTS_PER_SNAPSHOT = 30

# Cap on final graph node/edge count for the dump (still includes ALL counts
# in metadata so the graph stats are fully accurate; this only limits the
# explicit per-node/edge dump for the visualizer).
FINAL_GRAPH_MAX_NODES = 5000


def should_snapshot_within_doc(doc_idx: int, para_idx_in_doc: int, total_paras: int) -> bool:
    if doc_idx >= KEYFRAME_DOCS:
        return False
    if para_idx_in_doc == 0:
        return False  # doc-boundary will catch the start
    if para_idx_in_doc == total_paras - 1:
        return False  # doc-boundary will catch the end
    return para_idx_in_doc % KEYFRAME_EVERY == 0


def should_snapshot_at_doc_boundary(doc_idx: int, total_docs: int) -> bool:
    if doc_idx in ANCHOR_DOC_INDICES:
        return True
    if doc_idx == total_docs - 1:  # last doc, always snapshot
        return True
    if doc_idx < FREQUENT_DOCS:
        return True
    return (doc_idx % SPARSE_EVERY) == 0


# ----------------------------------------------------------------------
# Graph snapshot helpers
# ----------------------------------------------------------------------


def _top_entities(memory: GraphMemoryV4, k: int = TOP_ENTITIES_PER_SNAPSHOT) -> list[dict]:
    """Top-k entities by mention count, with last-seen-step."""
    items = []
    for name, count in memory._entity_mention_count.items():
        items.append({
            "name": name,
            "mention_count": int(count),
            "last_step": int(memory._entity_last_step.get(name, -1)),
        })
    items.sort(key=lambda x: (-x["mention_count"], -x["last_step"], x["name"]))
    return items[:k]


def _recent_event_steps(memory: GraphMemoryV4, k: int = RECENT_EVENTS_PER_SNAPSHOT) -> list[int]:
    """Last k event steps stored in the graph (by step ID)."""
    steps = []
    for node, data in memory._graph.nodes(data=True):
        if data.get("type") == "event":
            steps.append(int(data.get("step", -1)))
    steps.sort(reverse=True)
    return steps[:k]


def _edge_type_counts(memory: GraphMemoryV4) -> dict:
    counts = {"temporal": 0, "mentions": 0, "mentioned_in": 0, "other": 0}
    for _, _, data in memory._graph.edges(data=True):
        t = data.get("edge_type", "other")
        if t in counts:
            counts[t] += 1
        else:
            counts["other"] += 1
    return counts


def _node_counts(memory: GraphMemoryV4) -> dict:
    n_event = 0
    n_entity = 0
    for _, data in memory._graph.nodes(data=True):
        if data.get("type") == "event":
            n_event += 1
        elif data.get("type") == "entity":
            n_entity += 1
    return {"events": n_event, "entities": n_entity, "total": n_event + n_entity}


def make_snapshot(
    memory: GraphMemoryV4,
    global_step: int,
    doc_idx: int,
    paragraph_idx_in_doc: int,
    doc_title: str,
    snapshot_kind: str,
    cumulative_paragraphs_added: int,
    cumulative_paragraphs_seen: int,
) -> dict:
    """Build a lightweight snapshot dict (counts + top entities + recent events)."""
    edges = _edge_type_counts(memory)
    nodes = _node_counts(memory)
    return {
        "global_step": global_step,
        "doc_idx": doc_idx,
        "paragraph_idx_in_doc": paragraph_idx_in_doc,
        "doc_title": doc_title,
        "kind": snapshot_kind,  # "doc_start" | "doc_end" | "keyframe" | "anchor"
        "cumulative_paragraphs_added": cumulative_paragraphs_added,
        "cumulative_paragraphs_seen": cumulative_paragraphs_seen,
        "n_event_nodes": nodes["events"],
        "n_entity_nodes": nodes["entities"],
        "n_temporal_edges": edges["temporal"],
        "n_mention_edges": edges["mentions"],
        "n_mentioned_in_edges": edges["mentioned_in"],
        "top_entities": _top_entities(memory),
        "recent_event_steps": _recent_event_steps(memory),
        "n_unique_entities_seen": len(memory._entity_mention_count),
    }


def dump_final_graph(memory: GraphMemoryV4, max_nodes: int = FINAL_GRAPH_MAX_NODES) -> dict:
    """Full final-state dump for the viewer. Caps total nodes to max_nodes
    by sampling: keep all entities + the most-recent events up to budget.
    """
    g = memory._graph
    entity_nodes = []
    event_nodes = []
    for node, data in g.nodes(data=True):
        if data.get("type") == "entity":
            entity_nodes.append({
                "id": str(node),
                "type": "entity",
                "mention_count": int(memory._entity_mention_count.get(node, 0)),
                "last_step": int(memory._entity_last_step.get(node, -1)),
            })
        elif data.get("type") == "event":
            event_nodes.append({
                "id": str(node),
                "type": "event",
                "step": int(data.get("step", -1)),
                "observation": str(data.get("observation", ""))[:160],
            })

    # Keep all entities. Sample events: most-recent first.
    entity_nodes.sort(key=lambda n: -n["mention_count"])
    event_nodes.sort(key=lambda n: -n["step"])

    budget = max_nodes
    kept_entities = entity_nodes[:budget // 3]  # up to 1/3 of budget for entities
    budget -= len(kept_entities)
    kept_events = event_nodes[:budget]
    kept_node_ids = {n["id"] for n in kept_entities + kept_events}

    kept_edges = []
    for u, v, data in g.edges(data=True):
        if str(u) in kept_node_ids and str(v) in kept_node_ids:
            kept_edges.append({
                "source": str(u),
                "target": str(v),
                "edge_type": data.get("edge_type", "other"),
            })

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "kept_nodes": len(kept_node_ids),
        "kept_edges": len(kept_edges),
        "nodes": kept_entities + kept_events,
        "edges": kept_edges,
        "capped": g.number_of_nodes() > max_nodes,
    }


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


def load_config_params(config_name: str, benchmark: str) -> MemoryParamsV4:
    """Map config name -> MemoryParamsV4 with text_mode_entities=True.

    Stage 3 corpus ingestion always opts in to the text-mode entity
    gate bypass — without it, the Bayesian importance gate (calibrated
    for grid-world) suppresses novel proper nouns indefinitely and the
    graph is mostly events with no entity nodes. See chapter §3.6.
    """
    if config_name == "v4-canonical":
        params = vec_to_params(CANONICAL_THETA_VEC)
    elif config_name == "v4-tuned":
        # Prefer wide if available, else narrow.
        params = None
        for fname in [
            f"tuned_theta_wide_{benchmark}.json",
            f"tuned_theta_{benchmark}.json",
        ]:
            path = ROOT / "results" / "stage3" / fname
            if path.exists():
                data = json.loads(path.read_text())
                vec = data.get("tuned_theta_vec")
                if isinstance(vec, list) and len(vec) == 10:
                    params = vec_to_params(np.asarray(vec, dtype=np.float64))
                    break
        if params is None:
            raise FileNotFoundError(f"No tuned theta for {benchmark!r}")
    else:
        raise ValueError(f"Unknown config: {config_name!r}")
    params.text_mode_entities = True
    return params


# ----------------------------------------------------------------------
# Main ingestion loop
# ----------------------------------------------------------------------


def run_corpus_ingestion(
    benchmark: str,
    config: str,
    limit_docs: int | None,
    seed: int,
    out_dir: Path,
    progress_every_docs: int = 25,
) -> dict:
    print(f"\n{'=' * 78}")
    print(f"  CORPUS INGESTION  benchmark={benchmark}  config={config}  seed={seed}")
    print(f"  limit_docs={limit_docs or 'ALL'}")
    print(f"{'=' * 78}")

    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=limit_docs))
    total_docs = len(docs)
    print(f"  Loaded {total_docs} docs.")

    # Build memory with the config's params.
    params = load_config_params(config, benchmark)
    memory = GraphMemoryV4(params)
    print(f"  Memory: V4 with params="
          f"theta_store={params.theta_store:.3f} "
          f"theta_novel={params.theta_novel:.3f} "
          f"w_embed={params.w_embed:.3f} "
          f"w_recency={params.w_recency:.3f}")

    snapshots: list[dict] = []
    global_step = 0  # global paragraph index across all docs
    cumulative_paragraphs_added = 0  # actually stored (after theta_store gate)
    cumulative_paragraphs_seen = 0

    doc_boundaries: list[dict] = []  # for meta.json

    t_start = time.time()
    last_progress_t = t_start

    for doc_idx, doc in enumerate(docs):
        doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        paragraphs = doc.get("paragraphs", [])
        doc_start_global_step = global_step
        n_event_at_doc_start = memory._graph.number_of_nodes()

        # Doc-start snapshot (cheap)
        if should_snapshot_at_doc_boundary(doc_idx, total_docs):
            snapshots.append(make_snapshot(
                memory, global_step, doc_idx, 0, doc_title,
                snapshot_kind="doc_start",
                cumulative_paragraphs_added=cumulative_paragraphs_added,
                cumulative_paragraphs_seen=cumulative_paragraphs_seen,
            ))

        for para_idx, paragraph in enumerate(paragraphs):
            # Format observation: include doc title prefix on first paragraph
            if para_idx == 0:
                obs = f"[{doc_title}] {paragraph}"
            else:
                obs = paragraph
            event = Event(step=global_step, observation=obs, action="read")
            n_event_before = memory._graph.number_of_nodes()
            memory.add_event(event, episode_seed=seed)
            n_event_after = memory._graph.number_of_nodes()
            if n_event_after > n_event_before:
                cumulative_paragraphs_added += 1
            cumulative_paragraphs_seen += 1
            global_step += 1

            # Within-doc keyframe?
            if should_snapshot_within_doc(doc_idx, para_idx, len(paragraphs)):
                snapshots.append(make_snapshot(
                    memory, global_step, doc_idx, para_idx, doc_title,
                    snapshot_kind="keyframe",
                    cumulative_paragraphs_added=cumulative_paragraphs_added,
                    cumulative_paragraphs_seen=cumulative_paragraphs_seen,
                ))

            # Safety net
            if len(snapshots) >= MAX_SNAPSHOTS:
                print(f"  [WARN] MAX_SNAPSHOTS={MAX_SNAPSHOTS} reached at doc_idx={doc_idx}, para_idx={para_idx}")
                break

        # Doc-end snapshot (always, when boundary criterion met)
        if should_snapshot_at_doc_boundary(doc_idx, total_docs):
            snapshots.append(make_snapshot(
                memory, global_step, doc_idx, len(paragraphs) - 1, doc_title,
                snapshot_kind="doc_end",
                cumulative_paragraphs_added=cumulative_paragraphs_added,
                cumulative_paragraphs_seen=cumulative_paragraphs_seen,
            ))

        doc_boundaries.append({
            "doc_idx": doc_idx,
            "doc_title": doc_title,
            "global_step_start": doc_start_global_step,
            "global_step_end": global_step - 1,
            "n_paragraphs": len(paragraphs),
            "events_added": memory._graph.number_of_nodes() - n_event_at_doc_start,
        })

        # Progress
        now = time.time()
        if (doc_idx + 1) % progress_every_docs == 0 or doc_idx == total_docs - 1:
            elapsed = now - t_start
            dt = now - last_progress_t
            last_progress_t = now
            print(
                f"  doc {doc_idx+1:>5}/{total_docs:<5}  "
                f"global_step={global_step:>7}  "
                f"|events|={_node_counts(memory)['events']:>6}  "
                f"|entities|={_node_counts(memory)['entities']:>5}  "
                f"snapshots={len(snapshots):>4}  "
                f"elapsed={elapsed:>6.1f}s  dt={dt:.1f}s"
            )

        if len(snapshots) >= MAX_SNAPSHOTS:
            break

    elapsed = time.time() - t_start

    # Final full-graph dump
    print(f"\n  Building final graph dump...")
    final_graph = dump_final_graph(memory)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots_path = out_dir / "snapshots.json"
    snapshots_path.write_text(json.dumps(snapshots, indent=2, default=str))

    final_graph_path = out_dir / "final_graph.json"
    final_graph_path.write_text(json.dumps(final_graph, indent=2, default=str))

    meta = {
        "_manifest": build_manifest(seed=seed, extra={
            "experiment": "stage3_corpus_ingestion",
            "benchmark": benchmark,
            "config": config,
        }),
        "benchmark": benchmark,
        "config": config,
        "seed": seed,
        "n_docs": total_docs,
        "limit_docs": limit_docs,
        "total_paragraphs_seen": cumulative_paragraphs_seen,
        "total_paragraphs_stored": cumulative_paragraphs_added,
        "store_ratio": (cumulative_paragraphs_added / max(cumulative_paragraphs_seen, 1)),
        "final_n_event_nodes": _node_counts(memory)["events"],
        "final_n_entity_nodes": _node_counts(memory)["entities"],
        "final_n_edges": memory._graph.number_of_edges(),
        "final_n_unique_entities": len(memory._entity_mention_count),
        "n_snapshots": len(snapshots),
        "elapsed_seconds": elapsed,
        "params": {
            "theta_store": params.theta_store,
            "theta_novel": params.theta_novel,
            "theta_erich": params.theta_erich,
            "theta_surprise": params.theta_surprise,
            "theta_entity": params.theta_entity,
            "theta_temporal": params.theta_temporal,
            "theta_decay": params.theta_decay,
            "w_graph": params.w_graph,
            "w_embed": params.w_embed,
            "w_recency": params.w_recency,
        },
        "doc_boundaries": doc_boundaries,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"\n  Saved {snapshots_path.name}, {meta_path.name}, {final_graph_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Final graph: {meta['final_n_event_nodes']} events, "
          f"{meta['final_n_entity_nodes']} entities, "
          f"{meta['final_n_edges']} edges")
    print(f"  Store ratio: {meta['store_ratio']:.2%} "
          f"({meta['total_paragraphs_stored']}/{meta['total_paragraphs_seen']})")

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, choices=sorted(ADAPTERS.keys()),
        help="Which benchmark to ingest.",
    )
    parser.add_argument(
        "--config", default="v4-tuned", choices=["v4-tuned", "v4-canonical"],
        help="Memory configuration (v4-tuned uses per-benchmark tuned θ).",
    )
    parser.add_argument(
        "--limit-docs", type=int, default=None,
        help="Limit number of docs (default: all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", default=None,
        help="Output dir (default: results/stage3/corpus_traces/{benchmark}__{config})",
    )
    parser.add_argument(
        "--progress-every-docs", type=int, default=25,
        help="Print a progress line every N docs.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = OUT_ROOT / f"{args.benchmark}__{args.config}"
    else:
        out_dir = Path(args.out_dir)

    try:
        run_corpus_ingestion(
            benchmark=args.benchmark,
            config=args.config,
            limit_docs=args.limit_docs,
            seed=args.seed,
            out_dir=out_dir,
            progress_every_docs=args.progress_every_docs,
        )
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
