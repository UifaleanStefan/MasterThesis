"""
Build frontend-consumable graph evolution data from corpus_traces/.

For each `results/stage3/corpus_traces/<benchmark>__<config>[__<suffix>]`
directory, emit a slimmed JSON to `web/public/data/graph_evolution/`
that the GraphEvolution.tsx viewer can load.

Transformations:

* **Snapshot timeline** (snapshots.json -> timeline.json):
  Per-snapshot counts, top-K entities, doc title — but truncated and
  formatted for fast loading. Each entry adds `first_seen_step` for
  every entity ever referenced (derived by scanning snapshots in
  order; first appearance wins). Used by the viewer to animate node
  appearance.

* **Final graph** (final_graph.json -> graph.json):
  Nodes + edges as-shipped, but with `first_seen_step` annotated on
  each entity node by cross-referencing the timeline above. Event
  nodes' first_seen_step is their `step` (deterministic).

* **Manifest** (meta.json -> manifest.json):
  Distilled benchmark + config + params + doc_boundaries for the UI.

Output structure:
    web/public/data/graph_evolution/
        index.json                              # list of available runs
        <run_id>/
            timeline.json
            graph.json
            manifest.json

Usage:
    python scripts/build_graph_evolution_frontend.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "results" / "stage3" / "corpus_traces"
OUT_ROOT = ROOT / "web" / "public" / "data" / "graph_evolution"


def derive_entity_first_seen(snapshots: list[dict]) -> dict[str, int]:
    """For every entity that ever appears in top_entities, record the
    earliest global_step at which we saw it. Used for graph-build
    animation in the viewer."""
    first_seen: dict[str, int] = {}
    for snap in snapshots:
        step = snap.get("global_step", 0)
        for ent in snap.get("top_entities", []):
            name = ent.get("name")
            if name and name not in first_seen:
                first_seen[name] = step
    return first_seen


def slim_snapshot(snap: dict) -> dict:
    """Trim per-snapshot payload for fast frontend load."""
    return {
        "step": snap["global_step"],
        "doc_idx": snap["doc_idx"],
        "para_idx": snap["paragraph_idx_in_doc"],
        "doc_title": snap.get("doc_title", "")[:80],
        "kind": snap.get("kind", ""),
        "added": snap.get("cumulative_paragraphs_added", 0),
        "seen": snap.get("cumulative_paragraphs_seen", 0),
        "n_events": snap["n_event_nodes"],
        "n_entities": snap["n_entity_nodes"],
        "n_temporal_edges": snap.get("n_temporal_edges", 0),
        "n_mention_edges": snap.get("n_mention_edges", 0),
        "top_entities": [
            {
                "name": e["name"],
                "count": e["mention_count"],
                "last_step": e["last_step"],
            }
            for e in snap.get("top_entities", [])[:15]
        ],
        "n_unique_entities_seen": snap.get("n_unique_entities_seen", 0),
    }


def annotate_graph(graph: dict, first_seen: dict[str, int]) -> dict:
    """Add `first_seen_step` to every node in the final graph."""
    nodes = []
    for n in graph.get("nodes", []):
        nid = n["id"]
        if n.get("type") == "event":
            # Event node IDs are like "event_<step>"; parse the step.
            try:
                step = int(nid.split("_", 1)[1])
            except Exception:
                step = n.get("step", 0)
            n = {**n, "first_seen_step": step}
        else:
            # Entity node: look up first_seen step from timeline.
            n = {**n, "first_seen_step": first_seen.get(nid, 0)}
        nodes.append(n)

    edges = []
    for e in graph.get("edges", []):
        # Edge timing: max of endpoints' first_seen.
        # (We'll compute this lazily client-side via the node first_seen
        # map; for now keep the edges raw.)
        edges.append(e)

    return {
        "total_nodes": graph.get("total_nodes", len(nodes)),
        "total_edges": graph.get("total_edges", len(edges)),
        "kept_nodes": graph.get("kept_nodes", len(nodes)),
        "kept_edges": graph.get("kept_edges", len(edges)),
        "capped": graph.get("capped", False),
        "nodes": nodes,
        "edges": edges,
    }


def slim_manifest(meta: dict) -> dict:
    """Distill meta.json for the frontend (drop git/python noise)."""
    return {
        "benchmark": meta["benchmark"],
        "benchmark_category": meta.get("benchmark_category", "uncategorized"),
        "config": meta["config"],
        "v4_variant": meta.get("v4_variant", ""),
        "order": meta.get("order", "canonical"),
        "shuffle_seed": meta.get("shuffle_seed"),
        "w_recency_zero": meta.get("w_recency_zero", False),
        "seed": meta.get("seed", 42),
        "n_docs": meta["n_docs"],
        "limit_docs": meta.get("limit_docs"),
        "total_paragraphs_seen": meta.get("total_paragraphs_seen", 0),
        "total_paragraphs_stored": meta.get("total_paragraphs_stored", 0),
        "store_ratio": meta.get("store_ratio", 0.0),
        "final_n_event_nodes": meta.get("final_n_event_nodes", 0),
        "final_n_entity_nodes": meta.get("final_n_entity_nodes", 0),
        "final_n_edges": meta.get("final_n_edges", 0),
        "final_n_unique_entities": meta.get("final_n_unique_entities", 0),
        "params": meta.get("params", {}),
        "doc_boundaries": meta.get("doc_boundaries", []),
        "elapsed_seconds": meta.get("elapsed_seconds", 0),
    }


def process_run(run_dir: Path) -> dict | None:
    snap_path = run_dir / "snapshots.json"
    graph_path = run_dir / "final_graph.json"
    meta_path = run_dir / "meta.json"
    if not (snap_path.exists() and graph_path.exists() and meta_path.exists()):
        print(f"  [SKIP] {run_dir.name}: missing one of snapshots/final_graph/meta")
        return None

    snapshots_raw = json.loads(snap_path.read_text())
    graph_raw = json.loads(graph_path.read_text())
    meta_raw = json.loads(meta_path.read_text())

    first_seen = derive_entity_first_seen(snapshots_raw)
    timeline = [slim_snapshot(s) for s in snapshots_raw]
    graph = annotate_graph(graph_raw, first_seen)
    manifest = slim_manifest(meta_raw)

    out_dir = OUT_ROOT / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timeline.json").write_text(json.dumps(timeline, default=str))
    (out_dir / "graph.json").write_text(json.dumps(graph, default=str))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"  [OK] {run_dir.name}: "
          f"{len(timeline)} snapshots, {len(graph['nodes'])} nodes, "
          f"{len(graph['edges'])} edges -> {out_dir.relative_to(ROOT)}")

    return {
        "run_id": run_dir.name,
        "benchmark": manifest["benchmark"],
        "benchmark_category": manifest["benchmark_category"],
        "config": manifest["config"],
        "order": manifest.get("order", "canonical"),
        "n_docs": manifest["n_docs"],
        "n_snapshots": len(timeline),
        "n_nodes": len(graph["nodes"]),
        "n_edges": len(graph["edges"]),
    }


def main() -> int:
    if not TRACES_DIR.exists():
        print(f"[FAIL] no corpus_traces dir at {TRACES_DIR}")
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_dir in sorted(TRACES_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        result = process_run(run_dir)
        if result is not None:
            runs.append(result)

    # Top-level index for the viewer.
    index = {
        "version": 1,
        "runs": runs,
        "benchmarks": sorted({r["benchmark"] for r in runs}),
        "configs": sorted({r["config"] for r in runs}),
        "orders": sorted({r["order"] for r in runs}),
    }
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=2, default=str))
    print(f"\n  Index: {len(runs)} runs across "
          f"{len(index['benchmarks'])} benchmarks x "
          f"{len(index['configs'])} configs x "
          f"{len(index['orders'])} orders")
    print(f"  -> {OUT_ROOT / 'index.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
