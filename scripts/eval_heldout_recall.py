"""
Held-out recall@k + theta ablations for corpus mode (critique remediation).

Replaces the chapter's "recall lift" column — which was the CMA-ES
optimizer's own in-sample training fitness — with retrieval recall
measured on questions the tuner never saw, and adds two controls:

  * minilm-grid    — the V4 theta re-tuned on MultiHopKeyDoor under the
                     CURRENT sentence-transformers backend
                     (results/graphmemory_v4_cmaes_results.json). Separates
                     "corpus adaptation" from "embedding-backend upgrade":
                     the published canonical theta predates the encoder
                     swap, so shifts vs it conflate the two.
  * w_graph=0      — corpus-tuned theta with the graph term zeroed.
                     Tests whether the graph term carries retrieval load
                     (the four-shift's "w_graph rises from zero" claim).

No LLM calls — pure ingestion + retrieval, same objective semantics as
tuning/tune_v4t_corpus.py (recall@8 against gold paragraph steps, online
current_step = end of source doc; batch = end of corpus).

Question splits per benchmark (tuner scope = qa0 of docs 0..tune_limit-1):

    bench          eval docs   tune_limit   tuned-on Qs   held-out Qs
    financebench   150         50           50            100
    qasper         30          30           ~30           ~64
    cuad           10          30           10            122
    hotpotqa       100         30           30            70   (1 q/doc)
    longmemeval    100         30           30            70   (1 q/doc)

Output: results/stage3/heldout_recall_summary.json
        results/stage3/four_shift_rebaseline.json

Usage:
    python scripts/eval_heldout_recall.py
"""

from __future__ import annotations

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

from environment.benchmarks import get_adapter  # noqa: E402
from memory.event import Event  # noqa: E402
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4  # noqa: E402
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC, vec_to_params  # noqa: E402

K = 8
SEED = 42

# Eval-corpus size per benchmark = the Protocol A regime (FB/QASPER/CUAD)
# or the extended held-out regime (HQA/LME, where the original 10-doc eval
# had zero held-out questions).
EVAL_DOCS = {
    "financebench": None,   # all 150
    "qasper": 30,
    "cuad": 10,
    "hotpotqa": 100,
    "longmemeval": 100,
}
TUNE_LIMIT = {
    "financebench": 50, "qasper": 30, "cuad": 30,
    "hotpotqa": 30, "longmemeval": 30,
}

OUT_RECALL = ROOT / "results" / "stage3" / "heldout_recall_summary.json"
OUT_SHIFT = ROOT / "results" / "stage3" / "four_shift_rebaseline.json"

PARAM_NAMES = [
    "theta_store", "theta_novel", "theta_erich", "theta_surprise",
    "theta_entity", "theta_temporal", "theta_decay",
    "w_graph", "w_embed", "w_recency",
]


def params_from_dict(d: dict) -> MemoryParamsV4:
    """Build V4 params from a post-scaling param dict (no x4 re-scaling)."""
    return MemoryParamsV4(
        **{name: float(d[name]) for name in PARAM_NAMES}, mode="learnable",
    )


def load_thetas(bench: str) -> dict[str, MemoryParamsV4 | None]:
    """Theta variants to evaluate for one benchmark."""
    tfidf_canonical = vec_to_params(CANONICAL_THETA_VEC.copy())

    grid = json.load(open(ROOT / "results" / "graphmemory_v4_cmaes_results.json",
                          encoding="utf-8"))
    minilm_grid = params_from_dict(grid["v4"]["best_params"])

    tuned_path = ROOT / "results" / "stage3" / f"tuned_theta_v4t_corpus_{bench}.json"
    tuned = json.load(open(tuned_path, encoding="utf-8")) if tuned_path.exists() else {}
    tp = tuned.get("tuned_params")
    corpus_tuned = params_from_dict(tp) if tp else None

    wg0 = None
    if tp:
        tp0 = dict(tp)
        tp0["w_graph"] = 0.0
        wg0 = params_from_dict(tp0)

    return {
        "tfidf-canonical": tfidf_canonical,
        "minilm-grid": minilm_grid,
        "corpus-tuned": corpus_tuned,
        "corpus-tuned-wgraph0": wg0,
    }


def build_tasks(bench: str) -> tuple[list, list[dict]]:
    """Load eval docs and the full QA task list with global gold steps."""
    adapter = get_adapter(bench)
    docs = list(adapter.iter_documents(limit=EVAL_DOCS[bench]))
    tasks: list[dict] = []
    gstep = 0
    for doc_idx, doc in enumerate(docs):
        start = gstep
        paragraphs = doc.get("paragraphs", [])
        gstep += len(paragraphs)
        title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for qa_idx, qa in enumerate(doc.get("qa_pairs", []) or []):
            relevant = qa.get("relevant_paragraphs", []) or []
            if not relevant:
                continue
            tasks.append({
                "doc_idx": doc_idx,
                "qa_idx": qa_idx,
                "question": f"[Regarding {title}] {qa['question']}",
                "gold": {start + i for i in relevant if 0 <= i < len(paragraphs)},
                "end_of_doc": gstep,
            })
    return docs, tasks


def ingest(docs: list, params: MemoryParamsV4) -> GraphMemoryV4:
    params.text_mode_entities = True
    memory = GraphMemoryV4(params)
    gstep = 0
    for doc_idx, doc in enumerate(docs):
        title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for para_idx, paragraph in enumerate(doc.get("paragraphs", [])):
            obs = f"[{title}] {paragraph}" if para_idx == 0 else paragraph
            memory.add_event(Event(step=gstep, observation=obs, action="read"),
                             episode_seed=SEED)
            gstep += 1
    return memory


def recall_splits(memory: GraphMemoryV4, tasks: list[dict], tune_limit: int,
                  end_of_corpus: int) -> dict:
    """recall@K split tuned-on/held-out, under online and batch semantics."""
    buckets: dict[str, list[float]] = {
        "tuned_on_online": [], "held_out_online": [],
        "tuned_on_batch": [], "held_out_batch": [],
    }
    for t in tasks:
        tuned_on = (t["doc_idx"] < tune_limit and t["qa_idx"] == 0)
        for sem, step in (("online", t["end_of_doc"]), ("batch", end_of_corpus)):
            retrieved = memory.get_relevant_events(t["question"], current_step=step, k=K)
            hit = 1.0 if ({ev.step for ev in retrieved} & t["gold"]) else 0.0
            buckets[f"{'tuned_on' if tuned_on else 'held_out'}_{sem}"].append(hit)
    out = {}
    for name, xs in buckets.items():
        out[f"{name}_n"] = len(xs)
        out[f"{name}_recall"] = float(np.mean(xs)) if xs else None
    return out


def main() -> int:
    recall_report: dict = {"k": K, "seed": SEED, "eval_docs": EVAL_DOCS,
                           "tune_limit": TUNE_LIMIT, "benchmarks": {}}

    for bench in EVAL_DOCS:
        print(f"\n=== {bench} ===")
        docs, tasks = build_tasks(bench)
        end_of_corpus = sum(len(d.get("paragraphs", [])) for d in docs)
        n_tuned = sum(1 for t in tasks
                      if t["doc_idx"] < TUNE_LIMIT[bench] and t["qa_idx"] == 0)
        print(f"  {len(docs)} docs, {len(tasks)} QAs "
              f"({n_tuned} tuned-on / {len(tasks)-n_tuned} held-out)")

        bench_out: dict = {}
        for label, params in load_thetas(bench).items():
            if params is None:
                bench_out[label] = None
                continue
            t0 = time.time()
            memory = ingest(docs, params)
            res = recall_splits(memory, tasks, TUNE_LIMIT[bench], end_of_corpus)
            res["elapsed_s"] = round(time.time() - t0, 1)
            bench_out[label] = res
            ho = res["held_out_online_recall"]
            hb = res["held_out_batch_recall"]
            print(f"  {label:<22} held-out recall: online="
                  f"{ho if ho is not None else float('nan'):.3f}  "
                  f"batch={hb if hb is not None else float('nan'):.3f}  "
                  f"({res['elapsed_s']}s)")
        recall_report["benchmarks"][bench] = bench_out

    OUT_RECALL.write_text(json.dumps(recall_report, indent=2))
    print(f"\n[eval_heldout_recall] wrote {OUT_RECALL}")

    # ------------------------------------------------------------------
    # Four-shift re-baseline: tuned theta vs BOTH canonical baselines.
    # Direction scoring uses a noise floor: |delta| < 0.05 counts as "~"
    # (no shift), fixing the asymmetric sign-test criticism.
    # ------------------------------------------------------------------
    grid = json.load(open(ROOT / "results" / "graphmemory_v4_cmaes_results.json",
                          encoding="utf-8"))
    baselines = {
        "tfidf-canonical": {n: float(v) for n, v in zip(
            PARAM_NAMES,
            [0.293, 0.908, 0.198, 0.785, 0.285, 0.278, 0.668, 0.0, 1.079, 3.777],
        )},
        "minilm-grid": {n: float(grid["v4"]["best_params"][n]) for n in PARAM_NAMES},
    }
    FOUR = ["w_recency", "w_embed", "theta_store", "w_graph"]
    EXPECT = {"w_recency": "down", "w_embed": "up", "theta_store": "down", "w_graph": "up"}
    NOISE = 0.05

    shift_report: dict = {"noise_floor": NOISE, "expected": EXPECT,
                          "baselines": baselines, "benchmarks": {}}
    for bench in EVAL_DOCS:
        tuned_path = ROOT / "results" / "stage3" / f"tuned_theta_v4t_corpus_{bench}.json"
        if not tuned_path.exists():
            continue
        tuned = json.load(open(tuned_path, encoding="utf-8"))
        tp = tuned.get("tuned_params")
        if not tp:
            shift_report["benchmarks"][bench] = {"status": tuned.get("status", "missing")}
            continue
        per_baseline = {}
        for bname, base in baselines.items():
            shifts = {}
            n_match = 0
            for p in FOUR:
                delta = float(tp[p]) - base[p]
                if abs(delta) < NOISE:
                    direction = "flat"
                else:
                    direction = "up" if delta > 0 else "down"
                match = (direction == EXPECT[p])
                n_match += int(match)
                shifts[p] = {"base": base[p], "tuned": float(tp[p]),
                             "delta": delta, "direction": direction,
                             "matches_expected": match}
            per_baseline[bname] = {"shifts": shifts, "n_match_of_4": n_match}
        shift_report["benchmarks"][bench] = per_baseline

    OUT_SHIFT.write_text(json.dumps(shift_report, indent=2))
    print(f"[eval_heldout_recall] wrote {OUT_SHIFT}")

    print("\nFour-shift score (matches of 4, noise floor 0.05):")
    print(f"{'bench':<14} {'vs tfidf-canonical':>20} {'vs minilm-grid':>16}")
    for bench, pb in shift_report["benchmarks"].items():
        if "status" in pb:
            print(f"{bench:<14} {pb['status']:>20}")
            continue
        print(f"{bench:<14} {pb['tfidf-canonical']['n_match_of_4']:>20} "
              f"{pb['minilm-grid']['n_match_of_4']:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
