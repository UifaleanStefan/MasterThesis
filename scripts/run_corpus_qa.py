"""
Corpus-cumulative QA runner — Stage 3 Phase 2 (2/N) "magnum opus".

Runs LLM answer + judge against ONE persistent V4 memory instance that
has ingested every paragraph of every document in a benchmark.

Two QA modes (can run both in one pass):

* ONLINE (interleaved): for each doc K, ingest its paragraphs, then
  immediately answer its QAs against the CUMULATIVE memory built so
  far (docs 0..K). Produces an "answer-quality vs. corpus-seen-so-
  far" curve. The retrieval `current_step` is the global step at
  end-of-doc-K, so recency favors recent paragraphs (mostly the
  doc just ingested). Temporally analogous to "answer this question
  now, against everything I've read up to this point."

* BATCH (end-of-corpus): ingest ALL docs first, then loop over every
  QA and answer it against the FINAL memory. Retrieval
  `current_step` is the total step count at end-of-corpus, so
  recency uniformly favors the LAST documents ingested (regardless
  of which doc the question is about). Analogous to "build a
  knowledge base, then ask any question at deployment time."

Both modes share the same V4 instance during ingestion; they differ
only in WHEN the QA is run. Costs roughly double when running both
(each QA gets answered + judged twice).

Output: results/stage3/corpus_traces/{benchmark}__{config}/
  qa_online.json   — per-doc online QA results (skipped if --no-online)
  qa_batch.json    — end-of-corpus batch QA results (skipped if --no-batch)
  qa_summary.json  — aggregate: per-mode mean judge, online vs. batch
                     correlation, costs.

Cost gating: --limit-docs N caps both ingestion and QA to the first N
documents. Useful for Tier 1 smoke tests before committing to a
full-corpus run.

Usage:
    # Tier 1 smoke (~$0.10): FinanceBench full
    python scripts/run_corpus_qa.py --benchmark financebench --config v4-tuned

    # Tier 2: QASPER full (~$1.50 single config)
    python scripts/run_corpus_qa.py --benchmark qasper --config v4-tuned

    # Limit to first 50 docs of CUAD for iteration
    python scripts/run_corpus_qa.py --benchmark cuad --config v4-tuned --limit-docs 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm_agent import LLMAgent
from environment.benchmarks import ADAPTERS, get_adapter
from environment.benchmarks.base import document_fingerprint
from evaluation.document_qa_llm_judge import llm_judge_score_multi_ref
from evaluation.claude_judge_queue import write_judge_queue
from memory.attention_memory import AttentionMemory
from memory.bm25_memory import BM25Memory
from memory.dump_all_memory import DumpAllMemory
from memory.event import Event
from memory.flat_memory import FlatMemory
from memory.graph_memory import GraphMemory, MemoryParams as MemoryParamsV1
from memory.graph_memory_v3 import GraphMemoryV3, MemoryParamsV3
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from memory.graph_memory_v5 import GraphMemoryV5
from memory.rag_memory import RAGMemory
from memory.semantic_memory import SemanticMemory
from results.manifest import build_manifest
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC, vec_to_params

OUT_ROOT = ROOT / "results" / "stage3" / "corpus_traces"

# Mirrored from run_corpus_ingestion.py (Block B / critique #2).
DOMAIN_COHERENT = {"cuad", "qasper", "financebench"}
DOMAIN_INCOHERENT = {"hotpotqa", "narrativeqa", "longmemeval"}


def benchmark_category(benchmark: str) -> str:
    if benchmark in DOMAIN_COHERENT:
        return "domain_coherent"
    if benchmark in DOMAIN_INCOHERENT:
        return "domain_incoherent_control"
    return "uncategorized"


def load_config_params(config_name: str, benchmark: str) -> MemoryParamsV4:
    """V4ₜ ("V4-text") variant params. Same θ as V4 but with
    text_mode_entities=True. Used by v4t-canonical and v4t-tuned configs.
    Critique #7: variant naming must be explicit.

    For corpus-mode-tuned θ (Block B / critique #6), pass
    config_name='v4t-corpus-tuned' which loads from
    tuned_theta_v4t_corpus_<benchmark>.json.
    """
    if config_name in ("v4-canonical", "v4-tuned"):
        config_name = config_name.replace("v4-", "v4t-")

    if config_name == "v4t-canonical":
        params = vec_to_params(CANONICAL_THETA_VEC)
    elif config_name == "v4t-tuned":
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
    elif config_name == "v4t-corpus-tuned":
        # Block B / critique #6: θ tuned UNDER the corpus-cumulative regime.
        path = ROOT / "results" / "stage3" / f"tuned_theta_v4t_corpus_{benchmark}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No corpus-mode tuned θ for {benchmark!r}. "
                f"Run `python -m tuning.tune_v4t_corpus --benchmarks {benchmark}` first."
            )
        data = json.loads(path.read_text())
        vec = data.get("tuned_theta_vec")
        if not isinstance(vec, list) or len(vec) != 10:
            raise ValueError(f"Malformed corpus-tuned θ for {benchmark!r}")
        params = vec_to_params(np.asarray(vec, dtype=np.float64))
    else:
        raise ValueError(
            f"Unknown config: {config_name!r}. "
            f"Valid V4ₜ configs: v4t-canonical, v4t-tuned, v4t-corpus-tuned."
        )
    params.text_mode_entities = True
    return params


def _load_tuned_attention_temp(benchmark: str) -> float:
    """Load the tuned AttentionMemory temperature τ.

    Prefers the corpus-mode τ (tuned on the same corpus-cumulative recall@k
    objective as V4ₜ — fairness, audit B6; tuning/tune_attention_corpus.py),
    then falls back to the older per-doc τ (Phase 1.7), then 0.5."""
    for fname in (
        f"tuned_attention_corpus_{benchmark}.json",
        f"tuned_temperature_{benchmark}.json",
    ):
        path = ROOT / "results" / "stage3" / fname
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return float(data.get("tuned_temperature", 0.5))
            except Exception:
                continue
    print(f"  [WARN] no tuned tau for {benchmark!r}; using default 0.5")
    return 0.5


def build_memory(config_name: str, benchmark: str) -> Any:
    """Memory factory dispatch for the 12-config corpus suite.

    Configs (V4ₜ variants use text_mode_entities=True):

      Graph family (V4t = V4 + text_mode_entities):
        v4t-canonical       — grid-world θ
        v4t-tuned           — per-doc Phase 1.5/1.6 CMA-ES θ
        v4t-corpus-tuned    — corpus-mode CMA-ES θ (Block B)  ← HEADLINE
        v5t-corpus          — V5 (V4+attention gating) + text-mode entities
        v1-corpus           — GraphMemory V1 (3-D θ baseline)
        v3-corpus           — GraphMemory V3 (V2 + importance-scored storage)

      Attention family:
        attention-corpus        — AttentionMemory(τ=0.5) default
        attention-corpus-tuned  — AttentionMemory with per-bench tuned τ
                                  (Phase 1.7)

      Retrieval baselines:
        rag-corpus       — RAGMemory (dense MiniLM cosine)
        semantic-corpus  — SemanticMemory (TF-IDF cosine)
        bm25-corpus      — BM25Memory (Okapi BM25 sparse)
        flat-corpus      — FlatMemory(window=50) baseline
        dump-all         — no-selection upper bound (small corpora only)

    Returns: (memory_instance, params_or_None).
    """
    # Legacy alias normalization
    if config_name in ("v4-canonical", "v4-tuned"):
        config_name = config_name.replace("v4-", "v4t-")

    # V4ₜ family
    if config_name in ("v4t-canonical", "v4t-tuned", "v4t-corpus-tuned"):
        params = load_config_params(config_name, benchmark)
        return GraphMemoryV4(params), params

    # V5ₜ — V4 with attention gating, also runs in text-mode-entities path
    if config_name == "v5t-corpus":
        # Default to canonical θ; corpus-tuning V5 specifically is future work.
        params = vec_to_params(CANONICAL_THETA_VEC)
        params.text_mode_entities = True
        return GraphMemoryV5(params), params

    # V1 (3D θ baseline). MemoryParamsV1 has only theta_store, theta_entity,
    # theta_temporal — the original POC params. Default (1.0, 0.0, 1.0)
    # reproduces unparameterized behavior.
    if config_name == "v1-corpus":
        return GraphMemory(MemoryParamsV1()), None

    # V3 (V2 + importance-scored storage)
    if config_name == "v3-corpus":
        return GraphMemoryV3(MemoryParamsV3()), None

    # Attention family
    if config_name == "attention-corpus":
        return AttentionMemory(temperature=0.5), None
    if config_name == "attention-corpus-tuned":
        tau = _load_tuned_attention_temp(benchmark)
        return AttentionMemory(temperature=tau), None

    # Retrieval baselines
    if config_name == "rag-corpus":
        return RAGMemory(), None
    if config_name == "semantic-corpus":
        return SemanticMemory(), None
    if config_name == "bm25-corpus":
        return BM25Memory(), None
    if config_name == "bm25-corpus-tuned":
        # Fairness (audit B6): BM25 with (k1, b) tuned on the same
        # corpus-cumulative recall@k objective as V4ₜ. See
        # tuning/tune_bm25_corpus.py.
        path = ROOT / "results" / "stage3" / f"tuned_bm25_corpus_{benchmark}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No corpus-tuned BM25 for {benchmark!r}. "
                f"Run `python -m tuning.tune_bm25_corpus --benchmarks {benchmark}` first."
            )
        data = json.loads(path.read_text())
        return BM25Memory(k1=float(data["k1"]), b=float(data["b"])), None
    if config_name == "flat-corpus":
        return FlatMemory(window_size=50), None
    if config_name == "dump-all":
        return DumpAllMemory(), None
    # Published-system head-to-heads (Phase 3): faithful reimplementations behind
    # the same interface, so the comparison is retrieval-only under the identical
    # gpt-4o-mini answerer + Claude judge (see memory/hipporag_memory.py,
    # memory/letta_memory.py for the disclosed divergences).
    if config_name == "hipporag-corpus":
        from memory.hipporag_memory import HippoRAGMemory
        return HippoRAGMemory(), None
    if config_name == "letta-corpus":
        from memory.letta_memory import LettaMemory
        return LettaMemory(), None

    raise ValueError(
        f"Unknown config: {config_name!r}. Valid 12-config suite: "
        f"v4t-canonical, v4t-tuned, v4t-corpus-tuned, v5t-corpus, "
        f"v1-corpus, v3-corpus, attention-corpus, attention-corpus-tuned, "
        f"rag-corpus, semantic-corpus, bm25-corpus, flat-corpus, dump-all."
    )

# gpt-4o-mini pricing (per 1M tokens; sync'd with run_stage3_full.py).
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
MAX_ANSWER_TOKENS = 150


# ----------------------------------------------------------------------
# Prompt + scoring helpers (mirror run_stage3_full.py for comparability)
# ----------------------------------------------------------------------


_QA_SYSTEM_PROMPT = (
    "You are a question-answering assistant. You receive a question and relevant document passages. "
    "Answer the question concisely using only the provided context. "
    "Output only the answer text, no preamble or explanation."
)


def _try_tiktoken(model: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def _make_token_counter(model: str):
    enc = _try_tiktoken(model)
    if enc is not None:
        def _count(s: str) -> int:
            try:
                return len(enc.encode(s))
            except Exception:
                return max(1, len(s) // 4)
        return _count
    # Fallback: ~4 chars per token
    return lambda s: max(1, len(s) // 4)


def _build_qa_prompt(question: str, retrieved_events) -> tuple[str, str]:
    context_lines = []
    for ev in retrieved_events:
        context_lines.append(f"step {ev.step}: {ev.observation}")
    memory_context = "\n".join(context_lines)
    user_msg = (
        f"Relevant passages:\n{memory_context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return _QA_SYSTEM_PROMPT, user_msg


def _scope_question(question: str, doc_title: str) -> str:
    """Block A.2 / critique #5: prepend the document title to the question
    so corpus-mode retrieval has an explicit doc-scope signal.

    Per-doc framing implicitly meant "in this contract / paper / filing."
    Corpus mode breaks that — the same question gets matched against
    every other doc's paragraphs. Prepending the doc title restores
    the doc-scope signal as if it were part of the question text.
    """
    if not doc_title:
        return question
    # Strip the "[" "]" if title already has them (shouldn't, but safe).
    t = doc_title.strip("[]").strip()
    return f"[Regarding {t}] {question}"


# ----------------------------------------------------------------------
# Online + batch QA loops
# ----------------------------------------------------------------------


def answer_one_qa(
    agent: LLMAgent,
    memory: GraphMemoryV4,
    question: str,
    gold_answer,
    doc_title: str,
    relevant_global_steps: set[int],
    current_step: int,
    k: int,
    model: str,
    token_count,
    skip_judge: bool = False,
    uncapped_context: bool = False,
) -> dict:
    """Run one Q against the current memory state.

    Block A.2 (#5): the QUESTION used for retrieval and LLM generation is
    prepended with the doc title so corpus-mode questions retain the
    doc-scope context that the per-doc framing implicitly assumed.

    Block A.1 (#3): recall@k uses GLOBAL step indices for relevant
    paragraphs (caller maps doc-local indices via doc_start_step before
    calling here). Recall is 1.0 iff retrieved_steps intersects
    relevant_global_steps.

    `skip_judge=True` (Block E): skip the inline LLM judge call and
    write {predicted, gold} downstream to a queue for Claude-in-the-loop
    judging. The returned dict has judge_score=None when skipped.

    `uncapped_context=True` (dump-all fix): send ALL retrieved events to
    the prompt instead of the ContextFormatter's default 12-event cap.
    Questions whose uncapped prompt would exceed the model's context
    window are skipped before the API call and flagged
    `context_overflow=True` — they must not enter judge queues.
    """
    scoped_question = _scope_question(question, doc_title)
    retrieved = memory.get_relevant_events(scoped_question, current_step=current_step, k=k)
    retrieved_steps = [ev.step for ev in retrieved]
    sys_prompt, user_msg = _build_qa_prompt(scoped_question, retrieved)
    prompt_tokens_estimated = token_count(sys_prompt) + token_count(user_msg)

    # Block A.1: recall@k via global-step intersection. For uncapped
    # (dump-all) prompts this now reflects exactly what the model sees;
    # for capped configs retrieval is already bounded by k < 12.
    if relevant_global_steps:
        recall_at_k = 1.0 if (set(retrieved_steps) & relevant_global_steps) else 0.0
    else:
        recall_at_k = None

    base = {
        "question": question,                        # original (un-scoped) text
        "scoped_question": scoped_question,          # what was actually sent
        "gold_answer": gold_answer,
        "recall_at_k": recall_at_k,
        "retrieved_steps": retrieved_steps,
        "relevant_global_steps": sorted(relevant_global_steps) if relevant_global_steps else [],
        "current_step": current_step,
        "k": k,
        "prompt_tokens_estimated": prompt_tokens_estimated,
    }

    # 128K window minus completion budget and tokenizer-estimate slack.
    if uncapped_context and prompt_tokens_estimated > 124_000:
        base.update({
            "predicted": f"[CONTEXT_OVERFLOW: ~{prompt_tokens_estimated} prompt tokens "
                         f"exceed the 128K window — question not sent]",
            "judge_score": None,
            "context_overflow": True,
        })
        return base

    predicted = agent.answer_question(
        scoped_question, past_events=retrieved,
        max_events=None if uncapped_context else 12,
    )
    meta = getattr(agent, "last_answer_meta", {}) or {}

    if skip_judge:
        judge_score = None
    else:
        judge_score = llm_judge_score_multi_ref(predicted, gold_answer, model=model)

    base.update({
        "predicted": predicted,
        "judge_score": judge_score,
        "context_overflow": False,
        "answer_fallback": bool(meta.get("fallback")),
        "prompt_tokens_actual": meta.get("prompt_tokens"),
        "response_model": meta.get("response_model"),
    })
    return base


def _build_all_qa_inventory(docs: list) -> list[dict]:
    """Pre-walk all docs to assemble the full (doc_idx, qa_idx, q, gold) inventory
    BEFORE ingestion. Required for Protocol B (calibration) so the random sampler
    at each doc-end can pick from ALL future questions, including ones whose
    source docs have not been ingested yet.

    Each entry includes the doc_start_step (cumulative global step at which the
    doc's paragraphs begin) so retrieval recall@k works the same as in the
    incremental inventory in run_corpus_qa.
    """
    inventory = []
    cumulative_step = 0
    for doc_idx, doc in enumerate(docs):
        paragraphs = doc.get("paragraphs", []) or []
        qa_pairs = doc.get("qa_pairs", []) or []
        doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for qa_idx, qa in enumerate(qa_pairs):
            local_relevant = qa.get("relevant_paragraphs", []) or []
            global_relevant = {cumulative_step + i for i in local_relevant
                               if 0 <= i < len(paragraphs)}
            inventory.append({
                "doc_idx": doc_idx,
                "doc_title": doc_title,
                "qa_idx_in_doc": qa_idx,
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "relevant_global_steps": sorted(global_relevant),
                "doc_start_step": cumulative_step,
                "n_paragraphs": len(paragraphs),
            })
        cumulative_step += len(paragraphs)
    return inventory


def run_corpus_qa(
    benchmark: str,
    config: str,
    limit_docs: int | None,
    k: int,
    seed: int,
    model: str,
    do_online: bool,
    do_batch: bool,
    out_dir: Path,
    progress_every_docs: int = 25,
    w_recency_zero: bool = False,
    protocol: str = "online_batch",
    questions_per_doc: int = 10,
) -> dict:
    """Corpus-cumulative QA driver.

    Two protocols supported (controlled by the `protocol` arg):

    * ``protocol="online_batch"`` (default, the original behaviour) — runs
      Protocol A "memory retention" experiment:
        - ONLINE: after ingesting doc K, ask doc K's questions against the
          cumulative memory.
        - BATCH: after all docs ingested, ask every question once more
          against the final memory.

    * ``protocol="calibration"`` (Phase 1.9 Protocol B "honesty" experiment)
      — at each doc-end, sample ``questions_per_doc`` random questions from
      the **full** corpus question pool (mix of already-ingested-source-doc and
      not-yet-ingested-source-doc), then ask them. Each entry is tagged with
      ``expected_behavior``:
        - "answer" if the source doc has already been ingested (model should
          answer correctly);
        - "acknowledge_missing" if the source doc has NOT been ingested yet
          (model should admit it doesn't know).
      Plus the same end-of-corpus batch as protocol="online_batch" (writes
      ``qa_batch.json`` with the standard schema; predictions are
      byte-identical to a matched protocol="online_batch" run, so the dedupe
      transfer downstream can save Claude judging effort).
    """
    if protocol not in {"online_batch", "calibration"}:
        raise ValueError(f"unknown protocol={protocol!r}; choose 'online_batch' or 'calibration'")
    is_calibration = (protocol == "calibration")
    # Dump-all fix: the no-selection baseline must send EVERY stored event
    # to the prompt (the ContextFormatter's 12-event default previously
    # truncated it silently). Per-question context-window overflow is
    # guarded inside answer_one_qa.
    uncapped = (config == "dump-all")
    # In calibration mode, the "online" per-doc questions are replaced by
    # the random-sample calibration questions, so `do_online` (per-doc QA
    # of the just-ingested doc's questions) is force-off — the calibration
    # loop covers that slot.
    if is_calibration:
        do_online = False

    category = benchmark_category(benchmark)
    print(f"\n{'=' * 78}")
    print(f"  CORPUS QA  benchmark={benchmark} ({category})")
    print(f"  config={config}  seed={seed}  protocol={protocol}")
    print(f"  online={do_online}  batch={do_batch}  k={k}  limit_docs={limit_docs or 'ALL'}")
    if is_calibration:
        print(f"  calibration: {questions_per_doc} random q's per doc-end")
    if w_recency_zero:
        print(f"  w_recency_zero=True (Block B / critique #4)")
    print(f"{'=' * 78}")

    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=limit_docs))
    total_docs = len(docs)
    print(f"  Loaded {total_docs} docs.")

    # Calibration mode (Protocol B): pre-build the full question inventory so
    # the per-doc-end random sampler can draw from questions whose source docs
    # have not yet been ingested.
    all_qa_inventory: list[dict] = []
    if is_calibration:
        all_qa_inventory = _build_all_qa_inventory(docs)
        print(f"  Calibration: pre-built inventory of {len(all_qa_inventory)} questions "
              f"across {total_docs} docs (sampler will draw {questions_per_doc} per doc-end).")

    memory, params = build_memory(config, benchmark)
    if params is not None and w_recency_zero:
        params.w_recency = 0.0
        # Rebuild memory so retrieval uses the overridden w_recency.
        memory = GraphMemoryV4(params)
    if params is not None:
        print(f"  V4t params: "
              f"theta_store={params.theta_store:.3f} theta_entity={params.theta_entity:.3f} "
              f"w_embed={params.w_embed:.3f} w_recency={params.w_recency:.3f}")
    else:
        print(f"  Memory: {type(memory).__name__} (no tunable params; pure sparse retrieval)")

    agent = LLMAgent(model=model, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS, seed=seed)
    token_count = _make_token_counter(model)

    online_results: list[dict] = []
    calibration_results: list[dict] = []
    qa_inventory: list[dict] = []  # (doc_idx, qa_idx_in_doc, question, gold)

    global_step = 0
    cumulative_stored = 0
    t_start = time.time()

    def _node_count_o1(m) -> int:
        """Memory-agnostic O(1) graph-size probe. Used to detect whether
        add_event accepted (graph grew) or rejected (no change). V4's
        graph also tracks entity nodes, so this counts events+entities."""
        if hasattr(m, "_graph"):
            return m._graph.number_of_nodes()  # O(1) in NetworkX DiGraph
        if hasattr(m, "_events"):
            return len(m._events)
        return m.get_stats().get("n_nodes", 0)

    # Phase A: ingest doc K, then optionally answer its QAs (online mode).
    for doc_idx, doc in enumerate(docs):
        doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        paragraphs = doc.get("paragraphs", [])
        doc_start_step = global_step

        for para_idx, paragraph in enumerate(paragraphs):
            obs = f"[{doc_title}] {paragraph}" if para_idx == 0 else paragraph
            event = Event(step=global_step, observation=obs, action="read")
            n_before = _node_count_o1(memory)
            memory.add_event(event, episode_seed=seed)
            if _node_count_o1(memory) > n_before:
                cumulative_stored += 1
            global_step += 1

        # Inventory this doc's QAs.
        # Block A.1 (#3): map doc-local relevant_paragraphs to GLOBAL
        # step indices. The mapping is direct since the ingestion loop
        # increments global_step once per paragraph regardless of
        # whether the storage gate accepted the event.
        qa_pairs = doc.get("qa_pairs", []) or []
        for qa_idx, qa in enumerate(qa_pairs):
            local_relevant = qa.get("relevant_paragraphs", []) or []
            global_relevant = {doc_start_step + i for i in local_relevant
                               if 0 <= i < len(paragraphs)}
            qa_inventory.append({
                "doc_idx": doc_idx,
                "doc_title": doc_title,
                "doc_fingerprint": document_fingerprint(doc),
                "qa_idx_in_doc": qa_idx,
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "relevant_paragraphs_local": local_relevant,
                "relevant_global_steps": sorted(global_relevant),
                "n_paragraphs": len(paragraphs),
                "doc_start_step": doc_start_step,
                "doc_end_step": global_step - 1,
            })

        if do_online and qa_pairs:
            for qa_idx, qa in enumerate(qa_pairs):
                local_relevant = qa.get("relevant_paragraphs", []) or []
                global_relevant = {doc_start_step + i for i in local_relevant
                                   if 0 <= i < len(paragraphs)}
                r = answer_one_qa(
                    agent, memory, qa["question"], qa["answer"],
                    doc_title=doc_title,
                    relevant_global_steps=global_relevant,
                    current_step=global_step,  # cumulative end-of-doc-K
                    k=k, model=model, token_count=token_count,
                    skip_judge=True,  # Block E: write to judge queue downstream
                    uncapped_context=uncapped,
                )
                r["doc_idx"] = doc_idx
                r["qa_idx_in_doc"] = qa_idx
                r["mode"] = "online"
                r["docs_seen"] = doc_idx + 1
                online_results.append(r)

        # Calibration mode (Protocol B): after ingesting doc K, sample
        # `questions_per_doc` random questions from the FULL corpus pool
        # and ask each one. Each entry is tagged with `expected_behavior`
        # = "answer" if source doc has been ingested (source_doc_idx <= K)
        # or "acknowledge_missing" if not (source_doc_idx > K). The model
        # uses the same retrieval + LLM stack as protocol="online_batch";
        # only the question selection differs.
        if is_calibration and all_qa_inventory:
            # Deterministic per-(seed, doc_idx) sampling: same seed → same
            # sampled qids per doc-end, across configs. This lets us pair
            # questions across configs for paired analysis.
            rng = random.Random(seed * 31 + doc_idx)
            n_to_sample = min(questions_per_doc, len(all_qa_inventory))
            sampled_indices = rng.sample(range(len(all_qa_inventory)), n_to_sample)
            for q_idx in sampled_indices:
                qi = all_qa_inventory[q_idx]
                src_doc_idx = qi["doc_idx"]
                expected_behavior = (
                    "answer" if src_doc_idx <= doc_idx else "acknowledge_missing"
                )
                r = answer_one_qa(
                    agent, memory, qi["question"], qi["gold_answer"],
                    doc_title=qi["doc_title"],
                    relevant_global_steps=set(qi["relevant_global_steps"]),
                    current_step=global_step,  # cumulative end-of-doc-K
                    k=k, model=model, token_count=token_count,
                    skip_judge=True,
                    uncapped_context=uncapped,
                )
                r["doc_idx"] = src_doc_idx
                r["qa_idx_in_doc"] = qi["qa_idx_in_doc"]
                r["mode"] = "calibration"
                r["docs_seen"] = doc_idx + 1
                r["expected_behavior"] = expected_behavior
                r["source_doc_idx"] = src_doc_idx
                r["asked_after_doc_idx"] = doc_idx
                calibration_results.append(r)

        if (doc_idx + 1) % progress_every_docs == 0 or doc_idx == total_docs - 1:
            elapsed = time.time() - t_start
            cost = agent.session_cost_usd
            print(
                f"  doc {doc_idx+1:>5}/{total_docs:<5}  "
                f"global_step={global_step:>7}  "
                f"events_stored={cumulative_stored:>6}  "
                f"online_qa={len(online_results):>5}  "
                f"calib_qa={len(calibration_results):>5}  "
                f"cost=${cost:>6.4f}  elapsed={elapsed:>6.1f}s"
            )

    online_elapsed = time.time() - t_start
    online_cost = agent.session_cost_usd

    # Phase B: with FULLY-BUILT memory, run end-of-corpus batch QA.
    batch_results: list[dict] = []
    batch_t0 = time.time()
    if do_batch:
        print(f"\n  Running batch QA on {len(qa_inventory)} questions against final memory...")
        end_of_corpus_step = global_step
        for i, qi in enumerate(qa_inventory):
            r = answer_one_qa(
                agent, memory, qi["question"], qi["gold_answer"],
                doc_title=qi["doc_title"],
                relevant_global_steps=set(qi["relevant_global_steps"]),
                current_step=end_of_corpus_step,
                k=k, model=model, token_count=token_count,
                skip_judge=True,
                uncapped_context=uncapped,
            )
            r["doc_idx"] = qi["doc_idx"]
            r["qa_idx_in_doc"] = qi["qa_idx_in_doc"]
            r["mode"] = "batch"
            r["docs_seen"] = total_docs
            batch_results.append(r)
            if (i + 1) % 100 == 0 or i == len(qa_inventory) - 1:
                elapsed = time.time() - batch_t0
                cost = agent.session_cost_usd
                print(
                    f"  batch {i+1:>5}/{len(qa_inventory):<5}  "
                    f"cost=${cost:>6.4f}  elapsed={elapsed:>6.1f}s"
                )
    batch_elapsed = time.time() - batch_t0
    total_cost = agent.session_cost_usd

    # Aggregate. Note judge_score is None when skip_judge=True (Block E:
    # downstream Claude-in-the-loop judging). We still aggregate recall@k
    # since that's computable from the retrieval result alone.
    def _mean(values: list, key: str) -> float | None:
        xs = [v[key] for v in values if v.get(key) is not None]
        return float(np.mean(xs)) if xs else None

    def _n_overflow(values: list) -> int:
        return sum(1 for v in values if v.get("context_overflow"))

    def _n_fallback(values: list) -> int:
        return sum(1 for v in values if v.get("answer_fallback"))

    summary = {
        "_manifest": build_manifest(seed=seed, extra={
            "experiment": "stage3_corpus_qa",
            "benchmark": benchmark,
            "config": config,
            "model": model,
            "protocol": protocol,
        }),
        "benchmark": benchmark,
        "config": config,
        "model": model,
        "seed": seed,
        "k": k,
        "n_docs": total_docs,
        "limit_docs": limit_docs,
        "n_qa_total": len(qa_inventory),
        "protocol": protocol,
        "do_online": do_online,
        "do_batch": do_batch,
        "online": {
            "n": len(online_results),
            "mean_judge": _mean(online_results, "judge_score"),
            "mean_recall_at_k": _mean(online_results, "recall_at_k"),
            "n_with_judge": sum(1 for r in online_results if r.get("judge_score") is not None),
            "n_with_recall": sum(1 for r in online_results if r.get("recall_at_k") is not None),
            "n_context_overflow": _n_overflow(online_results),
            "n_answer_fallback": _n_fallback(online_results),
            "mean_prompt_tokens_actual": _mean(online_results, "prompt_tokens_actual"),
            "elapsed_seconds": online_elapsed,
        } if do_online else None,
        "batch": {
            "n": len(batch_results),
            "mean_judge": _mean(batch_results, "judge_score"),
            "mean_recall_at_k": _mean(batch_results, "recall_at_k"),
            "n_with_judge": sum(1 for r in batch_results if r.get("judge_score") is not None),
            "n_with_recall": sum(1 for r in batch_results if r.get("recall_at_k") is not None),
            "n_context_overflow": _n_overflow(batch_results),
            "n_answer_fallback": _n_fallback(batch_results),
            "mean_prompt_tokens_actual": _mean(batch_results, "prompt_tokens_actual"),
            "elapsed_seconds": batch_elapsed,
        } if do_batch else None,
        "uncapped_context": uncapped,
        "total_cost_usd": total_cost,
    }

    # Calibration (Protocol B) summary, only when protocol="calibration".
    if is_calibration:
        n_answer = sum(1 for r in calibration_results
                       if r.get("expected_behavior") == "answer")
        n_ack = sum(1 for r in calibration_results
                    if r.get("expected_behavior") == "acknowledge_missing")
        summary["protocol"] = "calibration"
        summary["questions_per_doc"] = questions_per_doc
        summary["calibration"] = {
            "n": len(calibration_results),
            "n_expected_answer": n_answer,
            "n_expected_acknowledge_missing": n_ack,
            "mean_recall_at_k": _mean(calibration_results, "recall_at_k"),
            "elapsed_seconds": online_elapsed,
            "n_with_judge": sum(1 for r in calibration_results
                                if r.get("judge_score") is not None),
        }

    # Online vs batch recall agreement (when both modes ran). Judge
    # agreement is computed downstream once Claude scores both queues.
    if do_online and do_batch and online_results and batch_results:
        online_map = {(r["doc_idx"], r["qa_idx_in_doc"]): r.get("recall_at_k")
                      for r in online_results}
        batch_map = {(r["doc_idx"], r["qa_idx_in_doc"]): r.get("recall_at_k")
                     for r in batch_results}
        paired = [(online_map[k], batch_map[k]) for k in online_map
                  if k in batch_map and online_map[k] is not None and batch_map[k] is not None]
        if paired:
            online_arr = np.array([p[0] for p in paired])
            batch_arr = np.array([p[1] for p in paired])
            summary["online_vs_batch_recall"] = {
                "n_paired": len(paired),
                "online_recall_mean": float(online_arr.mean()),
                "batch_recall_mean": float(batch_arr.mean()),
                "diff_mean": float((batch_arr - online_arr).mean()),
                "diff_std": float((batch_arr - online_arr).std(ddof=1)) if len(paired) > 1 else 0.0,
                "agreement_rate": float((online_arr == batch_arr).mean()),
            }

    # Block A: assign canonical qid to each record so Claude-in-the-loop
    # judge results merge back correctly.
    def _qid(rec: dict, mode_label: str) -> str:
        return (
            f"{benchmark}__{config}__{mode_label}__"
            f"doc{rec['doc_idx']}_qa{rec['qa_idx_in_doc']}__seed{seed}"
        )

    def _qid_calib(rec: dict) -> str:
        """Calibration qid disambiguates by `asked_after_doc_idx` because
        the same (source_doc_idx, qa_idx) can be sampled at multiple
        doc-end checkpoints. Without `asked_after`, qids would collide."""
        return (
            f"{benchmark}__{config}__calibration__"
            f"doc{rec['doc_idx']}_qa{rec['qa_idx_in_doc']}__"
            f"after{rec['asked_after_doc_idx']}__seed{seed}"
        )

    for rec in online_results:
        rec["qid"] = _qid(rec, "online")
    for rec in batch_results:
        rec["qid"] = _qid(rec, "batch")
    for rec in calibration_results:
        rec["qid"] = _qid_calib(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    if do_online:
        (out_dir / "qa_online.json").write_text(json.dumps(online_results, indent=2, default=str))
    if do_batch:
        (out_dir / "qa_batch.json").write_text(json.dumps(batch_results, indent=2, default=str))
    if is_calibration:
        (out_dir / "qa_calibration.json").write_text(
            json.dumps(calibration_results, indent=2, default=str))
    (out_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Block E: emit Claude-in-the-loop judge queues (one per mode).
    # Each entry has just what the judge needs: question, gold, predicted.
    # The merger downstream looks up by qid to splice judge_score back.
    # Calibration entries carry `expected_behavior` + `source_doc_idx` +
    # `asked_after_doc_idx` so the judge applies the calibration rubric
    # in `evaluation/claude_judge_protocol.md`.
    def _queue_entry(rec: dict) -> dict:
        entry = {
            "qid": rec["qid"],
            "benchmark": benchmark,
            "config": config,
            "mode": rec.get("mode"),
            "doc_idx": rec.get("doc_idx"),
            "qa_idx_in_doc": rec.get("qa_idx_in_doc"),
            "docs_seen": rec.get("docs_seen"),
            "question": rec["question"],
            "scoped_question": rec.get("scoped_question"),
            "gold_answer": rec["gold_answer"],
            "predicted": rec["predicted"],
            "retrieved_steps": rec.get("retrieved_steps"),
            "k": rec.get("k"),
        }
        # Calibration entries carry extra fields the judge needs.
        if rec.get("mode") == "calibration":
            entry["expected_behavior"] = rec.get("expected_behavior")
            entry["source_doc_idx"] = rec.get("source_doc_idx")
            entry["asked_after_doc_idx"] = rec.get("asked_after_doc_idx")
        return entry

    # Entries that never reached the model (context overflow) or whose
    # answer came from the no-API heuristic fallback must NOT be judged
    # as model output — exclude them from the queues; counts are in the
    # summary so the chapter can report them as infeasible.
    def _judgeable(r: dict) -> bool:
        return not r.get("context_overflow") and not r.get("answer_fallback")

    if do_online and online_results:
        write_judge_queue(
            f"{benchmark}__{config}__online__seed{seed}",
            (_queue_entry(r) for r in online_results if _judgeable(r)),
        )
    if do_batch and batch_results:
        # Calibration mode's batch component goes to a separate "__batch_calib"
        # path so it doesn't overwrite Protocol A's batch queue (the same
        # corpus seed=42 setting produces near-identical predictions, but
        # OpenAI's approximate determinism means Protocol A judgments would
        # become slightly stale if overwritten).
        batch_suffix = "__batch_calib" if is_calibration else "__batch"
        write_judge_queue(
            f"{benchmark}__{config}{batch_suffix}__seed{seed}",
            (_queue_entry(r) for r in batch_results if _judgeable(r)),
        )
    if is_calibration and calibration_results:
        write_judge_queue(
            f"{benchmark}__{config}__calibration__seed{seed}",
            (_queue_entry(r) for r in calibration_results if _judgeable(r)),
        )

    print(f"\n  Saved qa_online.json / qa_batch.json / qa_summary.json to {out_dir}")
    print(f"  Total cost: ${total_cost:.4f}")
    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
    if summary.get("online"):
        mj = summary["online"].get("mean_judge")
        mr = summary["online"].get("mean_recall_at_k")
        n = summary["online"]["n"]
        judge_status = _fmt(mj) if mj is not None else "pending Claude judging"
        print(f"  Online: n={n}  recall@k={_fmt(mr)}  judge={judge_status}")
    if summary.get("batch"):
        mj = summary["batch"].get("mean_judge")
        mr = summary["batch"].get("mean_recall_at_k")
        n = summary["batch"]["n"]
        judge_status = _fmt(mj) if mj is not None else "pending Claude judging"
        print(f"  Batch:  n={n}  recall@k={_fmt(mr)}  judge={judge_status}")
    if summary.get("calibration"):
        cal = summary["calibration"]
        n = cal["n"]
        n_ans = cal["n_expected_answer"]
        n_ack = cal["n_expected_acknowledge_missing"]
        mr = cal.get("mean_recall_at_k")
        print(f"  Calibration: n={n}  expected_answer={n_ans}  expected_acknowledge_missing={n_ack}  "
              f"recall@k={_fmt(mr)}  judge=pending Claude judging")
    if "online_vs_batch_recall" in summary:
        ovb = summary["online_vs_batch_recall"]
        print(f"  Online vs batch recall: diff={ovb['diff_mean']:+.4f}  agreement={ovb['agreement_rate']:.3f}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(ADAPTERS.keys()))
    parser.add_argument(
        "--config", default="v4t-tuned",
        choices=[
            # Graph family
            "v4t-canonical", "v4t-tuned", "v4t-corpus-tuned",
            "v5t-corpus", "v1-corpus", "v3-corpus",
            # Attention family
            "attention-corpus", "attention-corpus-tuned",
            # Retrieval baselines
            "rag-corpus", "semantic-corpus", "bm25-corpus",
            "bm25-corpus-tuned",
            "flat-corpus", "dump-all",
            # Published-system head-to-heads (faithful reimplementations)
            "hipporag-corpus", "letta-corpus",
            # Legacy aliases
            "v4-tuned", "v4-canonical",
        ],
        help="12-config corpus suite + dump-all special. See build_memory() docstring.",
    )
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-online", action="store_true", help="Skip online (per-doc) QA")
    parser.add_argument("--no-batch", action="store_true", help="Skip batch (end-of-corpus) QA")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--progress-every-docs", type=int, default=25)
    parser.add_argument(
        "--w-recency-zero", action="store_true",
        help="Block B / critique #4: override w_recency=0 to eliminate "
             "recency-bias confound when comparing online vs batch retrieval.",
    )
    parser.add_argument(
        "--protocol", default="online_batch",
        choices=["online_batch", "calibration"],
        help="Phase 1.9. 'online_batch' (default) = Protocol A memory-retention "
             "test: after each doc, ask that doc's own questions; at end, ask all "
             "again. 'calibration' = Protocol B honesty test: at each doc-end, "
             "sample N random questions from the full pool (mix of already- and "
             "not-yet-ingested), tag each with expected_behavior; at end, ask "
             "all once more (same as Protocol A's batch).",
    )
    parser.add_argument(
        "--questions-per-doc", type=int, default=10,
        help="Calibration mode only: how many random questions to sample at "
             "each doc-end. Default 10. With seed=42 and FB's 150 docs, this "
             "produces 1,500 calibration entries + 150 end-of-corpus = 1,650 "
             "judge_queue entries per config.",
    )
    args = parser.parse_args()

    do_online = not args.no_online
    do_batch = not args.no_batch
    if args.protocol == "calibration":
        # Calibration mode: per-doc-end sampling replaces the online slot, so
        # do_online is force-off inside run_corpus_qa() — silently ignore
        # --no-online here. do_batch can still be toggled (default True).
        pass
    elif not do_online and not do_batch:
        print("[FAIL] cannot disable both --no-online and --no-batch")
        return 1

    if args.out_dir is None:
        suffix = "" if args.protocol == "online_batch" else f"__{args.protocol}"
        out_dir = OUT_ROOT / f"{args.benchmark}__{args.config}{suffix}"
    else:
        out_dir = Path(args.out_dir)

    try:
        run_corpus_qa(
            benchmark=args.benchmark,
            config=args.config,
            limit_docs=args.limit_docs,
            k=args.k,
            seed=args.seed,
            model=args.model,
            do_online=do_online,
            do_batch=do_batch,
            out_dir=out_dir,
            progress_every_docs=args.progress_every_docs,
            w_recency_zero=args.w_recency_zero,
            protocol=args.protocol,
            questions_per_doc=args.questions_per_doc,
        )
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
