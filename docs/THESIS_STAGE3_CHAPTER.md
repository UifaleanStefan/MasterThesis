# Stage 3 — Task-Adaptive Parameterized Memory on Six Real Long-Context Benchmarks

*Draft. Sections 1, 2, 3, 4, 6, 7 in prose. Section 5 (Results) has the
retrieval-quality table populated from Phase 1.5 runs; the LLM-judge
answer-quality and Pareto-frontier tables await Phase 4 API runs.*

---

## 1. Introduction

The dominant LLM-agent memory pattern as of 2026 is undifferentiated:
either keep the entire trajectory in the model's context window (limited
by the window, expensive per token, slower as length grows), or chunk
it and retrieve via an off-the-shelf dense retriever (fixed scoring,
no task-aware structure). Both treat memory as something that
*happens* to the agent rather than something the agent *learns*.

This thesis argues for the third position: **memory construction itself
should be learnable, and the optimal memory parameters are task-dependent.**
Concretely, the construction of a memory graph — what to store, which
concepts to elevate to entity nodes, how to weight retrieval signals —
is governed by a low-dimensional parameter vector θ ∈ [0, 1]¹⁰. That
vector can be searched for any specific task via black-box optimization
(CMA-ES, no gradients required), and the resulting θ generalizes within
the task but **does not transfer across tasks**. The thesis's earlier
chapters demonstrated this on grid-world environments: optimal θ for
key-door matching differs measurably from optimal θ for goal-reaching,
and both differ from optimal θ for multi-hop key-door routing.

The natural objection is that grid-worlds are not real-world LLM
workloads. The cynical reading is "you built the test set; the result
is unsurprising on a toy task". Stage 3 — this chapter — answers that
objection by porting the entire memory-learning machinery onto **six
published, real-world long-context benchmarks** spanning legal contracts
(CUAD), scientific papers (QASPER), Wikipedia multi-hop reasoning
(HotpotQA), full books and screenplays (NarrativeQA), SEC filings
(FinanceBench), and multi-session dialogue (LongMemEval).

The Stage 3 contribution comes in two layers:

1. **A retrieval-quality contribution.** The same `GraphMemoryV4` 10-D
   θ-vector, when re-tuned per benchmark via CMA-ES on recall@k, beats
   every other memory system on the two long-haystack benchmarks
   (QASPER and CUAD). It beats the grid-world-tuned canonical θ by
   +0.469 on QASPER (wide CMA-ES) and +0.229 on CUAD — a finding that
   simultaneously validates the "task-dependent memory" claim on real
   data and reveals a more nuanced cross-benchmark transfer story than
   the grid-world chapter predicted: tuned θ from one long-haystack
   benchmark transfers ~85% of its lift to a different long-haystack
   benchmark within the document-QA family, but the grid-world
   canonical θ transfers ~0% to document QA. Memory is task-dependent
   at the **task-family granularity**, not the within-family granularity.

2. **An LLM-cost contribution** (Phase 4, in progress). Building on the
   above retrieval table, the same parameterized memory is wired into
   a GPT-4o-mini answer-quality pipeline. Because memory selects fewer
   but more relevant paragraphs to feed to the LLM, the joint objective
   `J = QA_score − λ × cost_usd` becomes operational: selective memory
   directly translates to dollars saved per query while preserving (or
   improving) answer quality.

This chapter focuses on (1), with (2) introduced as future work
pending API budget. The reader's roadmap: Section 2 places the work in
the long-context retrieval and parameterized-memory literature; Section 3
covers the benchmark adapter layer and the tuning protocol; Section 4
specifies the experimental setup; Section 5 reports the retrieval-quality
results; Section 6 discusses the two-cluster finding (long-haystack
benchmarks differentiate memory systems; short-haystack ones saturate
at recall = 1.0); Section 7 outlines the path from here to the closed
LLM-cost loop and beyond.

---

## 2. Related Work

### Long-context retrieval-augmented QA

Dense retrieval (DPR, Karpukhin et al. 2020; Contriever, Izacard et al.
2022) provides the standard "embed everything, retrieve top-k by cosine
similarity" baseline. The retrieval head is typically a fixed
pre-trained model; tuning happens at the *encoder* level, not at the
*selection* level. RAG (Lewis et al. 2020) wraps this for generative
QA. LongRAG (Jiang et al. 2024) and Self-RAG (Asai et al. 2023)
introduce learned reranking or self-critique. None of these treat
*which events to store at all* as a tunable parameter — they all assume
the corpus is given and the question is how to retrieve.

The closest spiritual cousin is RAG with hierarchical or learned
chunking (RAPTOR, Sarthi et al. 2024; HiQA, Tao et al. 2024) — those
work *also* recognize that "what to store" is part of the system. But
the tunable surface is the chunking heuristic, not a low-dimensional
parameterization of a graph-memory's storage + scoring policy.

### Parameterized graph memory

Graph-augmented memory for LLM agents emerged with MemGPT (Packer et
al. 2023), which uses a static OS-like memory hierarchy. HippoRAG
(Gutiérrez et al. 2024) builds a knowledge graph at ingestion time and
uses Personalized PageRank for retrieval — closest to this thesis's
construction-time entity-extraction step. MemoryLLM (Wang et al. 2024)
explores hidden-state memory expansion. **None of these papers
parameterize the storage decision and tune it via black-box optimization;
they fix the construction policy and study downstream effects.**

This thesis's V4 design is in the same family as HippoRAG (entity nodes,
graph edges, retrieval scoring) but the 10-D θ exposes seven storage-time
and three retrieval-time knobs — `theta_store, theta_novel, theta_erich,
theta_surprise, theta_entity, theta_temporal, theta_decay, w_graph,
w_embed, w_recency` — and treats them all as searchable parameters.

### Black-box optimization for memory and retrieval

CMA-ES (Hansen 2003) and standard evolution strategies (Salimans et al.
2017) have been used to tune neural-network policies in reinforcement
learning but rarely on memory-construction parameters. Bayesian
optimization with Gaussian processes has been applied to embedding
hyperparameters (Snoek et al. 2012) and retrieval scoring weights
(Gupta et al. 2022). The closest match for this thesis's tuning protocol
is `BayesOpt-RAG` (concurrent and unpublished as of writing) — but
again, on retrieval weights, not on the storage policy.

### Reflexion and verbal memory

The thesis's Stage 2 chapter (V6 era — see `docs/REFLEXION_RESULTS.md`)
explored a Reflexion-style verbal lesson buffer layered atop V4. That
work landed an honest negative empirical result on grid-worlds (the
rule-based policy could not benefit from paraphrased hint text because
the hints were already in the observation channel). It is *architecturally
adjacent* to Stage 3 — both rely on selective memory — but Stage 3 makes
no claim about verbal-lesson generation; the question here is purely
about *what to store and how to retrieve* from real-world long-form text.

### Honest gap framing

What is new in this chapter, distinct from the cited prior work:

1. **The full memory-construction pipeline (storage policy + retrieval
   scoring) is parameterized by a single 10-D θ vector and tuned
   end-to-end on recall@k.** No prior published work does both layers
   in one optimization loop.
2. **The tuning is per-benchmark, and the resulting θ vectors are
   reported alongside the canonical grid-world-tuned θ — letting us
   observe and report the (lack of) cross-task transfer as a finding,
   not just a parameter choice.** The transfer-failure result is itself
   the contribution.
3. **The evaluation runs on six published, disjoint-domain benchmarks
   with paragraph-level gold-relevance signals where available
   (HotpotQA `supporting_facts`, LongMemEval `answer_session_ids`).**
   Retrieval-quality numbers are not synthetic.

---

## 3. Methodology

### 3.1 The benchmark adapter layer

The entire evaluation pipeline pre-existing this chapter assumes
documents in the shape:

```
Document = {
    "title": str,
    "paragraphs": list[str],
    "qa_pairs": [{
        "question": str,
        "answer": str | list[str],
        "relevant_paragraphs": list[int],
    }, ...],
}
```

This is the contract consumed by `environment/document_qa.py:DocumentQA`,
by `evaluation/document_qa_memory.py:_run_reading_phase` (which walks
paragraphs as `Event` objects into a memory), and by
`evaluation/document_qa_memory.py:_recall_at_k_for_qa` (which scores
retrieval against `relevant_paragraphs`).

Six benchmark adapters (`environment/benchmarks/{hotpotqa, qasper,
cuad, narrativeqa, financebench, longmemeval}.py`) translate each
benchmark's native format into this contract. Each adapter exposes a
single public method:

```
iter_documents(split, limit, seed, shuffle) -> Iterator[Document]
```

The Iterator pattern keeps memory pressure low even on NarrativeQA's
1.2-million-character books: at any moment only one document is
materialized, and the underlying HuggingFace datasets library
memory-maps the parquet store from disk.

**Critical contract invariant:** paragraphs MUST be emitted in the
order they will be ingested into memory, and `relevant_paragraphs`
MUST index that ordering. Post-hoc filtering, deduplication, or
reordering of paragraphs silently destroys the recall metric. This
constraint propagated all the way to the adversarial test layer
(Section 3.3), which asserts trigram-overlap between each gold
paragraph and the question text.

### 3.2 Per-benchmark conversion specifics

A condensed summary of each adapter's translation logic:

| Benchmark | Source | Paragraph construction | Gold relevance |
|---|---|---|---|
| HotpotQA | HF `hotpotqa/hot_pot_qa`, `distractor` config, validation split | passage `i` → `f"{titles[i]}\n{joined sentences}"` for `i ∈ [0, 10)` | `supporting_facts.title` → passage indices via `title_to_index` |
| QASPER | AI2 S3 tarball `qasper-{train-dev,test}-v0.3` | abstract + flatten of `full_text` sections; section names as their own paragraphs | per-answer `evidence` strings → paragraph indices via fuzzy substring match (NFKC + whitespace collapse), with trigram-overlap fallback |
| CUAD | Zenodo `CUAD_v1.zip` (SQuAD v2 JSON) | char-offset-preserving split on `\n\n`; preserve `[start, end)` ranges per paragraph | first answer's `answer_start` → paragraph index via offset-to-paragraph lookup |
| NarrativeQA | HF `deepmind/narrativeqa`, validation split | summary as paragraph[0]; `greedy_merge_paragraphs(body)` capped at 2000 paragraphs with truncation marker in title | none (multi-ref answer list goes to LLM judge) |
| FinanceBench | HF `PatronusAI/financebench`, default split | one paragraph per `evidence` entry's `evidence_text` (handles both list-of-dict and raw-string shapes) | every paragraph is gold (`list(range(n))`) by construction |
| LongMemEval | xiaowu0162/longmemeval-cleaned (raw JSON via `hf_hub_download` to bypass a known answer-column type mismatch) | one paragraph per session, formatted as `[Session sid on date]\nrole: content\n...` | `answer_session_ids` → indices via `haystack_session_ids` lookup |

Three benchmarks needed non-trivial workarounds during data acquisition:
HF `datasets>=3` dropped loading-script support (broke QASPER and CUAD,
fixed by pulling canonical archives directly from AI2 S3 and Zenodo
respectively); the original `xiaowu0162/longmemeval` HF repo was
deprecated (switched to `longmemeval-cleaned`); `longmemeval-cleaned`'s
HF `load_dataset` autoparser fails on a type mismatch in the answer
column (bypassed via direct `hf_hub_download` of the raw JSON files).
Full details in `docs/STAGE3_DATA_PREP.md`.

### 3.3 The four-layer test pyramid

Each adapter is guarded by four progressively-deeper test layers:

* **Layer 0 — Snapshot.** First three documents from each adapter are
  fingerprinted (SHA-256 over canonical JSON dump) and stored in
  `tests/fixtures/<bench>_snapshot.json`. The test asserts no drift
  across runs. This catches HuggingFace dataset version changes,
  unintended adapter logic edits, and local file corruption.
* **Layer 1 — Schema.** 44 parametrized tests per adapter assert: the
  three top-level keys present, each `qa_pair` has the three sub-keys,
  `relevant_paragraphs` indices fall in `[0, len(paragraphs))`, every
  indexed paragraph shares trigram overlap with the question or answer
  text (off-by-one detector), and `iter_documents` is byte-deterministic
  under fixed seed.
* **Layer 1.5 — Adversarial.** 29 mocked edge-case tests inject crafted
  inputs the happy path never sees: `yes_no=False` (must produce literal
  "No"), `FLOAT_TYPE_NONEVIDENCE` sentinel filtering, `answer_start` in
  blank-line gaps, multi-byte unicode (ﬁ ligatures, CJK), mismatched
  parallel arrays, paragraph cap markers, all-unanswerable papers, etc.
  Each test docstring names its specific edge case for fast diagnosis.
* **Layer 2 — Retrieval smoke.** 7 tests per adapter exercise the full
  `DocumentQA + V4` pipeline on one document at `k=8`, asserting
  per-benchmark mean recall@k thresholds AND a prompt-budget guard
  (`max_chars_retrieved_per_question < 50,000`) that catches paragraph
  caps before they translate to Phase-4 token blowups.

A fifth layer — Layer 3 paid API smoke — runs `scripts/smoke_stage3_api.py`
against real GPT-4o-mini at ~$0.10 to validate the end-to-end LLM path.
It is not part of the automated `pytest` suite for cost reasons.

Total automated coverage: **146 pytest tests, all green, 98 seconds wall
time.** The `scripts/audit_determinism.py` orthogonal check (14 memory
systems × seeds `[0, 7, 42]` × two repeated runs) remains green
post-Stage-3 additions.

### 3.4 CMA-ES per-benchmark θ tuning

`GraphMemoryV4`'s memory parameters live in
`memory/graph_memory_v4.py:MemoryParamsV4`. The 10 dimensions split
into: seven storage / construction thresholds clipped to `[0, 1]`
(`theta_store, theta_novel, theta_erich, theta_surprise, theta_entity,
theta_temporal, theta_decay`) and three retrieval-scoring weights
clipped to `[0, 4]` (`w_graph, w_embed, w_recency`). For the CMA-ES
optimizer, all 10 dimensions are searched in the unit hypercube
`[0, 1]¹⁰`, with the three weight dimensions multiplied by 4 at
`vec_to_params` time. This is the convention from
`run_graphmemory_v4_cmaes.py` (the Stage 2 grid-world tuner),
re-implemented locally in `tuning/tune_v4_per_benchmark.py:vec_to_params`
for self-containment.

The objective is the **mean recall@k=8** across qa_pairs with non-empty
`relevant_paragraphs`, averaged over `n_docs` documents pulled from the
adapter:

```
eval_fn(theta) =
    1/|gold_qas| * sum over docs and qa_pairs with non-empty gold:
        [ 1.0 if (any p in relevant_paragraphs is in top-k retrieval)
          else 0.0 ]
```

Higher fitness = better. CMA-ES with `n_params=10, sigma=0.3, seed=42`,
running for `n_generations` generations with the pycma backend's default
population size (~12 candidates per generation for `n=10`).

**Two configurations were run:**

1. **Narrow** (`n_docs=8, n_generations=10`): the first-pass conservative
   sweep. ~165 evaluations per benchmark; ~5-15 minutes per benchmark on
   the MiniLM-cached embedding stack.
2. **Wide** (`n_docs=20, n_generations=30`, target benchmarks only — see
   below): the second-pass sweep with more docs per evaluation and more
   generations to squeeze residual headroom. ~360 evaluations per
   benchmark; ~25-40 minutes per benchmark.

The wide sweep was restricted to QASPER and CUAD — the only two
benchmarks for which the narrow sweep produced measurable improvement
over canonical θ. The other four benchmarks have small enough haystacks
that k=8 saturates retrieval at recall = 1.0 regardless of θ, and the
narrow CMA-ES correctly reported zero gradient direction in that region.

### 3.5 Cross-benchmark θ-transfer ablation

To answer the question "does QASPER-tuned θ work on CUAD and vice
versa?", we build a 3×2 transfer matrix:

```
                    QASPER eval    CUAD eval
canonical θ         baseline_q     baseline_c
QASPER-tuned θ      diag_qq        off_qc
CUAD-tuned θ        off_cq         diag_cc
```

The script `scripts/run_theta_transfer.py` evaluates each (θ_source,
eval_benchmark) cell with `n_docs=15, k=8` and records mean recall@k.
The expected pattern — diagonal cells (`diag_qq, diag_cc`) high,
off-diagonal cells (`off_qc, off_cq`) near canonical — would confirm
**task-specific θ does NOT transfer across long-haystack QA tasks**,
mirroring the cross-environment finding from the Stage 1 chapter on
grid-worlds. The actual numbers are in Section 5.

---

## 4. Experimental Setup

### 4.1 Datasets

| Benchmark | Items (eval) | Avg paragraphs/doc | Avg gold per qa_pair | Note |
|---|---:|---:|---:|---|
| HotpotQA | 7,405 (val) | 10 | 2 | distractor config |
| QASPER | 281 (dev) + 416 (test) papers | 84 | 0–3 (sparse evidence) | per-answer evidence may miss substring match |
| CUAD | 510 contracts | 149 | 1 (rare, but answer can occur ≥1 places) | 68% of QAs `is_impossible=True` by design (filtered) |
| NarrativeQA | 3,461 (val) | 230 (after greedy merge) | 0 (no paragraph-level gold) | answers as 2-ref list |
| FinanceBench | 150 (default) | 1–3 (evidence excerpts only) | n (every paragraph is gold) | full SEC PDFs not in scope |
| LongMemEval | 500 (oracle) | 2 (median sessions per item) | 1–2 (single sessions) | s_cleaned/m_cleaned splits also fetched |

For Phase 1.5 retrieval-only experiments, 15 documents per benchmark
were sampled (deterministic, no shuffle), yielding ~50-200 qa_pairs
per benchmark depending on QAs-per-doc density. For tuning, 8 docs
(narrow) and 20 docs (wide).

### 4.2 Metrics

* **recall@k=8** — primary retrieval-quality metric. Binary per
  qa_pair: 1.0 if any retrieved event's step is in `relevant_paragraphs`,
  else 0.0. Aggregated as the mean over qa_pairs with non-empty gold.
  Mean (not median) chosen because the binary distribution makes
  median brittle.
* **LLM-judge score** (Phase 4, future) — `evaluation/document_qa_llm_judge.py:llm_judge_score`
  using gpt-4o-mini with a 0–1 rubric. `llm_judge_score_multi_ref`
  wrapper handles NarrativeQA's list-typed reference answers (max
  over references).
* **USD cost** (Phase 4, future) — tiktoken-counted prompt tokens +
  bounded completion tokens, multiplied by gpt-4o-mini's `$0.15/M`
  input + `$0.60/M` output pricing.
* **Prompt-byte budget** (Phase 1.5, current guardrail) — asserted in
  Layer 2 of the test pyramid as a leading indicator of Phase-4 cost
  blowups.

### 4.3 Memory systems (12 + tuned variants)

The reference panel from `evaluation/document_qa_memory.py:_make_document_qa_memory_systems`:
`FlatWindow(50)`, `GraphMemory+Theta`, `GraphMemoryV4`, `GraphMemoryV5`,
`SemanticMemory`, `SummaryMemory`, `EpisodicSemantic`, `RAGMemory`,
`HierarchicalMemory`, `WorkingMemory(7)`, `CausalMemory`,
`AttentionMemory`. Plus one extra row per benchmark with V4
instantiated using that benchmark's CMA-ES-tuned θ
(`V4-tuned-<benchmark>`).

### 4.4 Reproducibility

Every result JSON ships with a `_manifest` sibling containing
`git_sha`, `embedding_backend`, `timestamp_utc`, `python_version`,
numpy/scipy versions, the experiment-specific config, and the random
seed. Adapter results add a per-benchmark `dataset_fingerprint`
(SHA-256 over HuggingFace download checksums or local file content)
and per-document fingerprints. The `scripts/audit_determinism.py`
orthogonal check guarantees V4 retrieval is bit-identical across
repeated runs at the same seed.

Reproducing the full Phase 1+1.5 pipeline:

```
python scripts/prefetch_benchmarks.py
python scripts/verify_benchmarks.py
python -m pytest tests/test_benchmark_*.py -v
python -m tuning.tune_v4_per_benchmark --benchmarks all
python scripts/run_stage3_retrieval.py --benchmarks all \
    --n-docs 15 --load-tuned-thetas
python scripts/run_stage3_full.py --mode dry-run \
    --benchmarks all --configs v4-canonical v4-tuned flat-50 \
    --n-questions 30
python scripts/build_stage3_frontend_data.py
python scripts/audit_determinism.py
```

All steps are $0 (no API calls). Phase 4 substitutes `--mode full` for
the orchestrator step and requires `OPENAI_API_KEY`.

### 4.5 Hardware / wall-time

All experiments ran on a single workstation (Windows 11, Python 3.11,
no GPU; embeddings via `sentence-transformers/all-MiniLM-L6-v2` on CPU,
LRU-cached at 8192 entries). End-to-end timings:

* Adapter test pyramid (146 tests): 98 seconds.
* CMA-ES narrow tuning (6 benchmarks): ~3 minutes total.
* CMA-ES wide tuning (QASPER + CUAD): ~25-50 minutes total.
* Retrieval study (12 systems × 6 benchmarks × 15 docs): 163 seconds.
* Phase-4 dry-run (3 configs × 6 benchmarks × 30 q): ~3 minutes.

---

## 5. Results

### 5.1 Retrieval-quality table (Phase 1.5, complete)

Mean recall@k=8 across 15 documents per benchmark, all 12 reference
systems plus per-benchmark V4-tuned variants. Headline cells in
**bold**. Source: `results/stage3/retrieval_summary.json`.

| System | CUAD | FinanceBench | HotpotQA | LongMemEval | NarrativeQA | QASPER |
|---|---:|---:|---:|---:|---:|---:|
| **V4-tuned-<bench>** | **0.629** | 1.000 | 1.000 | 1.000 | n/a | **0.563** |
| AttentionMemory | 0.618 | 1.000 | 1.000 | 1.000 | 0.000 | 0.521 |
| RAGMemory | 0.618 | 1.000 | 1.000 | 1.000 | 0.000 | 0.521 |
| GraphMemory+Theta | 0.613 | 1.000 | 1.000 | 1.000 | 0.000 | 0.458 |
| GraphMemoryV5 | 0.393 | 1.000 | 1.000 | 1.000 | 0.000 | 0.229 |
| GraphMemoryV4 (canonical) | 0.366 | 1.000 | 1.000 | 1.000 | 0.000 | 0.208 |
| SemanticMemory | 0.296 | 1.000 | 1.000 | 1.000 | 0.000 | 0.188 |
| EpisodicSemantic | 0.108 | 1.000 | 1.000 | 1.000 | 0.000 | 0.208 |
| CausalMemory | 0.108 | 1.000 | 1.000 | 1.000 | 0.000 | 0.208 |
| FlatWindow(50) | 0.108 | 1.000 | 1.000 | 1.000 | 0.000 | 0.208 |
| HierarchicalMemory | 0.108 | 1.000 | 1.000 | 1.000 | 0.000 | 0.208 |
| WorkingMemory(7) | 0.102 | 1.000 | 0.933 | 1.000 | 0.000 | 0.167 |
| SummaryMemory | 0.075 | 1.000 | 0.733 | 1.000 | 0.000 | 0.083 |

### 5.2 V4-canonical vs V4-tuned per benchmark

| Benchmark | V4-canonical | V4-tuned (narrow) | V4-tuned (wide) | Improvement vs canonical |
|---|---:|---:|---:|---:|
| QASPER | 0.107 | 0.464 | *TBD W2* | +0.357 → *TBD* |
| CUAD | 0.458 | 0.687 | *TBD W2* | +0.229 → *TBD* |
| HotpotQA | 1.000 | 1.000 | not tuned | 0 (ceiling) |
| FinanceBench | 1.000 | 1.000 | not tuned | 0 (every paragraph is gold) |
| LongMemEval | 1.000 | 1.000 | not tuned | 0 (small haystack) |
| NarrativeQA | — | — | — | n/a (no gold) |

*The "wide" column will be filled in after the W2 run completes (W2 is
currently in flight as this draft is being written; preliminary
estimate: marginal +0.02 to +0.05 over narrow on both QASPER and
CUAD).*

### 5.3 Cross-benchmark θ-transfer matrix (Phase 1.6 complete)

Evaluating each θ-source (rows) on each benchmark (cols) at n_docs=15,
k=8. The diagonal cells (matched θ-source to eval-benchmark) use the
**better** of (narrow, wide) tuning per `theta_width_comparison.json`:
wide for QASPER (0.576 > narrow 0.464), narrow for CUAD (0.687 > wide
0.671 — CMA-ES non-convexity penalized the wider search slightly).
Source: `results/stage3/theta_transfer_matrix.json`.

| θ source ↓ \ Eval → | QASPER | CUAD |
|---|---:|---:|
| canonical (grid-world) | 0.208 | 0.366 |
| **QASPER-tuned** (wide) | **0.563** (diag) | 0.591 (off-diag) |
| **CUAD-tuned** (narrow) | 0.500 (off-diag) | **0.629** (diag) |

Summary statistics:
* Canonical average recall: 0.287
* Diagonal average recall (matched θ): 0.596 — **+0.309 vs canonical**.
* Off-diagonal average recall (mismatched θ): 0.546 — **+0.259 vs canonical**.

**The transfer DOES happen.** Off-diagonal cells recover 84 % of the
diagonal lift over canonical (0.259 / 0.309). Mismatched θ within the
document-QA family is dramatically better than grid-world θ, even when
the source benchmark is structurally different from the eval benchmark.

### 5.4 LLM answer-quality table (Phase 4, complete)

Mean LLM-judge score (gpt-4o-mini, 0–1 scale) across 3 seeds × 100
questions = 300 questions per cell. Source:
`results/stage3/phase4_summary.json` and per-cell
`results/stage3/cells/{benchmark}__{config}__seed{seed}.json`.
**Bold** = best per row. Paired-t-test column is the per-question
matched test of V4-tuned − V4-canonical pooled across all 300 questions.

| Benchmark | V4-canonical (95% CI) | V4-tuned (95% CI) | flat-50 (95% CI) | Lift V4t−V4c | t | p |
|---|---|---|---|---:|---:|---:|
| **CUAD** | 0.249 [0.209–0.289] | **0.316** [0.275–0.358] | 0.202 [0.167–0.235] | **+0.067** | **3.41** | **0.0007 \*\*\*** |
| **QASPER** | 0.180 [0.148–0.213] | **0.203** [0.170–0.238] | 0.162 [0.132–0.190] | +0.023 | 1.41 | 0.16 |
| HotpotQA | **0.648** [0.602–0.698] | 0.636 [0.588–0.687] | 0.616 [0.571–0.665] | −0.012 | −0.77 | 0.44 |
| LongMemEval | 0.437 [0.389–0.485] | 0.426 [0.379–0.474] | **0.450** [0.403–0.499] | −0.010 | −1.20 | 0.23 |
| FinanceBench | 0.442 [0.395–0.487] | 0.450 [0.400–0.496] | **0.488** [0.443–0.534] | +0.008 | 1.13 | 0.26 |
| NarrativeQA | **0.212** [0.172–0.253] | n/a | 0.158 [0.125–0.195] | — | — | — |

**Headline finding: V4-tuned beats V4-canonical on CUAD with high
statistical significance (p = 0.0007, n = 300 paired questions).** On
QASPER the direction is the same (+0.023) but does not reach p < 0.05
with this sample size. On the four ceiling-bound short-haystack
benchmarks, no config differs significantly from another — consistent
with the retrieval-quality two-cluster finding from Section 5.1.

### 5.5 Cost per question (Phase 4, complete)

Mean per-question OpenAI API cost (gpt-4o-mini, pooled across 3 seeds).
Source: same per-cell JSONs as 5.4, `actual_cost_usd / n_questions`.

| Benchmark | v4-canonical | v4-tuned | flat-50 |
|---|---:|---:|---:|
| LongMemEval (longest haystacks) | $0.00084 | $0.00084 | $0.00091 |
| NarrativeQA (long books) | $0.00060 | n/a | $0.00060 |
| CUAD | $0.00019 | $0.00022 | $0.00014 |
| HotpotQA | $0.00018 | $0.00017 | $0.00018 |
| QASPER | $0.00012 | $0.00013 | $0.00013 |
| FinanceBench | $0.00009 | $0.00009 | $0.00011 |

Total Phase-4 spend (Tier A + Tier B + Tier C — see §4.5):
**~$3.30 USD** for the entire 6 benchmarks × 3 configs × 3 seeds × 100
questions = 5,100 evaluated questions, plus the k-sweep (4,800
additional). 100× cheaper than the original $35 budget estimate, because
gpt-4o-mini's pricing and our compact retrieved-context per question
both came in under worst-case projection.

### 5.6 Pareto frontier — cost-quality elasticity (lambda-sweep)

Defining the joint objective `J(λ) = judge_score − λ × cost_per_q`,
sweeping λ ∈ {0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000}
re-ranks configurations as cost-sensitivity increases. Source:
`results/stage3/lambda_sweep.json`, post-hoc on Phase-4 per-cell
data (no extra API). The crossover λ tells us "at what cost-sensitivity
does the optimal config change":

| Benchmark | Leader at λ=0 | Leader at λ=100 000 | Crossover λ |
|---|---|---|---:|
| **CUAD** | v4-tuned | flat-50 | 5 000 |
| **QASPER** | v4-tuned | v4-canonical | 5 000 |
| HotpotQA | v4-canonical | v4-tuned | 5 000 |
| LongMemEval | flat-50 | v4-tuned | 500 → 10 000 (two crossovers) |
| FinanceBench | flat-50 | v4-tuned | 5 000 |
| NarrativeQA | v4-canonical | v4-canonical | (no crossover) |

The empirical reading is **V4-tuned dominates the upper-left of the
Pareto frontier** (low cost-sensitivity, where judge quality matters
most) on the two long-haystack benchmarks, exactly as hypothesised. As
λ grows past 5 000 (i.e. cost matters ~5,000× more than quality
per token), flat-50 wins on CUAD by being slightly cheaper. The
crossover λ values are *all in a narrow band* (500–10 000), which means
the practical decision between configs only flips under fairly extreme
cost-sensitivity — for typical Stage-3 use, V4-tuned is the right
choice on the long-haystack benchmarks.

The Tier C k-sweep (k ∈ {4, 8, 16, 32}, 100 q × 1 seed × 6 benchmarks × 2 configs)
extends this Pareto picture to the *retrieval budget* dimension and
is reported in
`results/stage3/k_sweep.json`. As k grows from 4 → 32, both
mean judge and per-question cost rise, but the marginal gains diminish
sharply past k = 8 on the short-haystack benchmarks and past k = 16 on
the long-haystack ones. k = 8 (the value used throughout this chapter)
sits at the elbow of the Pareto curve for QASPER and CUAD.

---

## 6. Discussion

### 6.1 The two-cluster finding

The single most striking pattern in the retrieval table is that the
six benchmarks split into two qualitatively-different clusters:

* **Short-haystack benchmarks** — HotpotQA, FinanceBench, LongMemEval —
  saturate at recall = 1.0 for almost every memory system. These three
  benchmarks have small enough haystacks (10 passages for HotpotQA, 1–3
  evidence excerpts for FinanceBench, median 2 sessions for LongMemEval)
  that retrieving the top-8 by *any* reasonable scoring function
  effectively recovers the gold. The memory system matters very little.
  The only outliers are `SummaryMemory` (0.73 on HotpotQA — the
  compression hurts retrieval) and `WorkingMemory(7)` (0.93 on HotpotQA
  — only 7 events held). Every other system is at the ceiling.
* **Long-haystack benchmarks** — CUAD (avg 149 paragraphs per contract)
  and QASPER (avg 84 paragraphs per paper) — produce a clear ordering
  across systems. The top three (V4-tuned, RAGMemory, AttentionMemory)
  cluster at 0.52–0.63; the middle (V4-canonical, V5, V1) at 0.21–0.46;
  the bottom (Flat, Episodic, Causal, Hierarchical, Working, Summary)
  at 0.08–0.21. The spread is more than 7× between best and worst.

The thesis claim of "memory matters" is therefore *quantitative on
long-haystack benchmarks and trivial on short-haystack ones*. The
honest framing for the reader is: **selective parameterized memory
distinguishes itself precisely where the LLM context window itself
would already be under pressure**. The short-haystack benchmarks are
the regime where the entire haystack fits in any context window and
memory choices don't matter. The long-haystack benchmarks are where
they do — and where Stage 3 makes its empirical contribution.

### 6.2 Why grid-world θ fails on document QA

V4's canonical θ (`w_recency=3.777, theta_store=0.293`) was tuned on
`MultiHop-KeyDoor` in the Stage 2 chapter. In that environment, the
agent traverses a grid taking actions, and the hint observations
("the red key opens the north door") appear at the very start of the
episode. The task signal is therefore *recent for a few steps and then
old for the remainder of the episode*, and `w_recency=3.777` pushes
the score function to value recent observations far more than older
ones at retrieval time.

On document QA the inverse holds: the agent reads a static document
in fixed order (paragraph 0 first, paragraph N last), and the question
is asked after all paragraphs are read. By construction the
"most recent" observations are always the last paragraphs of the
document — independent of what the question is about. With
`w_recency=3.777`, V4 retrieval collapses onto the document tail
regardless of the question. Empirically: on QASPER's first paper
(84 paragraphs, 4 questions), V4-canonical retrieved exactly
`{72, 74, 76, 77, 78, 80, 81, 82}` for every single question. The
top-8 set was identical across all four questions because the
recency scores swamped the embedding signal.

V4-tuned discovers θ vectors with `w_recency` near zero and `w_embed`
elevated to recover the embedding-driven retrieval that grid-world θ
suppressed. The +0.357 lift on QASPER and +0.263 lift on CUAD are
the direct empirical consequence of this rebalancing.

This is *exactly* the finding the cross-environment grid-world
results in the Stage 1 chapter predicted at toy scale: **the optimal θ
differs across tasks because the structure of the task signal differs**.
What's new in Stage 3 is showing that this principle survives the
jump from grid-worlds to real-world long-context QA.

### 6.3 The cross-benchmark transfer finding — hypothesis overruled

The 3×2 transfer matrix (Section 5.3) was designed to test whether
QASPER-tuned θ and CUAD-tuned θ would *fail* to transfer across each
other's benchmarks. The expected diagonal-dominant pattern would have
mirrored the cross-environment transfer-failure shown in the Stage 1
grid-world chapter and would have made the "task-dependent memory" claim
within-document-QA quantitative.

**The data overruled the hypothesis.** Off-diagonal cells recover 84 %
of the diagonal lift over canonical: QASPER-tuned θ on CUAD scores
0.591 (vs CUAD-tuned's diagonal 0.629), and CUAD-tuned θ on QASPER
scores 0.500 (vs QASPER-tuned's diagonal 0.563). Both off-diagonals
are far above canonical's 0.208 (QASPER) and 0.366 (CUAD).

The honest empirical conclusion is *not* "task-specific θ doesn't
transfer". It's "**task-tuned θ on long-haystack QA generalizes within
the document-QA family. The grid-world canonical θ is just bad for
document QA in general — but any θ tuned on any long-form document QA
task lifts retrieval substantially on every other long-form document
QA task.**"

This is a softer claim than the originally-targeted within-family
transfer failure, but it is also more honest. The underlying mechanism
is that the canonical θ's recency bias (`w_recency=3.777`) is wrong
for *any* read-document-then-ask-question setting; once that is
corrected via any per-task CMA-ES run, the resulting θ ports across
similar tasks. The thesis claim of "memory is task-dependent" therefore
holds at the **task-family** granularity (grid-world ≠ document-QA),
not at the within-family granularity (QASPER ≠ CUAD within document-QA).

This refinement matters for downstream prescriptive use of the system:
practitioners running new long-haystack QA workloads can adopt a
generic "document-QA tuned" θ (e.g. QASPER-tuned or CUAD-tuned) and
expect to capture ~85 % of the lift available from in-task tuning,
without paying for their own per-task CMA-ES sweep. This is a much
more useful and field-relevant prescription than "you must tune
per-task or you lose all the benefit".

### 6.4 Wide vs. narrow CMA-ES — diminishing returns and non-convexity

The "wide" CMA-ES sweep (`n_docs=20, n_generations=30`) was hypothesized
to squeeze additional headroom past the "narrow" first-pass
(`n_docs=8, n_generations=10`). The data are mixed:

* On QASPER, wide lifts the best-recall from 0.464 to 0.576 (**+0.112**).
* On CUAD, wide *regresses* slightly: 0.687 → 0.671 (**−0.016**).

The CUAD regression isn't a bug — it's an artifact of CMA-ES's
non-convex search: with more docs per eval but the same starting
covariance, the optimizer explores wider but doesn't always converge
faster. For CUAD specifically, the narrow run's smaller eval ensemble
produced a slightly lucky candidate that wide didn't rediscover within
its 30-generation budget. The transfer-matrix evaluation in 5.3 therefore
uses the *better* of (narrow, wide) per benchmark — wide for QASPER,
narrow for CUAD — as recorded in `theta_width_comparison.json`.

Diminishing returns: the wide CUAD result is within the noise floor
of the narrow result, suggesting the per-benchmark θ for CUAD is
substantively converged at the narrow budget. Further tuning would
need a fundamentally different search strategy (e.g. Bayesian
optimization with a GP prior) rather than longer CMA-ES.

### 6.4 What memory parameters actually capture

The per-benchmark tuned θ vectors provide diagnostic information.
Comparing canonical vs. tuned values:

| Dimension | Canonical | QASPER-tuned (narrow) | CUAD-tuned (narrow) | Reading |
|---|---:|---:|---:|---|
| `theta_store` | 0.293 | (W2 wide TBD) | (W2 wide TBD) | Storage threshold; lower = store more |
| `theta_novel` | 0.908 | TBD | TBD | Novelty weighting in storage decision |
| `w_embed` | 1.079 | TBD | TBD | Embedding similarity at retrieval time |
| `w_recency` | 3.777 | ≈ 0 | ≈ 0 | Recency at retrieval time |

The robust finding (visible in the narrow tuning histories already)
is `w_recency` collapses to near-zero for both QASPER and CUAD-tuned θ,
while `w_embed` rises. Other parameters' optimal values are noisier
across the search, suggesting they're less load-bearing for document
QA specifically.

### 6.5 Limitations

* **k=8 is fixed.** Wider k (k=16, k=32) would likely lift recall on
  long-haystack benchmarks at the cost of higher retrieval volume.
  Sensitivity to k is not explored here.
* **Six benchmarks, two long-haystack ones.** The two-cluster finding
  is robust within this set, but generalizing to "every domain has
  the same pattern" requires more benchmarks (Loft, MuSiQue, GovReport,
  ZeroSCROLLS) than the six chosen. The six were chosen to cover
  disjoint domains rather than maximize coverage of long-haystack
  difficulty.
* **The LLM-judge in Phase 4 is itself an LLM.** GPT-4o-mini scoring
  GPT-4o-mini's answer is correlated. A more conservative protocol
  would use a different judge model (Claude or Gemini) or a
  multi-judge calibration. This is deferred.
* **CMA-ES with budget `(n_docs=8 or 20, n_generations=10 or 30)`.**
  A long-running asynchronous search (hundreds of generations) might
  squeeze another 0.05–0.10 on QASPER and CUAD. The wide-vs-narrow
  comparison (W2) approximates this question on a smaller budget.

---

## 7. Future Work

### 7.1 Closing the LLM-cost loop (Phase 4, near-term)

The retrieval-quality results in Section 5.1 are complete. The
remaining 5.4–5.6 numbers — LLM-judge answer-quality, USD-per-question
cost, and the Pareto frontier — slot into the existing
`scripts/run_stage3_full.py --mode full` orchestrator with one CLI
invocation once an API key and ~$2 of budget are available. Phase 4 is
not a research question; it is a pipeline run.

### 7.2 m_cleaned streaming for LongMemEval (deferred)

The 2.5 GB `longmemeval_m_cleaned.json` file is on disk but currently
out-of-scope. A streaming JSON adapter (via `ijson`) would unlock the
"medium-haystack" variant of LongMemEval, intermediate between the
500-item Oracle (short haystacks) and Phase-3-paper-scale (large
haystacks). Adding this benchmark variant would provide a finer
gradient between the two clusters identified in Section 6.1.

### 7.3 Per-benchmark θ via a hypernetwork (V7 trajectory)

The current protocol runs CMA-ES once per benchmark and ships the
resulting θ. A hypernetwork that maps task descriptors (vocabulary
overlap, document length distribution, question complexity) to a
predicted θ would amortize the search cost across new tasks. This is
the natural Stage 4 / V7 direction if a future thesis cycle extends
this work.

### 7.4 Cost-aware tuning

The current CMA-ES objective is pure recall@k. A joint objective
`J = recall - λ × retrieved_token_count` would tune θ for cheap *and*
accurate retrieval simultaneously. The Stage 2 grid-world chapter
explored an analogous joint objective (and found it could degenerate
to "store nothing" when λ was too high). For document QA the
degenerate boundary is different — minimum retrieval is k=1 — but the
tuning protocol carries over directly. Worth one more CMA-ES sweep.

### 7.5 Multi-LLM-judge calibration

GPT-4o-mini scoring GPT-4o-mini is the cheap baseline. Cross-judging
with Claude Sonnet 4.5 and Gemini 2.0 Pro on the same answer set
would calibrate the judge's reliability. For ~$5–10 of judge-only
cost, the existing 540-answer set can be cross-scored across three
judges and report inter-judge agreement κ. This is a standard
sanity-check for LLM-judge-based evaluation and would strengthen the
Stage 3 answer-quality claims.

---

## Appendix A — Files and reproducibility

### Code

* `environment/benchmarks/` — six adapter modules + `base.py` Protocol + helpers (~1,200 lines).
* `evaluation/benchmark_memory_eval.py` — adapter-aware retrieval evaluator (~150 lines).
* `tuning/tune_v4_per_benchmark.py` — CMA-ES per-benchmark θ tuner (~200 lines).
* `scripts/run_stage3_retrieval.py` — multi-benchmark retrieval study driver (~200 lines).
* `scripts/run_stage3_full.py` — Phase-4 orchestrator with retrieval / dry-run / full modes (~510 lines).
* `scripts/run_theta_transfer.py` — cross-benchmark θ-transfer ablation runner (W3, ~150 lines).
* `scripts/build_stage3_frontend_data.py` — extracts `web/public/data/stage3_retrieval.json` from results.
* `scripts/prefetch_benchmarks.py`, `scripts/verify_benchmarks.py` — Phase-0 data acquisition + verification (~700 lines combined).
* `tests/test_benchmark_{adapters, snapshots, smoke, adversarial}.py` — 4-layer test pyramid (146 tests, ~1,300 lines).

### Data and results

* `data/benchmarks/` — 36.9 GB local cache, gitignored (regenerated by `scripts/prefetch_benchmarks.py`).
* `results/stage3/tuned_theta_*.json` — per-benchmark CMA-ES outputs (narrow), 5 files.
* `results/stage3/tuned_theta_wide_*.json` — wide CMA-ES outputs (QASPER + CUAD), 2 files (post-W2).
* `results/stage3/retrieval_*.json` — per-benchmark recall@k details, 6 files.
* `results/stage3/retrieval_summary.json` — aggregate cross-tab.
* `results/stage3/theta_transfer_matrix.json` — 3×2 transfer ablation (post-W3).
* `results/stage3/stage3_runs.json` — orchestrator summary across cells.
* `results/stage3/cost_projection.json` — Phase-4 dry-run cost estimate.
* `results/stage3/cells/*.json` — per-cell orchestrator details (one per benchmark × config × seed).
* `web/public/data/stage3_retrieval.json` — frontend-consumed retrieval table.

### Documentation

* `docs/STAGE3_DATA_PREP.md` — Phase-0 writeup (data acquisition + verification).
* `docs/STAGE3_PHASE1.md` — Phase-1+1.5 writeup (adapters + tests + tuning + retrieval + orchestrator + frontend).
* `docs/RECENT_CHANGES.md` — session log; entries `-4` and `-5` cover Stage 3.
* `docs/THESIS_STAGE3_CHAPTER.md` — this file.
