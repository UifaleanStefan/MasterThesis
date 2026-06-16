# Learnable, Task-Adaptive Structured Memory for LLM Agents
### Master's Thesis — Overview, Abstract & Synthesis

**Author:** Stefan Uifalean · Bocconi University
**Compiled:** June 2026 (post-remediation honest synthesis)
**Repo:** https://github.com/UifaleanStefan/MasterThesis

> This document is the **master spine** of the thesis. It states the abstract,
> ties the three research stages into one argument, and gives the honest
> synthesis. Full detail lives in the stage documents it points to. Where an
> older document and this one disagree, **this document and
> `PROFESSOR_MEMO_2026_06_12.md` are authoritative.**

---

## Abstract

Memory in LLM-based agents is almost always *fixed*: a context window of a
given size, a RAG index with a fixed chunking and retrieval rule, or a vector
store that treats every event alike. Yet different tasks demand different memory
structures — a multi-hop question needs entity relations, a navigation task
needs spatial transitions, an end-of-corpus question needs recency-independent
recall. This thesis asks whether an agent can **learn how to construct its own
memory** — *what to store, which concepts to track as nodes, and how to score
retrieval* — and whether that learned structure is **task-dependent**.

We parameterize memory construction by a vector **θ** (storage probability,
entity-creation threshold, temporal-edge probability, and learnable retrieval
weights `w_graph / w_embed / w_recency`, up to 10 dimensions in the most
complete system, GraphMemoryV4) and optimize θ — not the policy, not the value
function — by black-box search (Evolution Strategy, then CMA-ES). The work
proceeds in three stages. **Stage 1** (grid POC) shows the optimizer recovers
*different* θ for different tasks and that adaptive θ beats the fixed-memory
baseline (e.g. MultiHopKeyDoor success 2.5%→27.5%). **Stage 2** scales to a
12-memory-system × 4-environment benchmark with ablation, zero-shot θ-transfer,
and sensitivity analysis; the 10-D GraphMemoryV4 reaches the top performance
cluster, and θ tuned for one long-haystack task transfers to another but *not*
across task families. **Stage 3** moves to real LLMs (gpt-4o-mini answerer)
doing corpus-cumulative question answering over **six** real long-context
benchmarks (FinanceBench, QASPER, CUAD, HotpotQA, LongMemEval, NarrativeQA),
with a corpus-mode CMA-ES re-tuning of θ and an independent Claude
cross-vendor judge.

The central Stage-3 finding, stated honestly after an adversarial self-audit:
**corpus-cumulative tuning shifts θ toward recency-independent, embedding-driven
retrieval, and this produces a real end-of-corpus ("batch") accuracy lift that
survives on held-out questions for FinanceBench (+0.335, p_holm < 0.0001), CUAD
(+0.135), HotpotQA (+0.540) and LongMemEval (+0.165), but is not significant for
QASPER.** The advantage is specifically a *batch / end-of-corpus* effect (when
no recency cue helps); in *online* mode, where the queried document is freshly
ingested, the recency-heavy canonical θ is competitive or better. A
full-context "dump-all" baseline, once a truncation bug was fixed, is
**statistically tied with selective retrieval on accuracy** but ~18× more
expensive — so the case for learned selective memory at this scale is
*efficiency*, not an accuracy cliff.

---

## 1. Contribution

The contribution is **not** a better RL policy or a new LLM. It is the claim and
demonstration that **memory construction is itself a learnable, task-adaptive
object**: a small parameter vector θ governs storage/abstraction/retrieval, the
optimal θ differs by task, and θ can be discovered automatically. The thesis
substantiates this at three increasing levels of realism (toy grid → benchmark
suite → real-LLM long-context QA) and — unusually for a student project —
subjects its own headline results to an adversarial audit and reports the
corrected numbers (§5, §6).

---

## 2. Stage 1 — Proof of concept (grid worlds)

A NetworkX graph memory (event nodes, entity nodes, temporal + mention edges)
with θ = (θ_store, θ_entity, θ_temporal). Default θ = (1, 0, 1) reproduces the
fixed-memory baseline. An Evolution Strategy (12 gen × 6 candidates × 40
episodes) optimizes mean reward.

| Environment | Learned θ (store, entity, temporal) | Baseline success | Learned success |
|---|---|---:|---:|
| Key-Door | (0.116, 0.000, 0.819) | 17.5% | 30.0% |
| Goal-Room | (1.000, 0.220, 1.000) | 70.0% | 80.0% |
| MultiHop-KeyDoor | (1.000, 0.487, 0.843) | 2.5% | **27.5%** |

The learned θ vectors are **visibly different per task** (Key-Door discards most
events but keeps temporal order; Goal-Room stores everything) — the first,
cleanest evidence that optimal memory structure is task-dependent. Detail:
`docs/THESIS_STORY.md`, `docs/POC_RESULTS.md`.

---

## 3. Stage 2 — Benchmark, ablation, transfer (12 systems × 4 environments)

GraphMemory grows to V4 (10-D θ adding learnable retrieval weights, learned
importance scoring, and Bayesian entity decay), tuned by CMA-ES (30 gen × 50
episodes). Against 11 other memory systems on the hardest grid task
(MultiHopKeyDoor), V4 **reaches the top performance cluster** (mean reward 0.178
vs the prior best EpisodicSemantic 0.173, +75% over its own untuned start).

> **Honest caveat:** 0.178 vs 0.173 is a **statistical tie**, not a clean "#1" —
> a single-seed point estimate, and EpisodicSemantic's 95% bootstrap CI
> [0.120, 0.220] contains 0.178; the ranking also does not survive a MiniLM
> retrieval backend. The defensible claim is "the 10-D parameterization reaches
> the top cluster," not "uniquely best." (Corrected June-12;
> `docs/GRAPHMEMORY_V4_RESULTS.md`.)

Ablation confirms each θ component carries load on at least one task; the
zero-shot **θ-transfer** experiment shows task-tuned θ generalizes *within* a
task family (long-haystack QA) but transfers ~0% across families
(grid → document QA) — i.e. memory structure is task-dependent, which is the
thesis claim, observed rather than assumed. Detail:
`docs/BENCHMARK_RESULTS.md`, `docs/ABLATION_RESULTS.md`, `docs/TRANSFER_RESULTS.md`.

---

## 4. Stage 3 — Real LLMs on six long-context benchmarks

Corpus-cumulative QA: documents are ingested one at a time into the V4ₜ memory;
questions are answered *online* (right after the source doc) and *batch* (at
end-of-corpus, against the full accumulated memory, k=8 selective retrieval). A
gpt-4o-mini answerer is scored by an independent **Claude** cross-vendor judge
on a 5-point rubric. θ is re-tuned per benchmark on the corpus-cumulative
recall objective (CMA-ES, no LLM in the loop).

**End-of-corpus (batch) judge means, corpus-tuned vs canonical θ:**

| Benchmark | canonical | corpus-tuned | lift | held-out significance |
|---|---:|---:|---:|---|
| FinanceBench | 0.243 | 0.645 | **+0.402** | survives, p_holm < 0.0001 |
| HotpotQA (n=100) | 0.215 | 0.755 | **+0.540** | survives |
| QASPER | 0.250 | 0.415 | +0.165 | **n.s. after Holm** |
| CUAD | 0.023 | 0.184 | +0.161 | survives (+0.135) |
| LongMemEval (n=100) | 0.165 | 0.330 | +0.165 | survives |
| NarrativeQA | 0.400 | 0.400 | 0.000 | tuning objective undefined¹ |

The mechanism is consistent across benchmarks: corpus tuning drives `w_recency`
toward 0 and `w_embed` up, producing a memory that does **not** depend on the
queried document being recent — exactly what end-of-corpus QA needs. In *online*
mode (fresh document) the recency-heavy canonical θ is competitive or better, so
the lift is genuinely a batch-mode effect, not a uniform win.

Baselines and an upper bound were run in the same harness: dense (RAG/MiniLM)
and sparse (BM25) retrieval are strong non-tuned baselines; the full-context
**dump-all** upper bound, after a truncation-cap bug was fixed, scores
0.607–0.689 on FinanceBench — **statistically tied with corpus-tuned on
accuracy** but at ~18× the cost. Full detail, statistics, and the interactive
dashboard data: `docs/THESIS_STAGE3_CHAPTER.md`.

¹ NarrativeQA's adapter emits no paragraph-level gold, so the recall@k tuning
objective is undefined by construction (not a convergence failure); "corpus
-tuned" there falls back to canonical θ.

---

## 5. Conclusion

Across three stages of increasing realism, the same claim holds: **what an agent
stores and how it scores retrieval can be learned, and the learned optimum is
task-dependent.** In the grids, the optimizer recovers visibly different θ per
task and beats fixed memory on the hardest one (2.5%→27.5%). On the benchmark
suite, a 10-D learnable memory reaches the top cluster and task-tuned θ
generalizes within but not across task families. On six real long-context LLM
benchmarks, re-tuning θ on the corpus-cumulative objective yields a
recency-independent memory whose end-of-corpus accuracy lift **survives on
held-out questions for four of six benchmarks**, is honestly reported as
non-significant on QASPER, and is undefined-by-construction on NarrativeQA.

The most important methodological lesson is the one the thesis applies to
itself: a fixed full-context baseline ("dump everything") is, once measured
correctly, *accuracy-competitive* with learned selective memory — so the honest
argument for learnable task-adaptive memory at this corpus scale is **cost and
scalability** (it matches accuracy at ~1/18th the token spend, and unlike
dump-all it does not break when the corpus exceeds the context window), not a
claim that selective retrieval is uniquely accurate.

---

## 6. Limitations & future work

- **Statistical power.** Most Stage-3 corpus cells are single-seed; HotpotQA and
  LongMemEval were re-run at 100-document scale, but multi-seed replication of
  the larger benchmarks is future work (corpus-mode online/batch eval is
  near-deterministic at temperature 0, so the gain would be CI tightening, not
  point-estimate change).
- **Corpus scope.** CUAD currently evaluates a subset of its 510 contracts;
  scaling the contract count is in progress.
- **Judge.** Content answers are judged one-by-one by Claude; refusal /
  acknowledgment answers keep rule-assisted scores, disclosed and validated on a
  300-entry sample (population-weighted error 0.028). A second independent judge
  and Cohen's κ remain future work.
- **Dump-all Protocol-B calibration** was not re-run uncapped (disclosed; the
  headline rests on the re-run Protocol-A numbers).
- **Domain incoherence.** HotpotQA/LongMemEval were pre-registered as
  domain-incoherent controls; their batch lift is reported as what the controls
  show, not as clean wins.

---

## 7. Honesty & reproducibility

An adversarial self-audit (June 12, 2026) of the Stage-3 evaluation found five
material issues and corrected all of them in the data and the write-up; see
`docs/PROFESSOR_MEMO_2026_06_12.md`. Provenance is machine-checked:
`python scripts/audit_judge_provenance.py` verifies all judge lines carry Claude
provenance, 0 duplicate qids, queue↔results parity for every cell, and discloses
the rule-assisted refusal/ack share. `python -m pytest tests/ -q` (214 tests)
and `python scripts/audit_determinism.py` both pass. All experiments are
seeded; result JSONs carry `_manifest` siblings with git SHA, embedding backend,
and dataset fingerprints.

**Document map:** core narrative `THESIS_STORY.md` (Stages 1–2) · Stage-3
chapter `THESIS_STAGE3_CHAPTER.md` · honest numbers `PROFESSOR_MEMO_2026_06_12.md`
· results detail `BENCHMARK_RESULTS.md`, `ABLATION_RESULTS.md`,
`TRANSFER_RESULTS.md`, `GRAPHMEMORY_V4_RESULTS.md`, `NEURAL_CONTROLLER_V2_RESULTS.md`.
