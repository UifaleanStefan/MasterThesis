# Defense Study Curriculum — 10 Lessons

A structured path through *everything* in the thesis, ordered so each lesson builds
on the last. Study one lesson at a time; at the end of each you should be able to
answer its **self-test** out loud and survive its **examiner traps**.

**How to use this with me (Claude):** say *"teach me Lesson N"* and I'll deliver the
full lecture, take your questions, and quiz you. Say *"next"* to advance. Say
*"drill me"* for rapid-fire Q&A on any lesson.

**Companion docs to keep open:**
- [SUBMITTED_THESIS_STUDY_GUIDE.md](docs/SUBMITTED_THESIS_STUDY_GUIDE.md) — the submitted version, chapter by chapter
- [SUBMITTED_VS_CURRENT.md](docs/SUBMITTED_VS_CURRENT.md) — what changed after submission
- Thesis chapters: [01 intro](thesis/chapters/01_introduction.tex) … [06 conclusion](thesis/chapters/06_conclusion.tex)
- The defense deck + speaker notes: [docs/defense_deck/](docs/defense_deck/)

---

## The arc in one paragraph

An agent's memory today is **hand-designed and frozen**. This thesis makes memory
construction itself a small **learnable vector θ** — what to store, how to abstract,
how to score retrieval — and optimizes θ with **black-box search** (no LLM in the
loop). Across three stages of rising realism (grid worlds → a 12-system benchmark →
six real LLM corpora) it shows two things: (RQ1) memory construction **can** be
learned, and (RQ2) the optimum is genuinely **task-dependent**. Measured honestly,
at corpus scale the win is **efficiency and graceful scaling**, not free accuracy.

---

## Lesson 1 — The Problem and the Big Idea
**Goal: explain *why* this thesis exists in 90 seconds.**

- Core ideas: memory in LLM agents = store + abstract + retrieve, all hand-designed
  and frozen. The recency trap (the motivating 500-contract example). Why a rule
  optimal for one task is harmful for another → **task-dependence**. The reframe:
  make memory construction a *learnable object* θ, optimize θ only (not the LLM).
- The two research questions (RQ1 learnable? RQ2 task-dependent?).
- Numbers to know: θ has up to **10 dimensions**; we optimize θ, never the LLM weights.
- Sources: [01_introduction.tex](thesis/chapters/01_introduction.tex); deck slides 1–4.
- Self-test: State both RQs. Give the motivating example. Why isn't this "just RAG"?
- Examiner trap: "Isn't a learnable retrieval weight trivial?" → it also decides
  *what to store* and *how to abstract*; and the finding is the task-dependence, not the mechanism.

## Lesson 2 — Anatomy of θ and GraphMemoryV4
**Goal: name all 10 parameters and what each controls.**

- The three groups: **storage** (θ_store, θ_novel, θ_surprise, θ_erich),
  **abstraction** (θ_entity, θ_temporal, θ_decay), **retrieval** (w_graph, w_embed, w_recency).
- The storage gate: store iff importance = θ_novel·nov + θ_erich·erich + θ_surprise·surp > θ_store.
- The memory graph: events as nodes, entities/edges from abstraction params.
- Numbers to know: 4 store + 3 abstract + 3 retrieve = 10; V1→V5 progression (what each added).
- Sources: [memory/graph_memory_v4.py](memory/graph_memory_v4.py); [03_methodology.tex](thesis/chapters/03_methodology.tex); `project_theta_progression` memory.
- Self-test: Which parameters have *retrieval leverage*? Write the storage rule from memory.
- Examiner trap: "Algorithm 2 vs code" — know the store rule matches `graph_memory_v4.py`
  (this was a corrected defect; θ_store is a threshold, θ_erich is in the sum).

## Lesson 3 — Retrieval as a Weighted Vote
**Goal: derive the retrieval score and explain each term.**

- score(item) = w_graph·g + w_embed·cos + w_recency·ρ; keep top-k (k=8).
- The three signals: graph link (0/1), embedding similarity (cos rescaled to [0,1]),
  recency ρ = 1/(1+Δsteps). θ sets how loud each votes.
- Why recency is the "trap"; what tuning does to the weights.
- Numbers to know: mechanism **w_recency 3.78→0.003, w_embed 1.08→2.63** under corpus tuning.
- Sources: [memory/retrieval.py](memory/retrieval.py) `retrieve_events_learnable`; deck slide 5.
- Self-test: Walk the contract example through the three votes. Why does tuning help end-of-corpus Qs?
- Examiner trap: "Is w_graph always inert?" → inert *as a retrieval term on our benchmarks*;
  the graph still scaffolds storage/abstraction; multi-hop tasks weren't stressed (Lesson 8/10).

## Lesson 4 — How θ Is Learned (the optimizer)
**Goal: justify black-box search and the recall@8 objective.**

- Evolution Strategy → CMA-ES. Why derivative-free: the store decision is **discrete**.
- The objective is **recall@8 of the gold evidence** — uses **no LLM**. Two payoffs:
  cheap tuning, and it *cannot be biased by the judge* that later scores answers.
- Numbers to know: recall@8; recall↔judge correlation **ρ=0.69** (validates the objective).
- Sources: [optimization/cma_es.py](optimization/cma_es.py); [tuning/tune_v4t_corpus.py](tuning/tune_v4t_corpus.py); [03_methodology.tex](thesis/chapters/03_methodology.tex).
- Self-test: Why not gradient descent? Why is an LLM-free objective methodologically important?
- Examiner trap: "You tuned on the test set?" → tuned on recall@k of gold evidence, held-out
  splits confirm the lift (Lesson 8); the judge never enters tuning.

## Lesson 5 — Stage 1: Grid Worlds (the task-dependence proof)
**Goal: use Stage 1 as the cleanest evidence for RQ2.**

- Environments (Key-Door, Goal-Room, MultiHopKeyDoor). Tune a small θ per task.
- The headline is the **difference between the recovered vectors**, not any single score:
  Key-Door discards but keeps order; Goal-Room stores everything.
- Numbers to know: hardest task **2.5% → 27.5%**; single-seed proof of concept (stated).
- Sources: [04_results.tex](thesis/chapters/04_results.tex) Stage 1; deck slide 7.
- Self-test: Why is "different vectors" stronger evidence for RQ2 than "higher score"?
- Examiner trap: "n=1 seed" → yes, proof of concept; determinism at T=0 measured later (Lesson 9).

## Lesson 6 — Stage 2: 12-System Benchmark, Ablation, Transfer
**Goal: say what works, *why*, and what doesn't.**

- Competitive **top-cluster tie** (0.178 vs 0.173, TF-IDF backend — backend-sensitive).
- Ablation: **θ_novel is load-bearing** (zero it → reward collapses); **w_graph inert at retrieval**.
- Transfer matrix (within vs across task families). Neural controller (~2k params) **matches, doesn't beat**.
- Numbers to know: 0.178 vs 0.173; neural MLP 50→32→10 ≈1,962 params, best fitness 0.233.
- Sources: [04_results.tex](thesis/chapters/04_results.tex) Stage 2; [C_full_tables.tex](thesis/appendices/C_full_tables.tex); deck slide 8.
- Self-test: State the ablation result and what it implies about the architecture.
- Examiner trap: "0.178 is your headline?" → it's a *tie*, reported as competitive; and it's
  from the TF-IDF backend (MiniLM shifts absolutes) — disclosed as backend-sensitive.

## Lesson 7 — Stage 3 Setup: Real LLMs and the Judge
**Goal: defend the evaluation design and the judging.**

- Corpus-cumulative QA; questions asked **end-of-corpus**. Answerer **gpt-4o-mini** (T=0),
  encoder **MiniLM**, judge **Claude, 1-by-1, 5-point rubric, cross-vendor**.
- Six benchmarks **pre-registered into 3 groups**: confirmatory (FB/CUAD/QASPER),
  controls (HotpotQA/LongMemEval), undefined (NarrativeQA, no gold).
- Judge integrity: provenance audit, refusal/ack classifier (validated), κ self-consistency.
- Numbers to know: 5-point set {0,.25,.5,.75,1}; κ=0.66; refusal-classifier pop-weighted error ≈0.028.
- Sources: [evaluation/](evaluation/); `scripts/audit_judge_provenance.py`; deck slide 6/8.
- Self-test: Why cross-vendor judging? Why pre-register the benchmark groups?
- Examiner trap: "Claude grading = circular?" → cross-vendor (answerer is OpenAI); κ is a
  *self-consistency* measure and we say so (no human/second-vendor judge — a stated limitation).

## Lesson 8 — Stage 3 Results and the Honest Reframe
**Goal: state the real finding, including what you *don't* claim.**

- Corpus-tuning lift on all three coherent benchmarks (FB 0.243→0.645, CUAD 0.028→0.172,
  QASPER 0.250→0.415, batch end-of-corpus). Held-out: **2 of 3 survive** Holm correction
  (FB, CUAD; QASPER n.s.). The mechanism (recency↓, embed↑).
- The audit twist: a fixed **dump-all** baseline is **statistically tied** on accuracy →
  the claim is **efficiency, not accuracy**: ~**18× cheaper**, dump-all overflows at **N≈11**,
  selective prompt flat at ~**704 tokens**.
- Numbers to know: the six numbers above; ρ=0.69; 2/3 held-out.
- Sources: [04_results.tex](thesis/chapters/04_results.tex) / [05_discussion.tex](thesis/chapters/05_discussion.tex); deck slides 8–9.
- Self-test: What is the honest headline claim in one sentence? Why is efficiency the real win?
- Examiner trap: "So it's not more accurate?" → correct, and we say so; the case is cost + it
  *structurally* runs where full-context can't.

## Lesson 9 — The Post-Submission Program
**Goal: know what's new since the submitted PDF and the statistics behind it.**

- Fair **corpus-tuned baselines** (tuned BM25) → benchmark-dependent win/tie/lose.
- **Head-to-heads** (HippoRAG, MemGPT/Letta) under the same harness; the **scale reshuffle**:
  @50 HippoRAG>V4t>Letta; @510 Letta>V4t>HippoRAG (all Holm-significant).
- **Full-corpus scaling** (CUAD 510, QASPER 281). **θ-predictability** (LOBO, ~0.80 lift recovered).
  **Multi-seed determinism** (std ≤0.012 across seeds {7,42,100}).
- Stats toolkit: bootstrap/cluster CI, Wilcoxon, Holm correction, Cohen's d.
- Sources: [SUBMITTED_VS_CURRENT.md](docs/SUBMITTED_VS_CURRENT.md); [evaluation/statistics.py](evaluation/statistics.py).
- Self-test: Explain the scale reshuffle and why it matters. What did θ-predictability show?
- Examiner trap: "Cherry-picked scale?" → we report the reshuffle *both* ways with significance.

## Lesson 10 — Honesty, Limitations, and Defending It
**Goal: turn the self-audit into a strength and pre-empt every hard question.**

- The self-audit philosophy: ties reported as ties, negatives kept (w_graph), numbers
  re-verified against committed data. The submitted-vs-current story (45→68pp, more accurate).
- Limitations: single answerer/encoder; Claude-only judge (κ, not independent); w_graph scope;
  QASPER n.s.; NarrativeQA undefined objective.
- Future work: multi-hop tasks that force traversal; independent judges; more seeds.
- Sources: [06_conclusion.tex](thesis/chapters/06_conclusion.tex); Appendix B honesty audit; [SUBMITTED_VS_CURRENT.md](docs/SUBMITTED_VS_CURRENT.md).
- Self-test: List five limitations and your one-line answer to each.
- Examiner trap: any "gotcha" — the honest, scoped answer is already in the thesis; know it cold.

---

## Master self-test (know these cold before the defense)
1. Both RQs, verbatim-ish, and the answer to each.
2. All 10 θ parameters by group.
3. The retrieval score equation and the three signals.
4. Why the objective is recall@8 and LLM-free.
5. Stage 1's "different vectors" argument (2.5→27.5).
6. Stage 2 ablation (θ_novel load-bearing; w_graph inert) + the tie.
7. Stage 3 lift numbers + 2/3 held-out + the mechanism.
8. The honest reframe (dump-all tie → efficiency; 18×; N≈11; 704 tokens).
9. The scale reshuffle and multi-seed determinism.
10. Five limitations with crisp rebuttals.
