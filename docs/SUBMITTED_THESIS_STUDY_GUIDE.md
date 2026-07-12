# Study Guide — the SUBMITTED thesis (commit 268845d, 45 pp)

> "Learnable, Task-Adaptive Structured Memory for LLM Agents" — Stefan Uifalean,
> MSc AI, Bocconi. This guide covers **only the version you handed in** (byte-identical
> to the sent PDF). Study this, not the expanded current repo version.

---

## 1. The one-sentence thesis
**Memory construction for an LLM agent — what to store, which concepts to track, and
how to score retrieval — can be made a small learnable parameter vector θ, optimized
by black-box search; the learned optimum is task-dependent; and, measured honestly, it
buys efficiency and graceful scaling rather than a free accuracy gain.**

The research question (memorize verbatim): *"Can an agent learn how to construct its own
memory — what to store, which concepts to track as nodes, and how to score retrieval —
and is the learned optimum task-dependent?"*

What it is **NOT**: not a new RL policy, not a new LLM. You optimize **θ only** — the
policy and the model weights are left untouched. That framing is your defense against
"isn't this just RAG tuning?"

## 2. The four contributions
1. **Formalizes memory construction as a parameter vector θ** (storage prob., entity/
   temporal thresholds, learnable retrieval weights w_graph/w_embed/w_recency — up to
   **10 dimensions** in the fullest system, GraphMemoryV4) and optimizes it by Evolution
   Strategy → CMA-ES.
2. **Shows task-dependence at three levels of realism** (Stages 1–3).
3. **Subjects its own headline to an adversarial self-audit** and reports corrected
   numbers — including that the "dump-all collapses" headline was a **bug**, corrected
   to an accuracy **tie**, reframing the contribution as **efficiency**, not accuracy.
4. (Implicit 4th) A validated, cross-vendor LLM-judge evaluation with held-out testing.

## 3. The three-stage arc (the spine — know this cold)

| Stage | Setting | What it proves | Headline number |
|---|---|---|---|
| **1** | Grid worlds (Key-Door, Goal-Room, MultiHopKeyDoor), 3-D θ, Evolution Strategy | The optimum is **task-dependent** (different θ per task) | Hardest task **2.5% → 27.5%** success |
| **2** | 12 memory systems × 4 environments; ablation, transfer, sensitivity; 10-D θ, CMA-ES | A learnable memory is **competitive**; characterizes what each parameter does | V4 **0.178** vs best baseline **0.173** = statistical **tie** (top cluster) |
| **3** | Real LLM (gpt-4o-mini) corpus-cumulative QA over **6** long-context benchmarks; Claude judge | Corpus tuning → **recency-independent** memory; end-of-corpus lift; efficiency vs dump-all | 2 of 3 domain-coherent benchmarks show a held-out lift |

### Stage 1 detail
- Learned θ = (θ_store, θ_entity, θ_temporal). The **point is the difference** between
  the recovered vectors: Key-Door **(0.116, 0.000, 0.819)** — discards most events but
  keeps temporal order; Goal-Room **(1.000, 0.220, 1.000)** — stores everything;
  MultiHopKeyDoor **(1.000, 0.487, 0.843)**.
- Success (baseline → learned): Key-Door 17.5→30, Goal-Room 70→80, MultiHopKeyDoor
  **2.5→27.5**. Report the hard one as a **+25-point absolute lift** (baseline near-zero,
  so a ratio would overstate). **Single-seed proof of concept** — say so.

### Stage 2 detail
- **V4 reaches the top cluster: 0.178 vs EpisodicSemantic 0.173** — a **statistical tie**,
  not a clean #1 (single-seed; EpisodicSemantic's 95% CI [0.120, 0.220] contains 0.178).
  Defensible claim = "reaches the top cluster, +75% over its own untuned start," **not**
  "uniquely best." ⚠️ *(This 0.178 is from the TF-IDF backend; under MiniLM the ranking
  shifts — a known backend-sensitivity flagged in the honesty audit. If pressed, concede
  it's a tie and backend-sensitive.)*
- **Ablation:** θ_novel is the **load-bearing** dimension (zeroing it collapses reward to
  0 — it gates the whole storage pipeline). The **graph-traversal term (w_graph) carries
  NO measurable load** — zeroing it barely changes reward. So the corpus-tuning signature
  is a **three-shift, not four-shift**; the graph term is a **storage scaffold, not a
  retrieval engine**. (This is your honest "negative result" — own it.)
- **Transfer:** strong within a task family (Goal-Room **0.69**, HardKeyDoor 0.17),
  **fails on a larger OOD task** (MegaQuestRoom **0.00**) — and the failure is task
  **scale**, not memory quality (retrieval there is still high-precision).
- **Sensitivity:** a **sharp peak** — reward collapses once θ_novel < ~0.3.
- **Neural controller** (~2,000-param MLP, same CMA-ES budget) **matches but does not
  exceed** the 10-scalar V4 → scalar θ is a strong, cheap baseline.

### Stage 3 detail — the heart of the thesis
Six benchmarks, ingested doc-by-doc; questions answered **online** (right after the
source doc) and **batch** (end-of-corpus). θ re-tuned per benchmark on **recall@8**
(no LLM in the tuning loop). Benchmarks are pre-partitioned into three types:

| Type | Benchmarks | Prediction |
|---|---|---|
| **Confirmatory** (domain-coherent) | FinanceBench, CUAD, QASPER | lift expected |
| **Controls** (domain-incoherent, pre-registered) | HotpotQA, LongMemEval | no lift predicted |
| **Undefined** | NarrativeQA | no paragraph gold → objective undefined |

**The headline table (batch judge means, canonical θ → corpus-tuned θ, held-out verdict):**

| Benchmark | n | canonical | corpus-tuned | lift | held-out |
|---|---|---|---|---|---|
| FinanceBench | 150 | 0.243 | **0.645** | +0.402 | **survives**, p_holm<10⁻⁴ |
| CUAD | 644 | 0.028 | 0.172 | +0.144 | **survives** (+0.135) |
| QASPER | 94 | 0.250 | 0.415 | +0.165 | **n.s.** after Holm |
| HotpotQA | 100 | 0.215 | 0.755 | +0.540 | (control) |
| LongMemEval | 100 | 0.165 | 0.330 | +0.165 | (control) |
| NarrativeQA | 10 | 0.400 | 0.400 | 0.000 | undefined |

**Lead claim: "two of three domain-coherent benchmarks (FinanceBench, CUAD) show a
significant held-out lift; QASPER does not."** Do NOT say "four of six" — that would
launder the controls into wins.

**The mechanism ("corpus-tuning signature"):** tuning drives **w_recency → 0** and
**w_embed up** (FB: w_rec 3.78→0.003, w_embed 1.08→2.63, θ_store 0.29→0.01). The memory
stops depending on the queried doc being recent and retrieves by **semantic match** —
exactly what end-of-corpus QA needs (the answer was seen long ago, so recency fetches
the *wrong* recent doc). This is a **batch-mode effect**: in online mode the recency-heavy
canonical θ is competitive.

**Construct validity (why the recall@8 tuning objective is legitimate):** join per-question
recall@8 with judge score over **n=2,170** questions → pooled **Spearman ρ=0.69**;
retrieved-gold questions score **0.62** vs **0.10** when missed. QASPER is the weakest link
(ρ=0.39) — consistent with its being the coherent benchmark that fails Holm.

**Dump-all — the honesty centerpiece:** a full-context "dump everything" baseline. On FB
it scores **0.689 online / 0.607 batch — statistically tied** with corpus-tuned
(0.678/0.645, CIs overlap) but at **~18× the token cost** ($2.68 vs $0.15 per 150 q). So
the value of learned memory here is **efficiency + scalability, not raw accuracy.**

**Scalability (the structural argument):** each CUAD contract ≈ 10.9K tokens, so dump-all's
prompt grows O(N) and **exceeds gpt-4o-mini's 128K window at N≈11 contracts**; at the full
N=510 it would need **5.55M tokens (~43×)**, forcing truncation. Selective retrieval (k=8)
stays flat at **~704 tokens** regardless of N. Beyond ~11 docs, retrieving the right k
items is a **necessity, not an optimization**.

## 4. Methodology you must be able to explain
- **θ (10-D for V4):** θ_store (storage bias), θ_novel (novelty weight — load-bearing),
  θ_surprise, θ_erich (importance/enrichment), θ_entity (entity-node threshold),
  θ_temporal (temporal-edge prob.), θ_decay, **w_graph, w_embed, w_recency** (retrieval).
- **Retrieval score:** `s(item,q) = w_graph·g(item,q) + w_embed·cos(e_item,e_q) + w_recency·ρ(item)` — a linear combination; retrieve top-**k=8**.
- **Optimizers:** Evolution Strategy for the small Stage-1 search; **CMA-ES** for Stages 2–3
  (handles noisy, non-separable, low-dim objectives; covariance adaptation). Stage-2 budget:
  **30 generations × 50 episodes/candidate**. **CMA-ES objective = pure recall@k — no LLM
  in the tuning loop** (this is why tuning is cheap and unbiased by the judge).
- **Answerer:** gpt-4o-mini, **temperature 0**, max 150 answer tokens. **Encoder:** MiniLM
  (all-MiniLM-L6-v2). **Judge:** a **Claude** model scoring 1-by-1 on a 5-point rubric
  {0, 0.25, 0.5, 0.75, 1.0} — **cross-vendor** (judge ≠ answerer) to avoid self-bias.
- **Stats:** bootstrap 95% CIs, held-out train/test split, Holm correction across the
  benchmark family.

## 5. The self-audit — your single biggest differentiator
The thesis **audits itself and reports the corrected (worse) numbers.** Rehearse these:
- **Dump-all "collapse" was a truncation bug** (a 12-event cap). Fixed → dump-all is an
  accuracy **tie**, not a collapse → contribution reframed to **efficiency**.
- **Stage-2 "#1" was an overclaim** → demoted to a statistical **tie** (0.178 vs 0.173).
- **"Four-shift" → "three-shift"**: the graph term (w_graph) was shown inert.
- **Judge reliability:** an independent blind second Claude pass on 180 questions →
  quadratic-weighted **Cohen's κ = 0.66** ("substantial"). Honestly framed as **judge
  self-consistency**, NOT human/cross-vendor agreement (both passes are Claude-class).
- **Refusal/ack answers** carry validated rule-assisted scores (population-weighted error
  **0.028** on a 300-entry hand-check), not 1-by-1 judgments.
If an examiner attacks a number, the winning move is: "yes — and we flag exactly that in
the self-audit; here's the corrected reading." You pre-empted the critique.

## 6. Numbers cheat-sheet (memorize)
- Grid hard task: **2.5% → 27.5%** (+25 pts).
- Stage-2 tie: **0.178 vs 0.173**; neural MLP ≈ 2,000 params, matches not exceeds.
- FB: **0.243 → 0.645** (+0.402); held-out **+0.335**, p_holm<10⁻⁴. CUAD: **0.028 → 0.172**
  (+0.144); held-out +0.135. QASPER: 0.250 → 0.415, **n.s.**
- Signature: **w_rec 3.78→0.003, w_embed 1.08→2.63, θ_store 0.29→0.01**.
- Construct validity: **ρ=0.69**, n=2,170, retrieved 0.62 vs missed 0.10.
- Dump-all: tied on FB (0.607–0.689), **~18×** cost; overflows at **N≈11**, **43×**/5.55M
  tokens at N=510; selective flat at **704 tokens**.
- κ = **0.66** (self-consistency); refusal error **0.028**; tuning ≈ **$18.5**.

## 7. Limitations (they WILL ask — have all six ready)
1. Stage-1 single-seed; several Stage-3 cells single-seed (near-deterministic at T=0, so
   the gap is CI-tightening not point-estimate risk).
2. CUAD headline at **50 of 510** contracts; auxiliary baselines at 10-contract pilot scale.
3. Single (cross-vendor) Claude judge; κ bounds **self-consistency**, not human agreement.
4. Refusal/ack answers rule-assisted, not 1-by-1 judged.
5. Single answerer (gpt-4o-mini) + single encoder (MiniLM) — no model/encoder generalization.
6. (Stage 2) Top-cluster ranking is backend-sensitive (TF-IDF vs MiniLM).

## 8. Future work (know the list)
(i) multi-seed replication of large Stage-3 cells; (ii) scaling all benchmarks to full
corpora + all configs; (iii) an independent/second-vendor or human judge with κ;
(iv) revisiting the neural controller with a bigger budget; (v) stronger answerer models;
(vi) making θ **adaptive within an episode** (per-step controller) — motivated by the
MegaQuestRoom OOD failure where a fixed θ breaks.

## 9. Likely examiner questions → crisp answers
- **"Isn't this just hyperparameter tuning of RAG?"** — We tune the *construction policy*
  (storage + abstraction + retrieval scoring) as one vector, per task, by derivative-free
  search with no reader in the loop, and we *measure* task-dependence and transfer. Prior
  work fixes these or learns them by gradient descent entangled with a specific reader.
- **"Your accuracy gains vanish against dump-all — so what's the point?"** — Correct, and we
  say so ourselves. The point at this scale is efficiency (~18× cheaper) and **scalability**:
  dump-all *cannot run* past ~11 CUAD contracts; selective memory is a necessity, not a
  luxury, and θ tunes how good that necessity is.
- **"Why does the lift fail on QASPER?"** — QASPER is paraphrase-heavy information-seeking;
  it's also the weakest recall↔judge link (ρ=0.39). We report it as non-significant rather
  than bury it — that's the 2-of-3 honesty.
- **"One judge — how do you trust the scores?"** — Cross-vendor (Claude judging gpt-4o-mini)
  removes self-bias; κ=0.66 substantial; and the tuning objective (recall@8) is LLM-free and
  correlates with judge score at ρ=0.69, so the causal chain doesn't depend on the judge alone.
- **"Single seed?"** — Corpus-mode is near-deterministic at T=0; the FB cell we did replicate
  has tiny cross-seed variance, so remaining single-seed cells are a CI-width issue.
- **"What's genuinely new vs MemGPT/HippoRAG?"** — Those fix their construction policy by
  hand; we make it a small optimized, per-task-refittable object, and we measure that no
  fixed policy is best across corpora.

## 10. Where to look in the repo while studying
- Chapters: `thesis/chapters/0{1..6}.tex` (but note: the *current* repo files are the
  EXPANDED version — for the submitted text use `git show 268845d:thesis/chapters/…`).
- The sent PDF: `Uifalean_Stefan_3310360_MSc_AI_Thesis_submission/…Thesis.pdf`.
- Data behind the numbers: `results/*.json`, `results/stage3/*.json`.
- The self-audit appendix: `thesis/appendices/B_honesty_audit.tex`.
</content>
