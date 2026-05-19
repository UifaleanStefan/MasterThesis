# Recent Changes — Session Log

**Purpose:** Record of work done on the thesis codebase (analysis, fixes, dashboard, figures).
**Last updated:** May 2026

---

## -6. Stage 3 Phase 1.6 — wider tuning, cross-benchmark theta transfer, thesis chapter draft (May 2026)

Three short post-Phase-1.5 threads that compound the headline result.

**Wider CMA-ES tuning** (`tuning/tune_v4_per_benchmark.py` gains `--out-suffix`):
ran QASPER + CUAD with `n_docs=20, n_generations=30` (vs the narrow Phase-1.5 defaults of 8 and 10).
QASPER lifted further: 0.464 -> **0.576** (+0.11 over narrow). CUAD marginally regressed: 0.687 -> 0.671
(CMA-ES non-convexity; narrow's smaller eval ensemble happened to land a slightly better candidate).
`scripts/compare_theta_widths.py` records the comparison in `theta_width_comparison.json` and flags
which width is preferred per benchmark. The transfer experiment downstream uses the better of
(narrow, wide) per benchmark — wide-QASPER + narrow-CUAD.

**Cross-benchmark theta-transfer ablation** (`scripts/run_theta_transfer.py`) — and the empirical
hypothesis got overruled. Built a 3 x 2 matrix to test whether QASPER-tuned theta and CUAD-tuned
theta would FAIL to transfer across each other's benchmarks (the Stage-1 within-family
transfer-failure hypothesis). Instead:

| theta source -> | QASPER eval | CUAD eval |
|---|---:|---:|
| canonical (grid-world) | 0.208 | 0.366 |
| QASPER-tuned (wide) | **0.563** | 0.591 |
| CUAD-tuned (narrow) | 0.500 | **0.629** |

Off-diagonal cells recover 84% of the diagonal lift over canonical (off-diag avg +0.259 vs
diagonal avg +0.309). The honest finding is **task-tuned theta on long-haystack QA generalizes
within the document-QA family. Grid-world theta is bad for document QA in general; any
in-family tuned theta lifts retrieval substantially across the family.** Memory is task-dependent
at the task-family granularity (grid-world != document-QA), not the within-family granularity
(QASPER != CUAD within document-QA).

**Thesis chapter draft** (`docs/THESIS_STAGE3_CHAPTER.md`, ~13 pages of prose) — full sections
1 (Introduction), 2 (Related Work), 3 (Methodology), 4 (Experimental Setup), 6 (Discussion), and
7 (Future Work). Section 5 (Results) ships with the live retrieval-quality table from Phase 1.5
plus the new transfer matrix from Phase 1.6, with placeholders for Phase 4's LLM answer-quality
+ cost numbers. Style mirrors `docs/THESIS_STORY.md` (direct, claim-driven, with numbers).

**Frontend**: `web/src/sections/Stage3.tsx` gains a `TransferMatrixPanel` rendering the 3 x 2
matrix with the diagonal cells highlighted (★) and summary stats showing diagonal-vs-off-diagonal
average lifts. The supporting `web/public/data/stage3_retrieval.json` now includes
`transfer_matrix` and `width_comparison` blocks. TypeScript clean.

**Regression**: 146/146 pytest tests still pass; determinism audit green.

Full writeup: [docs/THESIS_STAGE3_CHAPTER.md](THESIS_STAGE3_CHAPTER.md).

**Files added (Phase 1.6):** `scripts/compare_theta_widths.py`,
`scripts/run_theta_transfer.py`, `docs/THESIS_STAGE3_CHAPTER.md`,
`results/stage3/tuned_theta_wide_qasper.json`,
`results/stage3/tuned_theta_wide_cuad.json`,
`results/stage3/theta_width_comparison.json`,
`results/stage3/theta_transfer_matrix.json`.

**Modified:** `tuning/tune_v4_per_benchmark.py` (`--out-suffix`),
`scripts/build_stage3_frontend_data.py` (transfer + width sections),
`web/public/data/stage3_retrieval.json` (regenerated),
`web/src/sections/Stage3.tsx` (TransferMatrixPanel component).

---

## -5. Stage 3 Phase 1 + 1.5 — adapters, adversarial tests, theta tuning, retrieval study, orchestrator (May 2026)

Built the full Stage-3 evaluation stack on top of the six real-data
adapters from entry -4: a 4-layer test pyramid (snapshot + schema +
adversarial + retrieval smoke), CMA-ES per-benchmark θ tuning, a 12+1
memory-system retrieval study on real data, and a Phase-4 orchestrator
with a `--dry-run` cost projection mode.

**Headline empirical result (real data, no LLM):**

V4 with per-benchmark CMA-ES-tuned θ **beats every other memory system
and beats V4-canonical by +0.26 to +0.36 recall** on the two long-haystack
benchmarks:

| Benchmark | V4-canonical | V4-tuned | Δ |
|---|---:|---:|---:|
| CUAD   | 0.366 | **0.629** | **+0.263** |
| QASPER | 0.208 | **0.563** | **+0.355** |

The other four benchmarks (HotpotQA, FinanceBench, LongMemEval, NarrativeQA)
saturate at recall = 1.0 or have no paragraph-level gold — ceiling-bound
for any reasonable retriever. This produces a clean two-cluster
finding: **memory-system differentiation only emerges on long-haystack
tasks; on short ones, k=8 saturates retrieval regardless of the system**.

**Phase-4 cost projection (dry-run, tiktoken-measured):** ~$1.50 for the
canonical 6 × 3 × 30-question sweep — vastly under the original $60-70
plan estimate. Cap-and-retrieve keeps prompts compact.

**Test pyramid: 117 → 146 pytest tests, all green; determinism audit
green.** Added 29 adversarial tests (5 per adapter avg) that mock
edge-case inputs: yes_no=False, FLOAT_TYPE_NONEVIDENCE evidence sentinel,
answer_start in blank-line gaps, multi-byte unicode (ﬁ ligature, 中文),
mismatched parallel arrays, paragraph-cap truncation marker, all-
unanswerable papers, etc. Each test docstring names its specific edge
case for fast diagnosis.

**Frontend:** `web/src/sections/Stage3.tsx` updated to render the live
13×6 retrieval table sorted by long-haystack performance, with
V4-tuned highlighted and side-cards showing per-benchmark canonical →
tuned improvement. Data file: `web/public/data/stage3_retrieval.json`,
built by `scripts/build_stage3_frontend_data.py`.

Full writeup: [docs/STAGE3_PHASE1.md](STAGE3_PHASE1.md).

**Files added:** `tests/test_benchmark_adversarial.py`,
`tuning/__init__.py`, `tuning/tune_v4_per_benchmark.py`,
`evaluation/benchmark_memory_eval.py`,
`scripts/run_stage3_retrieval.py`, `scripts/run_stage3_full.py`,
`scripts/build_stage3_frontend_data.py`,
`web/public/data/stage3_retrieval.json`,
`results/stage3/` (tuned_theta_*.json, retrieval_*.json,
retrieval_summary.json, stage3_runs.json, cost_projection.json, cells/).

**Modified:** `web/src/sections/Stage3.tsx`.

**Phase 4 (real API)** is now a one-line invocation against
`scripts/run_stage3_full.py --mode full`. Awaits API key + budget.

---

## -4. Stage 3 data prep — six real long-context benchmarks fetched + verified (May 2026)

The thesis pivots from synthetic environments (grid-world + hand-authored
DocumentQA) to **real, published long-context benchmarks** for the
LLM-agent stage. Six benchmarks chosen to span disjoint domains:
NarrativeQA (fiction), QASPER (science), CUAD (law), HotpotQA
(Wikipedia / multi-hop), FinanceBench (finance), LongMemEval
(long-term dialogue memory).

**Phase 0 — fetch + verify (PASS):** all 6 benchmarks downloaded to
`data/benchmarks/` (36.9 GB on disk, gitignored). `scripts/prefetch_benchmarks.py`
handles idempotent fetch with merging manifest; `scripts/verify_benchmarks.py`
runs offline (`HF_DATASETS_OFFLINE=1`) and confirms every benchmark has
non-empty `(question, answer, source content)` for sampled items, with
length distributions reported. 6/6 OK.

**Three problems hit and fixed mid-flight:**
1. HF `datasets>=3` dropped loading-script support → QASPER and CUAD now
   pulled from canonical archives (AI2 S3 / Zenodo) instead of HF.
2. Original `xiaowu0162/longmemeval` HF repo deprecated → switched to
   `xiaowu0162/longmemeval-cleaned`.
3. `load_dataset` autoparse fails on longmemeval-cleaned (answer-column
   type mismatch) → bypassed via `huggingface_hub.hf_hub_download`
   directly on the raw JSON files.

**Two gold-relevance signals confirmed:** HotpotQA `supporting_facts` and
LongMemEval `answer_session_ids` both expose per-question gold relevance
labels — directly compatible with the existing precision@k / recall@k
metric path (same shape as `relevant_paragraph_indices` in synthetic
DocumentQA).

**Headline scale:** NarrativeQA documents reach **1,199,816 chars** on
the largest sample (full books); CUAD contracts reach 338 K chars
(p95 = 161 K); the other four are in 5 K–55 K range — the regime where
selective memory is the lever.

Full writeup: [docs/STAGE3_DATA_PREP.md](STAGE3_DATA_PREP.md).

**Files added:** `scripts/prefetch_benchmarks.py`,
`scripts/verify_benchmarks.py`, `data/benchmarks/manifest.json`,
`data/benchmarks/verification.json`, `docs/STAGE3_DATA_PREP.md`.
`requirements.txt` adds `datasets>=2.14`, `huggingface_hub>=0.20`.
`.gitignore` excludes `data/benchmarks/`.

Next phase: build six adapter modules in `environment/benchmarks/` that
translate each benchmark's native shape into the `DocumentQA`-compatible
`(title, paragraphs, qa_pairs)` contract.

---

## -3. Reflexion plan: GraphMemoryV6 + lesson buffer (April 2026)

After a research deep-dive (covered in three parallel agent reports), the
single highest-leverage 2024-2026 thesis improvement was identified as a
Reflexion-style verbal-lesson buffer to address V4's MegaQuest 0.0 reward
finding. The plan: prove the bottleneck is policy (Phase 0), build the
architecture (Phase 1-2), and run the canonical same-env retry experiment
(Phase 3).

**Phase 0 — oracle diagnosis (PASS):** OmniscientOracle (env-cheating)
hits 100% reward on MegaQuest across all 30 episodes; OracleHonest gets
0.03; ExplorationPolicy gets 0.00. Policy-bottleneck confirmed: env is
fully solvable with optimal info.

**Phase 1 — V6 architecture (PASS):** GraphMemoryV6 extends V4 with two
theta dimensions (`w_lesson`, `theta_lesson_decay`) and a new `Lesson`
node type. Strict-generalization invariant holds (`w_lesson=0` matches V4
bit-identically). pytest invariants 40 → 60.

**Phase 2 — ReflexionPolicy + runners (PASS):** policy wrapper that
injects retrieved lessons as synthetic past_events; persistent-memory
runner; same-env retry runner.

**Phase 3 — empirical lift (NEGATIVE RESULT):** on MultiHop-KeyDoor across
5 env layouts × 4 tries, V4-base / V6-w-lessons / V6-Reflexion are
statistically indistinguishable (p = 1.000, d = 0.000 for all pairs).

**Diagnosis:** ExplorationPolicy is rule-based and deterministic given
parsed hints; `heuristic_lesson()` paraphrases hints already visible in
observations; the lesson channel adds no new information for a rule-based
policy. The architecture is sound but its empirical lift is gated on
richer lesson generators (LLM-judged reflections — Stage 3) or LLM-agent
policies that can act on strategic verbal text beyond regex-parseable
hints.

**Defensible thesis claim:** V6 is a plumbing-complete, evidence-gated
contribution. The framework is in place (60 invariants, determinism audit
green); the empirical lift awaits the richer reflective signal an LLM
judge can provide.

Full writeup: [docs/REFLEXION_RESULTS.md](REFLEXION_RESULTS.md).

**Files added:** `agent/oracle_policy.py`, `agent/reflexion_policy.py`,
`memory/lesson.py`, `memory/graph_memory_v6.py`,
`evaluation/reflexion_eval.py`, `run_oracle_diagnosis.py`,
`run_reflexion_ablation.py`, `tests/test_v6_invariants.py`,
`docs/REFLEXION_RESULTS.md`.

V7 (skill library) deferred per the same architectural ceiling.

---

## -2. Frontend refactor — Vite + React replaces Streamlit dashboard (April 2026)

The 842-line Streamlit dashboard at `dashboard/` has been replaced with a
modern static site at `web/`. The Streamlit version produced functional
charts but couldn't deliver a "thesis-grade" presentation; the new site is
single-page scrollytelling with eleven sections covering the full thesis
arc, four interactive components, and a maximalist cinematic aesthetic.

**Stack:**
- Vite 8 + React 19 + TypeScript 6
- Tailwind CSS v4 (CSS-first `@theme` config via `@tailwindcss/vite`)
- framer-motion v12 for scroll choreography
- recharts v3 + custom SVG for charts; react-force-graph-2d for the
  animated event/entity graph

**Sections (the story arc):**
1. Hero — animated 10D θ vector that periodically morphs between TF-IDF
   and MiniLM optima over a constellation backdrop.
2. The Question — fixed memory ↔ learnable θ side-by-side diagram.
3. Architecture — three-card explainer + the live ThetaExplorer.
4. Progression V1 → V5 — clickable timeline with dim-by-dim explanations.
5. Benchmark — 12 × 4 heatmap, Pareto scatter, pairwise significance
   table, force-directed memory graph.
6. The MiniLM Pivot — side-by-side ThetaRadar (TF-IDF vs MiniLM) with
   delta breakdown and three story-callouts.
7. Ablation — interactive knockout panel with the theta_novel = 100%-
   degradation flash.
8. Transfer & Sensitivity — transfer cards + the MegaQuest A2 finding +
   2D reward landscape with learned-θ marker.
9. Neural Meta-Controller — 200-gen learning curve with σ secondary axis.
10. Stage 3 — formula display + ready-to-run config commands.
11. Reproducibility — stack callouts + provenance from the manifest.

**Four interactive components live:**
- ThetaExplorer: 10 sliders, live preview of stored fraction, predicted
  reward (bilinear interp on sensitivity grid), and dominant retrieval
  signal.
- EmbeddingToggle: React context that flips the V4 numbers shown
  throughout between TF-IDF (legacy) and MiniLM (current default).
- BenchmarkHeatmap: 12 × 4 matrix with per-cell precision rings, V4
  highlighted, click-through detail panel.
- MemoryGraph: react-force-graph-2d animated event/entity graph with
  a "store rate" slider and play/pause/reset controls.

**Data pipeline:**
- `scripts/build_web_data.py` runs as `npm run`'s `predev` / `prebuild`
  hook. Copies `results/*.json` into `web/public/data/` and slims
  `neural_controller_v2_results.json` from 12 MB to 26 KB by dropping
  the per-generation 5,674-float weight arrays. Builds an aggregated
  manifest with embedding backend, git_sha, and timestamps.

**Removed:**
- `dashboard/app.py`, `dashboard/charts.py`, `dashboard/copy.py`.
- `streamlit`, `pandas`, `plotly` from `requirements.txt`.

**Build:** `cd web && npm run build` produces `web/dist/` (~1 MB JS / 300 KB
gzipped). No GitHub Pages workflow yet (deferred); the static output can be
hosted on Netlify, Vercel, or `gh-pages` on demand.

---

## -1. PoC hardening pass (April 2026)

A Phase 1–4 implementation plan landed end-to-end. Highlights:

- **Embedding swap (Phase 3 / S1).** Default backend changed from
  31-token TF-IDF to ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim)
  with LRU caching. Legacy TF-IDF kept as `EMBEDDING_BACKEND=tfidf`. Neural
  controllers explicitly retain TF-IDF for their own input feature so their
  parameter count stays at 5,674 / 1,962.

  **Headline impact:** under MiniLM, V4's previously-published optimum
  (`w_recency=3.78`, `w_embed=1.08`) is no longer optimal. Quick CMA-ES finds
  `w_embed=3.75`, `w_recency=0.66` — similarity dominates retrieval when
  embeddings are semantically meaningful. Re-running V4 CMA-ES is required
  to produce the new canonical θ; in the meantime, the post-MiniLM
  unoptimized V4 sits at ~0.12 reward on MultiHopKeyDoor (vs 0.18 under
  TF-IDF). The thesis claim shifts from "V4 is #1" toward "V4 is competitive
  with the leading systems and the optimal θ depends on the embedding".

- **Reproducibility (Phase 1 / B1).** `evaluation/benchmark.save_benchmark_results`
  no longer strips per-episode rewards; new `results.manifest.build_manifest()`
  is wired into every `run_*.py` script. Output JSONs gain a `_manifest` block
  recording git_sha, embedding_backend, timestamp, seed, and run-specific args.

- **Pytest scaffolding (Phase 1 / B2).** New `tests/` directory with 40
  invariants: memory-system contract, V4 storage gating, MemoryParamsV4
  roundtrip, neural controller pack/unpack determinism, statistics helpers,
  LLM-judge fallback. Exposed and fixed a divide-by-zero bug in
  `evaluation.statistics.paired_ttest`.

- **CMA-ES infrastructure (Phase 1 / B3-B4).** New `optimization/cma_es.py`
  wraps `pycma` (BIPOP-CMA-ES restarts, automatic stop conditions) with the
  legacy minimal pure-numpy implementation as a fallback. Both back-ends
  expose `save_checkpoint` / `load_checkpoint` and `--resume-from <path>`
  on the long-running scripts; an interrupted NeuralV2 200-gen run no
  longer loses 15h of progress.

- **Determinism audit (Phase 1 / B5).** New `scripts/audit_determinism.py`
  runs every memory system twice with the same `episode_seed` and asserts
  identical retrievals. All 16 systems pass under both backends.

- **Statistical significance (Phase 3 / S4).** Replaced the hand-rolled
  t-distribution approximation in `evaluation/statistics.py` with
  `scipy.stats.t.sf` / `norm.sf`. New `scripts/run_pairwise_significance.py`
  loads per-episode rewards from the benchmark JSON and computes pairwise
  paired t-tests + bootstrap CI + Cohen's d for any baseline vs every
  other system. Output: `results/pairwise_significance.json`.

- **Stage 3 wiring (Phase 2 / S2 + A4).** Three configs in
  `experiments/document_qa_{episodic,v4,neural_v2}.yaml`. New
  `evaluation/document_qa_llm_judge.py` provides an LLM-as-judge scorer that
  gracefully falls back to keyword overlap when `OPENAI_API_KEY` is unset
  (so CI exercises the codepath without API cost). DocumentQA's `score_fn`
  parameter wires it in. Three additional documents (corporate_memo,
  soap_opera, science_survey) bring the library to 5 documents × ~30 paras
  × 8 QA pairs each. Stage 3 runs themselves remain deferred from the PoC.

- **Graph-vestigial framing (Phase 2 / S3).** New `--w-graph-sweep` flag on
  `run_ablation.py` produces `docs/figures/fig_w_graph_ablation.png` showing
  reward is flat in `w_graph`. Doc updates in THESIS_HANDOFF, GRAPHMEMORY_V4_RESULTS,
  AGENTS reframe V4 as "selective importance-scored storage + recency-weighted
  embedding retrieval" with the graph data structure as a typed-storage scaffold.

- **NeuralV2 warm-start (Phase 3 / A3).** New
  `NeuralMemoryControllerV2Small.initialize_to_constant_theta(target)` sets
  output-layer biases to `logit(target)` so the MLP outputs V4's learned θ for
  any input at initialization. CMA-ES then explores deviations from this
  baseline rather than starting from random init. New `--pretrain-from-v4`
  flag on `run_neural_controller_v2.py`.

- **MegaQuest investigation (Phase 3 / A2).** New `--clamp-w-recency`
  flag on `run_transfer.py` overrides the learned w_recency on MegaQuestRoom
  to test whether recency-dominated retrieval is what breaks long-horizon
  transfer.

- **Master reproduction (Phase 2 / C4).** `python reproduce_thesis.py
  --quick|--full` runs every experiment in dependency order; `--dry-run` and
  `--only` / `--skip` for selective re-runs.

- **CI (Phase 1 / C5).** GitHub Actions workflow at `.github/workflows/ci.yml`
  runs pytest + determinism audit + smoke tests on every push/PR.

- **Polish.** `report.txt` renamed to `report_poc_v1.txt` (frozen);
  `main.py` now writes `report_poc_current.txt`. Synthetic placeholder
  figures (Fig 12, 13_curves, 14, 15) moved to `docs/figures/draft/` with
  banner annotations in viz functions; dashboard updated with a separate
  "Draft (synthetic)" section. Empty `PROJECT_SUMMARY_FOR_CHATGPT.md` deleted.

**Files added:** `results/manifest.py`, `tests/conftest.py`,
`tests/test_memory_contract.py`, `tests/test_v4_invariants.py`,
`tests/test_statistics.py`, `tests/test_neural_controller.py`,
`tests/test_llm_judge.py`, `evaluation/document_qa_llm_judge.py`,
`scripts/audit_determinism.py`, `scripts/run_pairwise_significance.py`,
`reproduce_thesis.py`, `optimization/cma_es_minimal.py` (renamed from
`cma_es.py`), `optimization/cma_es.py` (new wrapper), `pytest.ini`,
`.github/workflows/ci.yml`, `experiments/document_qa_v4.yaml`,
`experiments/document_qa_neural_v2.yaml`,
`docs/figures/draft/fig{12,13,14,15}_*.png` (moved).

**Pending re-runs (require ~24h overnight compute):** full V4 CMA-ES at
canonical 30 gens × 50 eps × 200 eval, NeuralV2 200-gen with
`--pretrain-from-v4`, ablation/transfer/sensitivity at canonical episode
counts. The Stage 3 (real-LLM) experiments remain explicitly deferred.

---

## 0. NeuralControllerV2 200-gen results and figures (March 2026)

- **Run:** 200 generations, sigma=0.3, ~15.4 h training. MultiHop eval reward **0.19** (vs V4 scalar 0.178); MegaQuest zero-shot **0.0**.
- **Figures:** (1) **fig_neural_analysis** — conditional annotation (“Neural matches or exceeds scalar V4” when neural ≥ V4); right panel replaced with **transfer comparison** (MultiHop vs MegaQuest, Neural vs V4). (2) **fig_neural_v2_curves** — regeneratable from JSON (2-panel learning curve + sigma). (3) **fig_neural_transfer** — dedicated transfer bar chart (Neural vs V4 on both envs).
- **Interpretation:** See `docs/NEURAL_CONTROLLER_V2_RESULTS.md`: 200-gen run shows neural meta-controller can match scalar V4 on MultiHop with sufficient budget; transfer failure on MegaQuest supports task-dependent memory.

---

## 1. Bar Chart Fixes (March 2026)

Several bar charts were corrected and figures regenerated.

### 1.1 `fig_ablation_ranked` (Ablation importance ranking)

- **Issue:** Value labels for *negative* degradation (e.g. `graph_only` at −15.4%) were placed on the wrong side of the bar (to the right of zero instead of next to the bar).
- **Change:** In `generate_thesis_figures.py`, label position and horizontal alignment are now value-dependent:
  - **Positive values:** `x = val + 0.5`, `ha="left"` (label to the right of the bar).
  - **Negative values:** `x = val - 0.5`, `ha="right"` (label to the left of the bar).
- **Scope:** Applied to both panels (reward degradation and precision degradation).

### 1.2 `fig_neural_analysis` (Neural vs scalar comparison)

- **Issue:** Reward (≈0.03–0.17) and precision (≈0.63–1.0) were plotted on the same y-axis, making reward bars barely visible.
- **Change:** Dual y-axis in `generate_thesis_figures.py`:
  - **Left axis:** Mean reward (purple), with its own scale and labels.
  - **Right axis:** Retrieval precision (gray), with separate scale (0–1.15).
  - Reward and precision bars each use the correct axis; value labels and legends updated accordingly.

### 1.3 `regen_benchmark_figs.py` (Fig 5, 5b, 5c, 5d)

- **V4/V1 data for MultiHop:** Fig5 and Fig5c now merge optimized GraphMemoryV4 and GraphMemoryV1 from `results/graphmemory_v4_cmaes_results.json` when that file exists, so MultiHop bar charts show CMA-ES results instead of raw benchmark values.
- **Error filtering:** All bar/heatmap data now exclude systems whose result entry contains an `"error"` key (via `_is_valid_system_entry()`).
- **Heatmap (Fig 5b):**
  - **Systems:** Uses the *union* of systems across all environments (not only the first env’s systems).
  - **Title:** Episode counts corrected to “n=50 (Key-Door, Goal-Room, MultiHop); n=20 (MegaQuestRoom)” instead of “n=50 episodes each.”
- **Colors:** `GraphMemoryV1` added to the `COLORS` palette in `regen_benchmark_figs.py`.
- **Precision scatter (Fig 5c):** Same V4/V1 merge and validation as above so scatter uses CMA-ES MultiHop values when available.
- **Easy env (Fig 5d):** Key-Door and Goal-Room bar charts now filter out error entries before building systems lists.

### 1.4 Figure regeneration

- All figures were regenerated with `python regen_all_figures.py` (thesis figures with `--allow-missing`, then benchmark fig5 variants, then extended Fig 8–15).

---

## 2. Earlier Session Work (Summary)

The following was completed in prior sessions; captured here for a single “what we did” record.

### 2.1 Pipeline and robustness

- **RAGMemory fallback:** Benchmark and DocumentQA memory eval support skipping RAGMemory via a positional argument when `sentence_transformers` is broken (e.g. `python run_benchmark.py RAGMemory`).
- **Smoke tests:** `run_smoke_tests.py` added/updated for a quick pipeline check (grid, DocumentQA memory, DocumentQA+LLM path).
- **Missing-file check:** `generate_thesis_figures.py` checks for required JSONs and, without `--allow-missing`, exits with clear instructions; `regen_all_figures.py` uses `--allow-missing` so partial runs still produce figures.
- **Runner safeguard:** Runner/config behavior tightened so experiments don’t silently use wrong configs.
- **Benchmark/DocumentQA:** Optional `skip_systems` (e.g. RAGMemory) supported in benchmark and DocumentQA memory evaluation.

### 2.2 Documentation

- **RUNNING_EXPERIMENTS.md:** Added under `docs/` with exact commands and order for regenerating result files and figures, smoke tests, DocumentQA+LLM, and dashboard.
- **AGENTS.md:** Updated with project structure, benchmark table, pending experiments, and latest results summary.

### 2.3 Scripts and structure

- **Scripts moved:** `find_seed.py` and `test_new_systems.py` moved into `scripts/`.
- **regen_all_figures.py:** Single entry point to regenerate thesis figures, benchmark fig5 variants, and extended figures (Fig 8–15) using real data when JSONs exist.

### 2.4 Figures and thesis audit

- **Figure naming:** Extended fig13 variant saved as `fig13_memory_curves.png` to avoid overwriting the thesis fig13.
- **Sensitivity:** Annotation (sharp peak vs broad plateau) made data-driven from sensitivity results.
- **Pareto figure:** Title clarified to “top-left preferred.”
- **Fig 5b caption:** Uses dynamic system/env counts where applicable.
- **GraphMemoryV5:** Added to color mappings across figure scripts.
- **Synthetic data labels:** Fig 12, 14, 15 marked “(Illustrative — synthetic data)” where appropriate.
- **Landscape viz:** `viz/landscape_viz.py` legend only drawn when labeled artists exist.
- **Precision–reward story figure:** N/A note added for Key-Door/Goal-Room where precision is not defined.

### 2.5 Archive and handoff

- **Archive:** Phase-specific docs moved to `docs/archive/` (e.g. STEP1_HARDER_ENVIRONMENT.md, PHASE6_IMPLEMENTATION_PLAN.md).
- **THESIS_HANDOFF.md, PROJECT_SUMMARY, PROJECT_SUMMARY_FOR_CHATGPT:** Updated to reflect current structure and “what’s done / what’s next.”

### 2.6 Dashboard and visualizations

- **Streamlit dashboard:** `dashboard/app.py` with tabs: Overview, Benchmark, Ablation, Transfer, Sensitivity, DocumentQA Memory, Figures, Playground, Visualizations (list of all 21 figures with short explanations).
- **Config:** `.streamlit/config.toml` created; CORS/XSRF issues addressed (e.g. `enableCORS` removed if it caused problems).
- **Visualizations page:** Lists all figures with brief descriptions for quick reference.

---

## 3. Files Touched (Bar Chart Session)

| File | Changes |
|------|--------|
| `generate_thesis_figures.py` | Ablation label placement (positive/negative); neural comparison dual y-axis. |
| `regen_benchmark_figs.py` | V4/V1 merge for MultiHop, `_is_valid_system_entry`, heatmap systems union and title, GraphMemoryV1 color, precision scatter merge, easy-env filtering, `main()` loads V4 results once and passes to fig5/precision scatter. |

---

## 4. How to Regenerate Everything

After any further data or code changes:

```powershell
cd c:\Users\uifal\MasterThesis
python regen_all_figures.py
```

For thesis-only figures with missing JSONs allowed:

```powershell
python generate_thesis_figures.py --allow-missing
```

For benchmark fig5 variants only (requires `results/benchmark_results.json`; V4/V1 merge automatic if `results/graphmemory_v4_cmaes_results.json` exists):

```powershell
python regen_benchmark_figs.py
```

---

## 5. References

- **Commands and order:** `docs/RUNNING_EXPERIMENTS.md`
- **Agent and project rules:** `AGENTS.md`
- **High-level summary:** `docs/THESIS_HANDOFF.md`
