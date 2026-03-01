# Learnable Memory Construction for LLM Agents — Project Summary

**Date:** February 2026  
**Status:** POC complete + full thesis framework scaffolded  
**GitHub:** https://github.com/UifaleanStefan/MasterThesis

---

## 1. Thesis Core Idea

Most AI agents learn *what action to take* (policy) and *how good a state is* (value function). They do **not** learn *how to build their own memory representation*.

This thesis asks:

> **Can an agent learn how to construct its own memory differently depending on the task?**

Memory is represented as a **directed graph** (event nodes, entity nodes, temporal/mention edges). The *structure* of this graph — what gets stored, how entities are tracked, how sequential edges are formed — is controlled by a learnable parameter vector **θ = (θ_store, θ_entity, θ_temporal)**.

- **θ_store ∈ [0,1]:** Bernoulli probability that an event gets stored at all
- **θ_entity ∈ [0,1]:** Frequency-importance threshold — only entities seen more often than this fraction get a graph node
- **θ_temporal ∈ [0,1]:** Bernoulli probability of adding a temporal chain edge between consecutive events

Default θ = (1.0, 0.0, 1.0) exactly reproduces the unparameterized baseline — backward compatible.

Why this matters: in real LLM-based agents (RAG, MemGPT, LangMem), storing and retrieving everything is expensive in tokens and dollars. The optimal memory structure is task-dependent. A lore-reading agent needs causal chains; a navigation agent needs spatial transitions; a key-door agent needs entity-color relationships. **θ should be learned, not hardcoded.**

---

## 2. What Has Been Implemented (Complete Inventory)

### 2.1 Environments

| Environment | Grid | Steps | Difficulty | Key Feature |
|---|---|---|---|---|
| `ToyEnvironment` | 6×6 | 80 | Easy | 3 keys, 1 door, baseline |
| `GoalRoom` | 6×6 | 80 | Easy | Reach goal cell, no keys |
| `HardKeyDoor` | 10×10 | 300 | Medium | 5 keys, 3 doors, distractors |
| `QuestRoom` | 12×12 | 500 | Hard | 4 chained doors, 2 NPCs, distractor obs |
| `MultiHopKeyDoor` | 10×10 | 250 | Hard (**primary benchmark**) | 6 keys, 3 doors, 3 hint observations at steps 0-2, 3 distractor keys, 15% distractor obs |
| `MegaQuestRoom` | 20×20 | 1000 | Very Hard (new, scaffolded) | 6 doors, 4 NPCs, 10 real + 5 distractor keys |
| `DocumentQA` | — | variable | Real-world | Sequential reading + multi-hop QA on 40-paragraph documents |
| `TextWorldEnv` | — | 200 | Real-world | Microsoft TextWorld IF games (stub if not installed) |
| `MultiSessionEnv` | — | 50/session | Real-world | 20 sessions, persistent memory, story consistency |

**Key design principle for MultiHopKeyDoor:** Hint observations (`"You see a sign: the red key opens the north door"`) appear *only* at steps 0–2. By step 50+, they're gone. Without memory that retains them, the agent cannot match keys to doors — guaranteed 0% success. With good memory, it achieves 30%+.

### 2.2 Memory Systems (10 total)

All share the uniform interface: `add_event(event, episode_seed)`, `get_relevant_events(obs, step, k)`, `clear()`, `get_stats()`.

| # | System | File | Key Mechanism |
|---|---|---|---|
| 1 | `FlatMemory` | `memory/flat_memory.py` | Sliding window, last N events. Baseline floor. |
| 2 | `GraphMemory + θ` | `memory/graph_memory.py` | Graph with event/entity nodes, parameterized by θ. **Core contribution.** |
| 3 | `SemanticMemory` | `memory/semantic_memory.py` | Importance-weighted pool: NPC hints + entity richness + novelty score |
| 4 | `SummaryMemory` | `memory/summary_memory.py` | Compresses every 25 steps into a summary node |
| 5 | `EpisodicSemanticMemory` | `memory/episodic_semantic_memory.py` | **Dual store:** episodic sliding window + persistent semantic facts (hints never evicted). Best performer. |
| 6 | `RAGMemory` | `memory/rag_memory.py` | Dense embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`), cosine retrieval |
| 7 | `HierarchicalMemory` | `memory/hierarchical_memory.py` | **NEW.** 3 levels: raw (last 20) → episode summaries (every 25 steps) → long-term facts (never evicted) |
| 8 | `WorkingMemory` | `memory/working_memory.py` | **NEW.** 7-slot LRU-retrieval eviction (Miller's Law). Recently retrieved events survive. |
| 9 | `CausalMemory` | `memory/causal_memory.py` | **NEW.** Tracks event→action→outcome causal chains. Retrieves by causal relevance. |
| 10 | `AttentionMemory` | `memory/attention_memory.py` | **NEW.** Softmax scaled dot-product attention. Differentiable retrieval. |
| 11 | `NeuralMemoryController` | `memory/neural_controller.py` | **NEW.** MLP (~4,400 params) outputs context-dependent θ per observation. |

### 2.3 Retrieval

`memory/retrieval.py` provides 4 retrieval functions:

- `retrieve_events()` — graph-based (entity keyword matching + temporal chain)
- `retrieve_similar_events()` — TF-IDF cosine similarity
- `retrieve_relevant_events()` — hybrid union of both
- `retrieve_events_learnable()` — **weighted scoring:** `score = w_graph × graph_signal + w_embed × cosine + w_recency × (1/(1+Δstep))`

### 2.4 Optimization (4 methods, all new)

All in `optimization/`:

| Method | File | Description |
|---|---|---|
| **ES** (existing in `main.py`) | `main.py:run_phase7_es` | Simple Evolution Strategy: best candidate updates mean, σ decays |
| **CMA-ES** | `optimization/cma_es.py` | Full covariance matrix adaptation. Handles correlated θ. Scales to 4000+ params for NeuralController. |
| **Bayesian Optimization** | `optimization/bayesian_opt.py` | GP surrogate + Expected Improvement. Sample-efficient (critical when each eval costs LLM API calls). |
| **OnlineAdapter** | `optimization/online_adapter.py` | θ adapts *during* an episode: `StatisticsAdapter` (rule-based) and `GradientAdapter` (finite-difference). |
| **MetaLearner** | `optimization/meta_learner.py` | Reptile-style cross-task meta-learning. Finds θ_init that adapts quickly to any task in 3-5 ES steps. |

### 2.5 Agent Layer

| Component | File | Description |
|---|---|---|
| `ExplorationPolicy` | `agent/policy.py` | Rule-based: parses NPC hints, matches keys to doors, random exploration. Works for all grid envs. |
| `LLMAgent` | `agent/llm_agent.py` | **NEW.** GPT-4o/gpt-4o-mini wrapper. Tracks prompt/completion tokens + cost in USD. Fallback heuristic if no API key. |
| `ContextFormatter` | `agent/context_formatter.py` | **NEW.** 3 formats: `flat` (concat), `structured` (labeled by type), `compressed` (single-sentence summary). |
| Episode runners | `agent/loop.py` | `run_episode_with_memory`, `run_episode_with_any_memory` → returns `(success, events, stats_dict)` with `retrieval_tokens`, `memory_size`, `retrieval_precision` |

### 2.6 Evaluation Suite

| Module | File | What it does |
|---|---|---|
| `run_evaluation` | `evaluation/run.py` | Original: no_memory/graph/embedding/hybrid/learnable on ToyEnvironment |
| `run_memory_comparison` | `evaluation/run.py` | Compare N memory systems on any env, collects partial_score, retrieval_precision, tokens |
| `run_ablation_study` | `evaluation/ablation.py` | 8 ablation configs (zero each θ component), compute degradation vs full learned |
| `run_transfer_matrix` | `evaluation/transfer.py` | θ trained on task A → evaluated on task B, full cross-task matrix |
| `compute_sensitivity` | `evaluation/sensitivity.py` | 2D reward landscape over θ_store × θ_entity at fixed θ_temporal |
| `run_full_benchmark` | `evaluation/benchmark.py` | 10 systems × all envs × N episodes, with bootstrap CIs |
| `bootstrap_ci`, `cohens_d`, `paired_ttest` | `evaluation/statistics.py` | Full statistical significance suite |
| `CostTracker` | `evaluation/cost_tracker.py` | Per-episode USD cost logging, proxy cost from token counts |

### 2.7 Visualization (15 figures total)

| Figure | File | Content |
|---|---|---|
| Fig 1 | `viz/graph_viz.py` | Memory graph: fixed θ (dense) vs. learned θ (sparse), Key-Door episode |
| Fig 2 | `viz/es_curves.py` | ES learning curves: mean_j + efficiency per generation, 3 environments |
| Fig 3 | `viz/es_curves.py` | θ trajectory over ES generations (θ_store, θ_entity, θ_temporal) |
| Fig 4 | `viz/bandit_landscape.py` | Phase 6 bandit scatter: θ configs colored by reward |
| Fig 5 | `viz/memory_comparison.py` | Memory system comparison bar charts (score + precision) |
| Fig 6 | `viz/grid_viz.py` | MultiHopKeyDoor agent trajectory heatmap with key/door/hint annotations |
| Fig 7 | `viz/episode_curves.py` | Per-episode reward + tokens + memory_size curves (EpisodicSemantic) |
| Fig 8 | `viz/ablation_viz.py` | Ablation bar chart with degradation % annotations |
| Fig 9 | `viz/landscape_viz.py` | 2D reward heatmap over θ_store × θ_entity |
| Fig 10 | `viz/transfer_viz.py` | Cross-task transfer matrix heatmap |
| Fig 11 | `viz/pareto_viz.py` | Pareto frontier: reward vs. token cost, all systems |
| Fig 12 | `viz/cost_viz.py` | Online adaptation: θ_store over episode steps |
| Fig 13 | `viz/cost_viz.py` | Memory size growth per system over episode steps |
| Fig 14 | `viz/cost_viz.py` | LLM cost breakdown: prompt / memory / completion tokens |
| Fig 15 | `viz/cost_viz.py` | Multi-session score + memory retention over 20 sessions |

Figs 1–7 are generated from real experiment data. Figs 8–15 currently use synthetic data and are ready to be re-generated once the full experiments run.

### 2.8 Framework Infrastructure

| Component | File | Description |
|---|---|---|
| Config system | `config.py` | `ExperimentConfig` dataclass, YAML/JSON serializable. Preset configs for common experiments. |
| CLI runner | `runner.py` | `python runner.py --config experiments/foo.yaml` or `--preset benchmark`. Saves to DB. |
| Results DB | `results/db.py` | SQLite: every run → `runs` + `results` + `episodes` tables. Query `best_theta(env, system)`. |
| Experiment configs | `experiments/*.yaml` | `multihop_cmaes.yaml`, `benchmark.yaml`, `document_qa_llm.yaml` |

---

## 3. Key Experimental Results (POC)

### 3.1 The Main Finding: Memory Structure Is Task-Dependent

Evolution Strategy learned fundamentally different θ for each environment:

| Environment | Learned θ (store, entity, temporal) | Interpretation |
|---|---|---|
| Key-Door | (0.31, 0.19, 0.84) | Sparse storage — filter distractor events, keep entity graph |
| Goal-Room | (0.74, 0.65, 0.51) | Medium storage — navigation needs recent context; entity nodes matter less |
| MultiHopKeyDoor | (0.96, 0.38, 1.00) | Near-full storage — cannot miss early hints; entity threshold filters distractor keys |

These θ values differ significantly across tasks. Key-Door wants sparse storage; MultiHopKeyDoor wants to keep almost everything. This is the thesis claim: **optimal memory structure is task-specific and must be learned.**

### 3.2 Performance Gains from Learned θ

| Metric | Fixed θ=(1,0,1) | Learned θ (ES) | Improvement |
|---|---|---|---|
| Key-Door success | 17.5% | 30.0% | **+71%** |
| Goal-Room success | 70.0% | 80.0% | +14% |
| MultiHop success | 0.0% | 27.5% | **∞ (0→nonzero)** |
| MultiHop reward | 0.000 | 0.100 | **new** |

MultiHopKeyDoor is the most important result: **the fixed baseline completely fails (0%), while the learned θ achieves 27.5% full success and 0.100 mean reward.** This is because the fixed θ stores everything (θ_store=1.0, θ_entity=0.0) but without entity importance filtering, the distractor keys fill the graph and crowd out the signal. The learned θ uses θ_entity=0.38 to filter distractors while retaining the hint events.

### 3.3 Memory System Comparison (MultiHopKeyDoor, n=50 episodes)

| System | Partial Score | Ret. Precision | Ret. Tokens | Efficiency |
|---|---|---|---|---|
| **EpisodicSemantic** | **0.180** | **1.000** | 1960.5 | 0.000092 |
| SemanticMemory | 0.080 | 1.000 | 1964.0 | 0.000041 |
| GraphMemory+θ | 0.053 | 0.646 | 1962.6 | 0.000027 |
| RAGMemory | 0.047 | 0.531 | 1964.0 | 0.000024 |
| FlatWindow(50) | 0.000 | 0.060 | 1964.0 | 0.000000 |
| SummaryMemory | 0.000 | 0.023 | 1636.0 | 0.000000 |

**Why EpisodicSemantic wins:** Its semantic store retains NPC hints *permanently* (never evicted), while its episodic buffer provides recent context. When the agent reaches a door at step 150, it retrieves the hint from step 1 — always. FlatWindow loses hints after 50 steps. SummaryMemory compresses them away.

**Why RAGMemory underperforms:** Dense embeddings (`all-MiniLM-L6-v2`) assign non-zero similarity to all observations (the vocabulary is too small and the observations too repetitive). A rule-based keyword detector ("you see a sign") outperforms cosine similarity on this task.

**Key thesis point:** The right *inductive bias* (semantic fact extraction) beats sophisticated *retrieval methods* (dense embeddings). Different tasks will flip this.

### 3.4 Retrieval Precision as the Causal Metric

`retrieval_precision = hint_hits / (hint_queries)` — at each step where the agent needs a hint (standing at a door), does it retrieve one?

- EpisodicSemantic: **1.000** (retrieves the relevant hint 100% of the time)
- FlatWindow(50): **0.060** (hints evicted after 50 steps)
- SummaryMemory: **0.023** (compression destroys hint details)

This is more informative than success rate because it's deterministic — it isolates *memory quality* from *policy quality*.

---

## 4. Architecture Overview

```
Tasks                  Agent              Memory             Optimization
─────────────────────  ─────────────────  ─────────────────  ────────────────────
ToyEnvironment    ──►  ExplorationPolicy  GraphMemory+θ ──►  ES (existing)
GoalRoom          ──►  (rule-based)       EpisodicSemantic   CMA-ES (new)
MultiHopKeyDoor   ──►                     HierarchicalMem    Bayesian Opt (new)
MegaQuestRoom     ──►  LLMAgent (new)     WorkingMemory      OnlineAdapter (new)
DocumentQA        ──►  (GPT-4o/mini)      CausalMemory       MetaLearner (new)
TextWorldEnv      ──►                     AttentionMemory
MultiSessionEnv   ──►                     RAGMemory
                                          SemanticMemory
                                          SummaryMemory
                                          FlatMemory
                       ContextFormatter
                       (flat/structured/compressed)
                                                             Evaluation
                                                             ──────────────────
                                                             ablation.py
                                                             transfer.py
                                                             sensitivity.py
                                                             benchmark.py
                                                             statistics.py
                                                             cost_tracker.py
```

---

## 5. What the POC Proves vs. What Remains

### ✅ POC Proves (done, results in hand)

1. **θ parameterization works** — MemoryParams controls memory structure cleanly; default θ=(1,0,1) is backward-compatible.
2. **Learning is possible** — ES finds θ that outperforms fixed θ on all 3 environments.
3. **Task-dependence confirmed** — Key-Door θ, Goal-Room θ, and MultiHopKeyDoor θ are structurally different.
4. **Memory architecture matters** — EpisodicSemantic consistently outperforms FlatWindow, SummaryMemory, and RAGMemory on the hard benchmark.
5. **Retrieval precision causally links memory quality to task success** — a clean diagnostic metric for the thesis.
6. **Visualization suite complete** — 15 figures covering graphs, learning curves, trajectories, comparisons.

### 🔲 Full Thesis Still Needs (scaffolded, not yet run)

| Phase | What | Status |
|---|---|---|
| A | New memory architectures (Hierarchical, Working, Causal, Attention, Neural) | **Implemented**, not yet benchmarked |
| B | CMA-ES, Bayesian Opt, OnlineAdapter, MetaLearner | **Implemented**, not yet run |
| C | Real LLM integration (GPT-4o, cost tracking) | **Implemented**, needs API key |
| D | Long-horizon benchmarks (MegaQuestRoom, DocumentQA, TextWorld, MultiSession) | **Implemented**, not yet run |
| E | Full ablation/transfer/sensitivity/statistical tests | **Implemented**, not yet run |
| F | Extended figures (8–15) with real data | **Code ready**, using synthetic data |
| G | Config system, CLI runner, SQLite DB | **Implemented** |

---

## 6. Next Steps (Recommended Order)

### Immediate (within 1 month)
1. **Run the full benchmark** — all 10 memory systems × MultiHopKeyDoor × 100 episodes using `python runner.py --preset benchmark`. Update Figs 8–15 with real data.
2. **Run CMA-ES on MultiHopKeyDoor** — compare to ES. CMA-ES should converge faster and reach higher θ. `python runner.py --config experiments/multihop_cmaes.yaml`
3. **Run ablation study** — quantify the contribution of each θ component. This becomes Figure 8.

### Short-term (1–3 months)
4. **MegaQuestRoom experiments** — 1000-step episodes stress-test all memory systems. FlatWindow will definitely fail; what's the winner?
5. **Sensitivity analysis** — 2D reward heatmap over θ_store × θ_entity. Reveals whether the landscape is convex (good for optimization) or rugged (hard).
6. **Cross-task transfer** — does MultiHopKeyDoor θ generalize to MegaQuestRoom? This tests meta-learning.

### Long-term / The Single Most Important Experiment
7. **DocumentQA + GPT-4o-mini + CMA-ES** — optimize `J = QA_score − λ × cost_usd`. The first demonstration that memory construction parameters can be learned to minimize real LLM API cost while maintaining answer quality on a task that genuinely exceeds the context window.

---

## 7. File Map (Quick Reference)

```
d:\Bocconi\Thesis\
├── main.py                          ← All experiments: phases 4-7, memory comparison
├── runner.py                        ← CLI: python runner.py --config experiments/X.yaml
├── config.py                        ← ExperimentConfig dataclass, YAML serializable
├── requirements.txt                 ← networkx, numpy, scikit-learn, sentence-transformers, scipy, openai, pyyaml
│
├── environment/
│   ├── env.py                       ← ToyEnvironment, GoalRoom, HardKeyDoor, QuestRoom, MultiHopKeyDoor
│   ├── mega_quest.py                ← MegaQuestRoom (20×20, 1000 steps)
│   ├── document_qa.py               ← DocumentQA (sequential reading + multi-hop QA)
│   ├── textworld_env.py             ← TextWorld wrapper (stub fallback)
│   └── multi_session.py             ← MultiSessionEnv (20 sessions, persistent memory)
│
├── memory/
│   ├── graph_memory.py              ← GraphMemory + MemoryParams (θ_store, θ_entity, θ_temporal)
│   ├── flat_memory.py               ← FlatMemory (sliding window)
│   ├── semantic_memory.py           ← SemanticMemory (importance-weighted)
│   ├── summary_memory.py            ← SummaryMemory (periodic compression)
│   ├── episodic_semantic_memory.py  ← EpisodicSemanticMemory (BEST on MultiHop)
│   ├── rag_memory.py                ← RAGMemory (sentence-transformers)
│   ├── hierarchical_memory.py       ← HierarchicalMemory (3 levels)
│   ├── working_memory.py            ← WorkingMemory (7 slots, LRU-retrieval)
│   ├── causal_memory.py             ← CausalMemory (event→action→outcome chains)
│   ├── attention_memory.py          ← AttentionMemory (softmax, differentiable)
│   └── neural_controller.py         ← NeuralMemoryController (MLP → context-dependent θ)
│
├── optimization/
│   ├── cma_es.py                    ← CMA-ES (pure numpy, scales to 4000+ params)
│   ├── bayesian_opt.py              ← Bayesian Opt (GP surrogate + Expected Improvement)
│   ├── online_adapter.py            ← StatisticsAdapter + GradientAdapter (θ during episode)
│   └── meta_learner.py              ← Reptile meta-learning (cross-task θ_init)
│
├── agent/
│   ├── policy.py                    ← ExplorationPolicy (rule-based, hint-aware)
│   ├── llm_agent.py                 ← LLMAgent (GPT-4o wrapper, token + cost tracking)
│   ├── context_formatter.py         ← ContextFormatter (flat/structured/compressed)
│   └── loop.py                      ← Episode runners → (success, events, stats_dict)
│
├── evaluation/
│   ├── run.py                       ← run_evaluation + run_memory_comparison
│   ├── ablation.py                  ← run_ablation_study (8 ablation configs)
│   ├── transfer.py                  ← run_transfer_matrix (cross-task θ)
│   ├── sensitivity.py               ← compute_sensitivity (2D reward landscape)
│   ├── benchmark.py                 ← run_full_benchmark (10 systems × all envs)
│   ├── statistics.py                ← bootstrap_ci, paired_ttest, cohens_d
│   └── cost_tracker.py              ← CostTracker (USD cost per episode)
│
├── viz/                             ← 15 figures (Figs 1-7 real data, 8-15 ready for real data)
├── results/db.py                    ← SQLite results database
├── experiments/                     ← YAML configs (multihop_cmaes, benchmark, docqa)
│
├── docs/
│   ├── THESIS_STORY.md              ← Full narrative, research rationale, thesis chapter map
│   ├── POC_RESULTS.md               ← Detailed POC results and analysis
│   └── figures/                     ← All 15 PNG figures
│
└── report.txt                       ← Full experiment output (all phases + comparison)
```

---

## 8. Reproducibility

All experiments use seeded randomness:

- Environment: `random.Random(seed)` in constructor
- Policy: `random.Random(seed)` in constructor
- Memory (θ decisions): `random.Random(hash((episode_seed, step, "memory")))` — deterministic per (seed, step) pair
- ES: seeded numpy random state per generation

Running `python main.py` with the same seeds always produces the same report.txt.

---

## 9. Thesis Chapter Map

| Chapter | Content | Key Data Source |
|---|---|---|
| 1. Introduction | The problem: memory is fixed. The claim: it should be learned. LLM context cost as motivation. | — |
| 2. POC | Grid worlds, θ parameterization, ES, 6-system comparison, 7 figures | `report.txt`, `docs/figures/fig1-7` |
| 3. Memory Architecture Taxonomy | All 10 systems compared on MegaQuestRoom + TextWorld. Inductive bias analysis. | Phase A experiments |
| 4. Learning to Construct Memory | ES vs CMA-ES vs Bayesian vs OnlineAdapter. Convergence, sample efficiency, online adaptation. | Phase B experiments |
| 5. Scaling to Real Tasks | GPT-4o agent + DocumentQA + `J = score − λ × cost_usd` | Phase C+D experiments |
| 6. Generalization | Cross-task transfer, meta-learning, ablation studies | Phase B4 + E1 + E2 |
| 7. Discussion | Limitations, connection to MemGPT/LangMem/LangGraph, future: gradient-through-retrieval, production | — |

---

*This document was auto-generated from codebase exploration on 2026-02-24. For the most recent experimental results, see `report.txt` and `docs/POC_RESULTS.md`.*
