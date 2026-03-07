# THESIS HANDOFF DOCUMENT
**Learnable Memory Construction for RL Agents**  
**Last updated:** March 2026  
**Repo:** https://github.com/UifaleanStefan/MasterThesis  
**This document replaces the need to read AGENTS.md + ALGORITHMS_AND_FINDINGS.md + all result docs.**

---

## 1. What This Project Is

This is a Master's thesis in AI. The core research question is:

> **Can an agent learn how to construct its own memory representation, and should that structure be task-dependent?**

The thesis contribution is **not** a better RL policy. It is a parameterized graph memory system where a vector θ controls *what gets stored, how entities are tracked, and how retrieval is scored*. The key claim is that the optimal θ differs across tasks — meaning memory structure is task-dependent — and that θ can be discovered automatically via black-box optimization (CMA-ES).

The project has a clear narrative arc:
- **Stage 1 (DONE):** POC — grid worlds, scalar 3D θ, Evolution Strategy, 6-system comparison
- **Stage 2 (DONE):** Full benchmark — 12 memory systems × 4 environments (incl. MegaQuestRoom), GraphMemoryV4/V5, CMA-ES, ablation, transfer, sensitivity, DocumentQA memory recall@k (no LLM), neural meta-controller
- **Stage 3 (PENDING):** Real LLM — GPT-4o + DocumentQA, optimize J = QA_score − λ × cost_usd

---

## 2. What Has Been Built

```
d:\Bocconi\Thesis\
├── main.py                        POC experiments (Phases 4–7)
├── runner.py                      CLI experiment runner
├── run_benchmark.py               12-system × 4-env benchmark (skip systems via args, e.g. RAGMemory)
├── run_document_qa_memory.py      DocumentQA memory recall@k (no LLM; skip systems via args)
├── run_smoke_tests.py             Quick pipeline check (grid + DocQA memory + DocQA+LLM path)
├── run_graphmemory_v4_cmaes.py    CMA-ES for V4 (30 gens, 200 eval eps)
├── run_ablation.py                V4 ablation study (10 configs)
├── run_transfer.py                Zero-shot transfer (4 environments)
├── run_sensitivity.py             2D reward landscape (theta_novel × w_recency)
├── run_neural_controller_v2.py    NeuralControllerV2Small CMA-ES training
├── generate_thesis_figures.py     Master figure generation (7 publication figures; --allow-missing)
├── regen_all_figures.py           Regenerate all figures (thesis + benchmark + extended with real data)
│
├── memory/
│   ├── graph_memory.py            V1: 3D θ (store, entity, temporal). ORIGINAL — never modify.
│   ├── graph_memory_v2.py         V2: 6D θ — adds learnable retrieval weights
│   ├── graph_memory_v3.py         V3: 9D θ — adds learned importance scoring
│   ├── graph_memory_v4.py         V4: 10D θ — adds Bayesian entity decay. MOST COMPLETE.
│   ├── graph_memory_v5.py         V5: V4 + attention-based storage gating
│   ├── neural_controller.py       V1 NeuralController: 36-dim → 3D θ. ORIGINAL.
│   ├── neural_controller_v2.py    V2 NeuralController: 50-dim → 10D θ. 5,674 params.
│   ├── neural_controller_v2_small.py  Smaller MLP: 50→32→10. 1,962 params. Used for CMA-ES.
│   ├── flat_memory.py             Baseline: sliding window
│   ├── semantic_memory.py         Importance-weighted pool
│   ├── summary_memory.py          Periodic compression
│   ├── episodic_semantic_memory.py Dual store (episodic + semantic)
│   ├── rag_memory.py              Dense embeddings (MiniLM)
│   ├── hierarchical_memory.py     3-level store
│   ├── working_memory.py          7-slot LRU
│   ├── causal_memory.py           Event→action→outcome chains
│   └── attention_memory.py        Softmax attention retrieval
│
├── optimization/
│   ├── cma_es.py                  CMA-ES (scales to 4500+ params)
│   ├── bayesian_opt.py            GP surrogate + EI
│   ├── online_adapter.py          θ adapts within episode
│   └── meta_learner.py            Reptile cross-task meta-learning
│
├── environment/
│   ├── env.py                     ToyEnv, GoalRoom, HardKeyDoor, MultiHopKeyDoor
│   ├── mega_quest.py              MegaQuestRoom (20×20, 1000 steps)
│   ├── document_qa.py             DocumentQA (sequential reading + multi-hop QA)
│   └── multi_session.py           MultiSessionEnv (20 sessions, persistent memory)
│
├── evaluation/
│   ├── ablation.py                AblationConfigV4, run_ablation_study_v4
│   ├── transfer.py                run_v4_transfer_matrix
│   ├── sensitivity.py             compute_sensitivity_v4 (2D landscape)
│   ├── benchmark.py               run_full_benchmark (12 systems × 4 envs; optional skip_systems)
│   └── document_qa_memory.py      DocumentQA memory-quality eval (recall@k, no LLM)
│
├── results/
│   ├── benchmark_results.json     12 systems × 4 envs
│   ├── document_qa_memory_results.json  Recall@k per system (fantasy_lore)
│   ├── graphmemory_v4_cmaes_results.json  V4 CMA-ES (30 gens × 50 eps, 200 eval)
│   ├── ablation_results.json      10 ablation configs × 100 episodes
│   ├── transfer_results.json      4 environments × 100 episodes
│   ├── sensitivity_results.json   12×12 grid, 20 eps/cell
│   └── neural_controller_v2_results.json  30 gens × 20 eps, 100 eval
│
└── docs/
    ├── THESIS_HANDOFF.md          THIS FILE
    ├── RUNNING_EXPERIMENTS.md     Exact commands and order to regenerate results/figures
    ├── DEPENDENCIES.md            Optional deps (RAG), skip/fallback behaviour
    ├── archive/                   Historical phase docs (STEP1, PHASE6, THESIS_VISION)
    ├── AGENTS.md                  Full AI agent guide
    ├── ALGORITHMS_AND_FINDINGS.md All algorithms + findings
    ├── GRAPHMEMORY_V4_RESULTS.md  V4 CMA-ES detailed analysis
    ├── ABLATION_RESULTS.md        Ablation study findings
    ├── TRANSFER_RESULTS.md        Transfer test findings
    ├── SENSITIVITY_RESULTS.md     Sensitivity analysis findings
    ├── NEURAL_CONTROLLER_V2_RESULTS.md  Neural controller findings
    └── figures/                   All publication figures (22+ PNGs)
```

---

## 3. All Experimental Results

### 3.1 POC Results (grid worlds, 3D θ, Evolution Strategy)

| Environment | Fixed θ reward | Learned θ (ES) | Learned θ vector |
|---|---|---|---|
| Key-Door | 17.5% | 30.0% | (0.31, 0.19, 0.84) |
| Goal-Room | 70.0% | 80.0% | (0.74, 0.65, 0.51) |
| **MultiHop** | **0.0%** | **27.5%** | **(0.96, 0.38, 1.00)** |

θ values differ across tasks — this is the core thesis evidence.

### 3.2 Full Benchmark — MultiHopKeyDoor (10 systems, 50 episodes each)

| Rank | System | Mean Reward | Precision | Tokens | Mem Size |
|---|---|---|---|---|---|
| 1 | **GraphMemoryV4** | **0.178** | **0.997** | 1,754 | **10.0** |
| 2 | EpisodicSemantic | 0.173 | 1.000 | 1,964 | 39.3 |
| 3 | WorkingMemory(7) | 0.153 | 1.000 | 1,722 | 7.0 |
| 3 | AttentionMemory | 0.153 | 1.000 | 1,952 | 248.5 |
| 5 | SemanticMemory | 0.133 | 1.000 | 1,964 | 80.0 |
| 6 | HierarchicalMemory | 0.127 | 1.000 | 1,964 | 39.3 |
| 7 | CausalMemory | 0.100 | 1.000 | 1,964 | 43.0 |
| 8 | RAGMemory | 0.053 | 0.482 | 1,964 | 250.0 |
| 9 | GraphMemoryV1 | 0.033 | 0.578 | 1,962 | 239.5 |
| 10 | FlatWindow(50) | 0.000 | 0.028 | 1,964 | 50.0 |
| 11 | SummaryMemory | 0.000 | 0.010 | 1,636 | 38.0 |

V4 is **#1** with 22× smaller memory footprint than V1 (10 vs 218 events).

### 3.3 CMA-ES Optimization — GraphMemoryV4 vs V1 (200 eval episodes)

| Metric | V1 (3D θ) | V4 (10D θ) | Improvement |
|---|---|---|---|
| Mean reward | 0.1017 | **0.1783** | +75.3% |
| Retrieval precision | 0.632 | **0.997** | +57.7% |
| Mean memory size | 218.2 | **10.0** | 22× smaller |
| Mean tokens | 1,958 | 1,754 | −10.4% |
| Optimization time | — | 35.4 min | 30 gens × 50 eps |

**Learned V4 θ:** theta_store=0.293, theta_novel=0.908, theta_erich=0.198, theta_surprise=0.785, theta_entity=0.285, theta_temporal=0.278, theta_decay=0.668, w_graph=0.000, w_embed=1.079, w_recency=3.777

### 3.4 Ablation Study (100 episodes per config)

| Config | Mean Reward | Reward Degradation | Precision |
|---|---|---|---|
| Full V4 (reference) | 0.163 | 0.0% | 0.999 |
| no_novelty (theta_novel=0) | 0.000 | **100%** | 0.000 |
| store_all (no filtering) | 0.000 | **100%** | 0.024 |
| v1_equivalent | 0.013 | 91.8% | 0.208 |
| no_erich (theta_erich=0) | 0.073 | 55.1% | 1.000 |
| no_embed (w_embed=0) | 0.123 | 24.5% | 0.912 |
| no_decay (theta_decay=0) | 0.137 | 16.3% | 0.996 |
| graph_only | 0.140 | 14.3% | 1.000 |
| no_surprise (theta_surprise=0) | 0.173 | -6.1% | 1.000 |
| no_recency (w_recency=0) | 0.170 | -4.1% | 1.000 |

**Key finding:** theta_novel is the load-bearing pillar (100% degradation when removed).

### 3.5 Zero-Shot Transfer (100 episodes per environment)

| Environment | Mean Reward | Mean Tokens | Notes |
|---|---|---|---|
| MultiHopKeyDoor (source) | 0.163 | 1,745 | In-distribution |
| GoalRoom | **0.690** | 47 | Strong positive transfer |
| HardKeyDoor | 0.160 | 1,382 | Moderate transfer |
| MegaQuestRoom | **0.000** | 7,908 | Complete failure (OOD) |

**Key finding:** V4 theta transfers well to simpler/similar tasks but fails completely on harder OOD tasks. Confirms task-dependence hypothesis.

### 3.6 Sensitivity Analysis (12×12 grid, theta_novel × w_recency)

- **Best cell:** theta_novel=0.909, w_recency=3.636 → reward=0.200
- **Broad plateau:** reward stays ≥ 0.150 across a wide region (robust, not brittle)
- **theta_novel dominates:** low theta_novel (<0.3) always gives reward ≤ 0.05 regardless of w_recency

### 3.7 NeuralControllerV2Small Training (30 gens × 20 eps, 100 eval)

| System | Reward | Precision | Memory Size |
|---|---|---|---|
| V4 Scalar (reference) | 0.178 | 0.997 | 10.0 |
| NeuralV2Small (1,962 params) | 0.033 | 0.352 | 41.7 |
| V1 Scalar (reference) | 0.102 | 0.632 | 218.2 |

**Key finding:** Neural controller underperforms scalar V4 by 81%. Root cause: sigma=0.05 too small for 1,962D weight space, 30 gens insufficient. Valuable negative result confirming expressivity-trainability tradeoff.

---

## 4. Key Findings (Thesis-Quality Claims)

1. **Memory structure is task-dependent.** The optimal θ for MultiHopKeyDoor (theta_novel=0.908, w_recency=3.777) differs substantially from GoalRoom. No single θ is universally optimal.

2. **Selective storage beats comprehensive storage.** V4 stores only ~10 events per episode (vs V1's 218) while achieving 75% higher reward. Filtering by novelty and surprise is more valuable than storing everything.

3. **theta_novel is the load-bearing pillar.** Removing the novelty feature causes 100% reward degradation. The system cannot function without novelty-based storage filtering.

4. **Learned memory does not zero-shot transfer to harder OOD tasks.** V4 theta achieves 0.69 reward on GoalRoom but 0.00 on MegaQuestRoom. Task-specific optimization is necessary for hard tasks.

5. **Neural meta-controllers face a trainability bottleneck.** A 1,962-parameter MLP trained via CMA-ES underperforms a 10-parameter scalar θ. The expressivity-trainability tradeoff is real and requires either larger training budgets or gradient-based optimization.

---

## 5. What to Do Next (Priority Order)

| Priority | Task | Estimated Runtime | Command |
|---|---|---|---|
| **1 (HIGH)** | DocumentQA + GPT-4o-mini + Bayesian Opt | ~2h (needs API key) | `python runner.py --config experiments/document_qa_llm.yaml` |
| **2 (HIGH)** | NeuralControllerV2 full training (sigma=0.3, 200 gens) | ~8h overnight | `python run_neural_controller_v2.py --generations 200 --sigma 0.3` |
| **3 (MEDIUM)** | GraphMemoryV5 — attention-based storage gating | ~2h dev + 1h run | New file: `memory/graph_memory_v5.py` |
| **4 (MEDIUM)** | Cross-task meta-learning (Reptile) | ~3h | `python runner.py --config experiments/meta_learning.yaml` |
| **5 (LOW)** | TextWorld integration | ~1 day dev | `environment/textworld_env.py` (stub exists) |
| **6 (LOW)** | MultiSessionEnv persistent memory test | ~2h | `environment/multi_session.py` (implemented) |

---

## 6. How to Run Things

```powershell
# Set working directory (always do this first)
Set-Location "d:\Bocconi\Thesis"

# Run the full POC experiment pipeline
python main.py

# Run the full 10-system benchmark (~75 seconds)
python run_benchmark.py

# Run V4 CMA-ES optimization (~35 minutes)
python run_graphmemory_v4_cmaes.py

# Run ablation study (~3 minutes)
python run_ablation.py

# Run zero-shot transfer test (~2 minutes)
python run_transfer.py

# Run sensitivity analysis (~5 minutes)
python run_sensitivity.py

# Run neural controller training (~80 minutes)
python run_neural_controller_v2.py

# Regenerate all thesis figures (from real data, ~10 seconds)
python generate_thesis_figures.py

# Run a specific experiment config
python runner.py --config experiments/multihop_cmaes.yaml

# Commit and push (PowerShell-compatible)
git add --all
$msg = @"
Your commit message here
"@
$msg | Out-File -FilePath ".\commit_msg.txt" -Encoding utf8
git commit -F ".\commit_msg.txt"
Remove-Item ".\commit_msg.txt"
git push origin master
```

---

## 7. Critical Design Decisions (Do NOT Change)

| Decision | Rationale |
|---|---|
| **Token usage = sum of retrieved events per episode** (not memory size) | Aligns with LLM context cost — what gets sent to the LLM is what costs money |
| **Entity importance = frequency-based** (Bayesian count model) | Task-agnostic — no task-specific strings in memory code |
| **Reproducibility via `random.Random(hash((episode_seed, step, "memory")))`** | Ensures experiments are reproducible without global state |
| **Default θ = (1.0, 0.0, 1.0)** for V1 | Reproduces the unparameterized baseline exactly. Backward compatible. |
| **No task-specific knowledge in memory systems** | Memory must learn from reward alone. Task logic lives only in `agent/policy.py` and `environment/` |
| **Separate files for each version** (V1, V2, V3, V4) | The progression V1→V4 is itself a thesis contribution. Never overwrite originals. |
| **CMA-ES sigma=0.3 for scalar θ, sigma=0.05 for neural** | Scalar θ needs wider exploration; neural weights are near zero initially |
| **MultiHopKeyDoor is the primary benchmark** | It requires multi-hop reasoning with hints at steps 0–2, creating genuine memory pressure |

---

## 8. Figures Reference

| Figure | File | Description | Data Source |
|---|---|---|---|
| fig_master_benchmark | `docs/figures/fig_master_benchmark.png` | 2×2 panel: reward ranking, reward vs precision, CMA-ES curves, memory size | Real data |
| fig_ablation_ranked | `docs/figures/fig_ablation_ranked.png` | Horizontal bar: degradation by feature removed | `ablation_results.json` |
| fig_transfer_annotated | `docs/figures/fig_transfer_annotated.png` | Transfer heatmap + token cost bars | `transfer_results.json` |
| fig_sensitivity_annotated | `docs/figures/fig_sensitivity_annotated.png` | 2D reward landscape with contours + optimum | `sensitivity_results.json` |
| fig_neural_analysis | `docs/figures/fig_neural_analysis.png` | Training curve + comparison + "what would be needed" | `neural_controller_v2_results.json` |
| fig11_pareto | `docs/figures/fig11_pareto.png` | Pareto front: reward vs token cost | Real data |
| fig13_memory_size | `docs/figures/fig13_memory_size.png` | Memory footprint + reward vs size scatter | Real data |
| fig1–fig7 | `docs/figures/fig1_*.png` … `fig7_*.png` | POC figures (memory graphs, ES curves, theta trajectory, etc.) | Real data |
| fig08_ablation_v4 | `docs/figures/fig08_ablation_v4.png` | Earlier ablation visualization | `ablation_results.json` |
| fig09_landscape_v4 | `docs/figures/fig09_landscape_v4.png` | Earlier sensitivity landscape | `sensitivity_results.json` |
| fig10_transfer_v4 | `docs/figures/fig10_transfer_v4.png` | Earlier transfer visualization | `transfer_results.json` |
| fig_neural_v2_curves | `docs/figures/fig_neural_v2_curves.png` | Earlier neural controller curves | `neural_controller_v2_results.json` |

Regenerate all figures at any time: `python generate_thesis_figures.py`
