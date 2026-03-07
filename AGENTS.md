# AGENTS.md — AI Agent Guide for the Learnable Memory Thesis Project

**For:** Any AI assistant (Cursor, Claude, GPT-4o, etc.) picking up this project  
**Last updated:** March 2026  
**Repo:** https://github.com/UifaleanStefan/MasterThesis  
**Read this before touching any code.**

---

## 1. What This Project Is

This is a Master's thesis in AI. The research question is:

> **Can an agent learn how to construct its own memory representation, and should that structure be task-dependent?**

This is NOT a standard RL project. The contribution is not a better policy. It is:

- A parameterized graph memory system where **θ controls what gets stored and how**
- Evidence that **optimal θ differs across tasks** (task-dependent memory)
- A progression from scalar θ → importance-scored θ → neural meta-controller

The thesis has a clear narrative arc: POC (grid worlds, scalar θ) → full framework (LLM agent, DocumentQA, real token cost). We are currently between these two stages.

---

## 2. Project Structure (Quick Reference)

```
d:\Bocconi\Thesis\
├── main.py                    ← All POC experiments (Phases 4-7, memory comparison)
├── runner.py                  ← CLI: python runner.py --config experiments/X.yaml
├── run_benchmark.py           ← Full 12-system × 4-env benchmark (run this for results)
├── run_document_qa_memory.py  ← DocumentQA memory recall@k (no LLM)
├── run_smoke_tests.py         ← Quick pipeline check (grid + DocQA memory + DocQA+LLM path)
├── config.py                  ← ExperimentConfig dataclass, YAML serializable
├── report.txt                 ← Full experiment output from last main.py run
├── requirements.txt
├── scripts/                   ← Dev utilities (run from project root)
│   ├── find_seed.py           ← Find seeds where EpisodicSemantic opens ≥1 door
│   └── test_new_systems.py    ← Functional test for Phase A memory systems + envs
│
├── memory/
│   ├── graph_memory.py        ← V1: ORIGINAL — 3D θ (store, entity, temporal). DO NOT MODIFY.
│   ├── graph_memory_v2.py     ← V2: 6D θ — adds learnable retrieval weights (w_graph, w_embed, w_recency)
│   ├── graph_memory_v3.py     ← V3: 9D θ — adds learned importance scoring (novelty, richness, surprise)
│   ├── graph_memory_v4.py     ← V4: 10D θ — adds Bayesian entity decay (theta_decay). MOST COMPLETE.
│   ├── graph_memory_v5.py     ← V5: V4 + attention-based storage gating.
│   ├── neural_controller.py   ← V1 NeuralController: MLP → 3D θ, 36-dim input. DO NOT MODIFY.
│   ├── neural_controller_v2.py← V2 NeuralController: MLP → 10D θ, 50-dim input. MOST ADVANCED.
│   ├── flat_memory.py         ← Baseline floor (sliding window)
│   ├── semantic_memory.py     ← Importance-weighted pool
│   ├── summary_memory.py      ← Periodic compression
│   ├── episodic_semantic_memory.py ← BEST PERFORMER on MultiHop (dual store)
│   ├── rag_memory.py          ← Dense embeddings (sentence-transformers MiniLM)
│   ├── hierarchical_memory.py ← 3-level store (raw → summaries → permanent)
│   ├── working_memory.py      ← 7-slot LRU-retrieval eviction
│   ├── causal_memory.py       ← event→action→outcome chains
│   └── attention_memory.py    ← Softmax attention retrieval
│
├── optimization/
│   ├── cma_es.py              ← CMA-ES (scales to 4500+ params for NeuralControllerV2)
│   ├── bayesian_opt.py        ← GP surrogate + Expected Improvement
│   ├── online_adapter.py      ← θ adapts within an episode (StatisticsAdapter, GradientAdapter)
│   └── meta_learner.py        ← Reptile cross-task meta-learning
│
├── environment/
│   ├── env.py                 ← ToyEnvironment, GoalRoom, HardKeyDoor, QuestRoom, MultiHopKeyDoor
│   ├── mega_quest.py          ← MegaQuestRoom (20×20, 1000 steps)
│   ├── document_qa.py         ← DocumentQA (sequential reading + multi-hop QA)
│   ├── textworld_env.py       ← TextWorld wrapper (stub if not installed)
│   └── multi_session.py       ← MultiSessionEnv (20 sessions, persistent memory)
│
├── agent/
│   ├── policy.py              ← ExplorationPolicy (rule-based, hint-aware)
│   ├── loop.py                ← Episode runners → (success, events, stats_dict)
│   ├── llm_agent.py           ← LLMAgent (GPT-4o wrapper, cost tracking)
│   └── context_formatter.py   ← ContextFormatter (flat/structured/compressed)
│
├── evaluation/
│   ├── run.py                 ← run_evaluation + run_memory_comparison
│   ├── benchmark.py           ← run_full_benchmark (12 systems × 4 envs)
│   ├── document_qa_memory.py  ← DocumentQA memory-quality (recall@k, no LLM)
│   ├── ablation.py            ← run_ablation_study
│   ├── transfer.py            ← run_transfer_matrix
│   ├── sensitivity.py         ← compute_sensitivity (2D reward landscape)
│   ├── statistics.py          ← bootstrap_ci, paired_ttest, cohens_d
│   └── cost_tracker.py        ← CostTracker (USD cost per episode)
│
├── results/
│   ├── db.py                  ← SQLite results database
│   └── benchmark_results.json ← REAL benchmark data (12 systems × 4 envs)
│
├── experiments/
│   ├── multihop_cmaes.yaml
│   ├── benchmark.yaml
│   ├── document_qa_llm.yaml
│   └── neural_controller_v2_cmaes.yaml  ← NEW: train NeuralControllerV2 via CMA-ES
│
├── viz/                       ← 15 figures (fig1-7 real data, fig8-15 synthetic placeholders)
│
└── docs/
    ├── AGENTS.md              ← THIS FILE
    ├── PROJECT_SUMMARY.md     ← Full implementation inventory + thesis chapter map
    ├── ALGORITHMS_AND_FINDINGS.md ← All algorithms documented + experimental findings
    ├── BENCHMARK_RESULTS.md   ← Full analysis of the 10-system benchmark
    ├── THESIS_STORY.md        ← Research narrative and thesis arc
    ├── POC_RESULTS.md         ← POC experimental results
    └── figures/               ← All 15+ PNG figures
```

---

## 3. The Memory System Hierarchy (Most Important to Understand)

There are two parallel lineages:

### Lineage 1: Scalar θ GraphMemory (the core contribution)

```
graph_memory.py (V1)     — 3D θ = (store, entity, temporal). ORIGINAL. Backward compatible.
graph_memory_v2.py (V2)  — 6D θ = V1 + (w_graph, w_embed, w_recency). Retrieval now learnable.
graph_memory_v3.py (V3)  — 9D θ = V2 + (novel, erich, surprise). Storage now importance-scored.
graph_memory_v4.py (V4)  — 10D θ = V3 + (theta_decay). Entity importance now Bayesian + temporal.
graph_memory_v5.py (V5)  — V4 + attention-based storage gating (softmax over recent embeddings).
```

**V4 is the most complete scalar θ system.** V5 adds attention-based gating at storage time. Each version is a separate file — originals are never modified.

### Lineage 2: Neural θ (context-dependent θ per observation)

```
neural_controller.py     — MLP: 36-dim input → 3D θ output. Wraps V1 GraphMemory.
neural_controller_v2.py  — MLP: 50-dim input → 10D θ output. Wraps V4 GraphMemory.
                           5,674 parameters. Trained via CMA-ES.
```

**NeuralControllerV2 is the most ambitious system.** It decides all 10 θ dimensions per observation from task-agnostic features (novelty, entity richness, surprise, entropy, etc.).

### The 10 Benchmark Competitors

| # | System | MultiHop Rank | Precision | Notes |
|---|---|---|---|---|
| 1 | **GraphMemoryV4** | **#1 (0.178)** | **0.997** | **NEW — was #8** |
| 2 | EpisodicSemantic | #2 (0.173) | 1.000 | Previous #1 |
| 3 | WorkingMemory(7) | #3 (0.153) | 1.000 | |
| 4 | AttentionMemory | #3 (0.153) | 1.000 | |
| 5 | SemanticMemory | #5 (0.133) | 1.000 | |
| 6 | HierarchicalMemory | #6 (0.127) | 1.000 | |
| 7 | CausalMemory | #7 (0.100) | 1.000 | |
| 8 | RAGMemory | #8 (0.053) | 0.482 | |
| 9 | GraphMemoryV1 | #9 (0.033) | 0.578 | |
| 10 | FlatWindow | #10 (0.000) | 0.028 | |
| 11 | SummaryMemory | #11 (0.000) | 0.010 | |

**GraphMemoryV4 is now the #1 system on MultiHopKeyDoor** (reward=0.178, precision=0.997), surpassing EpisodicSemantic. Achieved via CMA-ES on 10D theta with 30 gens × 50 eps. See `docs/GRAPHMEMORY_V4_RESULTS.md` for full analysis.

**NeuralControllerV2Small** (50->32->10 MLP, 1,962 params): 30-gen run achieved reward=0.033 (below V4). **200-gen run** (sigma=0.3, ~15 h) achieved **reward=0.19** on MultiHop — matches or exceeds scalar V4 (0.178). See `docs/NEURAL_CONTROLLER_V2_RESULTS.md`.

---

## 4. Key Experimental Results (What We Know)

### POC Results (real data, in report.txt)

| Environment | Fixed θ reward | Learned θ (ES) | θ learned |
|---|---|---|---|
| Key-Door | 17.5% | 30.0% | (0.31, 0.19, 0.84) |
| Goal-Room | 70.0% | 80.0% | (0.74, 0.65, 0.51) |
| **MultiHop** | **0.0%** | **27.5%** | **(0.96, 0.38, 1.00)** |

The θ values differ across tasks — this is the core thesis evidence.

### Benchmark Results (real data, in results/benchmark_results.json)

- Retrieval precision = 1.000 is the gating factor on MultiHop
- Pearson r ≈ 0.96 between precision and reward across 10 systems
- No universal winner across environments (SummaryMemory #1 on Goal-Room, #10 on MultiHop)
- All 4 new architectures (Working, Attention, Hierarchical, Causal) achieve precision=1.000

---

## 5. What Has NOT Been Run Yet (Pending Experiments)

These are implemented but have never been executed:

| Experiment | Command / Config | What it tests | Priority | Status |
|---|---|---|---|---|
| ~~CMA-ES on MultiHop with V4 GraphMemory~~ | `run_graphmemory_v4_cmaes.py` | Does 10D θ push GraphMemory to precision=1.000? | HIGH | **DONE** — reward=0.178, precision=0.997, #1 ranking |
| ~~Ablation study~~ | `run_ablation.py` | Which θ component contributes most? | HIGH | **DONE** — theta_novel is critical (100% degradation), theta_erich is 2nd (55%) |
| ~~NeuralControllerV2Small training~~ | `run_neural_controller_v2.py` | Can MLP meta-controller beat scalar V4? | HIGH | **DONE** — reward=0.033 (vs V4=0.178), underperforms due to optimization challenges |
| ~~Zero-shot transfer test~~ | `run_transfer.py` | Does learned V4 theta transfer? | HIGH | **DONE** — GoalRoom=0.69, HardKeyDoor=0.16, MegaQuestRoom=0.00 |
| ~~Sensitivity analysis~~ | `run_sensitivity.py` | Is the reward landscape convex or rugged? | Medium | **DONE** — broad plateau (not sharp peak), theta_novel dominates |
| DocumentQA + Bayesian Opt | `experiments/document_qa_llm.yaml` | Real LLM cost optimization | **HIGH (needs API key)** | Pending |
| ~~NeuralControllerV2 full training~~ | `run_neural_controller_v2.py --generations 200 --sigma 0.3` | Full neural controller with larger budget | HIGH | **DONE** — reward=0.19 (≥ V4 0.178), MegaQuest 0.0 |

---

## 6. Coding Rules (MUST FOLLOW)

### Never modify originals
- `memory/graph_memory.py` — V1 original, never touch
- `memory/neural_controller.py` — V1 original, never touch
- When improving a system, create a new file: `graph_memory_v5.py`, etc.

### File naming convention
- GraphMemory variants: `memory/graph_memory_v{N}.py`
- NeuralController variants: `memory/neural_controller_v{N}.py`
- Corresponding MemoryParams: `MemoryParamsV{N}` inside the same file

### Interface contract (all memory systems must implement)
```python
def add_event(self, event: Event, episode_seed: int | None = None) -> None
def get_relevant_events(self, observation: str, current_step: int, k: int = 8) -> list[Event]
def clear(self) -> None
def get_stats(self) -> dict
```

### Reproducibility
- All stochastic decisions in memory construction use `random.Random(hash((episode_seed, step, "memory")))`
- Never use `random.random()` directly in memory code

### No task-specific hardcoding in memory
- Memory systems must NOT contain strings like `"see a sign:"`, `"guard says"`, `"red key"`, etc.
- These are task-specific. Memory must learn from reward alone.
- Task-specific logic belongs ONLY in `agent/policy.py` and `environment/`

### PowerShell compatibility
- Use `;` not `&&` to chain commands: `Set-Location "d:\Bocconi\Thesis"; python script.py`
- Use `$msg = @"..."@` + `$msg | Out-File commit_msg.txt` for multiline git commit messages
- Never use bash heredoc syntax (`<<'EOF'`) — it doesn't work in PowerShell

---

## 7. How to Run Things

```powershell
# Run the full POC experiment pipeline
Set-Location "d:\Bocconi\Thesis"; python main.py

# Run the full 10-system benchmark (takes ~75 seconds)
Set-Location "d:\Bocconi\Thesis"; python run_benchmark.py

# Run a specific experiment config
Set-Location "d:\Bocconi\Thesis"; python runner.py --config experiments/multihop_cmaes.yaml

# Full list of commands and order to regenerate all results/figures
# See docs/RUNNING_EXPERIMENTS.md

# Smoke tests (quick pipeline check)
Set-Location "d:\Bocconi\Thesis"; python run_smoke_tests.py

# Run functional tests for GraphMemory variants
Set-Location "d:\Bocconi\Thesis"; python test_graph_memory_variants.py

# Regenerate thesis figures (requires results/*.json; use --allow-missing for partial)
Set-Location "d:\Bocconi\Thesis"; python generate_thesis_figures.py

# Regenerate all figures (thesis + benchmark fig5 + extended Fig 8–15 with real data when available)
Set-Location "d:\Bocconi\Thesis"; python regen_all_figures.py

# Regenerate benchmark figures from real data
Set-Location "d:\Bocconi\Thesis"; python regen_benchmark_figs.py

# Fig 6 & 7 (grid trajectory, episode curves) require live runs:
Set-Location "d:\Bocconi\Thesis"; python regen_figs.py

# Interactive dashboard (Streamlit; requires streamlit, pandas):
Set-Location "d:\Bocconi\Thesis"; streamlit run dashboard/app.py

# Commit and push
Set-Location "d:\Bocconi\Thesis"
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

## 8. The θ Parameter Vectors (Quick Reference)

| Version | Dimensions | Parameters |
|---|---|---|
| V1 (original) | 3D | θ_store, θ_entity, θ_temporal |
| V2 | 6D | + w_graph, w_embed, w_recency |
| V3 | 9D | + θ_novel, θ_erich, θ_surprise |
| V4 (full) | 10D | + θ_decay |
| NeuralControllerV2 | ~5,674 weights → 10D output | MLP meta-controller |

All V2/V3/V4 classes have `from_vector(v)` and `to_vector()` methods for CMA-ES integration.

---

## 9. What's Next (Recommended Order)

### Immediate (run existing code)
1. **Run CMA-ES on MultiHopKeyDoor with GraphMemoryV4** — does 10D θ push it to precision=1.000? This is the most important pending experiment. `experiments/multihop_cmaes.yaml` needs updating to use `GraphMemoryV4` and `n_params=10`.
2. **Run ablation study** — which θ component matters most? `evaluation/ablation.py` is ready.
3. **Regenerate Figs 8–15 with real data** — currently all synthetic placeholders.

### Short-term (new experiments)
4. **Train NeuralMemoryControllerV2** — CMA-ES on 5,674 weights. Config at `experiments/neural_controller_v2_cmaes.yaml`. This is the thesis's most ambitious experiment.
5. **Zero-shot transfer test** — train on MultiHop, evaluate on MegaQuestRoom without retraining.
6. **Sensitivity analysis** — 2D reward heatmap over θ_store × θ_entity.

### Long-term (needs OpenAI API key)
7. **DocumentQA + GPT-4o-mini + Bayesian Opt** — optimize `J = QA_score − λ × cost_usd`. The single most important experiment for the thesis. This is the first real demonstration that memory construction parameters can be learned to minimize LLM API cost.

---

## 10. Thesis Chapter Map

| Chapter | Content | Key Data |
|---|---|---|
| 1. Introduction | The problem, the claim, LLM cost motivation | — |
| 2. POC | Grid worlds, 3D θ, ES, 6-system comparison, 7 figures | `report.txt`, `docs/figures/fig1-7` |
| 3. Memory Architecture Taxonomy | All 10 systems on MegaQuestRoom + TextWorld | Phase A experiments |
| 4. Learning to Construct Memory | ES vs CMA-ES vs Bayesian vs OnlineAdapter. V1→V4 progression. NeuralControllerV2. | Phase B experiments |
| 5. Scaling to Real Tasks | GPT-4o + DocumentQA + `J = score − λ × cost_usd` | Phase C+D experiments |
| 6. Generalization | Cross-task transfer, meta-learning, ablation | Phase B4 + E1 + E2 |
| 7. Discussion | Limitations, MemGPT/LangMem connection, future work | — |

---

## 11. Key Design Decisions (Don't Revisit Without Good Reason)

- **Token usage = sum of retrieved events per episode** (not memory size). This aligns with LLM context cost.
- **Entity importance = frequency-based** (count/total), not task-specific. Generalization requires this.
- **Reproducibility via `random.Random(hash((episode_seed, step, "memory")))`** — never deviate.
- **Default θ = (1.0, 0.0, 1.0)** — reproduces the unparameterized baseline exactly. Backward compatible.
- **No task-specific knowledge in memory systems** — the system must learn from reward alone.
- **Separate files for each version** — never overwrite originals. The progression V1→V4 is itself a thesis contribution.

---

## 12. Documentation Files (Read Before Implementing)

| File | When to read |
|---|---|
| `docs/THESIS_HANDOFF.md` | **Single-document project summary** — read this first on a new machine |
| `docs/RECENT_CHANGES.md` | **Session log** — bar chart fixes, dashboard, figure audit, pipeline changes |
| `docs/RUNNING_EXPERIMENTS.md` | **Exact commands and order** to regenerate all result files and figures |
| `docs/DEPENDENCIES.md` | Optional deps (RAG/sentence_transformers), skip/fallback behaviour |
| `docs/PROJECT_SUMMARY.md` | Full inventory of everything implemented |
| `docs/ALGORITHMS_AND_FINDINGS.md` | All algorithms with pseudocode + experimental findings |
| `docs/BENCHMARK_RESULTS.md` | Full analysis of the 12-system × 4-env benchmark |
| `docs/THESIS_STORY.md` | Research narrative and rationale |
| `docs/POC_RESULTS.md` | POC phase results in detail |
| `results/benchmark_results.json` | Raw benchmark data (12 systems × 4 envs) |
| `report.txt` | Full output from last `python main.py` run |

---

## 13. Latest Results Summary

*All experiments complete as of March 2026. Numbers are from real data.*

### Key Numbers at a Glance

| Experiment | Best Result | File |
|---|---|---|
| V4 CMA-ES (200 eval eps) | reward=**0.178**, precision=**0.997**, mem=**10 events** | `results/graphmemory_v4_cmaes_results.json` |
| V1 baseline (200 eval eps) | reward=0.102, precision=0.632, mem=218 events | same file |
| Ablation — theta_novel removed | reward=**0.000** (100% degradation) | `results/ablation_results.json` |
| Ablation — theta_erich removed | reward=0.073 (55% degradation) | same file |
| Transfer — GoalRoom | reward=**0.690** (strong positive) | `results/transfer_results.json` |
| Transfer — MegaQuestRoom | reward=**0.000** (complete failure) | same file |
| Sensitivity — best cell | theta_novel=0.909, w_recency=3.636 → reward=**0.200** | `results/sensitivity_results.json` |
| NeuralV2Small (30 gens) | reward=0.033 (below V4) | same file |
| NeuralV2Small (200 gens) | reward=**0.19** (≥ V4 0.178), MegaQuest 0.0 | `results/neural_controller_v2_results.json` |

### What This Means for the Thesis

1. **GraphMemoryV4 is the current state-of-the-art** on MultiHopKeyDoor (reward=0.178, #1 of 11 systems)
2. **theta_novel is non-negotiable** — the system breaks without novelty-based storage filtering
3. **Memory is task-dependent** — V4 theta fails completely on MegaQuestRoom (OOD)
4. **Neural meta-controllers need more compute** — 30 gens with sigma=0.05 is insufficient for 1,962D space
5. **The next critical experiment is DocumentQA + GPT-4o** — this is the first real LLM cost demonstration

### Figures Generated (all from real data)

```
docs/figures/fig_master_benchmark.png    — 2×2 master summary (most important figure)
docs/figures/fig_ablation_ranked.png     — feature importance ranking
docs/figures/fig_transfer_annotated.png  — transfer heatmap with annotations
docs/figures/fig_sensitivity_annotated.png — 2D landscape with contours
docs/figures/fig_neural_analysis.png     — neural controller analysis
docs/figures/fig11_pareto.png            — Pareto front (reward vs token cost)
docs/figures/fig13_memory_size.png       — memory footprint comparison
```

Regenerate all: `python generate_thesis_figures.py`
