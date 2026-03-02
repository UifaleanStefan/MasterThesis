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
├── run_benchmark.py           ← Full 10-system × 3-env benchmark (run this for results)
├── config.py                  ← ExperimentConfig dataclass, YAML serializable
├── report.txt                 ← Full experiment output from last main.py run
├── requirements.txt
│
├── memory/
│   ├── graph_memory.py        ← V1: ORIGINAL — 3D θ (store, entity, temporal). DO NOT MODIFY.
│   ├── graph_memory_v2.py     ← V2: 6D θ — adds learnable retrieval weights (w_graph, w_embed, w_recency)
│   ├── graph_memory_v3.py     ← V3: 9D θ — adds learned importance scoring (novelty, richness, surprise)
│   ├── graph_memory_v4.py     ← V4: 10D θ — adds Bayesian entity decay (theta_decay). MOST COMPLETE.
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
│   ├── benchmark.py           ← run_full_benchmark (10 systems × all envs)
│   ├── ablation.py            ← run_ablation_study
│   ├── transfer.py            ← run_transfer_matrix
│   ├── sensitivity.py         ← compute_sensitivity (2D reward landscape)
│   ├── statistics.py          ← bootstrap_ci, paired_ttest, cohens_d
│   └── cost_tracker.py        ← CostTracker (USD cost per episode)
│
├── results/
│   ├── db.py                  ← SQLite results database
│   └── benchmark_results.json ← REAL benchmark data (10 systems × 3 envs × 50 episodes)
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
```

**V4 is the most complete scalar θ system.** Each version is a separate file — originals are never modified.

### Lineage 2: Neural θ (context-dependent θ per observation)

```
neural_controller.py     — MLP: 36-dim input → 3D θ output. Wraps V1 GraphMemory.
neural_controller_v2.py  — MLP: 50-dim input → 10D θ output. Wraps V4 GraphMemory.
                           5,674 parameters. Trained via CMA-ES.
```

**NeuralControllerV2 is the most ambitious system.** It decides all 10 θ dimensions per observation from task-agnostic features (novelty, entity richness, surprise, entropy, etc.).

### The 10 Benchmark Competitors

| # | System | MultiHop Rank | Precision |
|---|---|---|---|
| 1 | EpisodicSemantic | #1 (0.173) | 1.000 |
| 2 | WorkingMemory(7) | #2 (0.153) | 1.000 |
| 3 | AttentionMemory | #2 (0.153) | 1.000 |
| 4 | SemanticMemory | #4 (0.133) | 1.000 |
| 5 | HierarchicalMemory | #5 (0.127) | 1.000 |
| 6 | CausalMemory | #6 (0.100) | 1.000 |
| 7 | RAGMemory | #7 (0.053) | 0.482 |
| 8 | **GraphMemory+θ (V1)** | #8 (0.033) | 0.578 |
| 9 | FlatWindow | #9 (0.000) | 0.028 |
| 10 | SummaryMemory | #10 (0.000) | 0.010 |

**GraphMemory+θ currently ranks #8.** The V2/V3/V4 improvements are designed to push it into the precision=1.000 tier. This has not been run yet.

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

| Experiment | Command / Config | What it tests | Priority |
|---|---|---|---|
| CMA-ES on MultiHop with V4 GraphMemory | `python runner.py --config experiments/multihop_cmaes.yaml` | Does 10D θ push GraphMemory to precision=1.000? | **HIGH** |
| Ablation study | `evaluation/ablation.py` | Which θ component contributes most? | **HIGH** |
| NeuralControllerV2 training | `python runner.py --config experiments/neural_controller_v2_cmaes.yaml` | Can MLP meta-controller generalize? | HIGH |
| Zero-shot transfer test | Train on MultiHop, eval on MegaQuestRoom | Does learned policy transfer? | HIGH |
| Sensitivity analysis | `evaluation/sensitivity.py` | Is the reward landscape convex or rugged? | Medium |
| Cross-task transfer matrix | `evaluation/transfer.py` | Does MultiHop θ generalize? | Medium |
| DocumentQA + Bayesian Opt | `experiments/document_qa_llm.yaml` | Real LLM cost optimization | **HIGH (needs API key)** |

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

# Run functional tests for GraphMemory variants
Set-Location "d:\Bocconi\Thesis"; python test_graph_memory_variants.py

# Regenerate benchmark figures from real data
Set-Location "d:\Bocconi\Thesis"; python regen_benchmark_figs.py

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
| `docs/PROJECT_SUMMARY.md` | Full inventory of everything implemented |
| `docs/ALGORITHMS_AND_FINDINGS.md` | All algorithms with pseudocode + experimental findings |
| `docs/BENCHMARK_RESULTS.md` | Full analysis of the 10-system benchmark |
| `docs/THESIS_STORY.md` | Research narrative and rationale |
| `docs/POC_RESULTS.md` | POC phase results in detail |
| `results/benchmark_results.json` | Raw benchmark data (10 systems × 3 envs × 50 episodes) |
| `report.txt` | Full output from last `python main.py` run |
