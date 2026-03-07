# Recent Changes — Session Log

**Purpose:** Record of work done on the thesis codebase (analysis, fixes, dashboard, figures).  
**Last updated:** March 2026

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
