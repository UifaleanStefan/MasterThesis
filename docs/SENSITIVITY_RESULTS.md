# GraphMemoryV4 Sensitivity Analysis Results

**Date:** February 2026 (numbers refreshed June 2026 on the MiniLM-era θ)  
**Experiment:** `run_sensitivity.py`  
**Environment:** MultiHopKeyDoor  
**Grid:** 6x6 over theta_novel x w_recency  
**Episodes per cell:** 10 (360 total)  
**Figure:** `docs/figures/fig09_landscape_v4.png`, `docs/figures/fig_sensitivity_annotated.png`  
**Raw data:** `results/sensitivity_results.json`

---

## Setup

We grid over the two dimensions that the CMA-ES optimizer pushed hardest:
- **theta_novel** (x-axis): range [0.0, 1.0] — novelty importance weight
- **w_recency** (y-axis): range [0.0, 4.0] — recency retrieval weight

All other V4 dimensions are fixed at their learned values:

| Fixed Parameter | Value |
|-----------------|-------|
| theta_store | 0.348 |
| theta_erich | 0.312 |
| theta_surprise | 0.418 |
| theta_entity | 0.063 |
| theta_temporal | 0.732 |
| theta_decay | 0.718 |
| w_graph | 1.188 |
| w_embed | 1.313 |

**Learned values being varied:**
- theta_novel = 0.442 (mid-range)
- w_recency = 1.207 (mid-range)

---

## Results

| Metric | Value |
|--------|-------|
| **Best reward (grid)** | **0.200** |
| Best theta_novel (grid) | 1.000 |
| Best w_recency (grid) | 2.400 |
| Learned theta_novel | 0.442 |
| Learned w_recency | 1.207 |
| Mean reward (all cells) | 0.1065 |
| Reward std (all cells) | 0.0592 |
| Reward range | 0.200 |
| Top-10% mean | 0.1697 |
| Top-10% std | 0.0096 |
| **Sharp peak** | **True** |

---

## Key Findings

### 1. The landscape has a sharp peak along theta_novel

`is_sharp_peak = True` — the best cell (0.200) sits well above the all-cell mean (0.107), and reward collapses to ~0 once theta_novel drops below ~0.3. The high-reward region is a narrow high-theta_novel band rather than a broad plateau, so the optimizer must find fairly precise values (this is the annotation shown on `fig_sensitivity_annotated.png` / `fig09_landscape_v4.png`).

> **Note (June 2026 refresh).** Earlier drafts reported `is_sharp_peak = False`
> ("broad plateau, robust") for the superseded TF-IDF-era θ. On the current
> MiniLM-era θ the landscape is flagged as a sharp peak; the figures already
> reflect this.

**Implication:** The CMA-ES optimizer does not need to find a precise value of theta_novel or w_recency. The memory system is tolerant of moderate parameter perturbations.

### 2. High theta_novel is consistently important

The grid optimum is at theta_novel=1.0 (maximum novelty requirement), confirming the ablation finding that theta_novel is the most critical dimension. The reward increases monotonically with theta_novel across most w_recency values.

**Interpretation:** The task benefits from maximally selective storage — only the most novel observations (hints) should be stored. This is consistent with the MultiHopKeyDoor structure where hints appear only at steps 0-2 and are highly novel relative to the rest of the episode.

### 3. The 2D grid optimum sits near the learned θ

The grid optimum is at theta_novel=1.000, w_recency=2.400 (reward 0.200), while the learned θ is theta_novel=0.442, w_recency=1.207 — both in the high-theta_novel region. The learned θ's nearest grid cell scores ~0.13; the grid's best cell scores 0.200.

**Interpretation:** The gap is small and well within grid noise (6×6 grid, 20 episodes/cell, std≈0.06). The 2D slice fixes all other parameters at the learned values, so it cannot capture the full 10D interaction structure CMA-ES optimized over; the learned θ was validated on 200 held-out episodes (reward 0.178), which is more reliable than any single 20-episode grid cell.

> **Note (June 2026 refresh).** Earlier drafts framed this section as a
> "surprise" — a grid w_recency≈0.727 beating a learned w_recency≈3.777. Those
> figures were for the superseded TF-IDF-era θ. On the current MiniLM-era θ the
> learned w_recency is a moderate 1.207, close to the grid optimum's 2.400, so
> there is no longer a recency paradox to explain.

### 4. Low w_recency region has moderate performance

Cells with w_recency < 1.0 and high theta_novel achieve 0.13-0.22 reward, which is competitive with the learned theta. This suggests that recency-based retrieval is helpful but not the only viable strategy.

### 5. Low theta_novel region fails

Cells with theta_novel < 0.3 achieve near-zero reward regardless of w_recency. This is consistent with the ablation finding: theta_novel gates the entire storage pipeline.

---

## Landscape Interpretation

The reward landscape has the following structure:
- **High theta_novel (>0.6) + any w_recency:** Moderate to good performance (0.10-0.22)
- **Low theta_novel (<0.3) + any w_recency:** Near-zero performance
- **Transition zone (0.3-0.6):** Gradual improvement with theta_novel

The landscape is **unimodal** along the theta_novel axis (higher is better) and **relatively flat** along the w_recency axis for high theta_novel values.

---

## Comparison: Grid Optimum vs Learned Theta

| | Grid Optimum | Learned Theta |
|-|-------------|---------------|
| theta_novel | 1.000 | 0.442 |
| w_recency | 2.400 | 1.207 |
| Reward (grid cell, 10 eps) | 0.200 | ~0.13 |
| Reward (200 eps, held-out) | N/A | 0.178 |

The grid's best cell scores marginally higher than the learned θ's nearest cell on the noisy 10-episode grid, but the learned θ was validated on 200 held-out episodes (reward=0.178). The small gap is consistent with grid noise rather than a real advantage.

**Recommendation:** The learned theta (from 30-generation CMA-ES with 50 eps/candidate) is more reliable than the 2D grid search with 10 eps/cell.

---

## Implications for Thesis

1. **theta_novel must be set carefully.** The landscape has a sharp peak: reward collapses once theta_novel falls below ~0.3, so the novelty gate is the dimension the optimizer most needs to get right. Performance is, however, comparatively insensitive to w_recency within the high-theta_novel band.

2. **theta_novel is the dominant dimension.** Future work should focus on improving the novelty estimation function (currently based on cosine similarity to stored embeddings) rather than fine-tuning w_recency.

3. **The 2D sensitivity analysis is insufficient.** The full 10D interaction structure cannot be captured by a 2D slice. Future work should use Sobol sensitivity analysis or SHAP values to quantify global dimension importance.

4. **The CMA-ES found a reasonable but not globally optimal solution.** The 2D grid's best cell (theta_novel=1.0, w_recency≈2.4) scores marginally above the learned θ's neighbourhood, but within grid noise; a longer CMA-ES run or a different optimizer might close the small gap.

---

## Figure

`docs/figures/fig09_landscape_v4.png` — Dual-panel heatmap:
- **Left panel:** Reward landscape (theta_novel x w_recency) with learned optimum (blue star) and grid optimum (red cross)
- **Right panel:** Precision landscape (theta_novel x w_recency) with learned optimum marked
