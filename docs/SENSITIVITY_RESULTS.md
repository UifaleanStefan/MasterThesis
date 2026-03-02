# GraphMemoryV4 Sensitivity Analysis Results

**Date:** February 2026  
**Experiment:** `run_sensitivity.py`  
**Environment:** MultiHopKeyDoor  
**Grid:** 12x12 over theta_novel x w_recency  
**Episodes per cell:** 20 (2,880 total)  
**Runtime:** ~330 seconds (~5.5 minutes)  
**Figure:** `docs/figures/fig09_landscape_v4.png`  
**Raw data:** `results/sensitivity_results.json`

---

## Setup

We grid over the two dimensions that the CMA-ES optimizer pushed hardest:
- **theta_novel** (x-axis): range [0.0, 1.0] — novelty importance weight
- **w_recency** (y-axis): range [0.0, 4.0] — recency retrieval weight

All other V4 dimensions are fixed at their learned values:

| Fixed Parameter | Value |
|-----------------|-------|
| theta_store | 0.293 |
| theta_erich | 0.198 |
| theta_surprise | 0.785 |
| theta_entity | 0.285 |
| theta_temporal | 0.278 |
| theta_decay | 0.668 |
| w_graph | 0.000 |
| w_embed | 1.079 |

**Learned values being varied:**
- theta_novel = 0.908 (near the top of the range)
- w_recency = 3.777 (near the top of the range)

---

## Results

| Metric | Value |
|--------|-------|
| **Best reward (grid)** | **0.2167** |
| Best theta_novel (grid) | 1.000 |
| Best w_recency (grid) | 0.727 |
| Learned theta_novel | 0.908 |
| Learned w_recency | 3.777 |
| Mean reward (all cells) | 0.0719 |
| Reward std (all cells) | 0.0506 |
| Reward range | 0.2167 |
| Top-10% mean | 0.1600 |
| Top-10% std | 0.0271 |
| **Sharp peak** | **False (broad plateau)** |

---

## Key Findings

### 1. The landscape is a broad plateau, not a sharp peak

`is_sharp_peak = False` — the top-10% of cells have low variance (std=0.027), meaning many (theta_novel, w_recency) combinations achieve near-optimal performance. The learned theta is **robust**, not fragile.

**Implication:** The CMA-ES optimizer does not need to find a precise value of theta_novel or w_recency. The memory system is tolerant of moderate parameter perturbations.

### 2. High theta_novel is consistently important

The grid optimum is at theta_novel=1.0 (maximum novelty requirement), confirming the ablation finding that theta_novel is the most critical dimension. The reward increases monotonically with theta_novel across most w_recency values.

**Interpretation:** The task benefits from maximally selective storage — only the most novel observations (hints) should be stored. This is consistent with the MultiHopKeyDoor structure where hints appear only at steps 0-2 and are highly novel relative to the rest of the episode.

### 3. Surprising: w_recency = 0.727 beats the learned w_recency = 3.777

The grid optimum uses w_recency=0.727, while the learned theta uses w_recency=3.777. The grid achieves 0.217 reward vs the learned theta's ~0.163 on this 20-episode evaluation.

**Interpretation:** This discrepancy has several possible explanations:
1. **Evaluation noise** — 20 episodes per cell is noisy; the 0.217 may be a lucky seed
2. **Overfitting in CMA-ES** — the CMA-ES training (50 eps/candidate) may have overfit to specific training seeds that favor high recency
3. **Interaction effects** — w_recency=3.777 may only be optimal in combination with other specific parameter values not captured in this 2D slice

The 2D grid fixes all other parameters at learned values, so it cannot capture the full interaction structure. The CMA-ES found a 10D optimum; the 2D slice may not align with it.

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
| theta_novel | 1.000 | 0.908 |
| w_recency | 0.727 | 3.777 |
| Reward (20 eps) | 0.217 | ~0.163 |
| Reward (200 eps) | N/A | 0.178 |

The grid optimum achieves higher reward on the 20-episode evaluation, but the learned theta was validated on 200 held-out episodes (reward=0.178). The discrepancy suggests the grid result may be noisy.

**Recommendation:** The learned theta (from 30-generation CMA-ES with 50 eps/candidate) is more reliable than the 2D grid search with 20 eps/cell.

---

## Implications for Thesis

1. **The memory system is robust.** The broad plateau means that small perturbations to theta_novel or w_recency don't catastrophically degrade performance. This is a positive property for real-world deployment.

2. **theta_novel is the dominant dimension.** Future work should focus on improving the novelty estimation function (currently based on cosine similarity to stored embeddings) rather than fine-tuning w_recency.

3. **The 2D sensitivity analysis is insufficient.** The full 10D interaction structure cannot be captured by a 2D slice. Future work should use Sobol sensitivity analysis or SHAP values to quantify global dimension importance.

4. **The CMA-ES found a reasonable but not globally optimal solution.** The grid suggests there may be better configurations in the (theta_novel=1.0, w_recency~0.7) region. A longer CMA-ES run or a different optimizer might find this.

---

## Figure

`docs/figures/fig09_landscape_v4.png` — Dual-panel heatmap:
- **Left panel:** Reward landscape (theta_novel x w_recency) with learned optimum (blue star) and grid optimum (red cross)
- **Right panel:** Precision landscape (theta_novel x w_recency) with learned optimum marked
