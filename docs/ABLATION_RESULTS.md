# GraphMemoryV4 Ablation Study Results

**Date:** February 2026  
**Experiment:** `run_ablation.py`  
**Environment:** MultiHopKeyDoor  
**Episodes per config:** 100 (held-out, seed_offset=2000)  
**Figure:** `docs/figures/fig08_ablation_v4.png`  
**Raw data:** `results/ablation_results.json`

---

## Setup

The ablation study measures the contribution of each of the 10 theta dimensions in `MemoryParamsV4`. Starting from the best V4 theta found by CMA-ES (30 gens, 50 eps/candidate), we reset one group of dimensions at a time to its "uninformative default" and measure the reward drop.

**Learned V4 theta (baseline):**

| Parameter | Learned Value | Interpretation |
|-----------|--------------|----------------|
| theta_store | 0.293 | Moderate importance threshold |
| theta_novel | 0.908 | Very high novelty requirement |
| theta_erich | 0.198 | Low entity richness weight |
| theta_surprise | 0.785 | High surprise requirement |
| theta_entity | 0.285 | Moderate entity importance threshold |
| theta_temporal | 0.278 | Low temporal edge probability |
| theta_decay | 0.668 | Strong entity recency decay |
| w_graph | 0.000 | Graph traversal disabled |
| w_embed | 1.079 | Moderate embedding retrieval |
| w_recency | 3.777 | Recency dominates retrieval |

---

## Results

| Config | Mean Reward | Std | Precision | Mem Size | Degradation |
|--------|------------|-----|-----------|----------|-------------|
| **no_surprise** | **0.1733** | 0.180 | 1.0000 | 6.3 | -6.1% (better!) |
| **no_recency** | **0.1700** | 0.204 | 1.0000 | 9.9 | -4.1% (better!) |
| **full** | 0.1633 | 0.192 | 0.9993 | 9.8 | 0.0% (baseline) |
| graph_only | 0.1400 | 0.185 | 1.0000 | 10.0 | -14.3% |
| no_decay | 0.1367 | 0.184 | 0.9957 | 9.9 | -16.3% |
| no_embed | 0.1233 | 0.188 | 0.9123 | 10.2 | -24.5% |
| no_erich | 0.0733 | 0.139 | 1.0000 | 1.0 | -55.1% |
| v1_equivalent | 0.0133 | 0.066 | 0.2082 | 250.0 | -91.8% |
| **no_novelty** | **0.0000** | 0.000 | 0.0000 | 0.0 | -100% |
| **store_all** | **0.0000** | 0.000 | 0.0243 | 250.0 | -100% |

---

## Key Findings

### 1. theta_novel is the single most critical dimension

`no_novelty` (theta_novel=0.0) causes **complete failure** (reward=0.0, mem_size=0.0). This reveals a critical interaction: when theta_novel=0.0, the importance score formula produces zero importance for all events (novelty is the dominant term), so nothing is stored. The agent has no memory at all.

**Interpretation:** The novelty feature is not just helpful — it is the gating mechanism for the entire storage pipeline. Without it, the memory is empty.

### 2. theta_erich is the second most critical dimension

`no_erich` (theta_erich=0.0) causes a **55% degradation** (reward=0.073). When entity richness is disabled, only 1.0 events are stored on average (vs 9.8 for full). The entity richness feature is responsible for retaining the hint events (which contain multiple entities like "red key", "blue door") that are essential for solving MultiHopKeyDoor.

### 3. Embedding retrieval (w_embed) matters significantly

`no_embed` causes **24.5% degradation** and precision drops from 0.999 to 0.912. The embedding-based retrieval is important for finding relevant stored events when the observation is semantically similar but not identical.

### 4. Temporal decay (theta_decay) provides meaningful benefit

`no_decay` causes **16.3% degradation**. The Bayesian temporal decay correctly down-weights stale entity information, keeping entity importance estimates current.

### 5. Surprising: theta_surprise and w_recency are slightly *harmful*

Both `no_surprise` and `no_recency` achieve **slightly higher reward** than the full learned theta:
- `no_surprise`: 0.173 vs 0.163 (+6.1%)
- `no_recency`: 0.170 vs 0.163 (+4.1%)

This is likely due to:
1. **Noise in 100-episode evaluation** — the full theta was optimized on 50-episode training runs; these differences may not be statistically significant.
2. **Overfitting to training distribution** — the surprise and recency features may have been tuned to specific training seeds, with slight generalization cost.
3. **Simpler is sometimes better** — removing surprise makes storage less selective, which for this task may slightly help by retaining more hint events.

### 6. V1 equivalent is catastrophically worse

`v1_equivalent` achieves only 0.013 reward (91.8% degradation) with 250 stored events and 20.8% precision. This confirms that the V4 improvements (learned importance scoring, entity decay, learnable retrieval weights) are responsible for the massive performance gap between V1 and V4.

### 7. store_all fails completely

`store_all` (no importance filtering) stores 250 events but achieves 0.0 reward. This confirms that **selective storage is essential** — flooding memory with irrelevant observations makes retrieval impossible. The precision of 2.4% confirms that the retrieved events are almost never the relevant hints.

---

## Dimension Importance Ranking

Based on degradation when disabled:

1. **theta_novel** — CRITICAL (100% degradation, gates entire storage)
2. **theta_erich** — HIGH (55% degradation, retains hint events)
3. **w_embed** — MEDIUM-HIGH (24.5% degradation, semantic retrieval)
4. **theta_decay** — MEDIUM (16.3% degradation, entity recency)
5. **w_graph** — MEDIUM (14.3% degradation when forced to graph-only)
6. **theta_surprise** — LOW (slightly negative effect, may be noise)
7. **w_recency** — LOW (slightly negative effect, may be noise)
8. **theta_store** — INDIRECT (works through importance scoring)
9. **theta_entity** — INDIRECT (entity node creation threshold)
10. **theta_temporal** — LOW (temporal edge probability)

---

## Implications for Thesis

1. **Not all 10 dimensions are equally important.** A reduced 6D theta (theta_novel, theta_erich, w_embed, theta_decay, theta_store, theta_entity) might achieve similar performance with a simpler optimization landscape.

2. **The novelty feature is the load-bearing pillar.** Any future memory system should include a novelty-based storage gate.

3. **V1 is not just slightly worse — it is categorically inferior.** The 10x improvement from V1 to V4 is driven primarily by the learned importance scoring (theta_novel + theta_erich), not by the retrieval weight learning.

4. **The "store everything" baseline fails completely.** This directly supports the thesis motivation: without selective storage, memory is useless regardless of retrieval quality.

---

## Figure

`docs/figures/fig08_ablation_v4.png` — Dual-panel bar chart:
- **Top panel:** Mean reward per config with error bars and degradation % annotations
- **Bottom panel:** Retrieval precision per config

Color coding:
- Green: Full learned theta
- Blue: V1 equivalent
- Orange: Single-dim ablations
- Red: Store-all (no filter)
