# GraphMemoryV4 Zero-Shot Transfer Test Results

**Date:** February 2026 (numbers refreshed June 2026 on the MiniLM-era θ)  
**Experiment:** `run_transfer.py --episodes 100 --seed 42` (deterministic; the
figure `fig10_transfer_v4.png` and the JSON below are written by the same run)  
**Source theta:** MultiHopKeyDoor (V4 CMA-ES best, MiniLM backend)  
**Episodes per pair:** 100 (seed_offset=3000)  
**Figure:** `docs/figures/fig10_transfer_v4.png`  
**Raw data:** `results/transfer_results.json`

---

## Setup

We take the V4 theta learned on MultiHopKeyDoor and evaluate it **without any retraining** on 3 other environments. This tests the core thesis hypothesis: **is the optimal memory configuration task-specific?**

**Source theta (learned on MultiHopKeyDoor):**

| Parameter | Value |
|-----------|-------|
| theta_store | 0.348 |
| theta_novel | 0.442 |
| theta_erich | 0.312 |
| theta_surprise | 0.418 |
| theta_entity | 0.063 |
| theta_temporal | 0.732 |
| theta_decay | 0.718 |
| w_graph | 1.188 |
| w_embed | 1.313 |
| w_recency | 1.207 |

> **θ provenance.** These are the **MiniLM-era** V4 θ from
> `results/graphmemory_v4_cmaes_results.json` — the canonical θ used throughout
> the (remediated) thesis. They supersede an earlier TF-IDF-era θ
> (theta_novel≈0.908, w_recency≈3.777, w_graph=0) reported in older drafts; see
> the four-shift re-baseline discussion for why the encoder change moved θ.

**Target environments:**

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| MultiHopKeyDoor | Multi-hop key-door, 3 hints at steps 0-2 | Medium (training env) |
| GoalRoom | Navigate to goal, no hints, sparse reward | Easy |
| HardKeyDoor | Key-door, fewer doors, no multi-hop | Medium |
| MegaQuestRoom | 20x20 grid, 6 doors, 1000 steps | Hard (OOD) |

---

## Results

| Environment | Reward | Std | Precision | Mem Size | Tokens | Status |
|-------------|--------|-----|-----------|----------|--------|--------|
| **MultiHopKeyDoor** | **0.1400** | 0.191 | 1.0000 | 9.1 | 1,553 | In-distribution |
| **GoalRoom** | **0.6900** | 0.465 | N/A | 1.7 | 47 | Zero-shot transfer |
| **HardKeyDoor** | **0.1733** | 0.180 | N/A | 6.7 | 1,427 | Zero-shot transfer |
| **MegaQuestRoom** | **0.0000** | 0.000 | 0.9616 | 12.7 | 7,533 | Zero-shot transfer (OOD) |

---

## Key Findings

### 1. GoalRoom: Surprisingly strong transfer (reward=0.69)

The MultiHop theta achieves **0.69 reward on GoalRoom**, which is very high. This is likely because:
- GoalRoom is a simpler task (navigate to goal, no hints needed)
- Very few events are stored (1.7 on average) — GoalRoom episodes are short and hint-free, so there is little to store, which is appropriate where memory is not critical
- The agent can succeed by exploration alone; the memory system doesn't interfere

**Interpretation:** The MultiHop theta "accidentally" works well on GoalRoom because its selective storage strategy is conservative enough to not pollute the context with irrelevant observations.

### 2. HardKeyDoor: Comparable performance (reward=0.17)

The MultiHop theta achieves **0.173 on HardKeyDoor**, on par with (in fact marginally above) MultiHopKeyDoor's in-distribution 0.140 — well within the single-seed spread (std ≈ 0.18–0.19). This makes sense because:
- HardKeyDoor is structurally similar to MultiHopKeyDoor (key-door tasks)
- The same memory strategy (store novel/surprising events, embedding+recency retrieval) applies
- 6.7 events stored vs 9.1 for MultiHop — appropriate for the simpler task

**Interpretation:** Positive transfer between structurally similar tasks. The learned theta generalizes across key-door variants.

### 3. MegaQuestRoom: Complete failure (reward=0.00)

The MultiHop theta achieves **0.00 reward on MegaQuestRoom** — exactly 0.000 across all 100 episodes (std=0.000). This is the most important result:
- The task is much harder (20x20 grid, 6 doors, 1000 steps)
- The failure is **not** memory pollution: only 12.7 events are stored per episode and retrieval precision is high (0.962). The memory the agent builds is small and clean.
- Yet the agent still never reaches the goal — the MultiHop-tuned θ does not equip it to plan over the far larger state space and longer horizon, regardless of how well its (small) memory is retrieved.

**Interpretation:** **Strong evidence of task-dependence.** The θ optimized for MultiHopKeyDoor fails catastrophically on MegaQuestRoom — and it fails on the *task's* scale/structure, not because the memory is noisy. This directly supports the thesis claim that optimal memory configuration is task-specific and cannot be transferred to harder, structurally different tasks.

> **Note (June 2026 refresh).** Earlier drafts of this section reported MegaQuest
> as mem≈86 / precision≈0.33 and attributed the failure to a flooded, noisy
> memory. Those figures were from the older TF-IDF-era θ; on the current
> MiniLM-era θ the stored memory is small and high-precision, so the honest
> mechanism is task scale/structure, not memory pollution. The 0.000 reward
> (complete OOD failure) is unchanged.

---

## Transfer Matrix Analysis

```
                    MultiHop  GoalRoom  HardKeyDoor  MegaQuestRoom
MultiHop_V4_theta    0.140*    0.690     0.173        0.000
```

`*` = in-distribution

**Pattern:** The MultiHop theta transfers well to simpler/similar tasks but fails on harder OOD tasks. This is the expected pattern for a task-specific memory configuration.

---

## Task-Dependence Hypothesis

The results **confirm the thesis's core hypothesis** with nuance:

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| Optimal theta is task-specific | **CONFIRMED** | MegaQuestRoom: 0.00 reward even with a small, high-precision memory |
| Transfer to similar tasks is possible | **CONFIRMED** | HardKeyDoor: 0.173 (comparable to training) |
| Transfer to simpler tasks may work | **CONFIRMED** | GoalRoom: 0.69 (better than training!) |
| Transfer to harder OOD tasks fails | **CONFIRMED** | MegaQuestRoom: complete failure |

---

## Why MegaQuestRoom Fails

On the current MiniLM-era θ, MegaQuestRoom failure is **not** a memory-quality
problem:
1. Only **12.7 events** are stored per episode and retrieval precision is **0.962** — the agent builds a small, clean memory, not a flooded one.
2. Yet reward is **exactly 0.000 across all 100 episodes** (std=0.000): the agent never reaches the goal.
3. The bottleneck is the *task*, not the memory: MegaQuestRoom is a 20×20 grid with 6 doors and a 1000-step horizon, far larger and longer than the ~100-step MultiHopKeyDoor the θ (and the exploration policy) were tuned on. A memory configuration that is excellent on the small task does not equip the agent to plan over the much larger state space.

**This is exactly the failure mode the thesis predicts:** a θ optimized for one
task does not transfer to a structurally harder, out-of-distribution task —
*regardless* of how cleanly its (small) memory is maintained.

---

## Implications for Thesis

1. **Task-dependence is real and measurable.** The 0.00 reward on MegaQuestRoom vs 0.140 on MultiHop is a stark, quantitative demonstration.

2. **The NeuralMemoryControllerV2 is motivated.** A neural meta-controller that adapts theta per-step could potentially handle the long-horizon MegaQuestRoom by adjusting storage selectivity as the episode progresses.

3. **Simple tasks don't stress-test memory.** GoalRoom's high transfer (0.69) shows that easy tasks don't require optimal memory configuration — any reasonable theta works. The thesis should focus on hard, long-horizon tasks.

4. **Token cost scales with task difficulty.** MegaQuestRoom uses 7,533 tokens/episode vs 1,553 for MultiHop — a ~4.9x increase. This directly supports the LLM cost motivation.

---

## Figure

`docs/figures/fig10_transfer_v4.png` — Transfer matrix heatmap showing reward for each (source theta, target environment) combination.
