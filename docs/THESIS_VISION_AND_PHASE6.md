# Thesis Vision and Phase 6 — Learnable Memory Creation

This document captures the Master thesis context, research goal, current system state, and the design for Phase 6 (learnable memory creation). It is intended to align implementation with the conceptual contribution.

---

## 1. Thesis title and research question

**Thesis title (conceptual):**  
*Learning Task-Adaptive Structured Memory for Long-Horizon Decision-Making*

**Core research question:**  
Most AI agents learn a **policy** (what action to take) and a **value function** (how good a state is), but they do **not** learn **how to build their own memory representation**. In most systems:

- Memory is **fixed**
- The **structure** of stored information is **predefined**
- All events are stored the **same way**
- The graph schema is **static**

This thesis asks:  
**Can an agent learn how to construct its own memory differently depending on the task?**

---

## 2. Why this matters

In real-world LLM-based agents (e.g. systems using OpenAI-style APIs):

- Memory is implemented as: tokens in context window, retrieval-augmented generation, vector database recall.
- **Problems:** storing everything is expensive; retrieving too much increases token usage; not all tasks require the same structure of memory.

**Examples:**

| Task | Important for memory | Unimportant |
|------|----------------------|-------------|
| Key–door matching | Color entities, key–door relationships | Movement steps |
| Navigation | Spatial transitions, room connectivity | Fine-grained object colors |

**Implication:** memory construction itself should be **adaptive** — i.e. learnable and task-dependent.

---

## 3. Current system (Phases 1–5) — What is already built

The codebase implements the following. No code changes are required for this section; it serves as the baseline for Phase 6.

### 3.1 Environment

- **File:** `environment/env.py`
- 6×6 grid, partial observability (local text observations only).
- **Delayed dependency:** agent must pick the key whose color matches the door; door color is observed only when at the door.
- Actions: move (N/S/E/W), pickup, use_door. Success = use_door with correct key; max 80 steps.

### 3.2 Memory representation

- **File:** `memory/graph_memory.py`
- **Graph-based memory:** NetworkX directed graph with:
  - **Event nodes:** `event_{step}` with attributes: type, step, observation, action, embedding, event.
  - **Entity nodes:** e.g. `red_key`, `blue_door` (from rule-based extraction in `memory/entity_extraction.py`).
  - **Temporal edges:** `event_t → event_{t+1}`.
  - **Mention edges:** event ↔ entity (bidirectional).

**Current behavior:** Every step, `add_event(event)` always:

1. Creates an event node
2. Extracts entities from the observation
3. Creates entity nodes if they do not exist
4. Adds event–entity edges
5. Adds a temporal edge to the previous event

So **what to store**, **which entities to create**, and **whether to add temporal edges** are currently **fixed**, not parameterized.

### 3.3 Retrieval

- **File:** `memory/retrieval.py`
- **Graph retrieval:** entity-based traversal (door → matching key → events mentioning that key).
- **Embedding retrieval:** TF-IDF embedding + cosine similarity, top-k.
- **Hybrid:** union of graph + embedding results.
- **Learnable retrieval (Phase 5):** weighted scoring
  - `score(event) = w_graph * graph_signal + w_embed * embedding_similarity + w_recency * recency_score`
  - Top-k by score; weights tunable (e.g. 1.5, 1.0, 0.2 gave ~27% success vs ~19–22% for graph/hybrid).

So we already learn **how to select** memories; we do **not** yet learn **how to build** memory.

### 3.4 Agent loop and evaluation

- **Files:** `agent/loop.py`, `agent/policy.py`, `evaluation/run.py`, `main.py`
- Episode loop: observe → (optionally) retrieve from memory → decide action → store event in memory.
- Policy uses `past_events` to infer door color and pick the matching key when memory is used.
- Evaluation compares: no_memory, graph, embedding, hybrid, learnable (with several weight configs); 100 episodes; report in `report.txt`.

---

## 4. Phase 6 — Learnable memory creation (design)

Goal: make **memory construction** parameterized and learnable, while keeping the existing graph format and retrieval interface.

### 4.1 What “learnable memory creation” means

Memory construction involves (at least):

1. **What to store** — which events get a node
2. **How to represent it** — what goes in the node (unchanged for now)
3. **How to connect it** — which entities and temporal edges to add
4. **When to compress or ignore** — optional later (e.g. merging or dropping old events)

Currently all of the above are fixed. Phase 6 makes **selected parts** learnable via a parameter vector **θ**.

### 4.2 Parameterization (minimal, controlled)

Introduce learnable parameters:

| Parameter   | Domain   | Role |
|------------|----------|------|
| **θ_store**   | [0, 1]   | Probability of storing an event (Bernoulli). If 0, no event node, no new edges. |
| **θ_entity**  | [0, 1]   | Threshold for creating entity nodes. Only create entity if importance(entity) > θ_entity. |
| **θ_temporal**| [0, 1]   | Controls temporal edge: e.g. add with probability θ_temporal, or use as edge weight. |

Optional later: θ_entity_edge, θ_abstraction (out of scope for the minimal design).

### 4.3 How memory creation changes

**1) Event storage decision**

- For each new event: `store_event ~ Bernoulli(θ_store)`.
- If **false:** do not create event node, do not add any edges for this step. Memory stays smaller; retrieval only sees previously stored events.

**2) Entity node creation**

- For each entity extracted from the observation:
  - Compute **importance(entity)** (e.g. frequency in trajectory so far, or task-signal such as “key”/“door”).
  - If **importance > θ_entity:** create entity node (if not exists) and add event–entity edges.
  - Otherwise: do not create/connect that entity. Graph structure becomes sparser and task-adaptive.

**3) Temporal edge**

- Instead of always adding `event_{t-1} → event_t`:
  - Add with probability **θ_temporal**, or
  - Always add but attach weight θ_temporal (if we later support weighted traversal).
- Allows a continuum from strong sequential memory to more sparse temporal links.

### 4.4 Objective function

- **Reward:** `R = 1` if episode success, else `0`.
- **Token usage (proxy):** e.g. total number of “tokens” retrieved across the episode (or total nodes/events in memory at end of episode, or total retrievals × k).
- **Total objective:**  
  `J(θ) = R - λ * token_usage`  
  with small λ (regularization). Trade-off: more memory → potentially higher reward, but more tokens → penalty. Goal: **efficient** memory.

### 4.5 Learning mechanism (two options)

- **Option A (recommended): Bandit-style optimization**
  - Treat θ as parameters to tune.
  - Sample or perturb θ, run N episodes, measure J(θ), update θ toward better configurations (e.g. grid search, random search, or simple evolutionary/gradient-free method).
  - Stable and easy to implement; good for a thesis prototype.

- **Option B (advanced): REINFORCE**
  - Treat storage/entity/temporal decisions as a stochastic policy given θ.
  - Update: `θ ← θ + α * (reward - baseline) * ∇_θ log P(memory decisions | θ)`.
  - More powerful but more complex; use only if Option A is insufficient or for an extension.

### 4.6 Required metrics

For each configuration (and for comparison across phases):

- **Success rate** (and optionally avg steps, avg steps on success).
- **Avg tokens** (or chosen token proxy per episode).
- **Efficiency:** e.g. Success / Tokens (or success rate per mean token).

Goals to demonstrate:

1. Learnable **retrieval** (Phase 5) improves performance over fixed retrieval.
2. Learnable **construction** (Phase 6) can further improve performance and/or reduce tokens.
3. Best performance-per-token comes from **learned** memory (construction + retrieval).

### 4.7 Research claim (formal)

The thesis is **not** “better RL policy.” It is:

- **Learning how to represent experience.**

Formal statement:

- `G_t = BuildMemory(trajectory, θ)`
- `π(a | s, Retrieve(G_t))`
- We optimize **θ** (and possibly retrieval weights).

So the contribution is **meta-level learning over memory representation**, not only over the policy.

### 4.8 Constraints

- Do **not** break the existing architecture: keep graph-based memory and current retrieval API.
- **Extend** `GraphMemory` to accept θ (and optionally a “mode”: fixed vs learnable).
- Backward compatibility: when θ is not used or “fixed” mode is selected, behavior equals current Phases 1–5.
- Environment and policy logic remain unchanged; only memory construction and (optionally) evaluation/optimization code are extended.

---

## 5. Experiment design (Phase 6)

Compare, over 100–300 episodes:

1. **Fixed memory + learnable retrieval** (current Phase 5 baseline).
2. **Learnable memory + learnable retrieval** (θ_store, θ_entity, θ_temporal used; retrieval as in Phase 5).
3. **Learnable memory + token penalty** (optimize J(θ) = R - λ * token_usage).

Report: success rate, avg steps, avg tokens (or proxy), efficiency (e.g. success/tokens), and optionally variance across seeds.

---

## 6. Final vision

Demonstrate that **agents should not only learn to act, but also what to remember and how to structure it.** This aligns with:

- Adaptive cognition
- Efficient LLM-based systems
- Scalable long-horizon agents

Phase 6 implements the first step: **parameterized, learnable memory construction** (θ_store, θ_entity, θ_temporal) plus an objective and a simple optimization procedure, while keeping the rest of the pipeline unchanged.
