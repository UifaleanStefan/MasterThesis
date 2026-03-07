*Archived; see THESIS_HANDOFF and PROJECT_SUMMARY for current state.*

# Phase 6 — Implementation Plan (Learnable Memory Creation)

This document proposes a concrete, step-by-step implementation plan for Phase 6. It is meant for discussion: we can adjust order, scope, or metrics before coding.

---

## 1. Design principles

- **Extend, do not replace:** Keep `GraphMemory`, `add_event`, and all retrieval functions. Add an optional "theta mode" and parameters.
- **Backward compatibility:** When θ is not passed or "fixed" mode is used, behavior is identical to Phase 5.
- **Minimal first:** Implement one parameter at a time, validate, then add the next (θ_store → θ_entity → θ_temporal).
- **Single place for construction:** All parameterized decisions live in `GraphMemory.add_event` (or a small helper it calls), so the agent loop stays unchanged except for passing θ and mode.

---

## 2. Token usage (proxy)

We need a scalar "token_usage" per episode for the objective `J(θ) = R - λ * token_usage`. Options:

| Option | Definition | Pros | Cons |
|--------|------------|------|------|
| **A** | Sum over steps of (number of events retrieved at that step) | Directly reflects "retrieval cost" | Depends on retrieval mode and k |
| **B** | Number of event nodes in graph at end of episode | Simple, no retrieval dependency | Does not measure retrieval cost |
| **C** | Number of event nodes + number of entity nodes at end | Captures graph size | Slightly more bookkeeping |

**Recommendation:** Start with **Option B** (count of event nodes at end of episode). It is easy to implement, correlates with "how much we stored," and is independent of retrieval. We can add Option A later (e.g. in evaluation) if we want to penalize "retrieval volume" explicitly. Option C can be added as an extra metric without changing the objective.

---

## 3. Where θ lives and how it is passed

- **New object:** Introduce a small dataclass or dict, e.g. `MemoryParams` or `theta`, with fields:
  - `theta_store: float` (in [0,1])
  - `theta_entity: float` (in [0,1])
  - `theta_temporal: float` (in [0,1])
  - Optional: `mode: Literal["fixed", "learnable"]`
- **GraphMemory:**
  - `GraphMemory.__init__(self, params: MemoryParams | None = None)`.
  - If `params is None` or `mode == "fixed"`: behavior equals current `add_event` (always store, all entities, all temporal edges).
  - If `mode == "learnable"`: use θ in `add_event` as below.
- **Agent loop:** When creating or resetting memory for an episode, pass the same `params` so that each episode can be run with a specific θ (for evaluation or bandit updates).

No change to environment or policy; only `GraphMemory` and the caller (loop / evaluation) need to know about θ.

---

## 4. Step-by-step implementation order

### Step 1: θ_store (event storage decision)

- In `GraphMemory.add_event(event)`:
  - If learnable and `theta_store` is set: before creating the event node, sample `store = Bernoulli(theta_store)`. If `store == False`, return immediately (no node, no edges for this step).
  - If fixed or `store == True`: proceed as today (create node, entities, temporal edge).
- **Randomness:** Use a dedicated RNG (e.g. seeded by episode or step) so that runs are reproducible when needed.
- **Metrics:** Log "events stored" vs "events offered" per episode; end-of-episode graph size (event count) is the storage footprint.

**Validation:** Run with θ_store = 1.0 → same as Phase 5. Run with θ_store = 0.0 → empty graph, no memory. Run with θ_store = 0.5 → roughly half the events stored; check that success rate and graph size are sensible.

---

### Step 2: θ_entity (entity node creation threshold)

- **Importance:** We need `importance(entity)` for each extracted entity. Minimal options:
  - **Frequency:** Count how many times this entity (e.g. `red_key`) has appeared in the trajectory so far (or in the graph). Normalize to [0,1] (e.g. divide by max count so far, or use a simple formula like `min(1.0, count / K)`).
  - **Task-signal:** For key–door task, keys and doors could get fixed higher importance (e.g. 1.0) and "nothing of interest" could get 0. So we only create nodes for task-relevant entities when θ_entity is high.
- In `add_event`: after deciding to store the event (Step 1), for each entity in `extract_entities(observation)`:
  - Compute `importance(entity)`.
  - If `importance(entity) > theta_entity`: create entity node if not exists, add event–entity edges.
  - Else: skip that entity for this event.
- **Backward compatibility:** When mode is "fixed," treat θ_entity as 0 (always create all entities).

**Validation:** θ_entity = 0 → all entities as now. θ_entity = 1.0 → no entity nodes (only event nodes). Sweep θ_entity and check success rate and entity count.

---

### Step 3: θ_temporal (temporal edge)

- Currently we always add `event_{t-1} → event_t`. With θ_temporal:
  - **Option A:** Add temporal edge with probability `theta_temporal` (Bernoulli per step).
  - **Option B:** Always add the edge but store a weight `theta_temporal` on it (for future weighted traversal; retrieval would need to use it).
- **Recommendation:** Start with **Option A** (Bernoulli) for simplicity; retrieval logic stays the same (all temporal edges are treated equally when present).
- In `add_event`: when adding the temporal edge to the previous event, if learnable: sample `add_edge ~ Bernoulli(theta_temporal)`; only add if True. If fixed: always add.

**Validation:** θ_temporal = 1.0 → same as now. θ_temporal = 0.0 → no temporal edges (only event and entity nodes and mention edges). Check success rate and graph connectivity.

---

### Step 4: Episode-level metrics and objective

- In the agent loop (or a wrapper around `run_episode_with_memory`):
  - After the episode: read `memory.get_stats()` → e.g. `n_events`, `n_entities`.
  - Compute `token_usage = n_events` (or chosen proxy).
  - Compute `reward = 1 if success else 0`.
  - Compute `J = reward - λ * token_usage` (λ small, e.g. 0.001 or 0.01).
- Return or log: `success`, `steps`, `n_events`, `token_usage`, `J` per episode.
- **Evaluation script:** Extend `evaluation/run.py` (or add a new script) to run episodes with a given θ and report these metrics; optionally aggregate over N episodes per θ.

---

### Step 5: Bandit-style optimization (Option A)

- **Goal:** Find a θ that maximizes (or improves) J(θ) over "fixed" memory.
- **Procedure (sketch):**
  1. Define a search space for θ (e.g. each component in [0.2, 0.4, 0.6, 0.8, 1.0] or sample uniformly).
  2. For each candidate θ (or a batch of candidates):
     - Run M episodes (e.g. M = 20 or 50) with that θ.
     - Average J over the M episodes → estimate of J(θ).
  3. Choose the θ with highest average J (or use a simple search: grid, random search, or one-step improvement from default θ = (1, 0, 1)).
- **Stability:** Use fixed env seed and policy seed for comparability; report mean and std of success rate and token_usage across episodes.
- **Deliverable:** A small module or script that (1) takes θ, (2) runs N episodes with learnable memory + learnable retrieval, (3) returns mean J, mean success rate, mean token_usage; and optionally a loop that tries several θ and prints the best.

---

### Step 6: Optional — REINFORCE (Option B)

- Defer until Option A is working and documented.
- Requires: (1) storing the log-probability of each stochastic decision (store event, create entity, add temporal edge) given θ; (2) after the episode, computing ∇_θ log P(decisions | θ); (3) updating θ with the policy gradient. More invasive and sensitive to hyperparameters; only add if the thesis needs a gradient-based method.

---

## 5. File-level changes (summary)

| File / component | Change |
|------------------|--------|
| `memory/graph_memory.py` | Add optional `MemoryParams`; in `add_event`, branch on mode and apply θ_store, θ_entity, θ_temporal. Keep existing behavior when params is None or mode is "fixed". |
| `memory/` (new or existing) | Optional: `memory/params.py` or in `graph_memory.py` define `MemoryParams` (dataclass/dict) and default `FIXED_PARAMS`. |
| `agent/loop.py` | When constructing or clearing memory for an episode, pass `params` into `GraphMemory(params=...)` so that evaluation/optimization can run with a given θ. |
| `evaluation/run.py` | Add optional `memory_params` (or `theta`) to `run_evaluation`; when provided, create `GraphMemory(params=...)` and report `n_events` (and optionally token_usage, J) per condition. Optionally add a "learnable_memory" condition that uses θ. |
| New script or section in `main.py` | Run bandit optimization: multiple θ configs, aggregate J and success rate, print best θ and comparison to fixed memory. |
| `report.txt` / report generation | Extend to include Phase 6: learnable memory results, best θ, success rate vs token_usage, efficiency. |

**Not changed:** `environment/`, `agent/policy.py`, retrieval function signatures (they still receive the same graph). Retrieval may see a sparser graph when θ_store or θ_entity are low; that is intended.

---

## 6. Open decisions for discussion

1. **Token proxy:** Confirm Option B (event count at end of episode) for the first version, or prefer Option A (retrieval count)?
2. **Importance for θ_entity:** Start with frequency in trajectory, or with a simple task-signal (e.g. key/door = 1, else 0)?
3. **θ_temporal:** Bernoulli (add edge with probability θ_temporal) vs always-add with weight?
4. **Bandit:** Grid search over a few values vs random search vs a simple evolutionary step?
5. **Default θ for "learnable":** When we first add the learnable mode, default θ = (1.0, 0.0, 1.0) so that we only add stochasticity step by step (first θ_store, then θ_entity, then θ_temporal)?
6. **Seeding:** Should θ_store / θ_entity / θ_temporal use the episode seed so that the same (env_seed, policy_seed, θ) gives the same memory construction across runs?

Once these are agreed, implementation can proceed in the order above (Step 1 → 2 → 3 → 4 → 5), with documentation and report updates as needed.
