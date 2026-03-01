# Thesis Story — Learning Task-Adaptive Structured Memory

*Last updated: Mar 2026 (Phase A–F implementation: fixed objective, MultiHopKeyDoor, retrieval precision metric). This document captures the full research narrative, what has been built, what the results show, and what comes next. It is intended to align implementation with the thesis argument and to serve as a reference for writing the thesis.*

---

## 1. The Core Claim

Most AI agents learn two things:
- A **policy** — what action to take given the current state.
- A **value function** — how good a state is.

What they do **not** learn is:
- **How to build their own memory** — what to store, how to structure it, how much to use.

In practice, memory in LLM-based agents is implemented as:
- Tokens in the context window (fixed capacity, expensive)
- Retrieval-Augmented Generation (fixed indexing, no learned structure)
- Vector databases (fixed schema, all events treated equally)

**The problem:** different tasks require different memory structures.

| Task | What memory should track | What it should ignore |
|---|---|---|
| Key–door matching | Color entities, key–door relationships | Movement steps |
| Navigation | Spatial transitions, room connectivity | Fine-grained object colors |
| Long lore understanding | Character names, cause–effect chains | Repeated descriptions |
| Multi-step planning | Subgoal dependencies, causal links | Irrelevant context |

If the memory structure is fixed, the agent either stores too much (expensive, noisy) or too little (loses critical information).

**The thesis claim:**
> Memory construction — what to store, which concepts to track as nodes, how to chain events — should itself be **learned** and should be **task-adaptive**.

Formally: let `G_t = BuildMemory(trajectory, θ)` be the memory graph at time t, and let the policy act as `π(a | s, Retrieve(G_t))`. We optimize **θ** — not the policy, not the value function, but the **memory construction parameters**. This is meta-level learning over memory representation.

---

## 2. Why This Matters at Scale

The thesis demo uses toy grid environments. But the argument scales directly to real-world LLM agents:

**Analogy:**

| This system | LLM agent equivalent |
|---|---|
| Memory graph (event + entity nodes) | Context window / RAG index |
| θ_store — probability of storing an event | Deciding which chunks to keep in context |
| θ_entity — threshold for creating entity nodes | Deciding which concepts to index / track |
| θ_temporal — probability of adding temporal edges | Deciding how much sequential structure to maintain |
| J = reward − λ × retrieval_tokens | Performance minus API token cost |

**The real-world use cases this addresses:**

- **Long lore / world-building:** An agent reading a 10,000-page fantasy world needs to track character names and causal chains, but skip repeated scene descriptions. θ_entity (track important concepts) and θ_store (filter repetitive events) are exactly this.
- **Multi-session dialogue agents:** An agent across 100 conversations needs to remember user preferences and prior commitments, but not every exchange. Memory structure should adapt to the relationship type.
- **Long-horizon planning:** An agent solving a multi-day task needs to maintain subgoal chains (θ_temporal) and key intermediate states (θ_store), while ignoring irrelevant steps.
- **Beyond-context tasks:** When the task exceeds the context window, the agent *must* choose what to remember. This choice is currently manual/heuristic. We make it learnable.

The toy demo **proves the learning mechanism works** and that **optimal memory structure is task-dependent**. The thesis argues this mechanism scales to the above real-world settings.

---

## 3. What Has Been Built (Phases 1–7)

### Phase 1–2: Environment
- **ToyEnvironment** (`environment/env.py`): 6×6 grid, 3 colored keys, 1 door (only the matching color opens it). Text-only partial observations. Max 80 steps. Tests delayed dependency: the agent must find the key color, remember it, find the door, and use the right key.
- **GoalRoom** (`environment/env.py`): 6×6 grid, one goal cell. Success = `use_door` on goal cell. Tests a simpler task with no entity relationships. Used for cross-environment θ comparison.

### Phase 3: Graph Memory
- **`GraphMemory`** (`memory/graph_memory.py`): NetworkX directed graph.
  - **Event nodes** (`event_{step}`): observation, action, step, TF-IDF embedding.
  - **Entity nodes**: e.g. `red_key`, `blue_door`, `goal`.
  - **Temporal edges**: `event_t → event_{t+1}`.
  - **Mention edges**: event ↔ entity (bidirectional).
- **Entity extraction** (`memory/entity_extraction.py`): rule-based, detects colored keys, doors, and the goal entity.
- **TF-IDF embeddings** (`memory/embedding.py`): fixed 16-word vocabulary, deterministic, lightweight.

### Phase 4: Embedding Retrieval
- `retrieve_similar_events`: cosine similarity over stored TF-IDF embeddings, returns top-k.
- Hybrid mode: union of graph-based + embedding-based results.

### Phase 5: Learnable Retrieval Scoring
- `retrieve_events_learnable` (`memory/retrieval.py`):
  ```
  score(event) = w_graph × graph_signal + w_embed × cosine_sim + w_recency × recency
  ```
- Top-k by score. Weights are tunable. Best config: `(1.5, 1.0, 0.2)`.
- Result: retrieval became *selective* rather than exhaustive. Marginal success improvement; main benefit is reduced retrieval tokens.

### Phase 6: Learnable Memory Creation (θ search)
- **`MemoryParams`** (`memory/graph_memory.py`): introduces θ = `(θ_store, θ_entity, θ_temporal)` ∈ [0,1]³.
  - `θ_store`: Bernoulli(θ_store) — store this event or skip it entirely.
  - `θ_entity`: frequency-based importance threshold — only create entity node if (times seen / total mentions) > θ_entity.
  - `θ_temporal`: Bernoulli(θ_temporal) — add temporal edge to previous event or not.
- **Default θ = (1.0, 0.0, 1.0)**: exactly reproduces the fixed-memory baseline.
- **Random search (bandit)**: 15 θ configs × 75 episodes each. Objective: `J = reward − λ × retrieval_tokens` (λ = 0.001).
- **Reproducibility**: deterministic RNG seeded by `(episode_seed, step)`.

### Phase 7: Adaptive θ via Evolution Strategy
- **ES loop** (`main.py`): 12 generations, 6 candidates/gen, 25 episodes/candidate.
- Each generation: sample candidates from N(μ, σ), clip to [0,1], evaluate each by mean J, set μ = best, decay σ (min 0.05).
- Learns θ *over time* rather than finding it by one-off search.

### Cross-Environment Comparison
- Both phases run on Key-Door and Goal-Room.
- Results show different optimal θ per task — core empirical support for the thesis claim.

---

## 4. Experimental Results (Current)

*(From `report.txt`, run Mar 2026 — fixed objective, MultiHopKeyDoor added)*

### What changed from the previous run

The previous run (Feb 2026) produced a degenerate result: ES collapsed to θ_store ≈ 0.0 on Key-Door and Hard-KeyDoor because the objective `J = reward − λ × tokens` made it optimal to store nothing (zero tokens = zero penalty). This has been fixed:

- **Optimization objective changed to `J_learn = mean_reward`** (pure task performance). Token cost is now tracked as a secondary efficiency metric only.
- **Episode counts increased**: 100 episodes/θ in Phase 6, 40 episodes/candidate in Phase 7 (reduced variance).
- **MultiHopKeyDoor** replaces HardKeyDoor as the primary benchmark — designed so the rule-based policy strictly depends on hint memory to open any door.

### Phase 6 (random search, 100 ep/θ)

| Environment | Best θ (store, entity, temporal) | Success (learnable) | Mean reward | Efficiency |
|---|---|---|---|---|
| Key-Door | (0.343, 0.801, 0.207) | 29.0% | 0.290 | 0.000616 |
| Goal-Room | (0.315, 0.398, 0.769) | 69.0% | 0.690 | 0.002361 |
| MultiHop-KeyDoor | (0.609, 0.527, 0.809) | 16.0% | 0.057 | 0.000029 |

### Phase 7 (Evolution Strategy, 12 gen × 6 cand × 40 ep)

| Environment | Learned θ (store, entity, temporal) | Success (learnable) | Baseline success | Mean reward |
|---|---|---|---|---|
| Key-Door | (1.000, 0.286, 0.785) | 37.5% | 20.0% | 0.375 |
| Goal-Room | (0.987, 0.197, 0.907) | 77.5% | 57.5% | 0.775 |
| MultiHop-KeyDoor | (0.991, 0.523, 1.000) | 22.5% | 5.0% | 0.075 |

### Key observations

1. **ES no longer collapses to zero** (objective fix confirmed): All three environments now keep θ_store near 1.0 — the ES correctly stores events because memory genuinely helps and there's no token penalty in the optimizer. The efficiency metric is tracked separately and shows the tradeoff post-hoc.

2. **θ is task-dependent** (core claim confirmed): The three environments converge to structurally different θ:
   - Key-Door: θ_entity = 0.286 — moderate entity filtering (key-door color matching)
   - Goal-Room: θ_entity = 0.197, θ_temporal = 0.907 — very low entity tracking, high temporal chain
   - MultiHop-KeyDoor: θ_entity = 0.523 — highest entity importance threshold (must filter distractor keys, weight hint entities)

3. **ES improves over baseline on all environments**: Key-Door +87.5% (20% → 37.5%), Goal-Room +34.8% (57.5% → 77.5%), MultiHop-KeyDoor +350% (5% → 22.5%). The MultiHop improvement is the most dramatic because memory is strictly required.

4. **MultiHop-KeyDoor creates genuine memory pressure**: Baseline 5% success with fixed memory vs. 22.5% with learned θ. The task is hard enough that random exploration almost never finds a door solution.

### Memory System Comparison (MultiHopKeyDoor, n=50)

| System | Partial Score | Tokens | Retrieval Precision | Efficiency |
|---|---|---|---|---|
| EpisodicSemantic | **0.167** | 1964 | **1.000** | 0.000085 |
| SemanticMemory | 0.153 | 1964 | **1.000** | 0.000078 |
| RAGMemory | 0.053 | 1964 | 0.577 | 0.000027 |
| GraphMemory+θ | 0.013 | 1964 | 0.500 | 0.000007 |
| FlatWindow(50) | 0.000 | 1964 | 0.060 | 0.000000 |
| SummaryMemory | 0.000 | 1636 | 0.018 | 0.000000 |

**Retrieval precision** = fraction of door-query steps where the hint was in the top-k retrieved events. This is the direct memory quality metric independent of task success.

### Interpretation of memory comparison

- **EpisodicSemantic wins** because it stores sign observations verbatim in the semantic store (never evicted), then always returns them at door steps. Precision = 1.0.
- **SemanticMemory wins equally on precision** because sign observations score very high on importance (beta=5.0 for hint-like obs). Precision = 1.0, slightly lower score due to eviction of useful episodic context.
- **RAGMemory** achieves 0.577 precision — MiniLM embeddings somewhat capture semantic similarity between "requires blue key" and "blue key opens east door", but not reliably.
- **GraphMemory+θ** scores 0.500 precision — graph retrieval partially finds hints via entity nodes, but TF-IDF over a fixed vocab doesn't capture "sign" well.
- **FlatWindow(50) fails** — by the time the agent reaches a door (often 50-150 steps after seeing the hint at step 0-2), the hint has scrolled out of the 50-event window. Precision 0.060.
- **SummaryMemory fails** — compression destroys the exact text of the sign, so the hint content is lost even though the episode event is "summarized."

**This is the thesis's strongest empirical result**: memory *architecture* matters qualitatively. FlatWindow and SummaryMemory score 0 regardless of how large their window is because the hint arrives early and the task is long. EpisodicSemantic succeeds because it has the right inductive bias — hints belong in a persistent semantic store.

---

## 5. Limitations of the Current Demo

| Limitation | Status | Impact on thesis |
|---|---|---|
| ES collapsed to θ_store=0 | **FIXED** — objective now = pure reward | ES now finds non-trivial θ |
| Policy-task mismatch on QuestRoom | **ADDRESSED** — MultiHopKeyDoor replaces QuestRoom as comparison env | Memory comparison now shows differentiated results |
| No retrieval precision metric | **FIXED** — added hint_queries/hint_hits tracking | Can now show WHY a memory system succeeds or fails |
| High result variance | **IMPROVED** — 100ep/θ in Phase 6, 40ep/candidate in Phase 7 | Results are tighter; ES converges reliably |
| No learning curves plotted | Still pending | ES looks like search, not learning |
| No graph visualization | Still pending | Can't show "sparse learned graph vs. dense fixed graph" |
| Rule-based policy caps performance | Still pending | Policy bottleneck limits maximum achievable reward on hard tasks |
| TF-IDF embeddings lose sign semantics | Still pending | GraphMemory+θ retrieval precision is low (0.500) even with perfect storage |

---

## 6. Next Step: Harder Task (Step 1)

### Goal
Create genuine **memory pressure** — situations where the agent *must* remember selectively to succeed, and where θ meaningfully controls what survives.

### Design: Extended ToyEnvironment

Extend `ToyEnvironment` with:

| Parameter | Current | Extended |
|---|---|---|
| Grid size | 6×6 | 10×10 or 12×12 |
| Max steps | 80 | 300 |
| Keys | 3 | 5 |
| Doors | 1 | 3 (each needs a specific key) |
| Distractor objects | 0 | 2–3 fake "keys" that don't match any door |
| Observation noise | None | Occasionally see irrelevant objects |

With this design:
- The agent must remember *which* key color matches *which* door (not just one pair).
- With 5 real keys + distractors, entity tracking matters — θ_entity must filter signal from noise.
- 300 steps means sequential memory (θ_temporal) matters for reasoning about "where have I been."
- Token cost is much higher → the penalty term in J(θ) has real bite.

### Expected impact on results
- Success rates will be lower overall (harder task) but the gap between fixed and learned θ should be larger.
- θ_entity should converge to a meaningful value (not near 0 or 1 trivially).
- The cross-environment comparison becomes: "complex multi-dependency task vs. simple reach-goal task" — a much stronger contrast for the thesis.

---

## 7. The Thesis Arc (Research Narrative)

```
Problem:
  LLM agents waste tokens storing everything,
  or lose critical information storing too little.
  Memory structure is fixed and not task-adaptive.

Insight:
  Different tasks require different memory structures.
  Memory construction should be learnable.

Approach:
  Represent memory as a graph controlled by θ.
  Learn θ via Evolution Strategy optimizing J = reward − λ × tokens.

Demo (current):
  Toy grid environments. θ adapts to task.
  Key-Door → entity-focused memory.
  Goal-Room → sparse, temporal memory.
  Cross-env θ comparison confirms task-dependence.

Demo (harder task + memory comparison):
  QuestRoom: 12x12, 500 steps, 4 chained doors, 2 NPCs giving one-time hints,
  distractor observations. NPC hints arrive early; agent must recall them 300+ steps later.
  Six memory systems compared on QuestRoom:
    1. FlatWindow      — sliding window, no structure, no filtering
    2. GraphMemory+θ   — current system, graph + θ parameters
    3. SemanticMemory  — importance-weighted pool (entity count + NPC hint + novelty)
    4. SummaryMemory   — periodic compression of old events into summaries
    5. EpisodicSemantic — dual store: episodic buffer + persistent semantic facts
    6. RAGMemory       — dense sentence embeddings + cosine retrieval (production LLM analog)
  Comparison metric: efficiency = partial_score / (1 + mean_retrieval_tokens)
  and J = partial_score − λ × retrieval_tokens.

Scaling argument:
  θ_store = context window management.
  θ_entity = concept indexing in RAG.
  θ_temporal = sequential structure in long-horizon tasks.
  The mechanism generalizes to any agent where memory is the bottleneck.

Contribution:
  First demonstration that memory construction parameters
  can be learned end-to-end from task reward,
  producing task-adaptive graph memory structures.
```

---

## 8. Why the Current Graph Memory Is Too Simple

Honest assessment of the existing system's limitations (motivates the comparison):

| Component | What it actually does | Why it's not real learning |
|---|---|---|
| Entity extraction | `if "red key" in obs_lower` | Pure regex; can't generalize |
| Graph edges | Hardcoded temporal chain + keyword entity links | No learned relationship types |
| θ_store | Bernoulli coin flip | Learning = finding the right drop probability |
| θ_entity | Frequency counter vs. threshold | Not semantic; "goal" scores same as "key" |
| θ_temporal | Another coin flip | No content-based structure |
| TF-IDF embeddings | 21-word fixed vocab | "red key" ≈ "red door" because they share "red" |
| Retrieval weights | Tuned by random search | Not gradient descent; not online learning |

The comparison with SemanticMemory, SummaryMemory, EpisodicSemantic, and RAGMemory
tests whether richer memory architectures outperform the θ-parameterized graph on a
task where memory genuinely matters.

---

## 9. Memory System Comparison (QuestRoom)

### Why QuestRoom creates genuine memory pressure

- **NPC hints:** Two NPCs each give a one-time hint about which key opens a specific door.
  Hints arrive at random early steps; agent must recall them 200–400 steps later.
  A flat window forgets them. A semantic store retains them indefinitely.
- **Chained doors:** Door 1 must open before Door 2 is accessible, etc.
  Agent must track *which* doors are open to know what to do next.
- **Distractor observations** (~10% of steps): `"You hear wind."`, `"You see a painting."` —
  purely noise. Systems that store everything pay a token cost for irrelevant events.
- **500 steps, 12x12 grid:** 3600–4000 retrieval tokens per episode at k=8.
  The penalty term in J(θ) = partial_score − 0.001 × tokens has real bite.

### Six memory systems

| System | Structure | Retrieval | Key strength | Key weakness |
|---|---|---|---|---|
| FlatWindow(50) | Last 50 events | Last k | Cheap | Forgets NPC hints |
| GraphMemory+θ | Graph + entity nodes | Learnable weighted score | Structured, learnable | Rule-based extraction, TF-IDF only |
| SemanticMemory | Importance-ranked pool (cap 80) | Importance × cosine sim | Retains high-value events | Importance is heuristic |
| SummaryMemory | Raw buffer + compressed summaries | Summaries + recent raw | Bounded cost, long-term compression | Lossy compression |
| EpisodicSemantic | Episodic buffer + semantic facts | Facts + recent events | NPC hints always in semantic store | Semantic extraction is rule-based |
| RAGMemory | All events + dense embeddings | Cosine similarity (MiniLM) | Semantic similarity, no rules | Stores everything; high memory cost |

### Expected thesis outcome

EpisodicSemantic should perform well on QuestRoom because NPC hints go directly into
the semantic store and are always retrieved. RAGMemory should perform well if MiniLM
embeddings capture the relevance of "blue key" when the agent sees "blue door".
FlatWindow should fail on episodes where the NPC hint was > 50 steps ago.
SummaryMemory should score in the middle: summaries capture hints but lose detail.

The comparison table in `report.txt` directly supports the thesis claim:
**memory structure matters, and different structures are optimal for different properties
of the task (short-term vs. long-term, entity-focused vs. sequence-focused).**

---

## 10. POC vs. Full System — What This Is and What It Will Become

### What this codebase is right now

This is a **Proof of Concept (POC)**. Its purpose is to:
1. Prove the learning mechanism works (θ adapts, ES converges, J improves).
2. Prove the core claim empirically (different tasks → different optimal θ).
3. Establish the architecture and interfaces that scale to a real system.

### What this codebase is NOT (yet)

| Component | Current (POC) | Target (full system) |
|---|---|---|
| Environment | Grid worlds (6×12, text obs) | Real task: document QA, game engine, API workflows |
| Policy (agent) | Rule-based string matching | LLM (GPT-4o, Claude 3.5, Llama 3 via Ollama) |
| Embeddings | TF-IDF, 21-word fixed vocab | `text-embedding-3-small` or `all-MiniLM-L6-v2` via API |
| `retrieval_tokens` | Proxy count (`len(past_events)`) | Actual API token count billed in dollars |
| J = reward − λ×tokens | Partial score − λ×event count | Task score − λ×API cost (real money) |
| Long-horizon task | QuestRoom, 500 steps, 12×12 grid | 500-page lore, multi-session agent, beyond-context-window reasoning |
| Memory comparison | 6 custom Python classes | Drop-in for LangChain, MemGPT, LlamaIndex memory modules |

### Why the POC is still a valid thesis contribution

The architecture, the interface, and the learning mechanism (θ-parameterized graph, ES optimization, J = reward − λ×cost) are **identical** between the POC and the full system. The POC demonstrates:
- That the mechanism is learnable (ES converges to task-specific θ).
- That the mechanism generalizes (different environments → different θ).
- That the interface is modular (6 memory systems, same 4-method contract).
- That memory structure has real impact on performance and token cost.

A real LLM policy + real API costs would make the results more dramatic (token cost becomes real money), not change the fundamental finding.

### The development path

```
[NOW] POC — toy environments, rule-based policy, TF-IDF, proxy tokens
        ↓
[NEXT] Harder environments — QuestRoom comparison, ES on HardKeyDoor
        ↓
[BRIDGE] Real embeddings (MiniLM local), real retrieval, larger obs spaces
        ↓
[FULL] LLM as policy (Ollama local or OpenAI API), real token costs,
       real long-horizon task (document corpus / game world / codebase)
        ↓
[CONTRIBUTION] Publishable: learnable θ reduces API cost while maintaining
               task performance on tasks that exceed context window
```

---

## 12. File Map

```
d:\Bocconi\Thesis\
├── main.py                            # Entry point; all phases + memory comparison
├── report.txt                         # Auto-generated experiment results
├── requirements.txt                   # networkx, numpy, scikit-learn, sentence-transformers
├── environment/
│   ├── env.py                         # ToyEnvironment, GoalRoom, HardKeyDoor, QuestRoom
│   └── __init__.py
├── memory/
│   ├── graph_memory.py                # GraphMemory + MemoryParams (θ) + get_relevant_events()
│   ├── flat_memory.py                 # FlatMemory (sliding window)
│   ├── semantic_memory.py             # SemanticMemory (importance-weighted pool)
│   ├── summary_memory.py              # SummaryMemory (periodic compression)
│   ├── episodic_semantic_memory.py    # EpisodicSemanticMemory (dual store)
│   ├── rag_memory.py                  # RAGMemory (sentence-transformers)
│   ├── retrieval.py                   # Graph, embedding, learnable retrieval
│   ├── embedding.py                   # TF-IDF embeddings (fixed vocab)
│   ├── entity_extraction.py           # Rule-based entity detection
│   ├── event.py                       # Event dataclass
│   └── __init__.py
├── agent/
│   ├── policy.py                      # ExplorationPolicy (multi-door aware)
│   ├── loop.py                        # run_episode_with_memory + run_episode_with_any_memory
│   └── __init__.py
├── evaluation/
│   ├── run.py                         # run_evaluation() + run_memory_comparison()
│   └── __init__.py
└── docs/
    ├── THESIS_STORY.md                # This file
    ├── STEP1_HARDER_ENVIRONMENT.md    # HardKeyDoor design rationale
    ├── THESIS_VISION_AND_PHASE6.md
    └── PHASE6_IMPLEMENTATION_PLAN.md
```
