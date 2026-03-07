# Project Summary — Learning Task-Adaptive Structured Memory

*For external analysis / thesis writing. Last updated: Mar 2026. For run instructions and command order see **THESIS_HANDOFF.md** and **docs/RUNNING_EXPERIMENTS.md**.*
*Paste this entire file when asking an LLM to help with thesis writing, analysis, or experiment interpretation.*

---

## 1. What this project is

A Master's thesis in AI demonstrating that **memory construction for AI agents should be learned, not fixed**.

**Thesis title (working):** Learning Task-Adaptive Structured Memory for Long-Horizon Decision-Making

**Core claim:**
> Most AI agents learn a policy (what to do) and a value function (how good a state is). They do not learn how to build their own memory. This thesis shows that memory construction parameters can be learned end-to-end from task reward, producing memory structures that adapt to the task — storing what matters, ignoring what doesn't, and doing so efficiently.

---

## 2. What we built (current POC)

This is a **proof of concept** — a controlled simulation that validates the learning mechanism. It is NOT yet a production LLM system. Each toy component maps directly to a real-world equivalent (see Section 5).

### 2.1 Environments

| Environment | Grid | Steps | Task | Memory pressure |
|---|---|---|---|---|
| ToyEnvironment (Key-Door) | 6×6 | 80 | Pick up key matching door color | Low — task solvable by random walk |
| GoalRoom | 6×6 | 80 | Reach goal cell, use_door | Minimal |
| MultiHopKeyDoor | 10×10 | 250 | 3 doors, 6 real keys, 3 distractor keys, hint observations at steps 0–2 | High — hints arrive at steps 0/1/2, must be recalled up to 250 steps later |
| HardKeyDoor | 10×10 | 300 | 3 doors, 5 keys, 2 distractors | Medium |
| QuestRoom | 12×12 | 500 | 4 chained doors, 2 NPCs giving one-time hints, distractor observations | Very high — NPC hints must be recalled 300+ steps later |

**Primary benchmark: MultiHopKeyDoor.** Designed so the rule-based policy *can* succeed if memory delivers hint events, but *cannot* succeed without them. This gives clean, verifiable memory dependency.

All environments use the same interface: `reset()`, `step(action)`, `get_actions()`, `.done`, `.success`, `.partial_score`.

Observations are short text strings, e.g.:
- `"You are in a room. You see a red key."`
- `"A guard says: the blue key opens the first door."`
- `"You hear the wind howling."` (distractor noise)

### 2.2 Memory systems implemented

Six memory architectures, all sharing the same uniform interface:
- `add_event(event, episode_seed=None)`
- `get_relevant_events(observation, current_step, k) -> list[Event]`
- `clear()` / `get_stats()`

| System | File | Core idea | Real-world analog |
|---|---|---|---|
| FlatWindow | `memory/flat_memory.py` | Last N events, sliding window | Simple context truncation |
| GraphMemory+θ | `memory/graph_memory.py` | Graph of event/entity nodes, θ-parameterized | Structured knowledge graph |
| SemanticMemory | `memory/semantic_memory.py` | Importance-weighted pool (entity count + NPC hint + novelty); evicts low-importance events | Priority queue over context |
| SummaryMemory | `memory/summary_memory.py` | Every K steps, compress old events into a summary node | Recursive summarization |
| EpisodicSemantic | `memory/episodic_semantic_memory.py` | Dual store: recent raw events (episodic) + extracted persistent facts (semantic) | Human episodic vs. semantic memory |
| RAGMemory | `memory/rag_memory.py` | Dense sentence embeddings (MiniLM-L6-v2) + cosine retrieval | Production RAG (Pinecone, Chroma) |

### 2.3 Learnable memory parameters (θ)

The `GraphMemory+θ` system introduces a parameter vector **θ = (θ_store, θ_entity, θ_temporal) ∈ [0,1]³**:

| Parameter | Meaning | Effect |
|---|---|---|
| θ_store | Bernoulli probability of storing each event | Low → filter most events (small memory) |
| θ_entity | Frequency-based importance threshold for entity nodes | High → only track frequent entities |
| θ_temporal | Bernoulli probability of adding temporal edge | Low → sparse sequential chain |

Default θ = (1.0, 0.0, 1.0) exactly reproduces the fixed-memory baseline. Reproducible via seeded RNG per (episode_seed, step).

### 2.4 Learning mechanism

**Phase 6 — Random search (bandit):** Sample 15 θ configs uniformly from [0,1]³. Run 100 episodes per config. Objective: `J_learn(θ) = mean_reward` (pure task performance — no token penalty in optimizer). Pick best θ. Token cost tracked separately as efficiency = reward / (1 + tokens).

**Phase 7 — Evolution Strategy (ES):** 12 generations × 6 candidates × 40 episodes. Each generation: sample candidates from N(μ, σ), clip to [0,1], evaluate by mean reward, set μ = best candidate, decay σ. Learns θ over time. Full per-generation learning curve (theta, reward, tokens, efficiency) is logged to `report.txt`.

**token_usage** in this system = `Σ_t len(retrieved_events_at_step_t)` — a proxy for LLM context cost. Not real API tokens.

### 2.5 Agent and policy

- **Policy:** Rule-based `ExplorationPolicy` — reads past events to identify door color, picks up matching key, uses door when carrying matching key. Random move otherwise.
- **Episode runner:** `run_episode_with_any_memory(env, policy, memory, k)` — universal runner accepting any memory class. Returns `(success, events, stats_dict)` with `retrieval_tokens`, `memory_size`, `reward`.
- **No LLM involved.** No API keys needed. Everything runs locally.

---

## 3. Experimental results

*(From `report.txt`, Mar 2026 — 3 environments, Phase 6 + Phase 7 + Memory System Comparison)*

### Phase 6 results (random search, 15 configs × 100 episodes)

| Environment | Best θ (store, entity, temporal) | Success (learnable) | Mean reward | Efficiency |
|---|---|---|---|---|
| Key-Door | (0.088, 0.598, 0.070) | 25.0% | 0.250 | 0.001070 |
| Goal-Room | (0.265, 0.838, 0.769) | 71.0% | 0.710 | 0.003097 |
| MultiHop-KeyDoor | (0.667, 0.890, 0.486) | 15.0% | 0.053 | 0.000027 |

### Phase 7 results (ES, 12 gen × 6 cand × 40 ep)

| Environment | Learned θ (store, entity, temporal) | Learnable success | Baseline success | Mean reward | Efficiency |
|---|---|---|---|---|---|
| Key-Door | (0.116, 0.000, 0.819) | 30.0% | 17.5% | 0.300 | 0.001029 |
| Goal-Room | (1.000, 0.220, 1.000) | 80.0% | 70.0% | 0.800 | 0.002885 |
| MultiHop-KeyDoor | (1.000, 0.487, 0.843) | 27.5% | 2.5% | 0.092 | 0.000047 |

### Cross-environment comparison (core thesis result)

- **Key-Door:** θ_store=0.12 (aggressive filtering), θ_entity=0.00 (no entity threshold), θ_temporal=0.82. Interpretation: most events filtered, temporal chain maintained, entity nodes created freely.
- **Goal-Room:** θ_store=1.00 (store everything), θ_entity=0.22 (light entity filter), θ_temporal=1.00. Interpretation: simple goal task — store all, maintain full chain.
- **MultiHop-KeyDoor:** θ_store=1.00, θ_entity=0.49 (strong entity filter), θ_temporal=0.84. Interpretation: must store everything (hints), but filter entity nodes strongly (suppress distractor keys).
- **Core finding confirmed:** θ is task-dependent — different tasks converge to structurally different memory policies. This is the primary empirical support for the thesis claim.

### ES improvement over baseline

| Environment | Baseline reward | Learnable reward | Improvement |
|---|---|---|---|
| Key-Door | 0.175 | 0.300 | +71% |
| Goal-Room | 0.700 | 0.800 | +14% |
| MultiHop-KeyDoor | 0.008 | 0.092 | +1000% |

### Memory system comparison (MultiHopKeyDoor, n=50 episodes)

| System | Partial Score (doors/3) | Tokens | Retrieval Precision | Efficiency |
|---|---|---|---|---|
| EpisodicSemantic | **0.180** | 1961 | **1.000** | 0.000092 |
| SemanticMemory | 0.100 | 1964 | **1.000** | 0.000051 |
| GraphMemory+Theta | 0.060 | 1964 | 0.548 | 0.000031 |
| RAGMemory | 0.047 | 1964 | 0.531 | 0.000024 |
| FlatWindow(50) | 0.000 | 1964 | 0.060 | 0.000000 |
| SummaryMemory | 0.000 | 1636 | 0.023 | 0.000000 |

**Retrieval precision** = fraction of door-query steps where a hint event was in the top-k retrieved events. This is a direct memory quality metric independent of task success.

**Why EpisodicSemantic wins:** It stores hint observations verbatim in its semantic store (never evicted). At door steps, semantic facts are always returned → precision = 1.0.

**Why FlatWindow fails:** Hint observations arrive at steps 0–2. By the time the agent reaches a door (often 100+ steps later), the hint has scrolled out of the 50-event window.

**Why SummaryMemory fails:** Periodic compression destroys the exact text of the hint observation. The summary notes "saw red_key" but not "the red key opens the north door."

---

## 4. What the system is NOT (important caveats)

- **No real LLM.** Observations are hardcoded strings. The policy is rule-based string matching. No GPT, no Claude, no API.
- **No real embeddings.** TF-IDF over a 21-word fixed vocabulary (except RAGMemory which uses MiniLM locally).
- **Token count is a proxy.** `retrieval_tokens` counts list lengths, not actual API tokens billed.
- **Tasks are toy.** 6×6 to 12×12 grids. Not the real use case.
- **θ learning is limited.** Random search + ES over 3 parameters is not deep learning. It is gradient-free optimization over a 3D space.

---

## 5. The real system this points toward

The POC validates the principle. The production version replaces each toy component:

| POC component | Production equivalent |
|---|---|
| Hardcoded text observations | Real environment: game engine, document corpus, API responses |
| Rule-based policy | LLM (GPT-4o, Claude 3.5, Llama 3) as reasoning agent |
| TF-IDF 21-word embeddings | `text-embedding-3-small` or `all-MiniLM-L6-v2` via API |
| `retrieval_tokens` count | Actual token count billed by OpenAI (cost in dollars) |
| J = partial_score − λ × count | J = task_score − λ × API_cost |
| QuestRoom 12×12 grid | Real long-form task: 500-page lore, legal corpus, codebase |
| Uniform interface memory systems | Drop-in replacement memory modules for any LLM agent framework |

The architecture, the interface, and the θ-learning mechanism are **identical**. Only the components change.

---

## 6. The thesis argument (what to tell ChatGPT)

**The problem:** LLM agents either store everything (expensive, noisy) or retrieve randomly (loses critical information). Memory structure is fixed. There is no mechanism to learn what to remember based on the task.

**The insight:** Different tasks require different memory structures. A key-door task needs entity tracking. A navigation task needs temporal chaining. A lore-understanding task needs NPC hint retention. One fixed memory structure cannot be optimal for all.

**The approach:** Parameterize memory construction with θ. Learn θ via Evolution Strategy optimizing J = reward − λ × token_cost. The same mechanism works for any environment that exposes a reward signal.

**The demo:** Toy environments. θ adapts per task. Key-Door → entity-focused memory. Goal-Room → sparse, recency-focused. QuestRoom → NPC hints retained by EpisodicSemantic, flat window fails.

**The claim:** Memory structure is task-dependent and learnable. This is a new contribution: not a better policy, not a better retrieval function, but **meta-level learning over memory representation**.

**The scaling argument:** In GPT-4 agents today, memory is managed manually (truncation, summarization, vector search). Our mechanism automates this: θ_store ↔ context truncation policy, θ_entity ↔ what to index in RAG, θ_temporal ↔ how much sequential structure to maintain. The same θ-learning loop applies, with J = task_success − λ × API_cost.

---

## 7. File map

```
d:\Bocconi\Thesis\
├── main.py                            # Orchestrates all phases; writes report.txt
├── report.txt                         # Auto-generated experiment results (paste to ChatGPT)
├── requirements.txt                   # networkx, numpy, scikit-learn, sentence-transformers
├── environment/
│   ├── env.py                         # ToyEnvironment, GoalRoom, HardKeyDoor, QuestRoom
│   └── __init__.py
├── memory/
│   ├── graph_memory.py                # GraphMemory + MemoryParams (θ) — current main system
│   ├── flat_memory.py                 # FlatMemory — sliding window baseline
│   ├── semantic_memory.py             # SemanticMemory — importance-weighted pool
│   ├── summary_memory.py              # SummaryMemory — periodic compression
│   ├── episodic_semantic_memory.py    # EpisodicSemanticMemory — dual store
│   ├── rag_memory.py                  # RAGMemory — MiniLM dense embeddings
│   ├── retrieval.py                   # Graph, embedding, learnable retrieval functions
│   ├── embedding.py                   # TF-IDF embeddings (fixed vocab, local)
│   ├── entity_extraction.py           # Rule-based entity detection
│   ├── event.py                       # Event dataclass (step, observation, action)
│   └── __init__.py
├── agent/
│   ├── policy.py                      # ExplorationPolicy (rule-based, reads past events)
│   ├── loop.py                        # run_episode_with_memory + run_episode_with_any_memory
│   └── __init__.py
├── evaluation/
│   ├── run.py                         # run_evaluation() + run_memory_comparison()
│   └── __init__.py
└── docs/
    ├── PROJECT_SUMMARY_FOR_CHATGPT.md # This file
    ├── THESIS_STORY.md                # Full narrative + design rationale
    ├── STEP1_HARDER_ENVIRONMENT.md    # HardKeyDoor design rationale
    ├── THESIS_VISION_AND_PHASE6.md    # Original thesis vision document
    └── PHASE6_IMPLEMENTATION_PLAN.md  # Phase 6 implementation plan
```

---

## 8. Next steps (what comes after the POC)

**Near-term (still POC, but stronger) — PARTIALLY DONE:**
- ✅ Run `main.py` on all 3 environments — complete
- ✅ Memory comparison table (6 systems on MultiHopKeyDoor) — complete
- ✅ ES learning curves — now in `report.txt` (all 12 generations per env)
- ✅ Efficiency correctly reported in EXPERIMENT SUMMARY
- ⬜ Graph visualization: render learned graph (sparse θ) vs. fixed graph (dense) — not yet
- ⬜ Local LLM integration (Ollama / Llama 3) as the policy — not yet

**Medium-term (bridge to real system):**
- Wrap a real LLM (e.g. Ollama + Llama 3 locally, or OpenAI API) as the policy
- Replace hardcoded observations with real text from a document corpus or game engine
- Replace TF-IDF with real embeddings (`text-embedding-3-small`)
- Measure actual token cost (not proxy count)

**Long-term (full thesis contribution):**
- Test on a task that genuinely exceeds context window (e.g. 500-page lore, multi-session dialogue)
- Show that learned θ reduces API cost while maintaining task performance
- Compare against production memory systems (LangChain memory, MemGPT, etc.)
- This is the publishable contribution

---

## 9. How to run

```bash
# Install dependencies (no API key needed)
pip install networkx numpy scikit-learn sentence-transformers

# Run the full pipeline
cd d:\Bocconi\Thesis
python main.py

# Output: console report + report.txt
# Takes ~15-30 minutes (3 environments × Phase 6 + Phase 7 + memory comparison)
```

---

## 10. GitHub

Repository: https://github.com/UifaleanStefan/MasterThesis
