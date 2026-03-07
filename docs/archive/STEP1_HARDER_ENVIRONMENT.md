*Archived; see THESIS_HANDOFF and PROJECT_SUMMARY for current state.*

# Step 1 — Harder Environment: Multi-Door Chain Task

*Goal: create genuine memory pressure so that θ meaningfully controls what the agent can and cannot remember.*

---

## Problem with the current environment

The 6×6, 80-step, 1-door environment is too easy. A random walker finds the key and door by chance within 80 steps often enough that memory only marginally helps. Specifically:

- θ_store = 0.02 (almost nothing stored) achieves 21% success → memory not required.
- Entity graph never has more than 3 nodes → θ_entity has little to filter.
- 80 steps fits entirely in short-term context → no "beyond context" pressure.

---

## Design: HardKeyDoor (Multi-Door Chain)

### Grid and episode
- **Size:** 10×10 (100 cells, vs. 36 currently)
- **Max steps:** 300 (vs. 80)

### Objects
- **5 keys** (red, blue, green, yellow, purple)
- **3 doors** (each requires exactly one specific key color)
  - e.g. Door A needs red key, Door B needs blue key, Door C needs green key
  - Yellow and purple keys are **distractors** — they don't open any door
- **Goal:** open all 3 doors (use the right key on each)

### Observations (text, partial)
Same format as ToyEnvironment:
- `"You are in a room. You see a red key."`
- `"You are in a room. You see a blue door."` (but which key opens it? Agent must remember.)
- `"You are in a room. You see nothing of interest."`
- `"You are in a room. You are carrying a green key."`
- `"You are in a room. You see a yellow key."` (distractor — picking it up wastes carrying capacity)

### Why memory matters now
1. **3 separate key-door pairs to track.** The agent must remember which color key goes with which door across potentially hundreds of steps.
2. **Distractors force selective storage.** The agent sees yellow/purple keys that are useless. θ_entity must learn to suppress distractor entities.
3. **Carrying capacity = 1 key.** If the agent picks up a distractor, it must drop it (swap) to pick up the right key. Memory about which keys are distractors saves time.
4. **300 steps.** Early observations about door colors may be hundreds of steps in the past. θ_temporal and θ_store must preserve the right events.

### Success
- Each door opened = +1 reward event. Full success = all 3 doors opened.
- Partial success possible (opened 1/3 or 2/3 doors).
- J(θ) = (doors_opened / 3) − λ × retrieval_tokens

### Interface (unchanged from ToyEnvironment)
- `reset() -> str`
- `step(action) -> (obs, done, success)`
- `get_actions() -> list`
- `.done`, `.success`

The only addition: an optional `.doors_opened` counter for richer reporting.

---

## What this changes in the pipeline

| Component | Change |
|---|---|
| `environment/env.py` | Add `HardKeyDoor` class |
| `environment/__init__.py` | Export `HardKeyDoor` |
| `memory/entity_extraction.py` | Add yellow, purple to COLORS |
| `memory/embedding.py` | Add "yellow", "purple" to VOCAB |
| `agent/policy.py` | Extend to handle multiple doors and distractors |
| `main.py` | Add `"Hard-KeyDoor"` to env loop |

---

## Expected experimental outcome

| Metric | Current (6×6, 1 door) | Expected (10×10, 3 doors) |
|---|---|---|
| Baseline success | ~8–28% | ~2–10% (much harder) |
| Fixed memory success | ~28% | ~15–25% |
| Learned θ success | ~32% | ~30–40% |
| Performance gap (fixed vs learned) | small | larger |
| θ_entity meaning | trivially small | must actively filter distractors |
| θ_store meaning | near 0 works fine | low θ_store → miss critical past info |
| Thesis claim support | weak | strong |

---

## Implementation order

1. Add `HardKeyDoor` to `environment/env.py`
2. Extend entity extraction and embedding vocab
3. Extend policy for multi-door logic
4. Add `"Hard-KeyDoor"` to `main.py` env loop
5. Re-run Phase 6 and Phase 7; update `report.txt`
6. Update `THESIS_STORY.md` with new results
