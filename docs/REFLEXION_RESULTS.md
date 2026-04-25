# GraphMemoryV6 — Reflexion-Augmented Memory: Architecture, Invariants, and Empirical Findings

**Date:** April 2026
**Plan reference:** `.claude/plans/okay-create-a-plan-curried-sparkle.md` (Reflexion plan, post-research-deep-dive)
**Commits:** Phase 0 (oracle), Phase 1 (V6 + Lesson), Phase 2 (ReflexionPolicy + runner), Phase 3 (same-env retry experiment)

---

## Motivation

After landing the post-MiniLM canonical results, the thesis had one
unresolved finding: V4 reaches retrieval precision = 0.94+ on MegaQuestRoom
but reward = 0.00. Memory is doing its job; the agent never reaches the
goals. The 2024-2026 research synthesis identified this as a textbook
"policy bottleneck under partial observability" — the canonical fix being
a verbal-lesson buffer (Reflexion, Voyager, Meta-Policy Reflexion).

The Reflexion plan added GraphMemoryV6 — V4 with a second memory tier for
verbal lessons synthesized at episode end — and tested whether the lesson
channel can lift V4's reward on long-horizon and within-task-retry settings.

## Phase 0 — Oracle diagnosis (PASS)

Before building V6, we needed to confirm the env is solvable at all and
that the bottleneck is provably policy.

`agent/oracle_policy.py` exposes two policies:

* **OraclePolicy** — observation-only snake-sweep + Manhattan path planner.
  Maintains self-coordinates from move deltas; falls back to random walk
  when self-coords drift due to wall-clamping.
* **OmniscientOraclePolicy** — env-cheating reference. Reads
  `env._agent_pos / _key_positions / _door_positions / _door_key_map`
  directly to compute optimal Manhattan paths through the full sequence.
  Used solely as the env-solvability upper bound.

`run_oracle_diagnosis.py` runs 30 episodes per configuration on
MegaQuestRoom. Result:

| Configuration | Reward | Doors / 6 | Precision |
|---|---:|---:|---:|
| ExplorationPolicy + V4         | 0.0000 | 0.00 | 0.870 |
| OracleHonest + V4              | 0.0333 | 0.20 | 0.799 |
| OracleHonest + FlatMemory(50)  | 0.0389 | 0.23 | 0.002 |
| **OmniscientOracle + FlatMemory** | **1.0000** | **6.00** | 0.173 |

**Diagnosis: POLICY-BOTTLENECK CONFIRMED.** The omniscient policy hits
100% reward across all 30 episodes — the env is fully solvable with full
information. The gap from 0.03 (honest oracle) to 1.00 (omniscient
oracle) is the "observation → spatial map → optimal plan" capability
that an honest agent lacks under partial observability.

## Phase 1 — V6 architecture (PASS — 60 invariants)

`memory/graph_memory_v6.py` extends V4 with two new dimensions of θ and
one new node type:

* **`MemoryParamsV6`** = V4's 10D + (`w_lesson`, `theta_lesson_decay`).
* **`Lesson` dataclass** (in `memory/lesson.py`): episode_seed, final_reward,
  text, relevant_entities, relevant_event_steps, success_marker.
* **Lesson nodes** in the same graph as events with `node_type="lesson"`,
  edges to entities (`lesson_mentions`/`mentioned_in_lesson`) and to
  events (`lesson_derived_from`).
* **`record_lesson()` / `end_episode()`** lifecycle hooks.
* **`get_relevant_lessons(observation, k)`** — separate retrieval channel
  scored by `w_lesson * cosine_sim * exp(-theta_lesson_decay * episodes_ago)`.
* **`clear()` deliberately preserves lessons** — they're the
  cross-episode learning channel by design. `reset_lesson_buffer()` is
  the explicit full reset.

Two reflection generators behind a uniform interface:

* `heuristic_lesson()` — template-based. Emits success / failure lessons
  from observed hints; returns `None` on middle-ground episodes.
* `llm_lesson()` — GPT-4o-mini summarizer with no-API-key fallback to
  the heuristic. Mirrors the `evaluation.document_qa_llm_judge` pattern.

**Invariants tested in `tests/test_v6_invariants.py` (18 tests):**

* Strict generalization: `w_lesson=0` reproduces V4 retrieval bit-identically.
* Lesson roundtrip: `record_lesson(L); get_relevant_lessons(...)` returns L.
* Decay reduces older episodes' lesson scores (controlled-text test).
* `clear()` keeps lessons, wipes events. `reset_lesson_buffer()` wipes lessons.
* `episode_index` survives `clear()` so decay is consistent.
* `MemoryParamsV6.from_vector(p.to_vector()) == p`.
* Heuristic generator: success / failure / middle / no-hint cases.
* LLM generator: graceful fallback to heuristic without API key.

Total pytest invariants rise from 40 → **60** (all passing). Determinism
audit covers V6 across all 17 systems.

## Phase 2 — ReflexionPolicy + persistent runner (PASS)

`agent/reflexion_policy.py` decorates any base policy with retrieved-lesson
context. On each step it retrieves top-k lessons via
`memory.get_relevant_lessons()`, wraps them as synthetic Events with
negative step indices and `is_hint=True`, and prepends them to past_events
before delegating to `base_policy.decide()`. The base policy never knows
about lessons — its existing hint regex (`the X key opens the Y door`) parses
lesson text exactly like an observation.

`evaluation/reflexion_eval.py` provides:

* `persistent_run_episodes()` — multi-episode loop against a SINGLE memory
  instance, paired by env_seed and policy_seed.
* `compare_variants()` — original 4-way comparison
  (V4-base / V6-no-lessons / V6-w-lessons / V6-Reflexion).
* `retry_run()` and `compare_retry_variants()` — same-env retry
  experiment (Reflexion's original setup).

`run_reflexion_ablation.py` drives the retry experiment with per-try
learning curves and pairwise significance on final-try rewards.

## Phase 3 — Same-env retry experiment (NEGATIVE RESULT)

The thesis-relevant test: same env layout, multiple attempts, lessons
accumulate within-env across tries. Reflexion's standard setup.

**Empirical result on MultiHop-KeyDoor (5 env layouts × 4 tries each):**

| Variant       | Overall mean | Final-try mean | Significance vs V4-base |
|---|---:|---:|---:|
| V4-base       | 0.1333 | 0.0667 | — |
| V6-w-lessons  | 0.1333 | 0.0667 | p = 1.000, d = 0.000 |
| V6-Reflexion  | 0.1333 | 0.0667 | p = 1.000, d = 0.000 |

The three variants are **statistically indistinguishable.** This is a
clean honest finding once explained:

1. The strict-generalization invariant holds — V4-base equals
   V6-w-lessons (lessons don't pollute event retrieval scoring).
2. `ExplorationPolicy` is rule-based and effectively deterministic given
   observed hints — almost all decisions are determined by the hint map
   parsed from `past_events`, not by RNG.
3. `heuristic_lesson()` paraphrases hints already visible in
   observations ("the red key opens the north door"). The lesson channel
   adds no new information that isn't already in the observation
   channel.
4. To produce strategic insight beyond hint-paraphrase, lessons would
   need to encode SPATIAL knowledge (where keys/doors are at specific
   coordinates) or INTER-TRY DIFFERENCES (which paths failed last
   time). Both require either an LLM judge or a learning-to-explore
   policy — out of scope for the rule-based grid-world setup.

## What this means for the thesis

The honest framing: **V6 is a structurally sound architecture whose
empirical lift is gated on richer lesson generators.**

The thesis can defensibly state:

> "We built and verified a Reflexion-style verbal-lesson buffer (V6) on
> top of V4. The strict-generalization invariant holds (V6 with
> `w_lesson = 0` matches V4 exactly), the lesson primitive round-trips
> cleanly, and decay works as designed. On grid-world envs with the
> rule-based ExplorationPolicy, however, the lesson channel cannot
> empirically lift reward because the policy is already deterministic
> given parsed hints — the heuristic lesson generator paraphrases the
> same hint information that is already in the observation channel.
>
> The architecture lifts cleanly into Stage 3 where an LLM agent could
> emit strategic verbal lessons (via `llm_lesson` with a real API key),
> and where lesson text could encode meta-strategy beyond regex-parseable
> hints. We document this as a *plumbing-complete, evidence-gated*
> contribution: the framework is in place, awaiting the richer
> reflective signal that an LLM judge can provide."

This is a more defensible claim than over-reaching with a synthetic
positive on a benchmark where the architecture can't produce one. Cited
prior work (Reflexion, Voyager, Meta-Policy Reflexion) all relied on LLM
agents — Reflexion specifically reports +22pp on AlfWorld with an LLM,
not a rule-based policy. The negative result on rule-based policy is
itself diagnostic, and matches the literature.

## What's NOT in scope (deferred)

* **V7 skill library** — would face the same architectural ceiling on
  rule-based policy. Deferred until LLM-agent integration.
* **Better honest oracle** — the snake-sweep oracle's 0.03 reward is
  itself diagnostic; spending more time on it doesn't change the headline.
* **CMA-ES tuning of V6 θ** — original Phase 3. Without empirical lift
  from the lesson channel, tuning the new dimensions is uninformative.

## Files added

* `agent/oracle_policy.py` — OraclePolicy (honest) + OmniscientOraclePolicy (cheating reference).
* `run_oracle_diagnosis.py` — Phase 0 diagnosis runner.
* `memory/lesson.py` — Lesson dataclass + heuristic / LLM generators.
* `memory/graph_memory_v6.py` — V6 inheriting from V4 + lesson tier.
* `agent/reflexion_policy.py` — wrapper that injects lessons as synthetic past_events.
* `evaluation/reflexion_eval.py` — persistent-memory + same-env retry runners.
* `run_reflexion_ablation.py` — CLI for the headline retry experiment.
* `tests/test_v6_invariants.py` — 18 invariants for V6 + Lesson lifecycle.
* `results/oracle_diagnosis.json`, `results/reflexion_retry.json` — raw data.

## How to reproduce

```powershell
# Phase 0 — confirm policy bottleneck (~20 seconds)
python run_oracle_diagnosis.py --episodes 30

# Phase 1+2 — invariants and contract tests
python -m pytest tests/test_v6_invariants.py tests/test_memory_contract.py -v

# Phase 3 — retry experiment (~10 seconds for the smoke; minutes for canonical)
python run_reflexion_ablation.py --n-env-seeds 5 --n-tries 4 --envs MultiHop-KeyDoor
python run_reflexion_ablation.py --n-env-seeds 10 --n-tries 8 --envs MultiHop-KeyDoor MegaQuestRoom

# Determinism audit (V6 included)
python scripts/audit_determinism.py
```
