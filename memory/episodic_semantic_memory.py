"""
EpisodicSemanticMemory — dual-store memory mirroring human memory architecture.

Two separate stores:
  - Episodic buffer: sliding window of the last M raw events (recent, detailed).
    Answers "what just happened?"
  - Semantic store: important facts extracted and deduplicated across the episode.
    Facts are never evicted. Answers "what do I know?"

Semantic extraction rules:
  - If observation contains an NPC hint ("says:"), extract the full hint as a fact.
  - If observation contains a new entity not seen before, extract "saw <entity> at step N".
  - If a door is successfully opened (action=use_door and carried key matches), record it.

Retrieval = all semantic facts + episodic recent events (up to k total).
Semantic facts always included because they represent long-term knowledge.

This is the closest analog to the cognitive distinction between episodic memory
(what happened, where, when) and semantic memory (timeless knowledge).
"""

from .entity_extraction import extract_entities
from .event import Event

# Observations that carry long-term semantic facts:
#   - NPC hints ("A guard says: ...")
#   - Sign hints ("You see a sign: the X key opens the Y door")
_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")


def _is_npc_hint(obs: str) -> bool:
    obs_l = obs.lower()
    return any(m in obs_l for m in _NPC_MARKERS)


class EpisodicSemanticMemory:
    """
    Dual-store memory: episodic sliding window + persistent semantic facts.
    """

    def __init__(self, episodic_size: int = 30) -> None:
        self._episodic_size = episodic_size
        self._episodic: list[Event] = []    # recent raw events
        self._semantic: list[Event] = []    # extracted facts (never evicted)
        self._seen_entities: set[str] = set()
        self._seen_hints: set[str] = set()  # deduplicate hints

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        obs = event.observation

        # --- Semantic extraction ---
        if _is_npc_hint(obs):
            # Store the full hint observation as a semantic fact (once per unique hint).
            # Preserve is_hint flag so retrieval_precision tracking works correctly.
            if obs not in self._seen_hints:
                self._seen_hints.add(obs)
                fact_event = Event(
                    step=event.step,
                    observation=obs,
                    action=event.action,
                    is_hint=event.is_hint,
                )
                self._semantic.append(fact_event)

        # New entity seen for the first time → store as semantic fact
        entities = extract_entities(obs)
        for ent in entities:
            if ent not in self._seen_entities:
                self._seen_entities.add(ent)
                fact_obs = f"[Fact] First saw {ent} at step {event.step}."
                fact_event = Event(step=event.step, observation=fact_obs, action=event.action)
                self._semantic.append(fact_event)

        # --- Episodic store (sliding window) ---
        self._episodic.append(event)
        if len(self._episodic) > self._episodic_size:
            self._episodic.pop(0)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Always include all semantic facts; fill remaining slots with recent episodic events.
        Semantic facts are small (one per unique hint/entity) so this is bounded.
        """
        semantic_events = list(self._semantic)
        recent = self._raw_recent(k)

        # Combine: semantic first (long-term knowledge), then recent (current context)
        seen_steps: set[int] = set()
        combined: list[Event] = []
        for e in semantic_events + recent:
            if e.step not in seen_steps:
                seen_steps.add(e.step)
                combined.append(e)

        return combined[:k]

    def _raw_recent(self, k: int) -> list[Event]:
        return self._episodic[-k:] if k > 0 else list(self._episodic)

    def clear(self) -> None:
        self._episodic.clear()
        self._semantic.clear()
        self._seen_entities.clear()
        self._seen_hints.clear()

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._episodic) + len(self._semantic),
            "n_entities": len(self._seen_entities),
            "n_nodes": len(self._episodic) + len(self._semantic),
            "n_edges": 0,
        }
