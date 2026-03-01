"""Event representation for memory system."""

from dataclasses import dataclass, field


@dataclass
class Event:
    """Single interaction step stored in memory."""

    step: int
    observation: str
    action: str
    is_hint: bool = field(default=False, compare=False, hash=False)
    """True for observations that are hint events (e.g. MultiHopKeyDoor signs).
    Used by retrieval_precision metric: did memory return the right hint when needed?"""

    def __hash__(self) -> int:
        return hash((self.step, self.observation, self.action))
