"""Event representation for memory system."""

from dataclasses import dataclass


@dataclass
class Event:
    """Single interaction step stored in memory."""

    step: int
    observation: str
    action: str

    def __hash__(self) -> int:
        return hash((self.step, self.observation, self.action))
