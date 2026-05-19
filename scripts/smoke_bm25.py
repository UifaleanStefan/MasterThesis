"""Quick BM25Memory smoke test — single-file sanity check."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.bm25_memory import BM25Memory
from memory.event import Event

mem = BM25Memory()
events = [
    Event(step=0, observation="Paris is the capital of France", action="read"),
    Event(step=1, observation="London is the capital of England", action="read"),
    Event(step=2, observation="The Eiffel Tower is in Paris", action="read"),
    Event(step=3, observation="Berlin is the capital of Germany", action="read"),
]
for e in events:
    mem.add_event(e)
print("stats:", mem.get_stats())

queries = [
    "what is in Paris",
    "what is the capital of Germany",
    "Eiffel Tower location",
]
for q in queries:
    print(f"\nquery: {q!r}")
    top = mem.get_relevant_events(q, current_step=10, k=2)
    for e in top:
        print(f"  step={e.step}: {e.observation}")

# Memory contract check
mem.clear()
assert mem.get_stats()["n_events"] == 0
print("\nclear() works.")
