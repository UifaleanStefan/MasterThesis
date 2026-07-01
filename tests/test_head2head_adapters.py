"""Interface + behavior tests for the Phase-3 head-to-head memory adapters
(faithful HippoRAG and Letta/MemGPT reimplementations).

These are retrieval-only reimplementations used solely for the published-system
comparison; they implement the same interface as the other memory configs
(add_event / get_relevant_events / get_stats) so the head-to-head runs under the
identical gpt-4o-mini answerer + Claude judge.
"""
from __future__ import annotations

import pytest

from memory.event import Event
from memory.hipporag_memory import HippoRAGMemory
from memory.letta_memory import LettaMemory

DOCS = [
    "[Acme Distributor Agreement] This DISTRIBUTOR AGREEMENT is between Acme Corp and Beta Inc, governed by the laws of Delaware.",
    "The term is five years commencing on the Effective Date, January 1 2020.",
    "Beta Inc shall not assign this agreement without Acme Corp prior written consent.",
    "[Zeta Supply Contract] This SUPPLY CONTRACT between Zeta Ltd and Omega LLC is governed by New York law.",
    "Omega LLC agrees to purchase 1000 units per quarter from Zeta Ltd.",
    "The warranty period is twenty-four months from delivery.",
    "Acme Corp reserves the right to terminate for convenience upon 30 days notice.",
    "Confidential information of Beta Inc shall be protected for three years.",
]


def _ingest(mem):
    for i, d in enumerate(DOCS):
        mem.add_event(Event(step=i, observation=d, action="read"), episode_seed=42)
    return mem


@pytest.mark.parametrize("cls", [HippoRAGMemory, LettaMemory])
def test_interface_contract(cls):
    mem = _ingest(cls())
    res = mem.get_relevant_events("What law governs the Acme distributor agreement?",
                                  current_step=len(DOCS), k=4)
    assert isinstance(res, list)
    assert 0 < len(res) <= 4
    assert all(isinstance(e, Event) for e in res)
    stats = mem.get_stats()
    for key in ("n_events", "n_entities", "n_nodes", "n_edges"):
        assert key in stats
    assert stats["n_events"] == len(DOCS)


@pytest.mark.parametrize("cls", [HippoRAGMemory, LettaMemory])
def test_deterministic_at_fixed_seed(cls):
    a = _ingest(cls()).get_relevant_events("governing law Acme", current_step=len(DOCS), k=5)
    b = _ingest(cls()).get_relevant_events("governing law Acme", current_step=len(DOCS), k=5)
    assert [e.step for e in a] == [e.step for e in b]


def test_hipporag_builds_entity_graph():
    """HippoRAG must build a schemaless phrase+passage KG (nodes and edges > 0)."""
    mem = _ingest(HippoRAGMemory())
    s = mem.get_stats()
    assert s["n_entities"] > 0
    assert s["n_nodes"] > s["n_events"]   # passage nodes + phrase nodes
    assert s["n_edges"] > 0


def test_hipporag_ppr_surfaces_topical_passage():
    """PPR seeded on query entities should rank an on-topic passage first."""
    mem = _ingest(HippoRAGMemory())
    res = mem.get_relevant_events("Which law governs the Acme Corp distributor agreement with Beta Inc?",
                                  current_step=len(DOCS), k=3)
    # the Acme/Delaware governing-law evidence is in doc 0 (title) or doc 2 (Acme/Beta)
    assert any(e.step in (0, 2) for e in res)


def test_letta_recency_core_is_present():
    """Letta's always-on recency core should surface the most recent passages."""
    mem = _ingest(LettaMemory())
    res = mem.get_relevant_events("unrelated query text", current_step=len(DOCS), k=4)
    steps = {e.step for e in res}
    # the recency core guarantees at least one of the most-recent events appears
    assert steps & {len(DOCS) - 1, len(DOCS) - 2, len(DOCS) - 3}
