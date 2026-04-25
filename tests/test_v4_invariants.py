"""
GraphMemoryV4 invariants — the headline thesis system.

These tests guard the design properties V4 was built to satisfy:
    1. MemoryParamsV4.from_vector(v.to_vector()) is identity.
    2. Default V4 params (theta_store=0, theta_novel=0, ...) store every event,
       reproducing the unparameterized baseline.
    3. With theta_novel=1.0 and theta_store=1.0 (impossible importance threshold),
       no events are stored.
    4. Same episode_seed and same event sequence => bit-identical retrieval results.
"""

from __future__ import annotations

import pytest

from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4


class TestMemoryParamsV4Roundtrip:
    def test_to_from_vector_identity(self):
        original = MemoryParamsV4(
            theta_store=0.293, theta_novel=0.908, theta_erich=0.198,
            theta_surprise=0.785, theta_entity=0.285, theta_temporal=0.278,
            theta_decay=0.668, w_graph=0.0, w_embed=1.079, w_recency=3.777,
        )
        roundtrip = MemoryParamsV4.from_vector(original.to_vector())
        assert roundtrip.to_vector() == original.to_vector()

    def test_from_vector_rejects_short_vector(self):
        with pytest.raises(ValueError):
            MemoryParamsV4.from_vector([0.5, 0.5, 0.5])  # only 3 dims

    def test_clipping_to_valid_ranges(self):
        # learnable mode clips theta_* to [0,1] and w_* to [0,4]
        p = MemoryParamsV4(theta_store=2.0, w_graph=10.0, mode="learnable")
        assert 0.0 <= p.theta_store <= 1.0
        assert 0.0 <= p.w_graph <= 4.0


class TestV4StorageBehavior:
    def test_default_params_store_everything(self, sample_events):
        """Default V4 (all thetas=0) accepts every event — matches V1 unparameterized."""
        mem = GraphMemoryV4(MemoryParamsV4())  # all storage thresholds = 0
        for ev in sample_events:
            mem.add_event(ev, episode_seed=42)
        stats = mem.get_stats()
        # The graph should contain at least one event node per added event
        assert stats.get("n_events", 0) >= len(sample_events)

    def test_high_importance_threshold_blocks_storage(self, sample_events):
        """
        With theta_novel>0 (engages the importance gate) and theta_store=1.0
        (importance must strictly exceed 1.0 to store), nothing should be stored
        because novelty is bounded at 1.0 and the gate is `importance <= theta_store`.

        Note: theta_novel=0 with no other weights triggers the fast-path that
        defaults to storing everything (the V1-baseline reproduction). So the
        gate-blocking test must set at least one weight > 0.
        """
        params = MemoryParamsV4(
            theta_store=1.0,
            theta_novel=1.0,
            theta_erich=0.0,
            theta_surprise=0.0,
            mode="learnable",
        )
        mem = GraphMemoryV4(params)
        for ev in sample_events:
            mem.add_event(ev, episode_seed=42)
        stats = mem.get_stats()
        assert stats.get("n_events", 0) == 0


class TestV4Determinism:
    def test_same_seed_same_retrieval(self, sample_events, query_observation):
        """Identical (seed, event sequence) must yield identical retrieval ordering."""
        params = MemoryParamsV4(
            theta_store=0.3, theta_novel=0.9, theta_surprise=0.7,
            w_recency=3.0, w_embed=1.0, mode="learnable",
        )

        def run_once() -> list[tuple[int, str]]:
            mem = GraphMemoryV4(params)
            for ev in sample_events:
                mem.add_event(ev, episode_seed=42)
            retrieved = mem.get_relevant_events(query_observation, current_step=10, k=3)
            return [(e.step, e.observation) for e in retrieved]

        a, b = run_once(), run_once()
        assert a == b

    def test_different_seeds_may_differ(self, sample_events, query_observation):
        """Sanity: with stochastic storage, different seeds *can* produce different graphs.
        We only assert the runs complete; equality is not required."""
        params = MemoryParamsV4(theta_temporal=0.5, mode="learnable")
        for seed in (0, 1, 2):
            mem = GraphMemoryV4(params)
            for ev in sample_events:
                mem.add_event(ev, episode_seed=seed)
            mem.get_relevant_events(query_observation, current_step=10, k=3)
