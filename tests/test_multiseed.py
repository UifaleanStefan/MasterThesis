"""Tests for scripts/aggregate_multiseed.summarize_group (audit A4-C)."""
from scripts.aggregate_multiseed import summarize_group


def test_cross_seed_stats():
    # seed1 mean 0.5, seed2 mean 1.0 -> cross mean 0.75, std 0.25, spread 0.5
    s = summarize_group({1: [1.0, 0.0], 2: [1.0, 1.0]})
    assert s["n_seeds"] == 2
    assert s["cross_seed_mean"] == 0.75
    assert s["cross_seed_std"] == 0.25
    assert s["cross_seed_spread"] == 0.5
    assert s["pooled_n"] == 4
    assert s["per_seed"]["1"]["mean"] == 0.5 and s["per_seed"]["2"]["mean"] == 1.0


def test_single_seed_has_no_cross_std():
    s = summarize_group({42: [0.5, 0.5, 1.0]})
    assert s["n_seeds"] == 1
    assert s["cross_seed_std"] is None
    assert s["cross_seed_spread"] is None
    assert s["cross_seed_mean"] == round(2.0 / 3, 4)


def test_identical_seeds_zero_std():
    # near-deterministic case: identical per-seed means -> std 0
    s = summarize_group({1: [1.0, 0.0], 2: [0.0, 1.0]})
    assert s["cross_seed_std"] == 0.0
