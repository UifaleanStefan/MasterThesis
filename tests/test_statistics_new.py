"""Tests for the Phase 1.7 statistics additions (Wilcoxon, Holm, cluster bootstrap)."""

import pytest

from evaluation.statistics import (
    cluster_bootstrap_ci,
    holm_bonferroni,
    wilcoxon_signed_rank,
)


class TestWilcoxonSignedRank:
    def test_identical_returns_nonsignificant(self):
        r = wilcoxon_signed_rank([0.5] * 10, [0.5] * 10)
        assert r["n_nonzero"] == 0
        assert r["p_two_sided"] == 1.0
        assert not r["significant"]

    def test_clearly_different_returns_significant(self):
        a = [0.1] * 30
        b = [0.9] * 30
        r = wilcoxon_signed_rank(a, b)
        assert r["significant"]
        assert r["p_two_sided"] < 0.01
        assert r["median_diff"] == pytest.approx(0.8)

    def test_no_difference_with_noise(self):
        a = [0.1, 0.5, 0.3, 0.7, 0.2]
        b = [0.2, 0.4, 0.4, 0.6, 0.3]
        r = wilcoxon_signed_rank(a, b)
        assert r["n_nonzero"] >= 4
        # Mixed direction → not significant
        assert not r["significant"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1, 2], [1, 2, 3])

    def test_returns_required_keys(self):
        r = wilcoxon_signed_rank([0.0, 0.5, 1.0], [0.5, 0.5, 1.0])
        for key in ["W", "W_plus", "W_minus", "p_two_sided", "p_one_sided",
                    "n_nonzero", "n_total", "median_diff", "significant"]:
            assert key in r


class TestHolmBonferroni:
    def test_empty_returns_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_p_unchanged(self):
        r = holm_bonferroni([0.04])
        assert len(r) == 1
        assert r[0]["p_raw"] == 0.04
        assert r[0]["p_adjusted"] == pytest.approx(0.04)
        assert r[0]["significant"]

    def test_multiple_comparisons_inflates_threshold(self):
        # 5 comparisons, one p=0.045 — Bonferroni-uncorrected significant, Holm-corrected NOT.
        ps = [0.045, 0.3, 0.5, 0.7, 0.8]
        r = holm_bonferroni(ps, alpha=0.05)
        # Smallest p × 5 = 0.225 > 0.05 → not significant under Holm.
        assert r[0]["p_adjusted"] == pytest.approx(0.225)
        assert not r[0]["significant"]

    def test_smallest_p_passes_when_truly_small(self):
        # p=0.001 in 5 comparisons → 0.005, still significant.
        ps = [0.001, 0.3, 0.5, 0.7, 0.8]
        r = holm_bonferroni(ps, alpha=0.05)
        assert r[0]["p_adjusted"] == pytest.approx(0.005)
        assert r[0]["significant"]
        # Others should not be significant.
        for entry in r[1:]:
            assert not entry["significant"]

    def test_monotonicity_under_stepdown(self):
        # The step-down enforces that adjusted-p is monotonically nondecreasing
        # in the sorted-p order.
        ps = [0.01, 0.012, 0.02, 0.3, 0.5]
        r = holm_bonferroni(ps)
        sorted_adj = sorted(r, key=lambda x: x["rank"])
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i]["p_adjusted"] >= sorted_adj[i - 1]["p_adjusted"]

    def test_returns_in_original_order(self):
        # Input is unsorted; output should be in input order (not sorted order).
        ps = [0.5, 0.001, 0.3]
        r = holm_bonferroni(ps)
        assert r[0]["p_raw"] == 0.5
        assert r[1]["p_raw"] == 0.001
        assert r[2]["p_raw"] == 0.3
        # The smallest p (0.001) gets rank=1.
        assert r[1]["rank"] == 1


class TestClusterBootstrapCi:
    def test_single_cluster_falls_back_to_iid(self):
        # If only 1 cluster, the function should still return a valid CI
        # (degenerate case — fallback to IID bootstrap).
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        ids = ["doc_0"] * 5
        r = cluster_bootstrap_ci(values, ids, n_resamples=100, seed=42)
        assert "ci_lower" in r and "ci_upper" in r
        # The point estimate is the mean.
        assert r["point_estimate"] == pytest.approx(0.3)

    def test_clustered_ci_wider_than_iid(self):
        # Construct a case where within-cluster correlation makes the
        # cluster bootstrap CI wider than the naive IID one.
        # 5 clusters of 10 obs each, where each cluster has the same value.
        values = []
        ids = []
        cluster_means = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, m in enumerate(cluster_means):
            values.extend([m] * 10)
            ids.extend([f"doc_{i}"] * 10)
        r_cluster = cluster_bootstrap_ci(values, ids, n_resamples=500, seed=42)
        # The cluster CI is the bootstrap over cluster choices.
        # Point estimate = mean of all 50 obs = 0.5.
        assert r_cluster["point_estimate"] == pytest.approx(0.5)
        assert r_cluster["n_clusters"] == 5
        assert r_cluster["n_observations"] == 50
        # CI should bracket the point estimate.
        assert r_cluster["ci_lower"] <= 0.5 <= r_cluster["ci_upper"]

    def test_empty_input(self):
        r = cluster_bootstrap_ci([], [], n_resamples=100, seed=42)
        assert r["n_clusters"] == 0
        assert r["n_observations"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cluster_bootstrap_ci([1, 2, 3], ["doc_0", "doc_1"], n_resamples=10)

    def test_determinism_across_runs(self):
        # Same seed → same CI.
        values = [0.1, 0.2, 0.3, 0.4, 0.5] * 4
        ids = ["a", "b", "c", "d"] * 5
        r1 = cluster_bootstrap_ci(values, ids, n_resamples=300, seed=42)
        r2 = cluster_bootstrap_ci(values, ids, n_resamples=300, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]
