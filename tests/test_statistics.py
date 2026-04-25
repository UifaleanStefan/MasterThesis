"""
Statistical helper invariants.

These tests pin down the behavior of bootstrap_ci, paired_ttest, and cohens_d
before Phase 3's switch from hand-rolled p-value approximations to scipy.stats.
"""

from __future__ import annotations

import statistics

import pytest

from evaluation.statistics import (
    bootstrap_ci,
    cohens_d,
    full_comparison,
    paired_ttest,
)


class TestBootstrapCI:
    def test_constant_values_yield_zero_width(self):
        result = bootstrap_ci([0.5] * 50)
        assert result["point_estimate"] == pytest.approx(0.5)
        assert result["ci_lower"] == pytest.approx(0.5)
        assert result["ci_upper"] == pytest.approx(0.5)
        assert result["ci_width"] == pytest.approx(0.0)

    def test_ci_brackets_point_estimate(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6] * 5
        result = bootstrap_ci(values, n_resamples=500)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_empty_input_yields_zero(self):
        result = bootstrap_ci([])
        assert result["point_estimate"] == 0.0
        assert result["n"] == 0


class TestPairedTTest:
    def test_identical_values_not_significant(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = paired_ttest(values, values)
        assert result["mean_diff"] == pytest.approx(0.0)
        assert not result["significant"]

    def test_clearly_different_means_are_significant(self):
        baseline = [0.1] * 30
        improved = [0.5] * 30
        result = paired_ttest(baseline, improved)
        assert result["significant"]
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.05

    def test_sign_of_mean_diff_matches_direction(self):
        a = [0.1, 0.2, 0.3]
        b = [0.5, 0.6, 0.7]
        result = paired_ttest(a, b)  # b - a > 0
        assert result["mean_diff"] > 0


class TestCohensD:
    def test_no_difference_yields_negligible(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = cohens_d(values, values)
        assert abs(result["d"]) < 0.01
        assert result["magnitude"] == "negligible"

    def test_large_separation_yields_large_effect(self):
        a = [0.0, 0.05, 0.1, 0.05, 0.0]
        b = [1.0, 0.95, 1.0, 0.9, 1.05]
        result = cohens_d(a, b)
        assert result["magnitude"] == "large"
        assert result["d"] > 0.8


class TestFullComparison:
    def test_returns_expected_keys(self):
        a = [0.1, 0.2, 0.3, 0.2, 0.1] * 5
        b = [0.3, 0.4, 0.5, 0.4, 0.3] * 5
        result = full_comparison(a, b, label_a="Baseline", label_b="Learned")
        assert "Baseline" in result
        assert "Learned" in result
        assert "ttest" in result
        assert "cohens_d" in result
        assert "improvement" in result
        assert "conclusion" in result
