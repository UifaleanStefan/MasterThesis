"""Tests for scripts/compare_fair_baselines.compare (audit B6 fairness analysis).

Focus on the verdict logic: direction must come from the MEAN paired difference
(judge scores are discrete, so the median diff is frequently 0 even when the
means clearly differ and Wilcoxon is significant)."""
from scripts.compare_fair_baselines import compare


def _q(scores):
    return {f"q{i}": s for i, s in enumerate(scores)}


def test_learned_clearly_wins():
    learned = _q([1.0] * 10)
    out = compare(learned, {"weak": _q([0.0] * 10)})
    t = out["paired_tests"][0]
    assert t["learned_mean_minus_baseline"] == 1.0
    assert t["significant_holm"] is True
    assert t["verdict"] == "learned wins"


def test_baseline_clearly_wins():
    learned = _q([0.0] * 10)
    out = compare(learned, {"strong": _q([1.0] * 10)})
    t = out["paired_tests"][0]
    assert t["learned_mean_minus_baseline"] == -1.0
    assert t["verdict"] == "baseline wins"


def test_tie_when_identical():
    learned = _q([0.0, 0.5, 1.0, 0.25, 0.75] * 2)
    out = compare(learned, {"same": dict(learned)})
    t = out["paired_tests"][0]
    assert t["learned_mean_minus_baseline"] == 0.0
    assert t["verdict"] == "tie (n.s.)"


def test_means_and_ci_reported():
    learned = _q([1.0, 1.0, 0.0, 1.0, 0.0])
    out = compare(learned, {"b": _q([0.0, 0.0, 0.0, 0.0, 0.0])})
    assert out["configs"]["v4t-corpus-tuned"]["mean"] == 0.6
    assert out["configs"]["v4t-corpus-tuned"]["ci"] is not None
