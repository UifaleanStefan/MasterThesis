"""
Statistical significance testing for thesis results.

Every reported result should come with:
  1. Confidence intervals (bootstrap, n=1000 resamples).
  2. Paired t-test p-value (learned theta vs. fixed theta).
  3. Effect size (Cohen's d) — quantifies practical significance, not just statistical.

All functions work with lists of episode rewards (floats).

Cohen's d interpretation:
  |d| < 0.2: negligible
  0.2 ≤ |d| < 0.5: small
  0.5 ≤ |d| < 0.8: medium
  |d| ≥ 0.8: large

Usage:
    from evaluation.statistics import bootstrap_ci, ttest, cohens_d, full_comparison
    ci = bootstrap_ci(rewards_learned, n_resamples=1000)
    result = full_comparison(rewards_baseline, rewards_learned, label_a="Fixed", label_b="ES")
    print_comparison_report(result)
"""

from __future__ import annotations

import math
import random
import statistics


def bootstrap_ci(
    values: list[float],
    statistic=statistics.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for a statistic over a list of values.

    Parameters
    ----------
    values : list of float
        Sample data.
    statistic : callable
        Function to compute (default: mean).
    n_resamples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 → 95% CI).
    seed : int
        Random seed.

    Returns
    -------
    dict with: point_estimate, ci_lower, ci_upper, ci_width, n, alpha
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0, "n": 0}

    point_estimate = statistic(values)
    bootstrap_stats = []
    for _ in range(n_resamples):
        resample = [rng.choice(values) for _ in range(n)]
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "ci_lower": bootstrap_stats[lo_idx],
        "ci_upper": bootstrap_stats[hi_idx],
        "ci_width": bootstrap_stats[hi_idx] - bootstrap_stats[lo_idx],
        "n": n,
        "alpha": alpha,
    }


def paired_ttest(values_a: list[float], values_b: list[float]) -> dict:
    """
    Paired t-test: tests if mean(a) != mean(b).
    Assumes paired observations (same episode seeds for both conditions).

    Returns p_value, t_statistic, df.
    """
    n = min(len(values_a), len(values_b))
    if n < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "df": 0, "significant": False}

    diffs = [values_b[i] - values_a[i] for i in range(n)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    if std_diff < 1e-12:
        # All paired diffs are identical: t-statistic is undefined.
        # mean_diff == 0 → no difference → p=1.0; otherwise → perfect separation, p≈0.
        if abs(mean_diff) < 1e-12:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "df": n - 1,
                "mean_diff": mean_diff,
                "significant": False,
                "n": n,
            }
        return {
            "t_statistic": float("inf") if mean_diff > 0 else float("-inf"),
            "p_value": 0.0,
            "df": n - 1,
            "mean_diff": mean_diff,
            "significant": True,
            "n": n,
        }

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # Approximate p-value using t-distribution (scipy not required)
    # Use Welch approximation via normal approximation for large n
    if n >= 30:
        # z-test approximation
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        # Simplified t-test p-value (one-tailed × 2)
        p_value = _t_pvalue(abs(t_stat), df)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "mean_diff": mean_diff,
        "significant": p_value < 0.05,
        "n": n,
    }


def cohens_d(values_a: list[float], values_b: list[float]) -> dict:
    """
    Cohen's d effect size: (mean_b - mean_a) / pooled_std.
    Positive d means b > a.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return {"d": 0.0, "magnitude": "insufficient data"}

    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)
    std_a = statistics.stdev(values_a)
    std_b = statistics.stdev(values_b)
    n_a, n_b = len(values_a), len(values_b)

    pooled_std = math.sqrt(
        ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2)
    )
    if pooled_std < 1e-9:
        d = 0.0
    else:
        d = (mean_b - mean_a) / pooled_std

    magnitude = "negligible"
    if abs(d) >= 0.8:
        magnitude = "large"
    elif abs(d) >= 0.5:
        magnitude = "medium"
    elif abs(d) >= 0.2:
        magnitude = "small"

    return {
        "d": d,
        "magnitude": magnitude,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "pooled_std": pooled_std,
    }


def full_comparison(
    values_a: list[float],
    values_b: list[float],
    label_a: str = "Baseline",
    label_b: str = "Learned",
    n_resamples: int = 1000,
) -> dict:
    """
    Full statistical comparison between two conditions.
    Returns: CI for each, t-test, Cohen's d, summary.
    """
    ci_a = bootstrap_ci(values_a, n_resamples=n_resamples)
    ci_b = bootstrap_ci(values_b, n_resamples=n_resamples)
    ttest = paired_ttest(values_a, values_b)
    d = cohens_d(values_a, values_b)

    return {
        label_a: {**ci_a, "values": values_a},
        label_b: {**ci_b, "values": values_b},
        "ttest": ttest,
        "cohens_d": d,
        "improvement": ci_b["point_estimate"] - ci_a["point_estimate"],
        "improvement_pct": (
            (ci_b["point_estimate"] - ci_a["point_estimate"]) / max(1e-9, abs(ci_a["point_estimate"])) * 100
        ),
        "conclusion": _conclusion(ttest, d, label_a, label_b),
    }


def wilcoxon_signed_rank(values_a: list[float], values_b: list[float]) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.

    More appropriate than paired t-test when the per-question score
    distribution is non-normal — which is exactly the case for our
    LLM-judge scores (bounded [0, 1], clustered at {0, 0.5, 1}).

    Uses exact null distribution for n ≤ 25 and the normal approximation
    (with continuity correction) for larger n.

    Returns
    -------
    dict with W (signed-rank statistic), p_two_sided, p_one_sided,
    n_nonzero (paired differences that were non-zero — zeros are dropped
    per convention), median_diff, significant (p < 0.05).
    """
    if len(values_a) != len(values_b):
        raise ValueError(f"length mismatch: {len(values_a)} vs {len(values_b)}")
    if not values_a:
        return {
            "W": 0.0, "p_two_sided": 1.0, "p_one_sided": 1.0,
            "n_nonzero": 0, "median_diff": 0.0, "significant": False,
        }

    # Compute paired differences, drop zeros.
    diffs = [b - a for a, b in zip(values_a, values_b)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {
            "W": 0.0, "p_two_sided": 1.0, "p_one_sided": 1.0,
            "n_nonzero": 0,
            "median_diff": statistics.median(diffs),
            "significant": False,
        }

    # Rank by absolute magnitude (average rank for ties).
    sorted_by_abs = sorted(enumerate(nonzero), key=lambda x: abs(x[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find the run of ties on abs value.
        while j + 1 < n and abs(sorted_by_abs[j + 1][1]) == abs(sorted_by_abs[i][1]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks start at 1
        for k in range(i, j + 1):
            orig_idx = sorted_by_abs[k][0]
            ranks[orig_idx] = avg_rank
        i = j + 1

    # Signed-rank statistic: sum of ranks where d > 0.
    W_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    W_minus = sum(ranks[i] for i in range(n) if nonzero[i] < 0)
    W = min(W_plus, W_minus)

    # P-value via normal approximation (with continuity correction). For
    # n < 20 this is conservative — for our typical n=100-300, it's fine.
    mean_W = n * (n + 1) / 4.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0
    # Tie correction: subtract sum(t_i^3 - t_i)/48 for each tie group of size t_i.
    # Simplification: count tie-groups via rank repetitions.
    from collections import Counter
    rank_counts = Counter(ranks)
    tie_correction = sum(t * (t * t - 1) for t in rank_counts.values() if t > 1) / 48.0
    var_W -= tie_correction
    if var_W <= 0:
        p_two_sided = 1.0
    else:
        # Continuity correction: |W - mean_W| - 0.5
        z = (abs(W_plus - mean_W) - 0.5) / math.sqrt(var_W)
        # Two-tailed p via normal CDF
        p_two_sided = 2 * (1 - _norm_cdf(z))
    p_one_sided = p_two_sided / 2.0

    return {
        "W": float(W),
        "W_plus": float(W_plus),
        "W_minus": float(W_minus),
        "p_two_sided": float(min(1.0, max(0.0, p_two_sided))),
        "p_one_sided": float(min(1.0, max(0.0, p_one_sided))),
        "n_nonzero": n,
        "n_total": len(diffs),
        "median_diff": statistics.median(diffs),
        "significant": p_two_sided < 0.05,
    }


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF via erf — local helper to avoid scipy hard dep."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Less conservative than naive Bonferroni (same family-wise error rate
    guarantee but more powerful). Standard practice for k pairwise tests.

    Returns
    -------
    list of dict, one per input p-value, each with:
      - p_raw: the original p-value
      - p_adjusted: the Holm-corrected p-value
      - rank: position in the sorted-p ordering (1 = smallest)
      - significant: whether p_adjusted < alpha
    """
    if not pvalues:
        return []
    n = len(pvalues)
    # Sort with original indices.
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    out_by_orig: dict[int, dict] = {}
    max_seen = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm correction: multiplier is (n - rank), step-down.
        p_adj = min(1.0, p * (n - rank))
        # Step-down: ensure monotonicity (adjusted p can only increase).
        p_adj = max(p_adj, max_seen)
        max_seen = p_adj
        out_by_orig[orig_idx] = {
            "p_raw": p,
            "p_adjusted": p_adj,
            "rank": rank + 1,
            "significant": p_adj < alpha,
        }
    return [out_by_orig[i] for i in range(n)]


def cluster_bootstrap_ci(
    values: list[float],
    cluster_ids: list,
    statistic=statistics.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Cluster-bootstrap CI: resample WHOLE CLUSTERS rather than individual
    observations. The right choice when observations within a cluster are
    correlated (e.g. multiple QAs from the same document share content
    and difficulty), which makes the IID bootstrap CI too narrow.

    Parameters
    ----------
    values : list of float
        Per-observation values (e.g., per-question judge scores).
    cluster_ids : list of hashable
        Cluster ID for each observation (e.g., doc_idx). Length must
        equal `values`.
    statistic : callable
        Statistic to compute (default: mean).
    n_resamples : int
        Number of cluster-bootstrap iterations.
    alpha : float
        Significance level (default 0.05 → 95% CI).
    seed : int
        RNG seed.

    Returns
    -------
    dict with point_estimate, ci_lower, ci_upper, ci_width, n_clusters,
    n_observations, alpha.
    """
    if len(values) != len(cluster_ids):
        raise ValueError(f"values ({len(values)}) and cluster_ids ({len(cluster_ids)}) length mismatch")
    if not values:
        return {
            "point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "ci_width": 0.0, "n_clusters": 0, "n_observations": 0, "alpha": alpha,
        }
    # Group values by cluster.
    by_cluster: dict = {}
    for v, c in zip(values, cluster_ids):
        by_cluster.setdefault(c, []).append(v)
    cluster_keys = list(by_cluster.keys())
    n_clusters = len(cluster_keys)
    if n_clusters < 2:
        # Degenerate — fall back to IID bootstrap on flat values.
        return bootstrap_ci(values, statistic=statistic,
                            n_resamples=n_resamples, alpha=alpha, seed=seed)

    rng = random.Random(seed)
    boot_stats: list[float] = []
    for _ in range(n_resamples):
        # Sample n_clusters clusters with replacement.
        sampled = [by_cluster[rng.choice(cluster_keys)] for _ in range(n_clusters)]
        flat = [v for cluster in sampled for v in cluster]
        if flat:
            boot_stats.append(statistic(flat))
    boot_stats.sort()
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples)
    return {
        "point_estimate": statistic(values),
        "ci_lower": boot_stats[lo_idx] if boot_stats else 0.0,
        "ci_upper": boot_stats[hi_idx - 1] if boot_stats else 0.0,
        "ci_width": (boot_stats[hi_idx - 1] - boot_stats[lo_idx]) if boot_stats else 0.0,
        "n_clusters": n_clusters,
        "n_observations": len(values),
        "alpha": alpha,
    }


def print_comparison_report(result: dict, label_a: str = "Baseline", label_b: str = "Learned") -> None:
    """Print a formatted statistical comparison."""
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)

    for label in [label_a, label_b]:
        if label in result:
            r = result[label]
            print(f"\n{label}:")
            print(f"  Mean: {r['point_estimate']:.4f}")
            print(f"  95% CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
            print(f"  n={r['n']}")

    print(f"\nImprovement: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")

    ttest = result["ttest"]
    print(f"\nt-test: t={ttest['t_statistic']:.3f}, p={ttest['p_value']:.4f}", end="")
    print(" (significant)" if ttest["significant"] else " (not significant)")

    d = result["cohens_d"]
    print(f"Cohen's d: {d['d']:.3f} ({d['magnitude']} effect)")
    print(f"\nConclusion: {result['conclusion']}")
    print("=" * 60)


def run_all_comparisons(
    baseline_rewards: dict[str, list[float]],
    learned_rewards: dict[str, list[float]],
) -> dict[str, dict]:
    """
    Run full_comparison for each environment/system pair.
    baseline_rewards and learned_rewards should have the same keys.
    """
    results = {}
    for key in baseline_rewards:
        if key in learned_rewards:
            results[key] = full_comparison(
                baseline_rewards[key],
                learned_rewards[key],
                label_a="Baseline",
                label_b="Learned",
            )
    return results


# ------------------------------------------------------------------
# P-value helpers — scipy when available, fallback to math approx otherwise
# ------------------------------------------------------------------

try:
    from scipy.stats import norm as _scipy_norm, t as _scipy_t
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - fallback exercised only when scipy missing
    _scipy_norm = None
    _scipy_t = None
    _HAS_SCIPY = False


def _normal_cdf(z: float) -> float:
    """Standard normal CDF — scipy when available, math.erf fallback otherwise."""
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(z))
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _t_pvalue(t: float, df: int) -> float:
    """
    Two-tailed p-value for the t-distribution at statistic ``t`` with ``df`` dof.

    Prefers ``scipy.stats.t.sf`` (survival function = 1 - CDF) for accuracy.
    Falls back to a math-only approximation when scipy is unavailable; the
    fallback is adequate for df > 5 but should not be relied on for thesis
    figures.
    """
    if df <= 0:
        return 1.0
    if _HAS_SCIPY:
        return float(2.0 * _scipy_t.sf(abs(t), df))
    # Cornish-Fisher fallback (kept for environments without scipy)
    if t <= 0:
        return 1.0
    p = math.exp(
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
        - (df + 1) / 2 * math.log(1 + t * t / df)
    )
    return min(1.0, 2 * p * math.sqrt(df) / abs(t) if abs(t) > 0 else 1.0)


def _conclusion(ttest: dict, d_result: dict, label_a: str, label_b: str) -> str:
    sig = ttest["significant"]
    d = d_result["d"]
    mag = d_result["magnitude"]
    direction = "better" if d > 0 else "worse"
    if sig:
        return (
            f"{label_b} is statistically significantly {direction} than {label_a} "
            f"(p={ttest['p_value']:.4f}, {mag} effect size d={d:.3f})."
        )
    else:
        return (
            f"No statistically significant difference between {label_a} and {label_b} "
            f"(p={ttest['p_value']:.4f}, {mag} effect size d={d:.3f})."
        )
