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
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 1e-9

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
# Helper math (no scipy required)
# ------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _t_pvalue(t: float, df: int) -> float:
    """
    Approximate two-tailed p-value for t-distribution.
    Uses a simple approximation adequate for df > 5.
    """
    # Cornish-Fisher approximation
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    # Incomplete beta function approximation
    if t <= 0:
        return 1.0
    p = math.exp(
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
        - (df + 1) / 2 * math.log(1 + t * t / df)
    )
    # Two-tailed approximation
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
