"""Inter-rater reliability for the LLM-as-judge: quadratic-weighted Cohen's kappa
between the primary judge (pass-1, committed scores) and an independent blind
second pass (pass-2).

Reads results/stage3/_judge_workdir/kappa/kappa_key.json (qid -> pass-1 score)
and scores__kappa__chunk*.json (qid -> pass-2 score). Writes
results/stage3/kappa_results.json and prints a summary.
"""
import glob, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KDIR = ROOT / "results" / "stage3" / "_judge_workdir" / "kappa"
CATS = [0.0, 0.25, 0.5, 0.75, 1.0]
IDX = {c: i for i, c in enumerate(CATS)}


def quadratic_weighted_kappa(pairs):
    """pairs: list of (score1, score2). Ordinal 5-point quadratic-weighted kappa."""
    k = len(CATS)
    n = len(pairs)
    O = [[0] * k for _ in range(k)]
    r1 = Counter(); r2 = Counter()
    for a, b in pairs:
        ia, ib = IDX[a], IDX[b]
        O[ia][ib] += 1
        r1[ia] += 1; r2[ib] += 1
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    E = [[r1[i] * r2[j] / n for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    kappa = 1.0 - num / den if den else 1.0
    return kappa, O


def main():
    key = {k: float(v) for k, v in json.loads((KDIR / "kappa_key.json").read_text()).items()}
    pass2 = {}
    for f in glob.glob(str(KDIR / "scores__kappa__chunk*.json")):
        for r in json.loads(Path(f).read_text("utf-8")):
            pass2[r["qid"]] = float(r["score"])
    pairs = [(key[q], pass2[q]) for q in key if q in pass2]
    n = len(pairs)
    exact = sum(1 for a, b in pairs if a == b) / n
    adjacent = sum(1 for a, b in pairs if abs(IDX[a] - IDX[b]) <= 1) / n
    mad = sum(abs(a - b) for a, b in pairs) / n
    kappa, conf = quadratic_weighted_kappa(pairs)

    def interp(kp):
        return ("almost perfect" if kp >= .81 else "substantial" if kp >= .61
                else "moderate" if kp >= .41 else "fair" if kp >= .21 else "slight")

    out = {
        "n": n, "missing": len(key) - n,
        "quadratic_weighted_cohen_kappa": round(kappa, 3),
        "interpretation": interp(kappa),
        "exact_agreement": round(exact, 3),
        "adjacent_agreement_within_1_level": round(adjacent, 3),
        "mean_absolute_score_difference": round(mad, 3),
        "confusion_matrix_pass1_rows_pass2_cols": conf,
        "categories": CATS,
        "note": "Pass-1 = primary Claude judge (committed). Pass-2 = independent "
                "blind Claude second pass (same five-point rubric, no access to "
                "pass-1 scores). Measures judge reliability / test-retest stability.",
    }
    (ROOT / "results" / "stage3" / "kappa_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "confusion_matrix_pass1_rows_pass2_cols"}, indent=2))


if __name__ == "__main__":
    main()
