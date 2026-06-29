"""
θ-from-descriptor predictability test — leave-one-benchmark-out (plan Phase 4).

Consumes the (descriptor, tuned-θ) pairs from tuning/gen_theta_subsamples.py and
asks the honest question: can a θ predicted from corpus descriptors recover the
recall lift that per-task tuning gives — WITHOUT tuning on the target?

Method (k-NN, robust for modest data):
  - z-score descriptors on the training pairs.
  - Leave-one-benchmark-out: hold out all slices of benchmark B; for each held-out
    slice, predict θ as the mean θ of its k nearest training-slice descriptors.
  - EVALUATE recall under the predicted θ on the held-out slice (re-run recall@k,
    no LLM), and compare to the slice's stored canonical and tuned recall.
  - recovered_lift_fraction = (recall_pred - recall_canon) / (recall_tuned - recall_canon).
    ~1 → predicted θ ≈ tuned θ (cheap transfer works);
    ~0 → predicted θ no better than the canonical default (θ NOT predictable).

Honest by construction: a low fraction is a real result that reinforces the
thesis's task-dependence finding; a high fraction is a genuine transfer win.

Output: results/stage3/theta_predictability.json

Usage:
    python -m optimization.theta_predict --k 3 --pool 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import get_adapter
from evaluation.statistics import bootstrap_ci
from tuning.gen_theta_subsamples import make_slice_eval_fn
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC

PAIRS = ROOT / "results" / "stage3" / "theta_subsamples.jsonl"
OUT = ROOT / "results" / "stage3" / "theta_predictability.json"


def load_pairs():
    if not PAIRS.exists():
        return []
    return [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3, help="k-NN neighbors")
    ap.add_argument("--pool", type=int, default=150, help="doc pool (must match gen)")
    ap.add_argument("--retr-k", type=int, default=8, help="recall@k retrieval depth")
    args = ap.parse_args()

    pairs = load_pairs()
    benches = sorted({p["benchmark"] for p in pairs})
    print(f"Loaded {len(pairs)} (descriptor, theta) pairs across {len(benches)} benchmarks: {benches}")
    if len(pairs) < 8 or len(benches) < 2:
        print("Not enough data yet for a leave-one-benchmark-out test (need >=2 benchmarks, "
              ">=8 pairs). Re-run after gen_theta_subsamples produces more.")
        return 0

    X = np.array([p["descriptor_vec"] for p in pairs], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    thetas = np.array([p["tuned_theta_vec"] for p in pairs], dtype=float)
    pair_bench = [p["benchmark"] for p in pairs]

    # Resumable per-slice cache (host is restart-prone; each completed slice is
    # persisted immediately so a teardown only costs the in-flight slice).
    SLICES = ROOT / "results" / "stage3" / "_theta_predict_slices.jsonl"
    done = {}
    if SLICES.exists():
        for l in SLICES.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[(r["benchmark"], r["slice_idx"])] = r
    print(f"Resuming: {len(done)} slices already evaluated.")

    # cache adapter doc pools lazily (reconstruct held-out slices for recall eval)
    pools = {}

    # Process cheapest benchmarks first (by mean paragraphs/doc) so a host
    # teardown banks the most completed slices; the slow large-doc corpora
    # (e.g. CUAD contracts) go last.
    cost = {}
    for p in pairs:
        cost.setdefault(p["benchmark"], p["descriptor"].get("mean_paras_per_doc", 0.0))
    held_order = sorted(benches, key=lambda b: cost.get(b, 0.0))
    print(f"Eval order (cheap->expensive): {held_order}")

    per_slice = []
    for held in held_order:
        tr = [i for i, b in enumerate(pair_bench) if b != held]
        te = [i for i, b in enumerate(pair_bench) if b == held]
        if not tr or not te:
            continue
        for i in te:
            p = pairs[i]
            key = (held, p["slice_idx"])
            if key in done:
                per_slice.append(done[key])
                continue
            # k-NN predict theta from TRAINING descriptors
            d = np.linalg.norm(Xz[tr] - Xz[i], axis=1)
            nn = [tr[j] for j in np.argsort(d)[:args.k]]
            theta_pred = thetas[nn].mean(0)

            if held not in pools:
                pools[held] = list(get_adapter(held).iter_documents(limit=args.pool))
            slice_docs = [pools[held][j] for j in p["slice_doc_indices"]]
            eval_fn, n_eval = make_slice_eval_fn(slice_docs, args.retr_k, seed=42)
            recall_pred = eval_fn(theta_pred)
            canon = p["canonical_recall"]
            tuned = p["tuned_recall"]
            denom = tuned - canon
            frac = ((recall_pred - canon) / denom) if denom > 1e-9 else None
            rec = {
                "benchmark": held, "slice_idx": p["slice_idx"],
                "recall_canonical": round(canon, 4),
                "recall_tuned": round(tuned, 4),
                "recall_predicted": round(float(recall_pred), 4),
                "recovered_lift_fraction": (round(float(frac), 4) if frac is not None else None),
            }
            with SLICES.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            per_slice.append(rec)
            print(f"  {held} slice {p['slice_idx']}: canon={canon:.3f} pred={recall_pred:.3f} "
                  f"tuned={tuned:.3f} recovered={rec['recovered_lift_fraction']}")

    fracs = [s["recovered_lift_fraction"] for s in per_slice
             if s["recovered_lift_fraction"] is not None]
    # also: does predicted beat canonical on average (paired)?
    pred = np.array([s["recall_predicted"] for s in per_slice])
    canon = np.array([s["recall_canonical"] for s in per_slice])
    tuned = np.array([s["recall_tuned"] for s in per_slice])

    summary = {
        "k_nn": args.k, "n_slices_evaluated": len(per_slice),
        "n_with_defined_fraction": len(fracs),
        "mean_recall_canonical": round(float(canon.mean()), 4) if len(canon) else None,
        "mean_recall_predicted": round(float(pred.mean()), 4) if len(pred) else None,
        "mean_recall_tuned": round(float(tuned.mean()), 4) if len(tuned) else None,
        "mean_recovered_lift_fraction": round(float(np.mean(fracs)), 4) if fracs else None,
        "recovered_fraction_ci": (bootstrap_ci(fracs) if len(fracs) >= 5 else None),
        "interpretation": (
            "fraction ~1 => predicted theta recovers the tuning lift (cheap transfer works); "
            "fraction ~0 => predicted theta is no better than the canonical default "
            "(theta NOT predictable from descriptors -> reinforces task-dependence)."
        ),
        "per_slice": per_slice,
    }
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in (
        "n_slices_evaluated", "mean_recall_canonical", "mean_recall_predicted",
        "mean_recall_tuned", "mean_recovered_lift_fraction")}, indent=2))
    print(f"-> saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
