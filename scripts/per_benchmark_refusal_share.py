"""Per-benchmark rule-assisted-share breakdown (audit critique A2, refinement).

The pooled C2 validation (results/stage3/_judge_workdir/c2_validation_summary.json)
established that rule-assisted refusal/ack scores carry a population-weighted mean
score error of 0.028 (near-exact ACK/REF buckets; error concentrated in the small
MIXED borderline bucket). That was a SINGLE pooled number. This script breaks it
down PER BENCHMARK: how much of each benchmark's judged population leans on the
rule-assisted classifier, so the reader can see which benchmarks' scores depend
most on it (and therefore inherit the 0.028 error bound scaled by that share).

Uses the EXACT same _RULE_PAT as scripts/audit_judge_provenance.py, over every
committed results.jsonl. Pure analysis; no LLM, no judging. The per-benchmark
UPPER-BOUND on mean score error = rule_assisted_share * 0.028 (the pooled
per-entry error), which is conservative because most rule-assisted entries are in
the near-exact ACK/REF buckets.

Output: results/stage3/per_benchmark_refusal_share.json
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "results" / "stage3" / "judge_queue"
POOLED_ERR = 0.028  # population-weighted per-entry error from C2 validation

# EXACT copy of audit_judge_provenance.py::_RULE_PAT (keep in sync).
_RULE_PAT = re.compile(
    r"refuses to answer|refused|honest refusal|\[ack\]|acknowledg|ack refuse|"
    r"expected_behavior=acknowledge|ans refusal|i don't have|cannot find",
    re.I,
)

BENCHMARKS = ["financebench", "qasper", "cuad", "hotpotqa", "longmemeval", "narrativeqa"]


def benchmark_of(cell_name: str) -> str | None:
    low = cell_name.lower()
    for b in BENCHMARKS:
        if b in low:
            return b
    return None


def main() -> int:
    tally: dict[str, dict[str, int]] = {b: {"total": 0, "rule": 0} for b in BENCHMARKS}
    unknown = 0
    for rj in QUEUE.glob("*/results.jsonl"):
        bench = benchmark_of(rj.parent.name)
        if bench is None:
            unknown += 1
            continue
        for line in rj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            tally[bench]["total"] += 1
            if _RULE_PAT.search(j.get("rationale", "") or ""):
                tally[bench]["rule"] += 1

    rows = {}
    for b in BENCHMARKS:
        tot, rule = tally[b]["total"], tally[b]["rule"]
        share = (rule / tot) if tot else 0.0
        rows[b] = {"total": tot, "rule_assisted": rule,
                   "rule_assisted_share": round(share, 4),
                   "error_upper_bound": round(share * POOLED_ERR, 5)}
    out = {"pooled_per_entry_error": POOLED_ERR,
           "note": "error_upper_bound = rule_assisted_share * pooled_per_entry_error (conservative; ACK/REF buckets are near-exact).",
           "per_benchmark": rows,
           "unknown_cells_skipped": unknown}
    path = ROOT / "results" / "stage3" / "per_benchmark_refusal_share.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{'benchmark':<14}{'total':>8}{'rule':>8}{'share':>9}{'err_ub':>9}")
    for b, r in rows.items():
        print(f"{b:<14}{r['total']:>8}{r['rule_assisted']:>8}{r['rule_assisted_share']:>9.3f}{r['error_upper_bound']:>9.4f}")
    print(f"\n-> saved {path}  (unknown cells skipped: {unknown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
