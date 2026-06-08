"""Phase 1.9 — LongMemEval v4t-tuned batch (10 entries)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "longmemeval__v4t-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/longmemeval__v4t-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.0, "PRED refuses ('don't have information about car')."),
    ("doc1_qa0", 0.0, "PRED talks about Effective Communication workshop; gold=Python webinar."),
    ("doc2_qa0", 1.0, "PRED 'bike first in February' matches gold."),
    ("doc3_qa0", 0.0, "PRED 'Dell XPS 13 first' but gold=Samsung — Y/N flip."),
    ("doc4_qa0", 1.0, "PRED implies 7 days via 'if meeting on Jan 17th' — matches gold."),
    ("doc5_qa0", 0.0, "PRED refuses ('need to find date of Ash Wednesday')."),
    ("doc6_qa0", 1.0, "PRED '14 days' matches gold exactly."),
    ("doc7_qa0", 0.0, "PRED 'marigold first' but gold=tomatoes — wrong order."),
    ("doc8_qa0", 0.0, "PRED 'March 18 → January 2 in 2022' wrong direction; gold=21 days."),
    ("doc9_qa0", 1.0, "PRED '4 days' matches gold."),
]


def main() -> None:
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: existing.add(json.loads(line)["qid"])
                except: pass
    added = 0; total = 0.0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing: continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"lme v4t-tuned batch added={added} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
