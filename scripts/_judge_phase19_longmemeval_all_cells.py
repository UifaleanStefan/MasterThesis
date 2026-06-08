"""Phase 1.9 Protocol A — LongMemEval 6 remaining cells (60 entries total)."""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# (results_path, [(qid, score, rationale), ...])
CELLS = [
    (
        "results/stage3/judge_queue/longmemeval__attention-corpus-tuned__batch__seed42/results.jsonl",
        [
            ("longmemeval__attention-corpus-tuned__batch__doc0_qa0__seed42", 1.0,
             "PRED 'GPS system, took back to dealership March 22nd' matches gold=GPS system not functioning correctly."),
            ("longmemeval__attention-corpus-tuned__batch__doc1_qa0__seed42", 0.0,
             "PRED 'Effective Time Management first' wrong; gold='Data Analysis using Python' webinar was first."),
            ("longmemeval__attention-corpus-tuned__batch__doc2_qa0__seed42", 0.0,
             "PRED 'car first in February' wrong; gold=bike was taken care of first."),
            ("longmemeval__attention-corpus-tuned__batch__doc3_qa0__seed42", 1.0,
             "PRED 'Samsung Galaxy S22 first on February 20th' matches gold."),
            ("longmemeval__attention-corpus-tuned__batch__doc4_qa0__seed42", 1.0,
             "PRED '7 days before team meeting' matches gold=7 days."),
            ("longmemeval__attention-corpus-tuned__batch__doc5_qa0__seed42", 0.0,
             "PRED '32 days' wrong; gold=30 days (31 acceptable)."),
            ("longmemeval__attention-corpus-tuned__batch__doc6_qa0__seed42", 0.0,
             "PRED '19 days' wrong; gold=14 days (15 acceptable)."),
            ("longmemeval__attention-corpus-tuned__batch__doc7_qa0__seed42", 0.0,
             "PRED 'marigold seeds started first' wrong; gold=Tomatoes."),
            ("longmemeval__attention-corpus-tuned__batch__doc8_qa0__seed42", 1.0,
             "PRED '21 days' matches gold=21 days."),
            ("longmemeval__attention-corpus-tuned__batch__doc9_qa0__seed42", 0.0,
             "PRED '14 days before Rack Fest' wrong; gold=4 days."),
        ],
    ),
    (
        "results/stage3/judge_queue/longmemeval__bm25-corpus__batch__seed42/results.jsonl",
        [
            ("longmemeval__bm25-corpus__batch__doc0_qa0__seed42", 1.0,
             "PRED 'GPS system, took back to dealership' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc1_qa0__seed42", 0.0,
             "PRED 'Effective Time Management first' wrong; gold=Data Analysis Python webinar."),
            ("longmemeval__bm25-corpus__batch__doc2_qa0__seed42", 0.0,
             "PRED 'car first in February' wrong; gold=bike."),
            ("longmemeval__bm25-corpus__batch__doc3_qa0__seed42", 1.0,
             "PRED 'Samsung Galaxy S22 first, February 20th' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc4_qa0__seed42", 1.0,
             "PRED '7 days' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc5_qa0__seed42", 1.0,
             "PRED '30 days from January 2nd to February 1st' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc6_qa0__seed42", 1.0,
             "PRED '14 days, started February 15th found March 1st' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc7_qa0__seed42", 0.0,
             "PRED 'marigold seeds first' wrong; gold=Tomatoes."),
            ("longmemeval__bm25-corpus__batch__doc8_qa0__seed42", 1.0,
             "PRED '21 days' matches gold."),
            ("longmemeval__bm25-corpus__batch__doc9_qa0__seed42", 1.0,
             "PRED '4 days (June 14 to June 18)' matches gold=4 days."),
        ],
    ),
    (
        "results/stage3/judge_queue/longmemeval__dump-all__batch__seed42/results.jsonl",
        [
            ("longmemeval__dump-all__batch__doc0_qa0__seed42", 0.0,
             "PRED refuses; gold=GPS system not functioning correctly."),
            ("longmemeval__dump-all__batch__doc1_qa0__seed42", 0.0,
             "PRED refuses/wrong; gold=Data Analysis using Python webinar."),
            ("longmemeval__dump-all__batch__doc2_qa0__seed42", 0.0,
             "PRED refuses; gold=bike."),
            ("longmemeval__dump-all__batch__doc3_qa0__seed42", 0.0,
             "PRED refuses; gold=Samsung Galaxy S22."),
            ("longmemeval__dump-all__batch__doc4_qa0__seed42", 1.0,
             "PRED '7 days before team meeting' matches gold."),
            ("longmemeval__dump-all__batch__doc5_qa0__seed42", 0.0,
             "PRED '33 days' wrong; gold=30 days (31 acceptable)."),
            ("longmemeval__dump-all__batch__doc6_qa0__seed42", 1.0,
             "PRED '14 days (February 15 to March 1)' matches gold."),
            ("longmemeval__dump-all__batch__doc7_qa0__seed42", 0.0,
             "PRED 'marigold seeds first' wrong; gold=Tomatoes."),
            ("longmemeval__dump-all__batch__doc8_qa0__seed42", 0.75,
             "PRED starts correct calculation (Feb 26→Mar 19, 21 days) but output truncated; reasoning is correct."),
            ("longmemeval__dump-all__batch__doc9_qa0__seed42", 1.0,
             "PRED '4 days (June 14 to June 18)' matches gold."),
        ],
    ),
    (
        "results/stage3/judge_queue/longmemeval__v4t-canonical__online__seed42/results.jsonl",
        [
            ("longmemeval__v4t-canonical__online__doc0_qa0__seed42", 0.0,
             "PRED 'context does not mention any issues' refuses; gold=GPS system not functioning correctly."),
            ("longmemeval__v4t-canonical__online__doc1_qa0__seed42", 1.0,
             "PRED 'Data Analysis using Python webinar first' matches gold."),
            ("longmemeval__v4t-canonical__online__doc2_qa0__seed42", 0.0,
             "PRED 'car first' wrong; gold=bike."),
            ("longmemeval__v4t-canonical__online__doc3_qa0__seed42", 1.0,
             "PRED 'Samsung Galaxy S22 first, February 20th' matches gold."),
            ("longmemeval__v4t-canonical__online__doc4_qa0__seed42", 1.0,
             "PRED '7 days' matches gold."),
            ("longmemeval__v4t-canonical__online__doc5_qa0__seed42", 0.0,
             "PRED '38 days' wrong; gold=30 days."),
            ("longmemeval__v4t-canonical__online__doc6_qa0__seed42", 0.0,
             "PRED '19 days' wrong; gold=14 days."),
            ("longmemeval__v4t-canonical__online__doc7_qa0__seed42", 1.0,
             "PRED 'tomatoes were started first' matches gold=Tomatoes."),
            ("longmemeval__v4t-canonical__online__doc8_qa0__seed42", 1.0,
             "PRED '21 days' matches gold."),
            ("longmemeval__v4t-canonical__online__doc9_qa0__seed42", 1.0,
             "PRED '4 days before Rack Fest' matches gold."),
        ],
    ),
    (
        "results/stage3/judge_queue/longmemeval__v4t-corpus-tuned__online__seed42/results.jsonl",
        [
            ("longmemeval__v4t-corpus-tuned__online__doc0_qa0__seed42", 0.0,
             "PRED 'context does not mention any issues' refuses; gold=GPS system not functioning correctly."),
            ("longmemeval__v4t-corpus-tuned__online__doc1_qa0__seed42", 1.0,
             "PRED 'Data Analysis using Python webinar first' matches gold."),
            ("longmemeval__v4t-corpus-tuned__online__doc2_qa0__seed42", 0.0,
             "PRED 'car first' wrong; gold=bike."),
            ("longmemeval__v4t-corpus-tuned__online__doc3_qa0__seed42", 1.0,
             "PRED 'Samsung Galaxy S22 first, February 20th' matches gold."),
            ("longmemeval__v4t-corpus-tuned__online__doc4_qa0__seed42", 1.0,
             "PRED '7 days' matches gold."),
            ("longmemeval__v4t-corpus-tuned__online__doc5_qa0__seed42", 0.0,
             "PRED '33 days' wrong; gold=30 days."),
            ("longmemeval__v4t-corpus-tuned__online__doc6_qa0__seed42", 0.0,
             "PRED '19 days' wrong; gold=14 days."),
            ("longmemeval__v4t-corpus-tuned__online__doc7_qa0__seed42", 1.0,
             "PRED 'tomatoes were started first' matches gold=Tomatoes."),
            ("longmemeval__v4t-corpus-tuned__online__doc8_qa0__seed42", 1.0,
             "PRED '21 days' matches gold."),
            ("longmemeval__v4t-corpus-tuned__online__doc9_qa0__seed42", 1.0,
             "PRED '4 days before Rack Fest' matches gold."),
        ],
    ),
    (
        "results/stage3/judge_queue/longmemeval__v4t-tuned__online__seed42/results.jsonl",
        [
            ("longmemeval__v4t-tuned__online__doc0_qa0__seed42", 0.0,
             "PRED 'context does not mention any issues' refuses; gold=GPS system not functioning correctly."),
            ("longmemeval__v4t-tuned__online__doc1_qa0__seed42", 1.0,
             "PRED 'Data Analysis using Python webinar first' matches gold."),
            ("longmemeval__v4t-tuned__online__doc2_qa0__seed42", 0.0,
             "PRED 'car first' wrong; gold=bike."),
            ("longmemeval__v4t-tuned__online__doc3_qa0__seed42", 1.0,
             "PRED 'Samsung Galaxy S22 first (pre-ordered Jan 28th)' matches gold."),
            ("longmemeval__v4t-tuned__online__doc4_qa0__seed42", 1.0,
             "PRED '7 days' matches gold."),
            ("longmemeval__v4t-tuned__online__doc5_qa0__seed42", 0.0,
             "PRED '38 days' wrong; gold=30 days."),
            ("longmemeval__v4t-tuned__online__doc6_qa0__seed42", 0.0,
             "PRED '19 days' wrong; gold=14 days."),
            ("longmemeval__v4t-tuned__online__doc7_qa0__seed42", 0.0,
             "PRED 'marigold seeds started first' wrong; gold=Tomatoes."),
            ("longmemeval__v4t-tuned__online__doc8_qa0__seed42", 1.0,
             "PRED '21 days' matches gold."),
            ("longmemeval__v4t-tuned__online__doc9_qa0__seed42", 1.0,
             "PRED '4 days before Rack Fest' matches gold."),
        ],
    ),
]


def main() -> None:
    for results_path_str, judgments in CELLS:
        results_path = Path(results_path_str)
        existing: set[str] = set()
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try: existing.add(json.loads(line)["qid"])
                    except: pass
        added = 0; total = 0.0; skipped = 0
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as f:
            for qid, score, rationale in judgments:
                if qid in existing:
                    skipped += 1; continue
                f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                    "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
                added += 1; total += score
        cell = results_path.parent.name
        print(f"{cell}: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
