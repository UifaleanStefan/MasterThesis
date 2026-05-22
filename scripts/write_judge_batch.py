"""
Tiny helper to append a batch of Claude-judged scores to a results.jsonl.

Usage:
    python scripts/write_judge_batch.py \
        --run-id financebench__v4t-corpus-tuned__online__seed42 \
        --batch-json /path/to/batch.json

batch.json shape: [{"qid": "...", "judge_score": 1.0, "rationale": "..."}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_ROOT = ROOT / "results" / "stage3" / "judge_queue"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-json", required=True)
    args = parser.parse_args()

    qdir = QUEUE_ROOT / args.run_id
    if not qdir.exists():
        print(f"[FAIL] no queue dir: {qdir}", file=sys.stderr)
        return 1

    batch_data = json.loads(Path(args.batch_json).read_text())
    if not isinstance(batch_data, list):
        print("[FAIL] batch JSON must be a list", file=sys.stderr)
        return 1

    out_path = qdir / "results.jsonl"
    n_written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for r in batch_data:
            if "qid" not in r or "judge_score" not in r:
                continue
            fh.write(json.dumps(r) + "\n")
            n_written += 1
    print(f"[OK] appended {n_written} results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
