"""B2 step: generate pass-2 work files for the four HQA/LME held-out batch cells
(100-doc re-run). Only emits a work file for cells whose queue has the new
100-question batch set (skips any still at the old n=10)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"
CELLS = [
    "hotpotqa__v4t-canonical__online__seed42",
    "hotpotqa__v4t-corpus-tuned__online__seed42",
    "longmemeval__v4t-canonical__online__seed42",
    "longmemeval__v4t-corpus-tuned__online__seed42",
]


def oneline(s):
    return re.sub(r"\s+", " ", str(s)).strip()


for cell in CELLS:
    q = QROOT / cell / "queue.jsonl"
    if not q.exists():
        print(f"{cell}: NO queue"); continue
    rows = [json.loads(L) for L in q.read_text("utf-8").splitlines()]
    if len(rows) < 50:
        print(f"{cell}: only {len(rows)} entries — skipping (run not complete)"); continue
    lines = []
    for i, e in enumerate(rows):
        lines.append(f"[{i}] qid={e['qid']}")
        lines.append(f"  EXPECT=- | Q: {oneline(e.get('question',''))[:300]}")
        lines.append(f"  GOLD: {oneline(e.get('gold_answer',''))[:300]}")
        lines.append(f"  PRED: {oneline(e.get('predicted',''))[:300]}")
    (WD / f"rejudge__{cell}__part00.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{cell}: wrote work file ({len(rows)} entries)")
