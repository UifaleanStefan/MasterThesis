"""Dump head-to-head judge queues into ~150-entry parts for the parallel Workflow
judging harness (format matches results/stage3/_judge_workdir/_wf_rejudge_*.js).

Usage: python scripts/_h2h_dump.py <cell1> <cell2> ...
Writes results/stage3/_judge_workdir/rejudge__<cell>__part<NN>.txt and prints,
per cell, the list of part ids so the Workflow can fan out over them.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WD = ROOT / "results" / "stage3" / "_judge_workdir"
QUEUE = ROOT / "results" / "stage3" / "judge_queue"
PART = 150


def flat(s: str, cap: int) -> str:
    import re
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:cap] + (" …[TRUNC]" if len(s) > cap else "")


def dump_cell(cell: str) -> list[str]:
    q = QUEUE / cell / "queue.jsonl"
    rows = [json.loads(l) for l in q.read_text(encoding="utf-8").splitlines() if l.strip()]
    parts = []
    for pi in range(0, len(rows), PART):
        nn = f"{pi // PART:02d}"
        chunk = rows[pi:pi + PART]
        lines = []
        for idx, e in enumerate(chunk):
            expect = e.get("expect", "-")
            lines.append(f"[{idx}] qid={e['qid']}")
            lines.append(f"    EXPECT={expect} | Q: {flat(e.get('question',''), 500)}")
            lines.append(f"    GOLD: {flat(e.get('gold_answer',''), 500)}")
            lines.append(f"    PRED: {flat(e.get('predicted',''), 700)}")
        (WD / f"rejudge__{cell}__part{nn}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        parts.append(nn)
    return parts


def main() -> int:
    manifest = {}
    for cell in sys.argv[1:]:
        parts = dump_cell(cell)
        manifest[cell] = parts
        print(f"{cell}: {len(parts)} part(s) {parts}")
    (WD / "_h2h_parts.json").write_text(json.dumps(manifest), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
