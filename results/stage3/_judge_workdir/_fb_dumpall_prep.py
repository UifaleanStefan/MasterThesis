"""Generate pass-2 work files for the fresh FB dump-all Protocol-A cells
(online + batch) which have queue.jsonl but no results.jsonl. Produces the
standard rejudge__<cell>__part00.txt work-file format the QA workflow reads."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"
CELLS = [
    "financebench__dump-all__online__seed42",
    "financebench__dump-all__batch__seed42",
]


def oneline(s):
    return re.sub(r"\s+", " ", str(s)).strip()


for cell in CELLS:
    q = (QROOT / cell / "queue.jsonl").read_text("utf-8").splitlines()
    lines = []
    for i, L in enumerate(q):
        e = json.loads(L)
        qid = e["qid"]
        question = oneline(e.get("question", ""))[:300]
        gold = oneline(e.get("gold_answer", ""))[:300]
        pred = oneline(e.get("predicted", ""))[:300]
        lines.append(f"[{i}] qid={qid}")
        lines.append(f"  EXPECT=- | Q: {question}")
        lines.append(f"  GOLD: {gold}")
        lines.append(f"  PRED: {pred}")
    out = WD / f"rejudge__{cell}__part00.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.name}: {len(q)} entries")
