"""C2 fix: regenerate the ACK1 stratum work file WITH the acknowledge_missing
context (my first harness stripped it, making the judge treat honest
abstentions as failures). The judge must know the source was unavailable so an
honest 'I don't have this' is the correct 1.0 answer."""
import json
import re
from pathlib import Path

WD = Path(__file__).resolve().parent
QROOT = WD.parents[0] / "judge_queue"
side = json.loads((WD / "c2_sidecar.json").read_text("utf-8"))
ack = {q: m for q, m in side.items() if m["stratum"] == "ACK1"}


def oneline(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# group ACK qids by cell, load queue payloads
lines = []
i = 0
for qid, m in sorted(ack.items()):
    cell = m["cell"]
    qf = QROOT / cell / "queue.jsonl"
    rec = None
    for L in qf.read_text("utf-8").splitlines():
        e = json.loads(L)
        if e["qid"] == qid:
            rec = e
            break
    if not rec:
        continue
    lines.append(f"[{i}] qid={qid}")
    lines.append(f"  EXPECT=acknowledge_missing | Q: {oneline(rec.get('question',''))[:300]}")
    lines.append(f"  GOLD: {oneline(rec.get('gold_answer',''))[:300]}")
    lines.append(f"  PRED: {oneline(rec.get('predicted',''))[:300]}")
    i += 1

(WD / "rejudge__c2_ack_recheck__part00.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote rejudge__c2_ack_recheck__part00.txt: {i} ACK entries (EXPECT=acknowledge_missing)")
