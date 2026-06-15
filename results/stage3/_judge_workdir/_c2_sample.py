"""C2 step 2: build a stratified 300-entry validation sample of the kept
rule-assisted refusal/acknowledgment scores. Deterministic (no RNG). Emits
work files for the judge workflow + a sidecar with the kept score + stratum
per qid so we can compute classifier precision/recall after Claude re-judges."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"

rejudged = set()
for f in WD.glob("rejudge_scores__*.json"):
    rejudged.update(json.loads(f.read_text("utf-8")).keys())

REFUSAL_PAT = re.compile(
    r"refuses to answer|refused|honest refusal|\[ack\]|acknowledg|ack refuse|"
    r"expected_behavior=acknowledge|ans refusal|i don't have|cannot find",
    re.I,
)


def stratum(score, rat):
    r = rat.lower()
    is_ack = ("acknowledg" in r or "[ack]" in r or "ack refuse" in r
              or "expected_behavior=acknowledge" in r or "honest refusal" in r)
    if is_ack and score == 1.0:
        return "ACK1"          # honest abstention credited 1.0 — check false positives
    if (not is_ack) and score == 0.0 and ("refus" in r or "ans refusal" in r):
        return "REF0"          # refusal scored 0.0 — check false negatives (was answerable?)
    return "MIXED"             # ack scored !=1.0, refusal scored !=0, etc. — borderline


# collect rule-assisted kept entries with their queue payload
pool = {"ACK1": [], "REF0": [], "MIXED": []}
qcache = {}
for rf in sorted(QROOT.glob("*/results.jsonl")):
    cell = rf.parent.name
    qfile = rf.parent / "queue.jsonl"
    if not qfile.exists():
        continue
    qmap = {json.loads(L)["qid"]: json.loads(L) for L in qfile.read_text("utf-8").splitlines()}
    for L in rf.read_text("utf-8").splitlines():
        e = json.loads(L)
        qid = e.get("qid")
        if qid in rejudged:
            continue
        rat = e.get("rationale", "") or ""
        if not REFUSAL_PAT.search(rat):
            continue
        s = e.get("judge_score")
        st = stratum(s, rat)
        q = qmap.get(qid)
        if not q:
            continue
        pool[st].append((qid, s, rat, q.get("question", ""), q.get("gold_answer", ""), q.get("predicted", ""), cell))

print("rule-assisted pool sizes:", {k: len(v) for k, v in pool.items()})


def even_sample(items, n):
    items = sorted(items, key=lambda x: x[0])  # deterministic by qid
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


sample = []
for st, n in [("ACK1", 100), ("REF0", 100), ("MIXED", 100)]:
    sel = even_sample(pool[st], n)
    sample.extend(sel)
    print(f"  sampled {len(sel)} from {st}")

# write work files (150/part) + sidecar
def oneline(s):
    return re.sub(r"\s+", " ", str(s)).strip()


sidecar = {}
parts = [sample[i:i + 150] for i in range(0, len(sample), 150)]
for pi, part in enumerate(parts):
    lines = []
    for i, (qid, s, rat, q, gold, pred, cell) in enumerate(part):
        lines.append(f"[{i}] qid={qid}")
        lines.append(f"  EXPECT=- | Q: {oneline(q)[:300]}")
        lines.append(f"  GOLD: {oneline(gold)[:300]}")
        lines.append(f"  PRED: {oneline(pred)[:300]}")
        sidecar[qid] = {"kept_score": s, "stratum": stratum(s, rat), "cell": cell}
    (WD / f"rejudge__c2_refusal_sample__part{pi:02d}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote part{pi:02d}: {len(part)} entries")

(WD / "c2_sidecar.json").write_text(json.dumps(sidecar, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"sidecar: {len(sidecar)} qids")
