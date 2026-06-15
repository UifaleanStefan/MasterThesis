"""C2 step 1: characterize the kept rule-assisted population — entries that were
NOT re-judged in C1 (their qids are absent from every rejudge_scores file).
Reports rationale-template clusters with score distributions so we can stratify
a validation sample."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"

# 1. all re-judged qids (pass-2)
rejudged = set()
for f in WD.glob("rejudge_scores__*.json"):
    rejudged.update(json.loads(f.read_text("utf-8")).keys())
print(f"re-judged qids: {len(rejudged)}")

# 2. scan all results.jsonl for kept (non-rejudged) entries
def bucket(rat: str) -> str:
    r = (rat or "").lower()
    if "acknowledg" in r or "not yet been ingested" in r or "honest refusal" in r:
        return "ACK_missing"
    if r.startswith("refused") or "refused;" in r or "declines" in r or "i don't have" in r or "i do not have" in r or "cannot find" in r or "unable to" in r:
        return "REFUSAL"
    if "does not match" in r or "wrong contract" in r or "andy north" in r:
        return "CONTENT_template"  # should mostly be re-judged already
    return "OTHER"

kept = defaultdict(list)   # bucket -> [(qid, score, rationale)]
tmpl = Counter()           # rationale (first 60 chars) -> count, among kept
n_total = n_kept = 0
for rf in QROOT.glob("*/results.jsonl"):
    for L in rf.read_text("utf-8").splitlines():
        e = json.loads(L)
        n_total += 1
        qid = e.get("qid")
        if qid in rejudged:
            continue
        n_kept += 1
        rat = e.get("rationale", "")
        b = bucket(rat)
        kept[b].append((qid, e.get("judge_score"), rat))
        tmpl[(rat or "")[:55]] += 1

print(f"total results lines: {n_total}; kept (non-rejudged): {n_kept}")
print("\n=== kept buckets (score dist) ===")
for b, items in sorted(kept.items(), key=lambda kv: -len(kv[1])):
    sc = Counter(s for _, s, _ in items)
    print(f"  {b}: n={len(items)}  scores={dict(sorted(sc.items(), key=lambda x:str(x[0])))}")
print("\n=== top 12 kept rationale templates ===")
for t, c in tmpl.most_common(12):
    print(f"  {c:6d}  {t!r}")
