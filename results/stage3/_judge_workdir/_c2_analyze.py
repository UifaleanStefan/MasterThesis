"""C2 step 3: compare Claude's independent re-judgments of the 300-entry sample
against the kept rule-assisted scores, and report classifier agreement /
error bounds per stratum."""
import json
from pathlib import Path
from collections import defaultdict

WD = Path(__file__).resolve().parent
side = json.loads((WD / "c2_sidecar.json").read_text("utf-8"))
claude = {}
for f in sorted(WD.glob("rejudge_scores__c2_refusal_sample__part*.json")):
    for qid, (s, _) in json.loads(f.read_text("utf-8")).items():
        claude[qid] = s
# ACK1 recheck (judged WITH acknowledge_missing context) overrides the flawed
# first-pass ACK scores for those qids.
ack_fix = WD / "rejudge_scores__c2_ack_recheck__part00.json"
if ack_fix.exists():
    for qid, (s, _) in json.loads(ack_fix.read_text("utf-8")).items():
        claude[qid] = s
    print(f"(applied {ack_fix.name} ACK recheck overrides)\n")

by = defaultdict(lambda: {"n": 0, "exact": 0, "within25": 0, "abs_sum": 0.0,
                          "false_pos": 0, "false_neg": 0})
rows = []
for qid, meta in side.items():
    if qid not in claude:
        continue
    kept = meta["kept_score"]
    c = claude[qid]
    st = meta["stratum"]
    d = by[st]
    d["n"] += 1
    d["exact"] += int(abs(c - kept) < 1e-9)
    d["within25"] += int(abs(c - kept) <= 0.25 + 1e-9)
    d["abs_sum"] += abs(c - kept)
    # false positive: classifier credited ack 1.0 but Claude says it's not a good answer
    if st == "ACK1" and c < 0.75:
        d["false_pos"] += 1
        rows.append(("FALSE_POS", qid, kept, c, meta["cell"]))
    # false negative: classifier zeroed a refusal but Claude finds a usable answer
    if st == "REF0" and c > 0.25:
        d["false_neg"] += 1
        rows.append(("FALSE_NEG", qid, kept, c, meta["cell"]))

print("=== C2 classifier validation (Claude re-judge vs kept rule-assisted) ===")
tot_n = tot_abs = 0
for st in ("ACK1", "REF0", "MIXED"):
    d = by[st]
    if not d["n"]:
        continue
    tot_n += d["n"]; tot_abs += d["abs_sum"]
    print(f"\n[{st}] n={d['n']}")
    print(f"  exact agreement:   {d['exact']}/{d['n']} = {d['exact']/d['n']:.1%}")
    print(f"  within 0.25:       {d['within25']}/{d['n']} = {d['within25']/d['n']:.1%}")
    print(f"  mean |claude-kept|: {d['abs_sum']/d['n']:.4f}")
    if st == "ACK1":
        print(f"  false-positive ACK (Claude<0.75): {d['false_pos']}/{d['n']} = {d['false_pos']/d['n']:.1%}")
    if st == "REF0":
        print(f"  false-negative REF (Claude>0.25): {d['false_neg']}/{d['n']} = {d['false_neg']/d['n']:.1%}")
print(f"\nOVERALL mean |claude-kept| over {tot_n} sampled rule-assisted entries: {tot_abs/tot_n:.4f}")
print("\n=== disagreements (first 15) ===")
for tag, qid, kept, c, cell in rows[:15]:
    print(f"  {tag} kept={kept} claude={c}  ...{qid[-40:]}")
print(f"... {len(rows)} total disagreements flagged")
