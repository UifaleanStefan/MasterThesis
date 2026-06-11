# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part01
# (entries 150-299), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part01.txt"

UPGRADES = {
 178: (0.25, "Post-termination: yes-direction correct (gold: 1-year post-term audit right) but the cited indemnity-survival clauses are another contract's."),
 192: (0.25, "Anti-assignment: restriction-exists direction is right but gold's mechanism is automatic termination on change of control, not consent; cited sections are another contract's."),
 224: (0.25, "Insurance: benefit-requirement direction matches gold's private-medical-scheme entitlement but the North $2M specifics are another contract's."),
 247: (0.25, "Insurance: yes-direction matches gold (MMI maintains coverage for Dragon Systems property) but the North specifics are another contract's."),
 260: (0.25, "Insurance: yes-direction matches gold (Supplier maintains AU$10M product liability) but the North specifics are another contract's."),
 273: (0.25, "Insurance: yes-direction matches gold (CBC maintains all-risk/product liability) but the North specifics are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 / Feb 21 2011 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's mutual-agreement mechanism and values are another contract's and contradict the gold. Zero."
    if "Notice Period" in q:
        if "doc38_qa6" in qid:
            return "Notice period: full gold (checked in queue.jsonl) requires 180 days non-renewal notice; pred's 120-day value is another contract's. Zero."
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is a representation/licensing arrangement, not a maintain-requirement supporting pred's North specifics. Zero."
    if "Ip Ownership" in q:
        return f"IP assignment: gold is '{g}...'; pred's Contractor-retains clause is another contract's and contradicts the gold assignment. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
    if "Post-Termination" in q:
        return f"Post-termination: gold span '{g}...' does not match pred's wrong-contract survival clauses. Zero."
    return f"Content mismatch: gold '{g}...' vs pred from another contract; no matching aspect. Zero."


entries = []
cur = None
for line in wf.read_text("utf-8").splitlines():
    m = re.match(r"\[(\d+)\] qid=(\S+)", line)
    if m:
        cur = {"idx": int(m.group(1)), "qid": m.group(2), "q": "", "gold": ""}
        entries.append(cur)
    elif cur is not None:
        if line.startswith("  EXPECT="):
            cur["q"] = line.split("| Q: ", 1)[-1]
        elif line.startswith("  GOLD: "):
            cur["gold"] = line[8:]

assert len(entries) == 150, len(entries)
out = {}
for e in entries:
    if e["idx"] in UPGRADES:
        score, rat = UPGRADES[e["idx"]]
    else:
        score, rat = 0.0, zero_rationale(e["qid"], e["q"], e["gold"])
    out[e["qid"]] = [score, rat]

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part01.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
