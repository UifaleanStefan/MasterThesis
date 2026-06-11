# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part02
# (entries 300-449), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part02.txt"

UPGRADES = {
 318: (0.25, "Insurance: requirement-direction matches gold (Franchisee participates in mandated workers-comp/insurance programs) but the North specifics are another contract's."),
 386: (0.25, "Insurance: yes-direction matches gold (Agency maintains $1M E&O) but the North specifics are another contract's."),
 393: (0.25, "Insurance: yes-direction matches gold (Supplier maintains A-rated CGL) but the North specifics are another contract's."),
 399: (0.25, "Insurance: yes-direction matches gold (Aimmune maintains clinical-trials/product liability) but the North specifics are another contract's."),
 403: (0.25, "Renewal: the one-year renewal duration matches gold's one-further-year extension, but the mutual-agreement mechanism and 120-day notice are another contract's."),
 407: (0.25, "Renewal: the one-year renewal duration matches gold's additional 12-month period, but the mutual-agreement mechanism is another contract's (gold renews automatically)."),
 433: (0.25, "Insurance: requirement-direction matches gold (each party increases coverage as prudent) but the North specifics are another contract's."),
 441: (0.25, "Insurance: yes-direction matches gold (Owner maintains $500M pollution coverage) but the North specifics are another contract's."),
 443: (0.25, "Insurance: yes-direction matches gold (A.M. Best A-8 carriers, Additional Insureds) but the North specifics are another contract's."),
 447: (0.25, "Insurance: yes-direction matches gold (both parties carry insurance incl. 1 year post-term) but the North specifics are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred's Florida claim is another contract's. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 / Feb 21 2011 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 / Dec 31 2012 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's renewal values/mechanism are another contract's and do not match. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's North-specific claims (or is not a maintain-requirement). Zero."
    if "Ip Ownership" in q or "Joint Ip" in q:
        return f"IP ownership: gold is '{g}...'; pred's Contractor-retains/North clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part02.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
