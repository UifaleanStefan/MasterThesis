# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part04
# (entries 600-749), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part04.txt"

UPGRADES = {
 611: (0.25, "Anti-assignment: restriction-exists direction per gold's annotated 1940-Act investment-adviser span but pred's consent sections are another contract's."),
 614: (0.25, "Insurance: yes-direction matches gold (Diplomat obtains product liability for KI) but the North specifics are another contract's."),
 621: (0.25, "Termination for convenience: direction matches gold's Pretzel Time 30-day suspension/termination right but the 15-day either-party clause is another contract's."),
 622: (0.25, "Insurance: yes-direction matches gold (Operations Manual coverages + 30-day insurer notice) but the North specifics are another contract's."),
 627: (0.25, "Renewal: the one-year renewal duration matches gold's automatic one-year periods, but the mutual-agreement mechanism is another contract's."),
 630: (0.25, "Renewal: the one-year renewal duration matches gold's successive one-year terms, but the XFN/mutual-agreement mechanism is another contract's."),
 641: (0.25, "Change of control: CoC-termination direction matches gold's Vendor right but pred's Section 16.9 assignment clause is another contract's."),
 646: (0.25, "Insurance: yes-direction per gold's Bachem coverage-statement obligation but the North specifics are another contract's."),
 651: (0.25, "Insurance: yes-direction matches gold ($5M ABG-acceptable coverage) but the North specifics are another contract's."),
 668: (0.25, "Change of control: CoC-termination direction matches gold's BSP notice right but pred's Section 16.9 assignment clause is another contract's."),
 669: (0.25, "Insurance: yes-direction per gold's BSP documentary-evidence obligation but the North specifics are another contract's."),
 674: (0.25, "Renewal: the mutual-written-agreement mechanism matches gold's deviation-by-mutual-agreement clause, but pred's one-year/120-day specifics are another contract's."),
 679: (0.25, "Insurance: yes-direction matches gold (each party adds other as additional insured on Clinical Trial Liability) but the North specifics are another contract's."),
 693: (0.25, "Price restrictions: restriction-exists direction matches gold's capped Payment-Schedule increases but pred cites another contract's JCC clause."),
 702: (0.25, "Insurance: yes-direction matches gold (Oak Ridge arranges product liability/warranty insurance) but the North specifics are another contract's."),
 710: (0.25, "Renewal: the 12-month renewal duration matches gold's automatic successive periods, but the mutual-agreement mechanism is another contract's."),
 720: (0.25, "Change of control: termination-on-ownership-change direction matches gold but pred's Section 16.9 assignment clause is another contract's."),
 730: (0.25, "Insurance: yes-direction matches gold (both maintain CGL with Products Liability) but the North specifics are another contract's."),
 742: (0.25, "Revenue/profit sharing: yes-direction matches gold (10% of License Fees to Rogers) but the 1% North royalty specifics are another contract's."),
 746: (0.25, "Renewal: the one-year renewal duration matches gold's automatic successive periods, but the mutual-agreement/120-day specifics are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred's claim is another contract's. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 / Feb 21 2011 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's renewal values/mechanism are another contract's and do not match. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's North-specific claims. Zero."
    if "Ip Ownership" in q:
        return f"IP ownership: gold is '{g}...'; pred's North/Contractor clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
    if "Change Of Control" in q:
        return f"Change of control: gold span '{g}...' does not match pred's wrong-contract assignment clause. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part04.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
