# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part03
# (entries 450-599), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part03.txt"

UPGRADES = {
 452: (0.25, "Insurance: yes-direction matches gold (Operator maintains $1M auto liability) but the North specifics are another contract's."),
 462: (0.25, "Insurance: yes-direction matches gold (STW insures, COFS additional insured) but the North specifics are another contract's."),
 476: (0.25, "Insurance: yes-direction matches gold (each party maintains liability insurance) but the North specifics are another contract's."),
 483: (0.25, "Insurance: yes-direction matches gold (GL insurance incl. clinical-trial coverage) but the North specifics are another contract's."),
 488: (0.25, "Insurance: yes-direction matches gold (Network Affiliate insures theatres/equipment) but the North specifics are another contract's."),
 516: (0.25, "Insurance: yes-direction matches gold (E&O $5M per claim) but the North specifics are another contract's."),
 520: (0.25, "Insurance: yes-direction matches gold (each party maintains insurance during and after activities) but the North specifics are another contract's."),
 524: (0.25, "Insurance: yes-direction matches gold (Customer insures stored finished Products) but the North specifics are another contract's."),
 529: (0.25, "Insurance: yes-direction per gold's annotated premium-non-expiry policy clause but the North specifics are another contract's."),
 538: (0.25, "Renewal: the mutual-negotiation mechanism matches gold's extension-by-negotiation, but pred's one-year/120-day specifics are another contract's."),
 547: (0.25, "Price restrictions: restriction-exists direction matches gold's at-cost-without-markup clause but pred cites another contract's JCC pricing clause."),
 555: (0.25, "Renewal: the one-year renewal duration matches gold's additional one-year extension, but the mutual-agreement mechanism and 120-day notice are another contract's."),
 558: (0.25, "Insurance: yes-direction per gold's annotated primary-insurance clause but the North specifics are another contract's."),
 571: (0.25, "Insurance: yes-direction matches gold (parties maintain GL insurance through 5 years post-sale) but the North specifics are another contract's."),
 575: (0.25, "Insurance: yes-direction matches gold (each party maintains statutory Workers' Comp) but the North specifics are another contract's."),
 578: (0.25, "Change of control: category-direction per gold's annotated Termination-Event stub but pred's Section 16.9 assignment clause is another contract's."),
 580: (0.25, "Insurance: yes-direction matches gold (Technology E&O $5M minimum) but the North specifics are another contract's."),
 584: (0.25, "Renewal: the one-year renewal duration matches gold's successive one-year periods, but the XFN/mutual-agreement mechanism is another contract's (gold renews automatically)."),
 585: (0.25, "Insurance: certificate-on-request implies maintained coverage matching the yes-direction, but the North specifics are another contract's."),
 592: (0.25, "Renewal: the one-year renewal duration matches gold's additional one-year period, but the mutual-agreement mechanism is another contract's (gold renews automatically)."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred's claim is another contract's. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 / Feb 21 2011 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 / Dec 31 2012 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's renewal values/mechanism are another contract's and do not match. Zero."
    if "Notice Period" in q:
        if "doc125_qa5" in qid:
            return "Notice period: full gold (checked in queue.jsonl) requires six weeks' registered-letter notice; pred's 120-day value is another contract's. Zero."
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is conditional/permissive or does not support pred's North-specific claims. Zero."
    if "Ip Ownership" in q:
        return f"IP ownership: gold is '{g}...'; pred's Contractor-retains clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
    if "Change Of Control" in q:
        return f"Change of control: gold span '{g}...' shows CoC is NOT an assignment (no restriction); pred's consent-required claim contradicts it. Zero."
    if "Non-Transferable" in q:
        return f"Non-transferable license: gold span '{g}...' shows broad sublicensing rights; pred's consent-required limitation contradicts it. Zero."
    if "Price Restrictions" in q:
        return f"Price restrictions: gold span '{g}...' does not support pred's wrong-contract pricing clause. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part03.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
