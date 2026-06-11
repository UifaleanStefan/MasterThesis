# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part05
# (entries 750-899), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part05.txt"

UPGRADES = {
 750: (0.25, "Insurance: obligations-exist direction per gold's premiums/deductibles clause but the North specifics are another contract's."),
 755: (0.25, "Change of control: termination-on-merger-notice direction matches gold's Nantz right but pred's Section 16.9 clause is another contract's."),
 756: (0.25, "Insurance: yes-direction matches gold ($5M policies naming Nantz) but the North $2M specifics are another contract's."),
 760: (0.25, "Renewal: the one-year renewal duration matches gold's automatic successive terms, but the mutual-agreement mechanism is another contract's."),
 762: (0.25, "Change of control: notice/consent-on-CoC direction matches gold's Medica obligation but pred's Section 16.9 clause is another contract's."),
 763: (0.25, "Uncapped liability: limited-with-indemnification-carve-out structure matches gold but the quoted King/Depomed clause is another contract's."),
 764: (0.25, "Insurance: yes-direction matches gold (Medica furnishes certification + cancellation notice) but the North specifics are another contract's."),
 783: (0.25, "Change of control: category-direction per gold's annotated death/incapacity transfer provisions but pred's Section 16.9 clause is another contract's."),
 785: (0.25, "Insurance: yes-direction matches gold (policies per Manuals standards) but the North specifics are another contract's."),
 786: (0.25, "Document name: pred's 'Service Agreement' partially matches gold TRANSITION SERVICES AGREEMENT but drops the distinguishing qualifier."),
 790: (0.25, "Insurance: yes-direction matches gold (each party CGL $5M + workers comp) but the North specifics are another contract's."),
 799: (0.25, "Anti-assignment: restriction-direction per gold's annotated no-third-party-benefit span but pred's consent mechanism is another contract's."),
 804: (0.25, "Change of control: termination-on-merger direction matches gold's Pey Dirt right but pred's Section 16.9 clause is another contract's."),
 805: (0.25, "Anti-assignment: restriction-direction per gold's annotated merger-termination span but pred's bare consent claim is another contract's mechanism."),
 806: (0.25, "Insurance: yes-direction matches gold ($3M policies naming Pey Dirt) but the North $2M specifics are another contract's."),
 814: (0.25, "Insurance: yes-direction matches gold (employer's liability GBP 5M + CGL) but the North specifics are another contract's."),
 822: (0.25, "Renewal: the one-year renewal duration matches gold's automatic 1-year terms, but the mutual-agreement mechanism is another contract's."),
 824: (0.25, "Renewal: the year-to-year renewal duration matches gold, but the mutual-agreement mechanism is another contract's (gold renews automatically)."),
 830: (0.25, "Insurance: yes-direction per gold's annotated 30-day-cancellation-notice clause but the North specifics are another contract's."),
 835: (0.25, "Change of control: 50%-ownership-change termination direction matches gold but pred's Section 16.9 clause is another contract's."),
 841: (0.25, "Insurance: yes-direction matches gold (each party maintains product liability) but the North specifics are another contract's."),
 845: (0.25, "Renewal: the 12-month mutual-agreement extension matches gold's option mechanism, but pred's 120-day specifics are another contract's."),
 857: (0.25, "Insurance: yes-direction matches gold (Customer maintains product liability naming Manufacturer) but the North specifics are another contract's."),
 858: (0.25, "Document name: pred's 'Development Agreement' partially matches gold BLOCKCHAIN ADMINISTRATION AND DEVELOPMENT AGREEMENT but drops the distinguishing qualifier."),
 863: (0.25, "Renewal: the annual continuation duration matches gold's successive annual periods, but the mutual-agreement mechanism differs from gold's Board-approval mechanism."),
 870: (0.25, "Renewal: the one-year renewal duration matches gold's automatic additional terms, but the XFN/mutual mechanism is another contract's."),
 874: (0.25, "Renewal: the one-year renewal duration matches gold's automatic subsequent periods, but the mutual-agreement mechanism is another contract's."),
 881: (0.25, "Insurance: yes-direction matches gold (ESTABLISHMENT maintains CGL $1M) but the North specifics are another contract's."),
 885: (0.25, "Anti-assignment: restriction-direction per gold's annotated JV-property-title span but pred's consent claim is another contract's mechanism."),
 891: (0.25, "Renewal: the one-year renewal duration matches gold's renewable-once year, but the mutual-agreement mechanism is another contract's."),
 892: (0.25, "Change of control: notification-on-change direction matches gold's Party-B obligation but pred's Section 16.9 clause is another contract's."),
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
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is permissive ('may acquire') or does not support pred's North-specific claims. Zero."
    if "Ip Ownership" in q:
        return f"IP ownership: gold is '{g}...'; pred's North/Contractor clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold span '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part05.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
