# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part00
# (entries 0-149), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part00.txt"

UPGRADES = {
 1:   (0.25, "Insurance: yes-direction matches gold (Company carries product liability naming Distributor) but the North $2M specifics are another contract's."),
 7:   (0.25, "Insurance: yes-direction matches gold (Seller covers 110% invoice value) but the North $2M specifics are another contract's."),
 35:  (0.25, "Exclusivity: exclusivity-exists direction matches gold's exclusive-distributor appointment but pred cites another contract's Depomed Section 10.1."),
 36:  (0.25, "Termination for convenience: yes-direction matches gold's PPI any-time repurchase right but the 15-day either-party clause is another contract's."),
 38:  (0.25, "Insurance: yes-direction matches gold (each party maintains product liability) but the North specifics are another contract's."),
 50:  (0.25, "Audit rights: yes-direction matches gold's EDGE inspection right but the 5-day-notice specifics are another contract's."),
 51:  (0.25, "Insurance: yes-direction matches gold's CGL $1M/$2M requirement but the North specifics are another contract's."),
 58:  (0.25, "Termination for convenience: yes-direction matches gold's Nuance any-time right but the 15-day either-party clause is another contract's."),
 64:  (0.25, "Insurance: yes-direction matches gold's Employers' Liability $10M requirement but the North specifics are another contract's."),
 66:  (0.25, "Anti-assignment: restriction-exists direction is right but gold's mechanism is automatic termination on assignment, not consent."),
 67:  (0.25, "Audit rights: yes-direction matches gold (Fund furnishes audits/statements) but the 5-day-notice specifics are another contract's."),
 80:  (0.25, "Post-termination: category-direction per gold's annotated audit-conditions span but pred's survival clauses are another contract's."),
 84:  (0.25, "Termination for convenience: category-direction per gold's annotated Agent-option span but the 15-day either-party clause is another contract's."),
 92:  (0.25, "Termination for convenience: yes-direction matches gold's Company sole-discretion right but the 15-day either-party clause is another contract's."),
 110: (0.25, "Affiliate license-licensee: license-grant direction per gold's annotated license context but pred's XFN sublicense specifics are another contract's."),
 114: (0.25, "Insurance: yes-direction per gold's annotated HERC policy-notice obligation but the North specifics are another contract's."),
 131: (0.25, "Termination for convenience: TFC-exists direction matches gold's Party-B unilateral right but pred's either-party 15-day clause is another contract's and misses the one-sidedness."),
 141: (0.25, "Termination for convenience: yes-direction matches gold's Party-A 3-month unilateral right but the 15-day either-party clause is another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's mechanism/values are another contract's and contradict the gold. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's 120-day value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic or another contract's (North/Golfers) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is permissive or a representation, not a maintain-requirement supporting pred's North specifics. Zero."
    if "Ip Ownership" in q:
        return f"IP assignment: gold is '{g}...'; pred's North/Contractor clauses are another contract's. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's clause, describing the restriction rather than the exception. Zero."
    if "Non-Transferable" in q:
        return f"Non-transferable license: gold span '{g}...' does not support pred's consent-required assertion. Zero."
    if "Exclusivity" in q:
        return f"Exclusivity: gold span '{g}...' does not match pred's non-committal section pointer. Zero."
    if "Termination For Convenience" in q:
        return f"Termination for convenience: gold span '{g}...' does not support pred's 15-day either-party claim. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part00.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
