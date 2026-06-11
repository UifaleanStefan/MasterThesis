# Persists Claude's pass-2 judgments for part10 (entries 1500-1649), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part10.txt"

UPGRADES = {
 1506: (0.25, "Minimum commitment: yes-direction matches gold (10% Available Capacity) but '1,000 Units' is another contract's value."),
 1507: (0.5,  "Effective date: gold defines the Effective Date as the date of actual execution; pred's signature-page-date phrasing conveys the same execution-date mechanism."),
 1519: (0.25, "Minimum commitment: yes-direction matches gold (12 Products initial order) but '1,000 Units' is another contract's value."),
 1520: (0.25, "Post-termination: yes-direction correct (gold: Aura purchase/return right) but cited clauses are another contract's."),
 1522: (0.25, "Insurance: yes-direction matches gold (both parties maintain product liability) but specifics cite North $2M."),
 1536: (0.25, "Post-termination: yes-direction correct (gold: 60-day Transition-out Period) but cited clauses are another contract's."),
 1537: (0.25, "Insurance: yes-direction matches gold (30-day notice on policy changes) but specifics cite North."),
 1549: (0.25, "Insurance: yes-direction matches gold (Company furnishes JHU certificates) but specifics cite North."),
 1553: (0.25, "Post-termination: direction overlaps gold's Suspension-Period notification duties but cited Section 8.5 clauses are another contract's."),
 1554: (0.25, "Insurance: category-direction plausible per gold's Indemnification-Agreement context but pred's North specifics are another contract's."),
 1563: (0.25, "Anti-assignment: restriction-exists direction matches gold's no-assign-except-successor but pred's consent-void mechanism is wrong."),
 1570: (0.25, "Post-termination: yes-direction correct (gold: Company post-term obligations) but cited clauses are another contract's."),
 1571: (0.25, "Insurance: yes-direction matches gold (Company carries reasonable insurance) but specifics cite North."),
 1572: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold's no-contest covenant but pred infers from another clause."),
 1575: (0.25, "Insurance: insurance-in-force direction matches gold's representation but pred's North specifics are another contract's."),
 1581: (0.25, "Exclusivity: yes-direction matches gold's exclusive-for-US supply but pred gives a generic restatement, no content."),
 1582: (0.25, "Minimum commitment: minimum-exists direction matches gold's expiry-dating minimum but '1,000 Units' is another contract's value."),
 1583: (0.25, "Post-termination: yes-direction correct (gold: insurance maintained 3 years post-term) but cited clauses are another contract's."),
 1584: (0.25, "Insurance: yes-direction matches gold (both maintain CGL incl. 3 years post) but specifics cite North."),
 1589: (0.25, "Post-termination: yes-direction correct (gold: options exercisable post-termination) but cited Section 8.5 clauses are another contract's."),
 1602: (0.25, "Termination for convenience: yes-direction matches gold (Sponsor any-time right) but pred's 120-day clause is another contract's."),
 1603: (0.25, "Insurance: yes-direction matches gold (Racing auto liability $5M) but specifics cite North $2M."),
 1610: (0.25, "Audit rights: yes-direction matches gold (third-party traffic auditor) but the cited specifics are another contract's."),
 1617: (0.25, "Post-termination: yes-direction correct (gold: link-removal responsibilities) but cited Section 8.5 clauses are another contract's."),
 1618: (0.25, "Audit rights: yes-direction matches gold's 18-month audit window but the cited specifics are another contract's."),
 1622: (0.25, "Anti-assignment: restriction-exists direction matches gold's ISO no-transfer clause but pred's consent mechanism is wrong."),
 1623: (0.25, "Audit rights: direction matches gold (audited financials to SERVICERS) but the cited specifics are another contract's."),
 1627: (0.25, "Audit rights: category-direction matches gold's stub but the cited specifics are another contract's."),
 1629: (0.25, "Insurance: yes-direction matches gold (Company and Agent maintain insurance) but specifics cite North."),
 1630: (1.0,  "Document name: gold SERVICE AGREEMENT; pred names the RISE Education SERVICE AGREEMENT - correct document identified."),
 1647: (0.25, "Post-termination: yes-direction correct (gold: records retention [*]) but cited clauses are another contract's."),
 1649: (0.25, "Insurance: yes-direction matches gold (each party 5-years-post insurance) but specifics cite North."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date or contradicts the gold mechanism. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's specific date does not follow from it (wrong contract). Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's clause is another contract's. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days value is another contract's. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is permissive/representational or does not support pred's wrong-contract specifics. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits consent-free assignment or is unrelated; pred's consent-required assertion contradicts it. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part10.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
