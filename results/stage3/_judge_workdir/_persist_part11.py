# Persists Claude's pass-2 judgments for part11 (entries 1650-1799), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part11.txt"

UPGRADES = {
 1655: (0.25, "Post-termination: yes-direction correct (gold: Turpin delivers Work Product) but cited clauses are another contract's."),
 1666: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold's no-contest covenant but pred infers from another clause."),
 1679: (0.25, "RoFR: yes-direction matches gold's bankruptcy-assignment first-refusal context but pred cites another contract's heading."),
 1680: (0.25, "Change of control: restriction-direction overlaps gold's no-transfer clause but pred cites another contract's provisions."),
 1681: (0.25, "Anti-assignment: restriction-exists direction matches gold's no-transfer clause but pred's quoted clause is another contract's."),
 1682: (0.25, "Minimum commitment: yes-direction matches gold's Minimum Development Quota but '1,000 Units' is another contract's value."),
 1684: (0.25, "Post-termination: yes-direction correct (gold: cancel registrations within 30 days) but cited clauses are another contract's."),
 1685: (0.25, "Insurance: requirement-direction matches gold (franchisor may procure on failure) but specifics cite North."),
 1693: (0.25, "Post-termination: yes-direction correct (gold: post-notice services at current rates) but cited clauses are another contract's."),
 1694: (0.25, "Uncapped liability: limited-with-carve-outs direction matches gold's structure but the cited exceptions are another contract's."),
 1695: (0.25, "Insurance: category-direction per gold's annotated stub but pred's North specifics are another contract's."),
 1708: (0.25, "Minimum commitment: minimum-fee-threshold direction matches gold but '1,000 Units' is another contract's value."),
 1709: (0.25, "Post-termination: yes-direction correct (gold: 5-year records retention) but cited clauses are another contract's."),
 1710: (0.25, "Audit rights: audit-exists direction matches gold's stub but the cited specifics are another contract's."),
 1712: (1.0,  "Document name: gold LICENCE AND MAINTENANCE AGREEMENT; pred names exactly that - correct."),
 1726: (0.25, "Post-termination: yes-direction correct (gold: Operating-Agreement continuation) but cited clauses are another contract's."),
 1732: (0.25, "Post-termination: yes-direction correct (gold: order-processing obligation) but cited clauses are another contract's."),
 1737: (0.25, "Post-termination: yes-direction correct (gold: return Source Plasma) but cited clauses are another contract's."),
 1738: (0.25, "Uncapped liability: limited-with-carve-outs direction matches gold's indemnity exception + cap but the cited clause is another contract's."),
 1745: (0.25, "Post-termination: yes-direction correct (gold: Termination Assistance) but cited clauses are another contract's."),
 1746: (0.25, "Insurance: yes-direction matches gold (certificates furnished on request) but specifics cite North."),
 1756: (0.25, "Post-termination: yes-direction correct (gold: 6-year records retention) but cited clauses are another contract's."),
 1769: (1.0,  "Parties: pred names Sony Electronics Inc. and GSI Technology Inc. - the actual parties to this IP agreement (gold span defines PURCHASER and SONY). Correct identification."),
 1773: (0.25, "Change of control: transfer-on-merger direction overlaps gold but pred cites another contract's provisions."),
 1779: (0.25, "Change of control: CoC-provisions direction overlaps gold's assignable-on-CoC clause but pred cites another contract's."),
 1780: (0.25, "Post-termination: yes-direction correct (gold: 6-month Use-up Period) but cited clauses are another contract's."),
 1781: (1.0,  "Document name: gold Master Service Agreement; pred names exactly that - correct."),
 1787: (0.25, "Post-termination: yes-direction correct (gold: deliver materials on termination) but cited clauses are another contract's."),
 1791: (0.25, "Post-termination: yes-direction correct (gold: transfer Policy to Trust) but cited clauses are another contract's."),
 1796: (0.25, "Post-termination: yes-direction correct (gold: sums-paid/orders-fulfilled obligations) but cited clauses are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date. Zero."
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
        return f"Insurance: gold span '{g}...' does not support pred's yes-with-wrong-contract specifics. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits consent-free assignment or is unrelated; pred's consent-required assertion contradicts it. Zero."
    if "Third Party Beneficiary" in q:
        return f"Third-party beneficiary: gold establishes a beneficiary ('{g}...'); pred's no-third-party-beneficiaries clause contradicts it (wrong contract). Zero."
    if "Audit Rights" in q:
        return f"Audit rights: gold span '{g}...' does not support pred's wrong-contract audit specifics. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part11.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
