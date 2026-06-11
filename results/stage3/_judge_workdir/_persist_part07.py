# Persists Claude's pass-2 judgments for part07 (entries 1050-1199), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part07.txt"

UPGRADES = {
 1062: (0.25, "Exclusivity: yes-direction matches gold (Manning Identification exclusive grant) but pred gives section pointers, no content."),
 1063: (0.25, "Anti-assignment: 'a protection exists' partially matches but gold gives a termination-on-merger right while pred asserts consent-required - mechanism wrong."),
 1064: (0.25, "Post-termination: yes-direction correct (gold: 90-day sell-off) but cited clauses are another contract's."),
 1065: (0.25, "Insurance: yes-direction and structure match gold (Company maintains $3M naming Pey Dirt) but pred cites the North contract's values."),
 1070: (0.25, "RoFN: yes-direction matches gold's LKPL RFM right but pred cites another contract's heading."),
 1075: (0.25, "Post-termination: yes-direction correct (gold: SEV 13-week repair liability) but cited clauses are another contract's."),
 1077: (0.25, "Insurance: yes-direction matches gold (employer's liability GBP 5M + CGL) but specifics cite North $2M."),
 1081: (0.25, "Minimum commitment: yes-direction matches gold (7,200 hours/year) but '1,000 Units' is another contract's value."),
 1090: (0.25, "Post-termination: yes-direction correct (gold: phase-out discussions) but cited clause 5/6 survival is another contract's."),
 1093: (0.25, "Cap on liability: 'limitation exists' direction matches gold's consequential-damages exclusion but cited fee-cap clause is another contract's."),
 1096: (0.25, "Insurance: insurance-exists direction matches gold's cancellation-notice clause but specifics cite North."),
 1107: (0.25, "Termination for convenience: yes-direction matches gold (Distributor at-will right) but pred's Contractor/Emerald clause is another contract."),
 1108: (0.25, "Post-termination: yes-direction correct (gold: Distributor continued selling) but cited clauses are another contract's."),
 1109: (0.25, "Insurance: yes-direction matches gold (each party product liability) but specifics cite North."),
 1115: (0.25, "Minimum commitment: yes-direction matches gold's guaranteed revenue but '1,000 Units' is another contract's value."),
 1124: (0.25, "Insurance: yes-direction matches gold (Customer product liability naming Manufacturer) but specifics cite North."),
 1137: (0.25, "Agreement date: pred quotes a genuinely related same-document clause referencing the Purchase Agreement date but extracts no date; gold is December 13, 1999."),
 1141: (0.25, "Post-termination: yes-direction correct (gold: support continues post-termination) but cited Section 8.5 clauses are another contract's."),
 1149: (0.25, "Insurance: yes-direction matches gold (ESTABLISHMENT CGL $1M) but specifics cite North $2M."),
 1156: (0.25, "Post-termination: yes-direction correct (gold: duties continue post-dissolution) but cited clauses are another contract's."),
 1172: (0.25, "Insurance: category-direction matches (gold annotates an insurance obligation stub) but pred's North specifics are another contract's."),
 1181: (0.25, "Minimum commitment: yes-direction matches gold's Quarterly Volume Commitment but '1,000 Units' is another contract's value."),
 1186: (0.25, "Insurance: yes-direction matches gold (certificate naming Logan's additional insured) but specifics cite North."),
 1191: (0.25, "Post-termination: yes-direction correct (gold: 3-year records retention) but cited clauses are another contract's."),
 1198: (0.25, "Post-termination: yes-direction correct (gold: Berkshire 30-day material use) but cited clauses are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "doc214_qa6" in qid or "doc216_qa5" in qid:
        return ("Governing law: the gold annotation names no jurisdiction (clause truncated in source); "
                "pred's Nevada is the recurring wrong-contract answer and cannot be supported. Zero.")
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
        return f"Insurance: gold span '{g}...' does not support pred's yes-with-wrong-contract specifics (permissive or unrelated clause). Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' does not support pred's consent-required assertion. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part07.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
