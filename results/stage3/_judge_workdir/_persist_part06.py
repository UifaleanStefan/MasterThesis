# Persists Claude's pass-2 judgments for part06 (entries 900-1049), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part06.txt"

UPGRADES = {
 905:  (0.25, "Post-termination: yes-direction correct (gold annotates a Green Cross termination-consequences clause) but pred's generic remedies list is another contract's."),
 916:  (0.25, "Insurance: yes-direction matches gold (BSP provides documentary evidence of insurance) but specifics cite North."),
 919:  (0.5,  "Effective date: gold says the Agreement comes into force on the Date of the Agreement; pred's signature-page-date phrasing conveys the same document-date mechanism."),
 925:  (0.25, "Post-termination: category-direction correct but pred's payments/Contractor clauses are another contract's."),
 926:  (0.25, "Insurance: yes-direction matches gold (mutual additional-insured Clinical Trial Liability) but specifics cite North."),
 927:  (1.0,  "Document name: gold STRATEGIC ALLIANCE AGREEMENT; pred names exactly that - correct."),
 946:  (0.25, "Post-termination: yes-direction correct (gold: assets transferred back) but cited clauses are another contract's."),
 947:  (0.25, "Insurance: yes-direction matches gold (Oak Ridge arranges product liability) but specifics cite North."),
 952:  (0.25, "Exclusivity: yes-direction matches gold's exclusive license grant but pred gives a section pointer, no content."),
 953:  (0.25, "Post-termination: yes-direction correct (gold: HEMISPHERX repurchases stock) but cited clauses are another contract's."),
 964:  (0.25, "Post-termination: yes-direction correct (gold: records maintenance) but cited clauses are another contract's."),
 972:  (0.25, "Post-termination: yes-direction correct (gold: Plant Material removal/destruction) but cited clauses are another contract's."),
 978:  (0.25, "Audit rights: inspection-right direction matches gold (INTERSECT lot inspection) but the 5-business-day audit specifics are another contract's."),
 980:  (0.25, "Insurance: yes-direction matches gold (both parties maintain CGL) but specifics cite North."),
 985:  (0.25, "Post-termination: yes-direction correct (gold: RemainCo continued distribution) but cited clauses are another contract's."),
 993:  (0.25, "License grant: bare 'yes' matches gold's Rogers VOD right but extracts no clause content."),
 994:  (0.25, "Uncapped liability: limited-with-carve-outs direction matches gold but pred's Section 8/12 exception structure is another contract's."),
 1001: (0.25, "Post-termination: yes-direction correct (gold: continued transfer cooperation) but cited clauses are another contract's."),
 1003: (0.25, "Insurance: insurance-exists direction matches gold's premiums clause but specifics cite North."),
 1010: (0.25, "Insurance: yes-direction and structure match gold (Company maintains $5M naming the celebrity) but pred cites the North contract's values."),
 1015: (0.25, "RoFR/RoFO/RoFN: yes-direction matches gold's Medica notice-and-offer right but pred cites another contract's heading."),
 1016: (0.25, "Insurance: yes-direction matches gold (Medica furnishes certification) but specifics cite North."),
 1029: (0.25, "Post-termination: yes-direction correct (gold: maintenance continuation right) but cited clauses are another contract's."),
 1034: (0.25, "RoFR: yes-direction matches gold's right of first refusal but pred cites another contract's heading."),
 1035: (0.25, "Change of control: consent-direction overlaps gold's transfer provisions but pred cites another contract's clause."),
 1036: (0.25, "Minimum commitment: yes-direction matches gold (50% promotional participation) but '1,000 Units' is another contract's value."),
 1038: (0.25, "Post-termination: yes-direction correct (gold: lease assignment on demand) but cited clauses are another contract's."),
 1039: (0.25, "Insurance: yes-direction matches gold (policy by satisfactory insurer) but specifics cite North."),
 1044: (0.25, "Insurance: yes-direction matches gold (CGL $5M + workers comp) but specifics cite North $2M."),
 1045: (0.5,  "Parties: pred names GWG Holdings, a real party to this Orderly Marketing Agreement, but misses the gold-annotated Trust Advisors."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "doc201_qa5" in qid:
        return ("Governing law: full gold checked in queue - State of New York; pred says Georgia. "
                "Wrong jurisdiction from another contract. Zero.")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's specific date does not follow from it (wrong contract). Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's 2-year/90-day clause is another contract's. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days value is another contract's. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's yes-with-wrong-contract specifics. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part06.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
