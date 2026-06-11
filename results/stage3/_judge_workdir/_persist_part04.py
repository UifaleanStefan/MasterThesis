# Persists Claude's pass-2 judgments for part04 (entries 600-749), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part04.txt"

UPGRADES = {
 614: (0.25, "Audit rights: yes-direction matches gold (audits conducted minimizing disruption) but the 5-business-day specifics are another contract's."),
 616: (0.25, "Insurance: yes-direction matches gold (parties increase coverage as prudent) but specifics cite North."),
 623: (0.25, "Insurance: yes-direction matches gold (Owner $500M pollution coverage) but specifics cite North $2M."),
 626: (0.25, "Insurance: yes-direction matches gold (A.M. Best A-8, Padres/SDBF additional insureds) but specifics cite North."),
 630: (0.25, "Insurance: yes-direction matches gold (both parties carry insurance) but specifics cite North."),
 637: (0.25, "Post-termination: yes-direction correct (gold: Operator transition cooperation) but cited clauses are another contract's."),
 638: (0.25, "Insurance: yes-direction matches gold (Operator auto insurance $1M) but specifics cite North."),
 649: (0.25, "Post-termination: yes-direction correct (gold: COFS termination-window obligations) but cited clauses are another contract's."),
 650: (0.25, "Audit rights: yes-direction matches gold (COFS premises entry) but the cited specifics are another contract's."),
 651: (0.25, "Insurance: yes-direction matches gold (STW insures, COFS additional insured) but specifics cite North."),
 665: (0.25, "Post-termination: yes-direction correct (gold: OntoChem furnishes deliverables) but cited clauses are another contract's."),
 666: (0.25, "Insurance: yes-direction matches gold (each party liability insurance) but specifics cite North."),
 675: (0.25, "Post-termination: yes-direction correct (gold: AbbVie sell-off right) but cited Section 8.5 clauses are another contract's."),
 676: (0.25, "Insurance: yes-direction matches gold (General Liability + Clinical Trial insurance) but specifics cite North $2M."),
 683: (0.25, "Post-termination: yes-direction correct (gold: NCM entry/recovery right) but cited clauses are another contract's."),
 684: (0.25, "Insurance: yes-direction matches gold (Network Affiliate maintains theatre insurance) but specifics cite North."),
 689: (0.25, "Cap on liability: 'limitation exists' direction matches gold's consequential-damages exclusion but cited fee-cap clause is another contract's."),
 698: (0.25, "Document name: pred 'Maintenance Agreement' captures part of the gold title 'SCHEDULE TO Software License, Customization and Maintenance Agreement'."),
 705: (0.25, "Post-termination: yes-direction correct (gold: transition negotiation) but cited clauses are another contract's."),
 714: (0.25, "Post-termination: yes-direction correct (gold: payments recognized 90 days post-term) but cited clauses are another contract's."),
 715: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold (Distributor recognizes Airspan copyright) but pred cites another contract's clause."),
 726: (0.25, "License grant: bare 'yes' matches gold's Vendor-to-AT&T license but extracts no clause content."),
 727: (0.25, "Post-termination: yes-direction correct (gold: equipment reimbursement) but cited clauses are another contract's."),
 730: (0.25, "Insurance: yes-direction matches gold (E&O $5M requirement) but specifics cite North $2M."),
 734: (0.25, "Post-termination: yes-direction correct (gold: Surgical delivers Work Product) but cited clauses are another contract's."),
 735: (0.25, "Insurance: yes-direction matches gold (each party maintains insurance) but specifics cite North."),
 740: (0.25, "Minimum commitment: yes-direction matches gold (FGI 2-4 weeks supply) but '1,000 Units' is another contract's value."),
 741: (0.25, "Post-termination: yes-direction correct (gold: compensation for unused inventory) but cited clauses are another contract's."),
 742: (0.25, "Insurance: yes-direction matches gold (Customer insures stored products) but specifics cite North."),
 749: (0.25, "Insurance: yes-direction matches gold's policy-continuation clause but specifics cite North."),
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
        return f"Renewal: gold is '{g}...'; pred's 2-year/90-day clause is another contract's. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days clause is another contract's. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's yes-with-wrong-contract specifics. Zero."
    if "Minimum Commitment" in q:
        return f"Minimum commitment: gold span '{g}...' does not support pred's wrong-contract '1,000 Units' value. Zero."
    if "Cap On Liability" in q:
        return f"Cap on liability: gold span '{g}...' does not support pred's wrong-contract cap clause. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part04.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
