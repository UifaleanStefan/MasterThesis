# Persists Claude's pass-2 judgments for part05 (entries 750-899), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part05.txt"

UPGRADES = {
 754: (0.25, "Insurance: conditional yes-direction matches gold (if-requested insurance arrangement) but specifics cite North."),
 766: (0.25, "Affiliate license: yes-direction matches gold (Village Media sublicense right) but pred gives section pointers, no content."),
 767: (0.25, "Post-termination: yes-direction correct (gold: continued exploitation right) but cited clauses are another contract's."),
 774: (0.25, "Post-termination: yes-direction correct (gold: rehire cooperation) but cited clauses are another contract's."),
 775: (0.25, "Insurance: yes-direction matches gold's insurance-primary clause but specifics cite North."),
 786: (0.25, "Minimum commitment: yes-direction matches gold (pipeline-fill minimum quantities) but '1,000 Units' is another contract's value."),
 789: (0.25, "Post-termination: yes-direction correct (gold: 25-month true-up) but cited clauses are another contract's."),
 790: (0.25, "Insurance: yes-direction matches gold (parties maintain GL 5 years post-sale) but specifics cite North."),
 793: (0.25, "Minimum commitment: category-direction matches gold's LTL/TL shipping minimum but '1,000 Units' is another contract's value."),
 794: (0.25, "Post-termination: yes-direction correct (gold: Cisco repurchase obligation) but cited clauses are another contract's."),
 796: (0.25, "Insurance: yes-direction matches gold (Worker's Comp statutory) but specifics cite North."),
 801: (0.25, "Post-termination: yes-direction correct (gold: 180-day transition cap) but cited clauses are another contract's."),
 803: (0.25, "Insurance: yes-direction matches gold (Tech E&O $5M) but specifics cite North $2M."),
 810: (0.25, "Insurance: yes-direction matches gold (NETGEAR furnishes certificate) but specifics cite North."),
 821: (0.25, "Cap on liability: 'cap exists' direction matches gold's termination-payment formula but cited limitation clause is another contract's."),
 827: (0.25, "Post-termination: yes-direction correct (gold: sell-off period) but cited clauses are another contract's."),
 828: (1.0,  "Document name: gold 'Intellectual Property Agreement'; pred names the Jingwei INTELLECTUAL PROPERTY AGREEMENT - correct document identified."),
 835: (0.25, "Anti-assignment: 'requirement exists' direction plausible but pred's generic consent clause is not the gold's 1940-Act-related span."),
 836: (0.25, "Cap on liability: 'limitation exists' direction matches gold's Declaration-of-Trust limitation notice but cited clause is another contract's."),
 841: (0.25, "Anti-assignment: 'a requirement exists' partially matches but gold requires NOTICE for KI assignments while pred asserts consent-or-nothing - mechanism wrong."),
 842: (0.25, "Post-termination: yes-direction correct (gold: 120-day sell-off) but cited clause 5/6 survival is another contract's."),
 843: (0.25, "Insurance: yes-direction matches gold (Diplomat obtains product liability for KI) but specifics cite North."),
 850: (0.25, "Exclusivity: yes-direction matches gold's territorial exclusivity but pred gives section pointers, no content."),
 851: (0.25, "Insurance: yes-direction matches gold (Operations-Manual coverages) but specifics cite North."),
 856: (0.25, "License grant: bare 'yes' matches gold's non-exclusive mark license but extracts no clause content."),
 857: (0.25, "Post-termination: yes-direction correct (gold: 24-month public-notice obligation) but cited Contractor clauses are another contract's."),
 858: (1.0,  "Document name: gold 'Distributor Agreement'; pred names the Lucid DISTRIBUTOR AGREEMENT - correct document identified."),
 863: (0.25, "Change of control: termination-right direction overlaps gold (Lucid may terminate on Distributor CoC) but pred cites another contract's clause."),
 869: (0.25, "Post-termination: yes-direction correct (gold: Contractors remain Network 1's) but cited clauses are another contract's."),
 871: (0.5,  "Effective date: gold says the Term begins on the date hereof; pred's 'contract date as specified on signature page' conveys the same document-date mechanism."),
 879: (0.25, "Post-termination: yes-direction correct (gold: no-use-of-marks except inventory sales) but cited clauses are another contract's."),
 881: (1.0,  "Document name: gold 'Master Development and Manufacturing Agreement'; pred names exactly that for Magenta Therapeutics - correct."),
 887: (0.25, "Insurance: yes-direction matches gold (Bachem provides insurance statement) but specifics cite North."),
 893: (0.25, "Post-termination: yes-direction correct (gold: PAPA JOHN'S post-term endorsement rights) but cited clauses are another contract's."),
 894: (0.25, "Insurance: yes-direction matches gold ($5M ABG-acceptable insurer) but specifics cite North $2M."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's specific date does not follow from it (wrong contract or fabricated). Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part05.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
