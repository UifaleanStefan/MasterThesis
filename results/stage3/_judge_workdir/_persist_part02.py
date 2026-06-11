# Persists Claude's pass-2 judgments for part02 (entries 300-449), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part02.txt"

UPGRADES = {
 300: (0.25, "Minimum commitment: yes-direction matches gold's delivery-tolerance commitment but '1,000 Units' is another contract's value."),
 301: (0.25, "Post-termination: yes-direction correct (gold: Vericel options + outstanding payments) but cited Article IV/clause 5 survival is another contract's."),
 303: (0.25, "Insurance: yes-direction matches gold (each party CGL + product liability) but specifics cite North $2M."),
 307: (0.25, "Termination for convenience: yes-direction matches gold (Company sole-discretion immediate termination) but pred's 15-day either-party clause is another contract."),
 309: (0.25, "Insurance: yes-direction matches gold (private medical scheme maintained for Executive) but pred cites North CGL specifics."),
 311: (0.5,  "Effective date: gold defines Commencement Date as the date of the Agreement; pred's 'contract date as specified on signature page' conveys the same meaning without quoting the span."),
 316: (0.25, "Post-termination: yes-direction correct (gold: Reseller retains Customer Agreements) but cited clauses are another contract's."),
 318: (0.25, "Insurance: yes-direction and $2M magnitude match gold (Reseller CGL+E&O $2M) but the cited North clause is another contract's."),
 322: (0.25, "Document name: pred 'Maintenance Agreement' captures part of the gold title 'Amendment n 01 to the Global Maintenance Agreement' but misses that it is an amendment."),
 335: (0.25, "Insurance: yes-direction matches gold (MMI maintains insurance on Dragon property) but specifics cite North."),
 341: (0.25, "Cap on liability: 'cap exists' direction matches gold's no-recourse-to-AIG-assets clause but cited limitation clause is another contract's."),
 353: (0.25, "Post-termination: yes-direction correct (gold: Supplier stock buy-back option) but cited clauses are another contract's."),
 354: (0.25, "Insurance: yes-direction matches gold (Supplier AU$10M product liability) but specifics cite North."),
 357: (0.5,  "Effective date: gold says effective as of the date first set forth above; pred's 'contract date as specified on signature page' conveys the document's stated date, close but not the exact mechanism."),
 361: (0.25, "License grant: 'a license exists' matches gold but pred reverses the direction (Contractor-to-Company instead of gold's Company-to-Contractor logo license)."),
 362: (0.25, "Post-termination: yes-direction correct (gold: orderly transfer of Notes administration) but cited clauses are another contract's."),
 363: (0.25, "Insurance: yes-direction matches gold (Contractor maintains adequate insurance) but specifics cite North."),
 370: (0.25, "Post-termination: yes-direction correct (gold: HSNS retains records 2 years post-term) but cited clauses are another contract's."),
 371: (0.25, "Cap on liability: 'limitation exists' direction matches gold's two-year action bar but cited limitation clause is another contract's."),
 375: (0.25, "Termination for convenience: yes-direction matches gold (CBC 24-month notice right) but pred's 15-day either-party clause is another contract."),
 377: (0.25, "Insurance: yes-direction matches gold (CBC all-risk + product liability) but specifics cite North."),
 385: (0.25, "Post-termination: yes-direction correct (gold: Manufacturer post-termination obligations) but cited clauses are another contract's."),
 387: (0.25, "Insurance: yes-direction matches gold (auto liability $2M requirement) but the cited North clause is another contract's."),
 394: (0.25, "Audit rights: yes-direction matches gold (audits per Quality Agreement) but the 5-business-day specifics are another contract's."),
 395: (0.25, "Cap on liability: 'limitations exist' direction matches gold's Section 2.2(b) limitation reference but cited clause is another contract's."),
 396: (0.25, "Insurance: yes-direction matches gold (certificate of insurance on request) but specifics cite North."),
 401: (0.25, "Irrevocable/perpetual license: bare 'yes' matches gold's irrevocable license grant but extracts no clause text."),
 415: (0.25, "Post-termination: yes-direction correct (gold annotates an upon-termination obligations clause) but cited clauses are another contract's."),
 419: (0.25, "Audit rights: yes-direction matches gold (audit with cost-shift on 5% discrepancy) but the cited specifics are another contract's."),
 420: (0.25, "Cap on liability: 'cap exists' direction matches gold's 12-month-fees cap but cited limitation clause is another contract's."),
 424: (0.25, "Termination for convenience: yes-direction matches gold (Company sole-discretion notice right) but pred's 15-day either-party clause is another contract."),
 425: (0.25, "RoFR/RoFO/RoFN: 'clause exists' direction matches the gold-annotated option grant but pred cites another contract's Combination-Products RoFN heading."),
 427: (0.25, "Post-termination: yes-direction correct (gold: settlement-agreement conditions) but cited survival clauses are another contract's."),
 432: (0.25, "Insurance: yes-direction matches gold (Association named additional insured) but specifics cite North."),
 436: (0.25, "Post-termination: yes-direction correct (gold: remaining Parties assume obligations) but cited clauses are another contract's."),
 437: (0.25, "Audit rights: yes-direction matches gold (F&ASC audit procedures) but the cited specifics are another contract's."),
 441: (0.25, "Post-termination: yes-direction correct (gold: eGain provides final Reports) but cited clauses are another contract's."),
 442: (0.25, "Uncapped liability: 'not uncapped' direction consistent with gold's consequential-damages exclusion but cited exception structure is another contract's."),
 443: (0.25, "Post-termination: yes-direction correct (gold: BKC post-termination option) but cited clauses are another contract's."),
 444: (0.25, "Audit rights: yes-direction matches gold (tax returns + records to BKC) but the cited specifics are another contract's."),
 445: (0.25, "Insurance: yes-direction matches gold (Franchisee participates in mandated insurance programs) but specifics cite North."),
 449: (1.0,  "Governing law: gold is California; pred states California governs - correct answer."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "doc75_qa5" in qid:
        return ("Governing law: full gold checked in queue - State of Israel; pred says Nevada. "
                "Wrong jurisdiction from another contract. Zero.")
    if "doc81_qa2" in qid:
        return ("Expiration: gold gives a relative one-year term with no stated date; pred's "
                "'March 22, 2002' is fabricated specificity not derivable from the clause. Zero.")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause. Zero."
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
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold permits consent-free assignment in divestiture cases ('{g}...'); pred's blanket consent requirement contradicts it. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part02.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
