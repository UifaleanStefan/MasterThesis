# Persists Claude's pass-2 judgments for part03 (entries 450-599), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part03.txt"

UPGRADES = {
 450: (0.25, "Post-termination: yes-direction correct (gold: Life Technologies license survives) but cited indemnity clauses are another contract's."),
 451: (0.25, "Insurance: yes-direction matches gold (certificate of insurance on request) but specifics cite North $2M."),
 452: (0.5,  "Governing law: gold is California AND Hong Kong; pred names California only - one of the two jurisdictions correct."),
 455: (0.25, "Post-termination: yes-direction correct (gold: ALFA AESAR inventory options) but cited clause 5/6 survival is another contract's."),
 461: (0.25, "Termination for convenience: yes-direction plausible per gold's notice clause but pred's 15-day either-party clause is another contract."),
 462: (0.25, "Anti-assignment: 'a requirement exists' partially matches but gold requires NOTICE for affiliate assignments while pred asserts consent-or-void - mechanism wrong."),
 463: (0.25, "Audit rights: yes-direction matches gold (inventory inspection access) but the 5-business-day specifics are another contract's."),
 464: (0.25, "Insurance: yes-direction matches gold's insurance-provisions clause but specifics cite North."),
 469: (0.25, "Post-termination: yes-direction correct (gold: 3-month Termination Assistance Services) but cited Section 8.5 clauses are another contract's."),
 474: (0.25, "RoFR/RoFO/RoFN: 'clause exists' matches the gold-annotated Option clause but pred cites another contract's Combination-Products heading."),
 480: (0.25, "Post-termination: yes-direction correct (gold: Fleet performs until [***]) but cited Contractor-fees clauses are another contract's."),
 481: (0.25, "Audit rights: yes-direction matches gold (Columbia inspection right) but the cited specifics are another contract's."),
 482: (0.25, "Insurance: yes-direction matches gold (each party maintains insurance post-term) but specifics cite North."),
 486: (0.25, "License grant: 'a license exists' matches gold's perpetual Code license but extracts no clause content."),
 487: (0.25, "Post-termination: yes-direction correct (gold: license converts to perpetual on default) but cited clauses are another contract's."),
 491: (1.0,  "Document name: gold REMARKETING AGREEMENT; pred names REMARKETING AGREEMENT - exact match."),
 513: (0.25, "Termination for convenience: yes-direction matches gold (60-day right after 7th anniversary) but pred's 15-day clause is another contract."),
 527: (0.25, "Post-termination: yes-direction correct (gold: Reseller keeps books 3 years post-term) but cited clauses are another contract's."),
 528: (0.25, "Insurance: yes-direction matches gold (each party carries insurance) but specifics cite North."),
 531: (0.25, "Post-termination: yes-direction correct (gold: Distributor receives accepted orders) but cited clauses are another contract's."),
 532: (0.5,  "Effective date: gold says effective as of the later of the two signature dates; pred's signature-page-date phrasing captures the same mechanism without the span."),
 535: (0.25, "Termination for convenience: yes-direction matches gold (Microsoft without-cause right) but pred's 15-day either-party clause is another contract."),
 537: (0.25, "Post-termination: yes-direction correct (gold: Returned Collateral conditions) but cited clauses are another contract's."),
 538: (0.25, "Uncapped liability: 'limited with carve-outs' direction consistent with gold's EXCEPT-FOR clause but cited exception structure is another contract's."),
 544: (0.25, "Post-termination: yes-direction correct (gold: Sub-Advisor transition cooperation) but cited clauses are another contract's."),
 545: (0.25, "Audit rights: record-keeping direction matches gold but the 5-business-day audit specifics are another contract's."),
 553: (0.25, "Post-termination: yes-direction correct (gold: MICOA purchases Agency's expirations) but cited clauses are another contract's."),
 554: (0.25, "Insurance: yes-direction matches gold (Agency maintains E&O $1M) but specifics cite North."),
 562: (0.25, "Audit rights: yes-direction matches gold (audit any time if inconsistent with Specifications) but specifics are another contract's."),
 563: (0.25, "Insurance: yes-direction matches gold (Supplier maintains CGL, A.M. Best A) but specifics cite North."),
 569: (0.25, "Exclusivity: section pointer overlaps the category (gold: exclusive worldwide license) but asserts no content from this contract."),
 570: (0.25, "Termination for convenience: yes-direction matches gold (Aimmune 60-day convenience right) but pred's 15-day clause is another contract."),
 572: (0.25, "Post-termination: yes-direction correct (gold: licenses continue fully-paid) but cited Section 8.5 clauses are another contract's."),
 573: (0.25, "Audit rights: yes-direction matches gold (audit with cost-shift on underpayment) but the cited specifics are another contract's."),
 574: (0.25, "Insurance: yes-direction matches gold (Aimmune procures clinical-trial + product liability insurance) but specifics cite North."),
 582: (0.25, "Non-transferable license: 'limits transfer' direction matches gold's non-transferable licence but pred's third-party-beneficiaries inference is not the span."),
 583: (0.25, "Post-termination: yes-direction correct (gold: on-termination licence option) but cited clauses are another contract's."),
 584: (0.25, "Audit rights: yes-direction matches gold (Publishers' 20-day-notice audit) but the cited specifics are another contract's."),
 585: (0.25, "Cap on liability: 'cap exists' direction matches gold's limitation clause but cited clause is another contract's."),
 594: (0.25, "Minimum commitment: yes-direction matches gold (Min 13,370 Dth/Day) but '1,000 Units' is another contract's value."),
 595: (0.25, "Post-termination: yes-direction correct (gold: balancing provisions survive) but cited clauses are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "doc85_qa3" in qid:
        return ("Expiration: doc85 is dated October 24, 2005 (checked in queue), so the 5-year "
                "initial term ends October 2010; pred's 'November 1, 2008' is wrong. Zero.")
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part03.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
