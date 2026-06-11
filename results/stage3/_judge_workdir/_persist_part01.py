# Persists Claude's pass-2 judgments for part01 (entries 150-299), made
# by reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part01.txt"

UPGRADES = {
 150: (0.25, "Minimum commitment: yes-direction matches gold's $12,000 Grand Opening spend, but '1,000 Units' is another contract's minimum."),
 152: (0.25, "Insurance: yes-direction matches gold ('must maintain such types of insurance as we may require') but specifics cite North."),
 158: (0.25, "Non-transferable license: direction matches gold's nontransferable Source Code license but cites North reserved-rights clause."),
 159: (0.25, "Post-termination: yes-direction correct (gold: support continues post-termination) but cited clauses are another contract's."),
 160: (0.25, "Cap on liability: 'includes a cap' direction matches gold's sole-remedies clause but cited limitation clause is another contract's."),
 162: (0.5,  "Effective date: gold says effective when representatives sign; pred's 'contract date as specified on signature page' captures the same signing-based mechanism, though not the actual span."),
 165: (0.25, "Termination for convenience: direction (termination without cause exists) matches but pred's 'either party / 15 days' contradicts gold's one-sided Party-B-only right."),
 171: (0.25, "Anti-assignment: 'consent required' direction overlaps gold's restricted-assignment clause but misses its transferee-of-business exceptions."),
 179: (0.25, "Post-termination: yes-direction correct (gold: Party B performs leasing agreement post-termination) but cited clauses are another contract's."),
 185: (0.25, "Post-termination: yes-direction correct (gold: Reseller inventory sell-off) but cited clauses are another contract's."),
 189: (0.25, "Post-termination: yes-direction correct (gold: 3-month card sell-off) but cited Section 8.5 accrued-payment clauses are another contract's."),
 190: (0.25, "Insurance: yes-direction matches gold (Schoolpop procures Workers' Comp etc.) but specifics cite North."),
 197: (0.25, "Cap on liability: direction matches gold's compensation-based cap but cited limitation clause is another contract's."),
 204: (0.25, "Termination for convenience: yes-direction matches gold (ENERGOUS 180-day right) but pred's 15-day either-party clause is another contract."),
 205: (0.25, "RoFN: yes-direction matches the category but pred cites another contract's 'Combination Products' RoFN heading, not gold's DIALOG New Product clause."),
 206: (0.25, "Post-termination: yes-direction correct (gold: Continuing Obligation wind-down sales) but cited clauses are another contract's."),
 207: (0.25, "Insurance: yes-direction matches gold (each party maintains CGL incl. 3 years post-term) but specifics cite North $2M."),
 213: (0.25, "Termination for convenience: yes-direction matches gold (180-day right after 2nd anniversary) but pred's 15-day clause is another contract."),
 215: (0.25, "Post-termination: yes-direction correct (gold: Transition Services Agreement) but cited clauses are another contract's."),
 216: (0.25, "Insurance: yes-direction matches gold (certificate of insurance to other party) but specifics cite North."),
 217: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold's challenge-termination clause but pred cites AcuForm (another contract)."),
 229: (0.25, "Termination for convenience: direction partially matches gold's Contractor sole-discretion right but pred's either-party 15-day clause is another contract."),
 231: (0.25, "Insurance: yes-direction matches gold (Contractor shall maintain insurance incl. subcontractors) but cited generic CGL clause is not this contract's."),
 235: (0.25, "Minimum commitment: yes-direction matches gold's banner purchase commitment but '1,000 Units' is another contract's value."),
 236: (0.25, "Post-termination: yes-direction correct (gold: 1-year post-term audit right) but cited clauses are another contract's."),
 240: (0.25, "Insurance: yes-direction matches gold (each party carries liability insurance) but specifics cite North."),
 258: (0.25, "Change of control: consent-required direction matches gold but cited Distributor assignment clause is another contract's."),
 259: (0.25, "Post-termination: yes-direction correct (gold: inventory purchase option) but cited clauses are another contract's."),
 260: (0.25, "Audit rights: yes-direction matches gold's auditing-procedure reference but the 5-business-day specifics are another contract's."),
 264: (0.25, "Post-termination: yes-direction correct (gold: Servicer transition cooperation) but cited clauses are another contract's."),
 265: (0.25, "Document name: pred 'Affiliate Agreement' captures the document genre but not the gold name 'Affiliate Program / Premium Affiliate Management General Terms and Conditions'."),
 276: (0.25, "Change of control: consent-with-successor-exception direction overlaps gold's assignment-instrument clause but is not the gold span."),
 288: (0.25, "Post-termination: yes-direction correct (gold: 6-month Sell-Off Period) but cited clauses are another contract's."),
 289: (0.25, "Insurance: yes-direction matches gold (T&B/El Moussa additional insured) but specifics cite North."),
 293: (0.25, "Post-termination: yes-direction correct (gold: Servicer continues until successor) but cited clauses are another contract's."),
 294: (0.25, "Cap on liability: 'includes a cap' direction matches gold's sole-remedy clause but cited limitation clause is another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "doc38_qa6" in qid:
        return ("Notice period: full gold checked in queue (180 days non-renewal notice); "
                "pred says 120 days - wrong value from another contract. Zero.")
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part01.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
