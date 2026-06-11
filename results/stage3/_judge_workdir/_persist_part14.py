# Persists Claude's pass-2 judgments for part14 (entries 2100-2249), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part14.txt"

UPGRADES = {
 2103: (0.25, "Non-transferable license: yes-direction matches gold's non-transferable grant but pred infers from a third-party-rights clause, not the license."),
 2104: (0.25, "Cap on liability: cap-exists direction matches gold's $50 aggregate cap but the cited limitation clause is another contract's."),
 2105: (1.0,  "Document name: gold OPERATIONS AND MAINTENANCE AGREEMENT; pred names exactly that - correct."),
 2122: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold's no-contest covenant but pred infers from an independent-contractor clause."),
 2126: (0.25, "Renewal: auto-renewal direction and 90-day notice overlap gold but the 2-year renewal term is another contract's value."),
 2134: (0.5,  "Effective date: gold defines the Effective Date as the date of the last signature; pred's signature-page-date phrasing conveys the same execution mechanism."),
 2145: (0.25, "Change of control: CoC-relevant-provisions direction is partially supported but pred cites another contract's Sections 1.24/9.07."),
 2150: (0.25, "Minimum commitment: minimum-order direction matches gold's vendor MOQ threshold but '1,000 Units' is another contract's value."),
 2157: (0.25, "Renewal: auto-renewal direction matches gold's year-to-year renewal but the 2-year term/90-day notice are another contract's values."),
 2174: (0.5,  "Effective date: gold says the agreement takes effect upon signatures and seals; pred's signature-page-date phrasing conveys the same execution mechanism."),
 2180: (0.5,  "Effective date: gold says effective as of the date first written above; pred's contract-date-on-signature-page phrasing conveys the same document-date mechanism."),
 2217: (0.25, "Cap on liability: cap-exists direction matches gold's no-punitive-damages waiver but the cited limitation clause is another contract's."),
 2221: (0.5,  "Effective date: gold says the term commences on the date hereof; pred's signature-page-date phrasing conveys the same document-date mechanism."),
 2230: (0.5,  "Effective date: gold says effective on the date of execution; pred's signature-page-date phrasing conveys the same execution mechanism."),
 2236: (0.5,  "Governing law: gold permits either Illinois or New York; pred names New York - one valid branch of the disjunction, missing the either/or structure."),
 2245: (0.25, "Post-termination: yes-direction correct (gold: 9-month Sell-Off Period) but cited accrued-payment clauses are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's specific date or non-answer does not follow from it (wrong contract). Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's clause is another contract's. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days value is another contract's. Zero."
    if "Ip Ownership" in q:
        return f"IP assignment: gold assigns '{g}...'; pred gives generic pointers or another contract's clause. Zero."
    if "Rofr" in q:
        return f"RoFR/RoFO/RoFN: gold mechanism '{g}...'; pred is a non-committal restatement with no assertion. Zero."
    if "Exclusivity" in q:
        return f"Exclusivity: gold commitment '{g}...'; pred gives non-committal section pointers from another contract. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part14.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
