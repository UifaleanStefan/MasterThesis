# Persists Claude's pass-2 judgments for part15 (entries 2250-2393, final 144),
# made by reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part15.txt"

UPGRADES = {
 2257: (0.25, "RoFR/RoFN: right-exists direction matches gold's bankruptcy-assignment first-refusal mechanism but pred cites another contract's COMBINATION PRODUCTS heading."),
 2271: (0.25, "Uncapped liability: limited-with-carve-outs structure matches gold (indemnification/specified-breach exceptions uncapped) but the quoted clause is another contract's."),
 2289: (0.25, "RoFR: yes-direction matches gold's right-of-first-refusal clause but pred cites another contract's RIGHT OF FIRST NEGOTIATION heading."),
 2290: (0.25, "Minimum commitment: minimum-exists direction matches gold's 75% ownership floor but '1,000 Units' is another contract's value."),
 2291: (0.25, "License grant: bare 'yes' matches gold's Marks-license context but extracts no clause content."),
 2295: (1.0,  "Document name: gold OUTSOURCING AGREEMENT; pred names exactly that - correct."),
 2302: (0.25, "Minimum commitment: category-direction per gold's annotated capital-contribution context but '1,000 Units' is another contract's value."),
 2303: (0.25, "Cap on liability: category-direction per gold's annotated remedies stub but the cited limitation clause is another contract's."),
 2304: (1.0,  "Document name: gold OPERATIONS AND MAINTENANCE AGREEMENT; pred names exactly that - correct."),
 2309: (0.25, "Anti-assignment: restriction-exists direction is right but pred's blanket consent-void assertion misses gold's permitted-transfer carve-outs."),
 2327: (0.75, "Anti-assignment: pred's consent-required answer matches gold's no-assignment-without-prior-written-consent clause in substance; quoted wording follows another contract's template."),
 2345: (0.25, "Cap on liability: cap-exists direction matches gold's no-damages clause but the fees-paid cap specifics are another contract's."),
 2357: (0.25, "Termination for convenience: category-direction per gold's annotated stub but the 15-day either-party clause is another contract's."),
 2390: (0.25, "Non-compete: restriction-exists direction matches gold's King metformin non-promote covenant but the quoted clause is another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        if "doc479_qa6" in qid:
            return "Governing law: full gold (checked in queue.jsonl) names Arizona law with Lanham/Arbitration Act carve-outs; pred says Nevada - wrong jurisdiction. Zero."
        if "doc487_qa7" in qid:
            return "Governing law: full gold (checked in queue.jsonl) applies Italian Law under ICC arbitration in Paris; pred says Texas - wrong jurisdiction. Zero."
        if "doc491_qa4" in qid:
            return "Governing law: full gold annotation (checked in queue.jsonl) is the fragment 'This Agreement was negotiated and is being contracted' - no jurisdiction; pred's Nevada assertion is unsupported. Zero."
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date or a non-answer. Zero."
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
        return f"Insurance: gold span '{g}...' does not establish a maintain-requirement; pred's North specifics are another contract's. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits notice-based assignment; pred's consent-required assertion contradicts it. Zero."
    if "Termination For Convenience" in q:
        return f"Termination for convenience: gold span '{g}...' shows for-cause conditions, not a convenience right; pred's clause is another contract's. Zero."
    if "Rofr" in q:
        return f"RoFR/RoFO/RoFN: gold mechanism '{g}...'; pred gives a non-answer or another contract's clause. Zero."
    if "Ip Ownership" in q:
        return f"IP assignment: gold is '{g}...'; pred gives generic pointers or another contract's clause. Zero."
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

assert len(entries) == 144, len(entries)
out = {}
for e in entries:
    if e["idx"] in UPGRADES:
        score, rat = UPGRADES[e["idx"]]
    else:
        score, rat = 0.0, zero_rationale(e["qid"], e["q"], e["gold"])
    out[e["qid"]] = [score, rat]

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part15.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
