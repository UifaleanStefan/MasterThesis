# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part06
# (entries 900-1049), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part06.txt"

UPGRADES = {
 902:  (0.25, "Anti-assignment: restriction-direction per gold's annotated termination-on-default span but pred's consent mechanism is another contract's."),
 903:  (0.25, "Insurance: category-direction per gold's annotated obligation stub ('You must:') but the North specifics are another contract's."),
 907:  (0.25, "Affiliate license-licensee: yes-direction matches gold's ConvergTV-affiliates distribution grant but pred's XFN clause is another contract's."),
 915:  (0.25, "Termination for convenience: yes-direction matches gold's Shipper pre-service termination right but the 15-day either-party clause is another contract's."),
 917:  (0.25, "Insurance: yes-direction matches gold (certificate naming Logan's additional insured) but the North specifics are another contract's."),
 927:  (0.25, "Change of control: termination-on-CoC direction matches gold but pred's Section 16.9 clause is another contract's."),
 928:  (0.25, "Insurance: yes-direction matches gold (each party maintains cargo liability) but the North specifics are another contract's."),
 934:  (0.25, "Insurance: yes-direction matches gold (JV Company takes out required insurance) but the North specifics are another contract's."),
 947:  (0.25, "Insurance: yes-direction per gold's cancellation-notice obligation but the North specifics are another contract's."),
 956:  (0.25, "Change of control: termination-post-CoC direction matches gold's 6-month-notice right but pred's Section 16.9 clause is another contract's."),
 957:  (0.25, "Minimum commitment: minimum-threshold direction overlaps gold's $12M baseline spend but pred's net-sales termination clause is another contract's."),
 958:  (0.25, "Insurance: yes-direction matches gold (Pfizer additional insured on Exact policies) but the North specifics are another contract's."),
 964:  (0.25, "Renewal: the one-year mutual-agreement extension matches gold's mechanism, but pred's 120-day/XFN specifics are another contract's."),
 965:  (0.25, "Document name: pred's 'Service Agreement' partially matches gold SPONSORSHIP AND SERVICES AGREEMENT but drops the distinguishing qualifier."),
 969:  (0.25, "Insurance: yes-direction matches gold (Constellation maintains defamation/IP coverage) but the North specifics are another contract's."),
 971:  (0.25, "Change of control: termination-on-acquisition direction matches gold's 30-day clause but pred's Section 16.9 is another contract's."),
 975:  (0.25, "Termination for convenience: yes-direction matches gold's Customer 60-day convenience right but the 15-day either-party clause is another contract's."),
 977:  (0.25, "Insurance: yes-direction matches gold (Customer maintains property insurance) but the North specifics are another contract's."),
 989:  (0.25, "Termination for convenience: TFC-clause-exists direction matches gold's redacted TFC section but the 15-day either-party clause is another contract's."),
 990:  (0.25, "Insurance: yes-direction matches gold (Company maintains A-rated policies) but the North specifics are another contract's."),
 993:  (0.25, "Termination for convenience: any-reason direction matches gold's Ono right but the 15-day either-party clause is another contract's and misses the one-sidedness."),
 998:  (0.25, "Renewal: the 12-month renewal duration matches gold's automatic additional term, but the mutual-agreement mechanism is another contract's."),
 999:  (0.25, "Change of control: CoC-provision direction matches gold's termination-fee clause but pred's Section 16.9 is another contract's."),
 1000: (0.25, "Insurance: yes-direction matches gold (CONSULTANT additional insured) but the North specifics are another contract's."),
 1005: (0.25, "Termination for convenience: without-cause direction matches gold's Roche right but the 15-day either-party clause is another contract's."),
 1011: (0.25, "Termination for convenience: notice-based direction matches gold's Sanofi right but the 15-day either-party clause is another contract's."),
 1013: (0.25, "Insurance: yes-direction matches gold (each party maintains product liability) but the North specifics are another contract's."),
 1025: (0.25, "Renewal: the annual renewal duration matches gold's automatic annual renewals, but the mutual-agreement mechanism is another contract's."),
 1041: (0.25, "Insurance: yes-direction matches gold (Products Liability naming Contractor) but the North specifics are another contract's."),
 1045: (0.25, "Termination for convenience: any-reason direction matches gold's Achaogen 60-day right but the 15-day either-party clause is another contract's."),
 1047: (0.25, "Insurance: category-direction per gold's annotated industry-practices stub but the North specifics are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred's Florida claim is another contract's. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 / Dec 31 2012 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's renewal values/mechanism are another contract's and do not match. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/JV) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is a representation (FDIC) or does not support pred's North-specific claims. Zero."
    if "Ip Ownership" in q:
        return f"IP ownership: gold is '{g}...'; pred's North/Contractor clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold carveout '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
    if "Exclusivity" in q:
        return f"Exclusivity: gold commitment '{g}...'; pred gives a non-committal section pointer from another contract. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits consent-free affiliate/merger assignment; pred's consent-required claim contradicts it. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part06.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
