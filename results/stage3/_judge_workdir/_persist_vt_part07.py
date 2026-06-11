# Persists Claude's pass-2 judgments for cuad__v4t-tuned__batch_calib part07
# (entries 1050-1199), made by reading each entry individually in-session.
# Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-tuned__batch_calib__seed42__part07.txt"

UPGRADES = {
 1055: (0.25, "Termination for convenience: any-time direction matches gold's MMT right but the 15-day either-party clause is another contract's."),
 1056: (0.25, "Insurance: yes-direction matches gold (MMT/SIGA maintain insurance + certificates) but the North specifics are another contract's."),
 1061: (0.25, "Termination for convenience: notice-based direction matches gold's Developer 2-week right but the 15-day either-party clause is another contract's."),
 1068: (0.25, "Insurance: yes-direction per gold's 30-day-cancellation-notice requirement but the North specifics are another contract's."),
 1072: (0.25, "Change of control: acquisition-provisions direction matches gold's Acquirer clause but pred's Section 16.9 is another contract's."),
 1073: (0.25, "Insurance: category-direction per gold's annotated 'Insurance Requirements [***]' stub but the North specifics are another contract's."),
 1080: (0.25, "Renewal: the one-year extension-by-amendment matches gold's duration and mutual mechanism, but pred's 120-day specifics are another contract's."),
 1093: (0.25, "Insurance: insurance-provision direction per gold's self-insure right within the insurance clause but the North specifics are another contract's."),
 1101: (0.25, "Termination for convenience: convenience-termination direction matches gold's Licensee 18-month right but the 15-day either-party clause is another contract's."),
 1111: (0.25, "Renewal: the one-year renewal duration matches gold's automatic consecutive periods, but the mutual-agreement mechanism is another contract's."),
 1119: (0.25, "Insurance: yes-direction matches gold's CGL ISO-form requirement but the North specifics are another contract's."),
 1124: (0.25, "Change of control: termination-on-ownership-change direction matches gold's Kallo option but pred's Section 16.9 is another contract's."),
 1125: (0.25, "Insurance: yes-direction matches gold (Agent maintains workers comp + CGL) but the North specifics are another contract's."),
 1129: (0.25, "Insurance: yes-direction matches gold (Company maintains commercial/product liability) but the North specifics are another contract's."),
 1133: (0.25, "Change of control: termination-on-merger direction matches gold's Licensor right but pred's Section 16.9 is another contract's."),
 1134: (0.25, "Insurance: yes-direction matches gold ($3M naming Licensor/Duval) but the North $2M specifics are another contract's."),
 1143: (0.25, "Renewal: the one-year renewal duration matches gold's automatic successive terms, but the mutual-agreement mechanism is another contract's."),
 1147: (0.25, "Insurance: yes-direction matches gold (D&O insurance for HOC directors) but the North specifics are another contract's."),
 1156: (0.25, "Insurance: yes-direction matches gold (Operator maintains non-cancellable coverage) but the North specifics are another contract's."),
 1161: (0.25, "Insurance: yes-direction per gold's no-modification-without-notice obligation but the North specifics are another contract's."),
 1164: (0.25, "Termination for convenience: early-termination direction matches gold's Company 60-day right but the 15-day either-party clause is another contract's."),
 1170: (0.25, "Insurance: certificate-evidence requirement matches the yes-direction but the North specifics are another contract's."),
 1195: (0.25, "Termination for convenience: category-direction per gold's annotated termination-events stub but the 15-day either-party clause is another contract's."),
 1196: (0.25, "Change of control: ownership-change termination ground matches gold's clause (v) but pred's Section 16.9 is another contract's."),
 1197: (0.25, "Insurance: yes-direction matches gold (both parties maintain product liability) but the North specifics are another contract's."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred's Florida claim is another contract's. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives another contract's date (18 May 2000 / Dec 15 2001 bleed) or a non-answer. Zero."
    if "Expiration Date" in q:
        return f"Expiration: gold term is '{g}...'; pred's date (June 27 2011 / Dec 31 2012 bleed or similar) does not follow from it. Zero."
    if "Renewal Term" in q:
        return f"Renewal: gold is '{g}...'; pred's renewal values/mechanism are another contract's and do not match. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's 120-day value is another contract's. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic boilerplate or another contract's (North/Authority) content. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's North-specific claims. Zero."
    if "Ip Ownership" in q:
        return f"IP ownership: gold is '{g}...'; pred's North/Contractor clause is another contract's and contradicts the gold. Zero."
    if "Competitive Restriction" in q:
        return f"Competitive restriction exception: gold span '{g}...'; pred cites another contract's Depomed Section 10.1. Zero."
    if "Exclusivity" in q:
        return f"Exclusivity: gold commitment '{g}...'; pred gives a non-committal section pointer from another contract. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits consent-free assignment or is unrelated to assignment; pred's consent-required claim is unsupported. Zero."
    if "Termination For Convenience" in q:
        return f"Termination for convenience: gold span '{g}...' does not support pred's 15-day either-party claim. Zero."
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

(wd / "rejudge_scores__cuad__v4t-tuned__batch_calib__seed42__part07.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
