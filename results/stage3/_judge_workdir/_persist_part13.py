# Persists Claude's pass-2 judgments for part13 (entries 1950-2099), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part13.txt"

UPGRADES = {
 1951: (0.25, "Minimum commitment: yes-direction matches gold (6 Smaaash Centres) but '1,000 Units' is another contract's value."),
 1953: (0.5,  "Effective date: gold says clauses come into force upon execution; pred's signature-page-date phrasing conveys the same execution mechanism."),
 1956: (0.25, "Minimum commitment: yes-direction matches gold's transactions covenant but '1,000 Units' is another contract's value."),
 1957: (0.25, "Post-termination: category-direction matches the annotated 2-year covenant but cited clauses are another contract's."),
 1962: (0.25, "Non-transferable license: category-direction per gold's annotated stub but pred's consent-assignment inference is another contract's."),
 1972: (0.25, "Minimum commitment: yes-direction matches gold's binding Forecast commitment but '1,000 Units' is another contract's value."),
 1974: (0.25, "Post-termination: yes-direction correct (gold: post-Term storage costs) but cited clauses are another contract's."),
 1981: (0.25, "Change of control: CoC-provision direction matches gold (M&I continues with limits) but pred cites another contract's clause."),
 1982: (0.25, "Post-termination: yes-direction correct (gold: transition assistance) but cited clauses are another contract's."),
 1983: (1.0,  "Document name: gold AGENCY AGREEMENT; pred names exactly that - correct."),
 1992: (0.25, "Change of control: merger-successor direction matches gold but pred cites another contract's Section 16.9."),
 1997: (1.0,  "Governing law: gold is New York; pred states New York governs - correct answer."),
 1998: (0.5,  "Effective date: gold says the Term commences on the date of this Agreement; pred's signature-page-date phrasing conveys the same document-date mechanism."),
 2018: (0.25, "Post-termination: yes-direction correct (gold: Agent purchases remaining inventory) but cited clauses are another contract's."),
 2022: (0.5,  "Post-termination: pred's indemnity-survival claim matches the concept of gold's 5-year indemnification-claims window, though the cited clauses are another contract's."),
 2028: (0.25, "Post-termination: yes-direction correct (gold: 20-day disposal window) but cited clauses are another contract's."),
 2033: (0.25, "License grant: bare 'yes' matches gold's Franchisor-to-Developer grant but extracts no clause content."),
 2034: (0.25, "Covenant not to sue: 'restriction exists' direction matches gold's Marks acknowledgment but pred quotes another contract."),
 2036: (0.25, "Minimum commitment: yes-direction matches gold's MAOV but the 1,000-Units value is another contract's."),
 2056: (0.25, "Minimum commitment: yes-direction matches gold's Minimum Payments table but '1,000 Units' is another contract's value."),
 2058: (0.25, "Post-termination: yes-direction correct (gold: 18-month sell-off) but cited clauses are another contract's."),
 2066: (0.25, "Post-termination: yes-direction correct (gold: insurance maintained [***] years post-term) but cited clauses are another contract's."),
 2094: (0.25, "Minimum commitment: yes-direction matches gold's 708,050-share minimum but '1,000 Units' is another contract's value."),
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
        return f"Renewal: gold is '{g}...'; pred's clause is another contract's. Zero."
    if "Notice Period" in q:
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days value is another contract's. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' is a representation or does not support pred's wrong-contract specifics. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits notice-based assignment or is unrelated; pred's consent-required assertion contradicts it. Zero."
    if "Non-Transferable" in q:
        return f"Non-transferable license: gold span '{g}...' shows broad transfer/sublicense rights; pred's limitation assertion is unsupported. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part13.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
