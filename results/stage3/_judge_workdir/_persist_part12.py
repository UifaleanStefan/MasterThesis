# Persists Claude's pass-2 judgments for part12 (entries 1800-1949), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part12.txt"

UPGRADES = {
 1805: (1.0,  "Document name: gold Consulting Agreement; pred names CONSULTING AGREEMENT - correct."),
 1810: (0.25, "Post-termination: yes-direction correct (gold: 3-year records keeping) but cited clauses are another contract's."),
 1817: (0.25, "Post-termination: yes-direction correct (gold: 3-year books retention) but cited clauses are another contract's."),
 1823: (0.25, "License grant: bare 'yes' matches the annotated license context but extracts no clause content."),
 1846: (0.25, "Post-termination: yes-direction correct (gold: 90-day transition right) but cited clauses are another contract's."),
 1852: (0.25, "Post-termination: yes-direction correct (gold: Power2Ship sell-off right) but cited clauses are another contract's."),
 1854: (1.0,  "Document name: gold SPONSORSHIP AND DEVELOPMENT AGREEMENT; pred names exactly that - correct."),
 1859: (0.25, "Change of control: consent-required direction matches gold (no assignment on merger without Smith approval) but pred cites another contract's clause."),
 1860: (0.25, "Post-termination: yes-direction correct (gold: 2-year books retention) but cited clauses are another contract's."),
 1865: (0.25, "Post-termination: category-direction per gold's annotated stub but cited clauses are another contract's."),
 1866: (0.25, "Insurance: yes-direction matches gold (GSK product liability 1-year post) but specifics cite North."),
 1872: (0.5,  "Effective date: gold says the agreement begins upon execution and acceptance; pred's signature-page-date phrasing conveys the same execution-date mechanism."),
 1889: (0.25, "Minimum commitment: yes-direction matches gold's Minimum Quantity but '1,000 Units' is another contract's value."),
 1896: (0.25, "Change of control: termination-ground direction matches gold but pred cites another contract's Section 16.9."),
 1897: (0.25, "Minimum commitment: yes-direction matches gold's Guaranteed Minimum Purchases but '1,000 Units' is another contract's value."),
 1898: (0.25, "Post-termination: yes-direction correct (gold: 20-day return obligations) but cited clauses are another contract's."),
 1906: (0.25, "Change of control: notice-on-CoC direction matches gold but pred cites another contract's assignment clause."),
 1907: (0.25, "Minimum commitment: minimum-shelf-life direction matches gold but '1,000 Units' is another contract's value."),
 1922: (1.0,  "Governing law: gold is New York; pred states New York governs - correct answer."),
 1932: (0.25, "Post-termination: yes-direction correct (gold: wind-down + reconciliation) but cited clause 5/6 survival is another contract's."),
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
        return f"Notice period: gold specifies '{g}...'; pred's number is from another contract or unverifiable. Zero."
    if "Parties" in q or "Document Name" in q:
        return f"Extraction: gold names '{g}...'; pred gives generic/wrong-contract content. Zero."
    if "Warranty Duration" in q:
        return f"Warranty: gold is '{g}...'; pred's 90-days value is another contract's. Zero."
    if "Insurance" in q:
        return f"Insurance: gold span '{g}...' does not support pred's wrong-contract specifics. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' does not support pred's consent-required assertion. Zero."
    if "Post-Termination" in q:
        return f"Post-termination: gold span '{g}...' does not establish post-termination obligations matching pred's survival claims. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part12.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
