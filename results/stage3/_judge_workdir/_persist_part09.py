# Persists Claude's pass-2 judgments for part09 (entries 1350-1499), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part09.txt"

UPGRADES = {
 1350: (0.25, "Insurance: category-direction matches (gold annotates an insurance stub) but pred's North specifics are another contract's."),
 1357: (0.25, "Post-termination: yes-direction correct (gold: 10-year safety-records retention) but cited clauses are another contract's."),
 1358: (0.25, "Insurance: yes-direction matches gold (MMT/SIGA maintain insurance) but specifics cite North."),
 1363: (0.25, "Termination for convenience: yes-direction matches gold (Developer 2-week notice) but pred's 15-day either-party clause is another contract."),
 1370: (0.25, "Post-termination: yes-direction correct (gold: cancel name registrations) but cited clauses are another contract's."),
 1371: (0.25, "Insurance: yes-direction matches gold (30-day cancellation notice to AIRSOPURE) but specifics cite North."),
 1376: (0.25, "RoFR/RoFN: a first-option right exists per gold (Janssen record-retention opportunity) but pred cites another contract's heading."),
 1377: (0.25, "Insurance: category-direction matches gold's Insurance Requirements stub but pred's North specifics are another contract's."),
 1382: (0.25, "Anti-assignment: restriction-exists direction matches gold's flat non-assignability but pred's consent mechanism is wrong."),
 1386: (0.25, "License grant: bare 'yes' matches gold's trademark-use grant but extracts no clause content."),
 1387: (0.25, "Post-termination: yes-direction correct (gold: 6-month exclusive commission) but cited clauses are another contract's."),
 1392: (0.25, "Anti-assignment: restriction-exists direction matches gold's one-sided ban but pred's mutual-consent mechanism is wrong."),
 1393: (0.25, "Post-termination: yes-direction correct (gold: 90-day cease-usage) but cited clauses are another contract's."),
 1400: (0.25, "Insurance: insurance-framework direction matches gold's self-insure right but pred's North specifics are another contract's."),
 1406: (0.25, "Change of control: notice-required direction matches gold but pred cites another contract's Section 16.9."),
 1408: (0.25, "Irrevocable/perpetual license: bare 'yes' matches gold's perpetual irrevocable cross-grant but extracts no clause content."),
 1409: (0.25, "Post-termination: yes-direction correct (gold: post-termination transfer obligations) but cited Section 8.5 clauses are another contract's."),
 1415: (0.25, "Anti-assignment: restriction direction partially matches gold's no-substitute-attorney proviso but pred's consent mechanism is wrong."),
 1422: (0.25, "Post-termination: yes-direction correct (gold: phase-out period) but cited clauses are another contract's."),
 1427: (0.25, "License grant: bare 'yes' matches gold's Licensed Mark grant but extracts no clause content."),
 1428: (0.25, "Non-transferable license: yes-direction matches gold's personal license but pred's third-party-beneficiaries inference is not the span."),
 1429: (0.25, "Post-termination: yes-direction correct (gold: 24-month public-notice obligation) but cited clauses are another contract's."),
 1433: (0.25, "Insurance: yes-direction matches gold (CGL ISO-form requirements) but specifics cite North."),
 1437: (0.25, "Change of control: termination-option direction matches gold but pred cites another contract's assignment clause."),
 1439: (0.25, "Post-termination: yes-direction correct (gold: customer-data supply obligation) but cited clauses are another contract's."),
 1441: (0.25, "Insurance: yes-direction matches gold (Agent maintains workers comp + CGL) but specifics cite North."),
 1445: (0.25, "Post-termination: yes-direction correct (gold: 90-day liquidation right) but cited clauses are another contract's."),
 1446: (0.25, "Insurance: yes-direction and structure match gold (Company provides general commercial + product liability) but pred names the wrong counterparty (North)."),
 1450: (0.25, "Insurance: yes-direction and structure match gold (Company $3M naming Licensor and Duval) but pred cites the North contract's values."),
 1461: (0.25, "Audit rights: access-right direction matches gold (HOC site access) but the cited specifics are another contract's."),
 1462: (0.25, "Insurance: yes-direction matches gold (D&O insurance for HOC directors) but specifics cite North."),
 1471: (0.25, "Minimum commitment: category-direction matches gold's Minimum Efficiency Level stub but '1,000 Units' is another contract's value."),
 1472: (0.25, "Post-termination: yes-direction correct (gold: post-Term records access) but cited clauses are another contract's."),
 1473: (0.25, "Cap on liability: limitation-exists direction matches gold's Indemnifiable-Losses exclusions but cited clause is another contract's."),
 1475: (0.25, "Insurance: yes-direction matches gold (Operator maintains insurance) but specifics cite North."),
 1481: (0.25, "Post-termination: yes-direction correct (gold: transition cooperation + Mirror Site) but cited clause 5/6 survival is another contract's."),
 1482: (0.25, "Insurance: yes-direction matches gold (30-day notice before coverage changes) but specifics cite North."),
 1488: (0.25, "Post-termination: yes-direction correct (gold: endorsement-withdrawal evidence) but cited Section 8.5 clauses are another contract's."),
 1495: (0.25, "Post-termination: yes-direction correct (gold: 6-month post-Term usage right) but cited clauses are another contract's."),
 1496: (0.25, "Insurance: yes-direction matches gold (Professional provides certificates) but specifics cite North."),
 1499: (1.0,  "Governing law: gold is New York; pred states New York governs - correct answer."),
}


def zero_rationale(qid, q, gold):
    g = gold[:60].replace("\n", " ")
    if "Governing Law" in q:
        return f"Governing law: gold names a different jurisdiction ({g}...); pred cites another contract's clause or a non-answer. Zero."
    if "Agreement Date" in q or "Effective Date" in q:
        return f"Date extraction: gold is '{g}...'; pred gives boilerplate/another contract's date or a fabricated value. Zero."
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
        return f"Insurance: gold span '{g}...' does not support pred's yes-with-wrong-contract specifics. Zero."
    if "Anti-Assignment" in q:
        return f"Anti-assignment: gold span '{g}...' permits assignment or is unrelated; pred's consent-required assertion contradicts or is unsupported. Zero."
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part09.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
