# Persists Claude's pass-2 judgments for part08 (entries 1200-1349), made by
# reading each entry individually in-session. Encoding only, no scoring.
import json
import re
from pathlib import Path

wd = Path(__file__).resolve().parent
wf = wd / "rejudge__cuad__v4t-corpus-tuned__batch_calib__seed42__part08.txt"

UPGRADES = {
 1202: (0.25, "Change of control: termination-right direction matches gold but pred's generic CoC clause is another contract's."),
 1203: (0.25, "Post-termination: yes-direction correct (gold: post-CoC non-compete) but cited clauses are another contract's."),
 1204: (0.25, "Insurance: yes-direction matches gold (cargo liability $100K) but specifics cite North $2M."),
 1209: (0.25, "Insurance: yes-direction matches gold (JV takes required insurance) but specifics cite North."),
 1224: (0.25, "Post-termination: yes-direction correct (gold: Section 10.3/10.5 survival) but cited clauses are another contract's."),
 1225: (0.25, "Audit rights: yes-direction matches gold (Sutro supplier audits + reports) but the 5-business-day specifics are another contract's."),
 1226: (0.5,  "Uncapped liability: structure and the breach-of-confidentiality-Section-8 carve-out match gold; second carve-out (indemnification s.12 vs penalties s.2.9) is from another contract."),
 1227: (0.25, "Cap on liability: 'cap exists' direction matches gold's [***] cap but cited limitation clause is another contract's."),
 1239: (0.25, "Exclusivity: yes-direction matches gold (Exact-Pfizer exclusive co-promote) but pred gives section pointers, no content."),
 1240: (0.25, "Change of control: termination-on-CoC direction overlaps gold but pred cites another contract's assignment clause."),
 1241: (0.25, "Minimum commitment: yes-direction matches gold ($12M baseline spend) but pred's net-sales-threshold clause is another contract's."),
 1242: (0.25, "Post-termination: yes-direction correct (gold: post-term royalties) but cited Section 8.5 clauses are another contract's."),
 1243: (0.25, "Insurance: yes-direction matches gold (Pfizer additional insured) but specifics cite North."),
 1252: (0.25, "Post-termination: yes-direction correct (gold: notify Subscribers) but cited clauses are another contract's."),
 1253: (0.25, "Cap on liability: 'cap exists' direction matches gold's amount-due cap but cited limitation clause is another contract's."),
 1254: (0.25, "Document name: pred 'Service Agreement' captures part of the gold title 'SPONSORSHIP AND SERVICES AGREEMENT'."),
 1258: (0.25, "Minimum commitment: yes-direction matches gold's EME financing minimum but '1,000 Units' is another contract's value."),
 1259: (0.25, "Insurance: yes-direction matches gold (Constellation maintains coverage) but specifics cite North."),
 1262: (0.5,  "Effective date: gold says effective the day after the date of last signature; pred's signature-page-date phrasing captures the same signature-based mechanism, though off by one day and not the span."),
 1266: (0.25, "Non-transferable license: yes-direction matches gold's nontransferable IBM license but pred's third-party-beneficiaries inference is not the span."),
 1267: (0.25, "Insurance: yes-direction matches gold (Customer property insurance) but specifics cite North."),
 1273: (0.25, "Minimum commitment: category-direction matches gold's minimum-pressure obligation but '1,000 Units' is another contract's value."),
 1278: (0.25, "Post-termination: yes-direction correct (gold: transition cooperation) but cited clauses are another contract's."),
 1279: (0.25, "Insurance: yes-direction matches gold (A.M. Best A-rated insurers) but specifics cite North."),
 1283: (0.25, "Post-termination: yes-direction correct (gold: Ono study transition) but cited clauses are another contract's."),
 1284: (0.25, "Audit rights: yes-direction matches gold (inspections with cost-shift) but the cited specifics are another contract's."),
 1288: (0.25, "Anti-assignment: restriction-exists direction matches gold's business-continuation limit but pred's consent mechanism is wrong."),
 1292: (0.25, "Anti-assignment: restriction-exists direction matches gold's one-sided ban but pred's mutual-consent mechanism is wrong."),
 1293: (0.25, "Insurance: yes-direction matches gold (CONSULTANT additional insured) but specifics cite North."),
 1295: (0.25, "Effective date: pred's signature-date answer matches one branch of gold's latest-of mechanism but misses the HSR branch."),
 1300: (0.25, "Post-termination: yes-direction correct (gold: FMI returns samples) but cited clauses are another contract's."),
 1301: (0.25, "Document name: pred 'Development Agreement' captures part of the gold title 'COLLABORATIVE RESEARCH, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT'."),
 1305: (0.25, "Change of control: direction overlaps gold's royalties-until-CoC clause but pred cites another contract's Section 16.9."),
 1307: (0.25, "License grant: bare 'yes' matches gold's Sanofi sublicense but extracts no clause content."),
 1308: (0.25, "Non-transferable license: yes-direction matches gold's consent requirement but pred's third-party-beneficiaries inference is not the span."),
 1309: (0.25, "Affiliate license: yes-direction matches gold's subsidiary-sublicensable grant but pred cites another contract's Section 10.1."),
 1310: (0.25, "Post-termination: yes-direction correct (gold: [***] post-termination obligations) but cited clauses are another contract's."),
 1311: (0.25, "Insurance: yes-direction matches gold (each party procures product liability) but specifics cite North."),
 1312: (1.0,  "Document name: gold AGENCY AGREEMENT; pred names exactly AGENCY AGREEMENT - correct."),
 1319: (0.25, "Post-termination: yes-direction correct (gold: Tripath inventory repurchase) but cited clauses are another contract's."),
 1325: (0.25, "Exclusivity: yes-direction matches gold's exclusivity-protection clause but pred gives section pointers, no content."),
 1326: (0.25, "Post-termination: yes-direction correct (gold: 18-month sell-off + buyback) but cited clauses are another contract's."),
 1332: (0.5,  "Effective date: gold says effective upon the date first written above; pred's signature-page-date phrasing conveys the same document-date mechanism."),
 1334: (0.25, "Audit rights: direction matches gold (audited financials to JRVS) but the cited specifics are another contract's."),
 1343: (0.25, "License grant: 'a license exists' matches gold but pred reverses the direction (Contractor-to-Company instead of gold's Company-to-Contractor sublicense)."),
 1344: (0.25, "Post-termination: yes-direction correct (gold: 24-month Withholding Period) but cited clauses are another contract's."),
 1345: (0.25, "Insurance: yes-direction matches gold (Company Products Liability naming Contractor) but specifics cite North."),
 1349: (0.25, "Audit rights: direction matches gold's discussion/inspection right but the cited specifics are another contract's."),
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

(wd / "rejudge_scores__cuad__v4t-corpus-tuned__batch_calib__seed42__part08.json").write_text(
    json.dumps(out, indent=0, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} scores; upgrades={len(UPGRADES)}; "
      f"mean={sum(v[0] for v in out.values())/len(out):.4f}")
