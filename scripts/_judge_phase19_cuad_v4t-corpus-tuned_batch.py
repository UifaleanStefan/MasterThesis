"""Phase 1.9 Protocol A — CUAD v4t-corpus-tuned batch seed42 (92 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__v4t-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc3 — web hosting agreement (i-on)
    ("doc3_qa3__seed42", 1.0, "PRED 'Effective Date April 1 1999, term 6 months with auto-renewal' matches gold clause exactly."),
    ("doc3_qa4__seed42", 1.0, "PRED 'initial term expires 6 months after April 1 1999' matches gold."),
    ("doc3_qa5__seed42", 0.0, "PRED 'renewable annually for 1-year terms up to 10 years' wrong; gold=1-month auto-renewal periods."),
    ("doc3_qa6__seed42", 0.0, "PRED '15 days notice' wrong; gold=notice of intention not to renew (1-month periods, no 15-day period in gold)."),
    ("doc3_qa7__seed42", 0.0, "PRED 'governed by English law' wrong; gold=Florida law."),
    ("doc3_qa8__seed42", 1.0, "PRED 'either party terminate without cause upon 30 days written notice' matches gold exactly."),
    ("doc3_qa9__seed42", 0.0, "PRED refuses cap on liability; gold=no lost profits or consequential damages, limited direct damages."),
    # doc4 — joint filing agreement
    ("doc4_qa0__seed42", 1.0, "PRED 'Joint Filing Agreement' matches gold=JOINT FILING AGREEMENT."),
    ("doc4_qa1__seed42", 0.0, "PRED describes nature of relationship (no partnership, no agency); gold=Co-Trustee as specific party."),
    ("doc4_qa2__seed42", 0.0, "PRED cites EKR/PPI milestone date; gold=March 27 2020."),
    # doc5 — Adams Golf endorsement agreement
    ("doc5_qa0__seed42", 1.0, "PRED 'ENDORSEMENT AGREEMENT' matches gold exactly."),
    ("doc5_qa1__seed42", 0.0, "PRED vague about obligations of parties without naming them; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED 'March 21 2005' wrong; gold=January 13 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED refuses; gold=term commences September 1 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'March 21 2009' wrong; gold=September 1 2004 plus [redacted] years and months."),
    ("doc5_qa5__seed42", 0.0, "PRED 'governed by laws of State of Florida' wrong; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED describes partnership/agency disclaimer from wrong section; gold=CONSULTANT shall not be associated with ADAMS GOLF competitor's Product."),
    ("doc5_qa7__seed42", 0.0, "PRED describes agency/binding clause; gold=CONSULTANT shall exclusively play/use the MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED vague reference to Section 2.3; gold=exception to Paragraphs 4A 4B 4C allowing CONSULTANT to endorse specified product."),
    ("doc5_qa9__seed42", 0.0, "PRED cites Dova's assignment clause with Affiliate exception; gold=Neither ADAMS GOLF nor CONSULTANT may assign without consent."),
    ("doc5_qa10__seed42", 0.0, "PRED refuses; gold=minimum professional golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED refuses; gold=not more than [*****] days per year for TV/radio/commercial appearances."),
    ("doc5_qa12__seed42", 0.0, "PRED refuses; gold=CONSULTANT grants ADAMS GOLF exclusive license to use CONSULTANT'S ENDORSEMENT."),
    # doc6 — consulting agreement (Kiromic/Gianluca Rotino)
    ("doc6_qa0__seed42", 1.0, "PRED 'Consulting Agreement' matches gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.75, "PRED 'Kiromic, Inc. and Gianluca Rotino' includes gold=Gianluca Rotino; gold only specifies one party name."),
    ("doc6_qa2__seed42", 0.25, "PRED 'effective as of July 1 2018' has correct month+year; gold=July 20 2018 (wrong day)."),
    ("doc6_qa3__seed42", 0.0, "PRED refuses; gold=effective as of July 1 2018."),
    ("doc6_qa4__seed42", 0.0, "PRED 'expires July 1 2021' wrong; gold=commences Effective Date, continues until termination."),
    ("doc6_qa5__seed42", 0.0, "PRED refuses; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED refuses non-compete clause; gold=Consultant use best efforts to segregate Company work."),
    ("doc6_qa7__seed42", 0.5, "PRED 'either party terminate upon 30 days written notice' correctly identifies bilateral termination but adds '30 days' not in gold; gold=prior written notice only."),
    ("doc6_qa8__seed42", 1.0, "PRED 'written consent of Company required for any assignment' matches gold exactly."),
    ("doc6_qa9__seed42", 0.0, "PRED refuses; gold=$400/hr 19hr monthly cap, preapproved expenses."),
    ("doc6_qa10__seed42", 0.25, "PRED describes license-back clause for incorporated inventions; gold=Consultant irrevocably assigns ALL IP to Company (broader than just license-back)."),
    ("doc6_qa11__seed42", 0.0, "PRED refuses; gold=if Consultant incorporates inventions, Company granted license."),
    ("doc6_qa12__seed42", 0.0, "PRED incorrectly says 'no irrevocable or perpetual license'; gold=incorporated-invention license clause exists."),
    ("doc6_qa13__seed42", 0.25, "PRED identifies surviving sections but misses specific property-delivery obligation; gold=deliver all property relating to Inventions upon termination."),
    # doc7 — amendment and termination of joint venture (Veoneer/Nissin Kogyo)
    ("doc7_qa0__seed42", 0.5, "PRED 'JOINT VENTURE AGREEMENT' missing 'AMENDMENT AND TERMINATION OF' prefix; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT."),
    ("doc7_qa1__seed42", 0.0, "PRED refuses; gold=Nissin Kogyo Co., Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED refuses; gold=October 30 2019."),
    ("doc7_qa3__seed42", 0.0, "PRED 'Effective Date July 1 2018' from wrong contract (Kiromic); gold=effective upon VNBJ Closing."),
    ("doc7_qa4__seed42", 0.0, "PRED 'expires 10 years from Effective Date' from wrong contract; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'governed by English law' wrong; gold=Japan law."),
    # doc8 — co-promotion agreement (Dova/Valeant)
    ("doc8_qa0__seed42", 0.25, "PRED 'Promotion Agreement related to Dova Pharmaceuticals' misses 'CO-' and garbles title; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.0, "PRED identifies Dova only, not Valeant; gold=Valeant."),
    ("doc8_qa2__seed42", 0.0, "PRED refuses; gold=September 26 2018."),
    ("doc8_qa3__seed42", 0.0, "PRED 'Effective Date August 10 2007' from wrong contract (EKR/PPI); gold='Effective Date' meaning set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED refuses specific term; gold=four-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED refuses; gold=[***] law (redacted in filing)."),
    ("doc8_qa6__seed42", 0.0, "PRED refuses; gold=Valeant/Affiliates shall not [***] in Territory other than the Product."),
    ("doc8_qa7__seed42", 0.0, "PRED refuses; gold=Dova grants Valeant co-exclusive right to Detail and promote Product."),
    ("doc8_qa8__seed42", 0.25, "PRED has competitive restriction exception concept citing Section 2.3.1(b); gold cites Section 2.3.2 restriction; wrong section reference but correct concept."),
    ("doc8_qa9__seed42", 0.0, "PRED refuses; gold=neither Valeant nor Dova shall solicit other's employees."),
    ("doc8_qa10__seed42", 0.25, "PRED 'either party terminate upon 30 days notice'; gold=either Party may terminate with [***] written notice; correct bilateral right but wrong (explicit) notice period."),
    ("doc8_qa11__seed42", 0.0, "PRED describes termination-on-change-of-control; gold=assignment to Affiliate without consent + Dova can assign in change of control."),
    ("doc8_qa12__seed42", 0.5, "PRED 'consent required from other Party except Affiliate or asset sale'; gold=Party must deliver written notice before assignment; correct anti-assignment concept but adds exceptions not present."),
    ("doc8_qa13__seed42", 0.0, "PRED refuses; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED refuses; gold=Quarterly Minimum Details calculation/adjustment."),
    ("doc8_qa15__seed42", 0.0, "PRED references IP section number without content; gold=Dova owns all right, title, interest in Product Materials."),
    ("doc8_qa16__seed42", 0.0, "PRED incorrectly states 'no license grant explicitly mentioned'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 0.5, "PRED 'may not be assigned without written consent of other Party'; gold=Valeant's rights non-transferable, non-assignable, non-delegable except to Affiliates; correct but misses Affiliate carve-out."),
    ("doc8_qa18__seed42", 0.0, "PRED refuses; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.0, "PRED refuses; gold=Dova has audit right during business hours with reasonable prior notice."),
    ("doc8_qa20__seed42", 0.0, "PRED refuses; gold=limitations don't apply to third-party indemnification or IP infringement/fraud damages."),
    ("doc8_qa21__seed42", 0.0, "PRED refuses; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination right."),
    ("doc8_qa22__seed42", 0.0, "PRED refuses; gold=termination compensation applies when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 0.0, "PRED refuses; gold=each Party maintains products liability, general commercial, business interruption insurance."),
    ("doc8_qa24__seed42", 0.0, "PRED refuses; gold=Valeant shall not do anything to impair Dova Trademarks and Copyrights."),
    # doc9 — amended and restated strategic licensing agreement (EKR/SkyePharma/Pacira)
    ("doc9_qa0__seed42", 0.5, "PRED 'PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING...' uses 'A_R' abbreviation; gold=AMENDED AND RESTATED STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT; partial."),
    ("doc9_qa1__seed42", 0.25, "PRED 'Pacira Pharmaceuticals, Inc. and EKR Therapeutics' identifies both parties; gold=F/K/A SKYEPHARMA, INC. (former name); partial match."),
    ("doc9_qa2__seed42", 0.0, "PRED refuses; gold=October 15 2009."),
    ("doc9_qa3__seed42", 1.0, "PRED 'Effective Date is August 10 2007' matches gold=August 10 2007 exactly."),
    ("doc9_qa4__seed42", 0.0, "PRED refuses; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 0.0, "PRED 'renews quarterly and annually for 1-year terms'; gold=consecutive 2-year periods auto-renewal."),
    ("doc9_qa6__seed42", 0.0, "PRED '15 days notice' wrong; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 0.0, "PRED refuses; gold=New York law."),
    ("doc9_qa8__seed42", 0.0, "PRED refuses; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product."),
    ("doc9_qa9__seed42", 0.0, "PRED refuses; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'either party terminate upon 30 days' from wrong contract; gold=after July 1 2015 PPI may terminate with 60 days written notice."),
    ("doc9_qa11__seed42", 0.0, "PRED vague about change-of-control definition; gold=ceasing to carry on whole or material part of business as trigger."),
    ("doc9_qa12__seed42", 0.75, "PRED 'consent required from other Party except Affiliate or successor' correctly identifies anti-assignment with exceptions; gold=no assignment without prior written consent (matches)."),
    ("doc9_qa13__seed42", 0.0, "PRED refuses; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED describes PPI ownership of existing IP; gold=EKR transfers NDA and regulatory documentation to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED refuses; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 0.0, "PRED refuses; gold=PPI grants EKR exclusive right/license to market, promote, sell Products in Territory."),
    ("doc9_qa17__seed42", 0.0, "PRED 'nonnegotiable, may not be transferred' from wrong contract; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED refuses; gold=PPI Improvements licensed to EKR; EKR Improvements licensed to PPI upon PPI-initiated termination."),
    ("doc9_qa19__seed42", 0.0, "PRED vague post-termination; gold=if EKR exercises Step-in Right, PPI cooperates at EKR's cost."),
    ("doc9_qa20__seed42", 0.25, "PRED correctly identifies audit rights exist but misses specific restriction; gold=PPI cannot re-inspect same Calendar Year after completed inspection."),
    ("doc9_qa21__seed42", 0.0, "PRED refuses; gold=limitation doesn't apply if EKR required to pay excess to third party per final judgment."),
    ("doc9_qa22__seed42", 0.0, "PRED refuses; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 0.0, "PRED refuses; gold=comprehensive product liability, general commercial liability, business interruption insurance."),
]


def main() -> None:
    assert len(JUDGMENTS) == 92
    qid_prefix = "cuad__v4t-corpus-tuned__batch__"
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: existing.add(json.loads(line)["qid"])
                except: pass
    added = 0; total = 0.0; skipped = 0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = qid_prefix + suffix
            if qid in existing:
                skipped += 1; continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"cuad v4t-corpus-tuned batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
