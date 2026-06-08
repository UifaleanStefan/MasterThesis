"""Phase 1.9 Protocol A — CUAD dump-all batch seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__dump-all__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# dump-all batch: model retrieves a promissory note (EKR/PPI October 2009) as context
# for almost every question across all 10 contracts. Near-zero performance.
# Only doc9 (EKR/PPI) incidentally matches for 3 questions.

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement (19 entries)
    ("doc0_qa0__seed42", 0.0, "PRED lists sections from promissory note (Place of Payment, Optional Prepayment, Defaults), no document name; gold=DISTRIBUTOR AGREEMENT."),
    ("doc0_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=Distributor."),
    ("doc0_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=7th day of September 1999."),
    ("doc0_qa3__seed42", 0.0, "PRED 'Effective Date not explicitly provided'; gold=term commences on date Company delivers last Sample."),
    ("doc0_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=10 years commencing on date Company delivers last Sample."),
    ("doc0_qa5__seed42", 0.0, "PRED 'Renewal Term not in provided passages'; gold=renewable annually for 1-year terms up to another 10 years."),
    ("doc0_qa6__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=laws of State of Illinois."),
    ("doc0_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Distributor shall not order/purchase Products from source other than Company."),
    ("doc0_qa8__seed42", 0.0, "PRED 'No-Solicit Of Customers not in provided passages'; gold=Distributor shall not interfere with Company's business relations."),
    ("doc0_qa9__seed42", 0.0, "PRED 'No-Solicit Of Employees not in provided passages'; gold=restriction on soliciting employees during Term and 12 months thereafter."),
    ("doc0_qa10__seed42", 0.0, "PRED 'ROFR not in provided passages'; gold=Distributor right to become exclusive Distributor of other Products."),
    ("doc0_qa11__seed42", 0.0, "PRED 'This Note shall be nonnegotiable. Further, this Note may not be transferred' from promissory note; gold=no assignment without prior written consent of Company."),
    ("doc0_qa12__seed42", 0.0, "PRED 'Price Restrictions not in provided passages'; gold=Company reserves right to increase/decrease price."),
    ("doc0_qa13__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=$250,000 minimum purchase order."),
    ("doc0_qa14__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Company appoints Distributor as exclusive distributor."),
    ("doc0_qa15__seed42", 0.0, "PRED 'Post-Termination Services not in provided passages'; gold=Company may repurchase inventory within 30 days."),
    ("doc0_qa16__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=Company unable/unwilling to promptly repair/replace Products."),
    ("doc0_qa17__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=Company carry reasonable product liability insurance."),
    ("doc0_qa18__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Distributor not to challenge IP during Term and 3 years thereafter."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement (12 entries)
    ("doc1_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from wrong contract; gold=Promotion and Distribution Agreement."),
    ("doc1_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=Distributor."),
    ("doc1_qa2__seed42", 0.0, "PRED 'Effective Date not provided'; gold=1 August 2011."),
    ("doc1_qa3__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=earlier of 2 years (31 July 2013) or Maximum Distribution Commitment."),
    ("doc1_qa4__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=English law."),
    ("doc1_qa5__seed42", 0.0, "PRED 'Change Of Control not in provided passages'; gold=notification/termination right upon Change of Control."),
    ("doc1_qa6__seed42", 0.0, "PRED 'nonnegotiable and nontransferable' from promissory note; gold=[*] may [*] rights/obligations without prior written consent of [*]."),
    ("doc1_qa7__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Google grants Distributor limited [*] license during Term."),
    ("doc1_qa8__seed42", 0.0, "PRED 'Audit Rights not in provided passages'; gold=Google may audit Distributor's records during Term and [*] after."),
    ("doc1_qa9__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=each party's total liability limited per Contract Year."),
    ("doc1_qa10__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=each party's total liability limited per Contract Year."),
    ("doc1_qa11__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=Google warrants for [*] period from date of supply."),
    # doc2 — Shenzhen LOHAS supply contract (6 entries)
    ("doc2_qa0__seed42", 0.0, "PRED lists promissory note sections; gold=SUPPLY CONTRACT."),
    ("doc2_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=The seller:."),
    ("doc2_qa2__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=Contract valid for 5 years."),
    ("doc2_qa3__seed42", 0.0, "PRED 'New York law' from promissory note; gold=People's Republic of China law or UN CISG."),
    ("doc2_qa4__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=within 7 days after arrival at destination for quality claims."),
    ("doc2_qa5__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement (10 entries)
    ("doc3_qa0__seed42", 0.0, "PRED lists promissory note sections (Governing Law, Nonnegotiability...); gold=WEB SITE HOSTING AGREEMENT."),
    ("doc3_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=Centrack International."),
    ("doc3_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=6th day of April, 1999."),
    ("doc3_qa3__seed42", 0.0, "PRED 'Effective Date not explicitly provided'; gold=term commences April 1, 1999."),
    ("doc3_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=April 1, 1999 plus 6 months."),
    ("doc3_qa5__seed42", 0.0, "PRED 'Renewal Term not in provided passages'; gold=auto-renews 1-month periods."),
    ("doc3_qa6__seed42", 0.0, "PRED 'Notice Period not in provided passages'; gold=notice required to stop 1-month auto-renewal."),
    ("doc3_qa7__seed42", 0.0, "PRED 'New York law' from promissory note; gold=Florida law."),
    ("doc3_qa8__seed42", 0.0, "PRED 'Termination For Convenience not in provided passages'; gold=terminate without cause 30 days written notice."),
    ("doc3_qa9__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=i-on not liable for lost profits or consequential damages."),
    # doc4 — joint filing agreement (3 entries)
    ("doc4_qa0__seed42", 0.0, "PRED lists promissory note sections; gold=JOINT FILING AGREEMENT."),
    ("doc4_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=Co-Trustee."),
    ("doc4_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=March 27, 2020."),
    # doc5 — Adams Golf endorsement agreement (13 entries)
    ("doc5_qa0__seed42", 0.0, "PRED lists promissory note sections; gold=ENDORSEMENT AGREEMENT."),
    ("doc5_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=January 13, 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED 'Effective Date not provided'; gold=term commences September 1, 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=September 1, 2004 plus [redacted] years and months."),
    ("doc5_qa5__seed42", 0.0, "PRED 'New York law' from promissory note; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=CONSULTANT not to associate with ADAMS GOLF competitor's Product."),
    ("doc5_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED 'Competitive Restriction Exception not in provided passages'; gold=exception to paragraphs 4A, 4B, 4C."),
    ("doc5_qa9__seed42", 0.0, "PRED 'This Note shall be nonnegotiable, may not be transferred' from promissory note; gold=Neither ADAMS GOLF nor CONSULTANT shall assign."),
    ("doc5_qa10__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED 'Volume Restriction not in provided passages'; gold=not more than [*****] days for TV/radio/commercial appearances."),
    ("doc5_qa12__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=CONSULTANT grants ADAMS GOLF exclusive endorsement license."),
    # doc6 — Kiromic consulting agreement (14 entries)
    ("doc6_qa0__seed42", 0.0, "PRED lists promissory note sections; gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.0, "PRED identifies Maker and Payee from promissory note; gold=Gianluca Rotino."),
    ("doc6_qa2__seed42", 0.0, "PRED references EKR/PPI promissory note note text; gold=July 20, 2018."),
    ("doc6_qa3__seed42", 0.0, "PRED 'Effective Date not provided'; gold=effective as of July 1, 2018."),
    ("doc6_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=commences Effective Date, continues until terminated."),
    ("doc6_qa5__seed42", 0.0, "PRED 'New York law' from promissory note; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=Consultant use best efforts to segregate Company work."),
    ("doc6_qa7__seed42", 0.0, "PRED 'Termination For Convenience not in provided passages'; gold=either Consultant or Company may terminate upon prior written notice."),
    ("doc6_qa8__seed42", 0.0, "PRED 'nonnegotiable and nontransferable, may not be transferred except to permitted transferee' from promissory note; gold=no assignment without express written consent of Company."),
    ("doc6_qa9__seed42", 0.0, "PRED 'Volume Restriction not in provided passages'; gold=$400/hr 19hr monthly cap."),
    ("doc6_qa10__seed42", 0.0, "PRED 'IP Ownership Assignment not in provided passages'; gold=Consultant irrevocably assigns all IP to Company."),
    ("doc6_qa11__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Consultant grants license if incorporating existing inventions."),
    ("doc6_qa12__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=incorporated-invention license clause."),
    ("doc6_qa13__seed42", 0.0, "PRED 'Post-Termination Services not in provided passages'; gold=deliver all property relating to Inventions upon termination."),
    # doc7 — amendment and termination of joint venture (Veoneer/Nissin Kogyo) (6 entries)
    ("doc7_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from wrong contract; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT."),
    ("doc7_qa1__seed42", 0.0, "PRED 'Parties not in provided passages'; gold=Nissin Kogyo Co., Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=October 30, 2019."),
    ("doc7_qa3__seed42", 0.0, "PRED 'Effective Date not in provided passages'; gold=effective upon VNBJ Closing."),
    ("doc7_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'New York law' from promissory note; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement (25 entries)
    ("doc8_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from wrong contract; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.0, "PRED identifies Maker/Payee from promissory note; gold=Valeant."),
    ("doc8_qa2__seed42", 0.0, "PRED 'October, 2009' from promissory note; gold=September 26, 2018."),
    ("doc8_qa3__seed42", 0.0, "PRED 'Effective Date not provided'; gold=Effective Date set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=4-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED 'New York law' from promissory note; gold=[***] law (redacted)."),
    ("doc8_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=Valeant/Affiliates shall not [***] in Territory."),
    ("doc8_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa8__seed42", 0.0, "PRED 'Competitive Restriction Exception not in provided passages'; gold=restrictions shall not apply to [***]."),
    ("doc8_qa9__seed42", 0.0, "PRED 'No-Solicit Of Employees not in provided passages'; gold=neither Valeant nor Dova shall solicit other's employees."),
    ("doc8_qa10__seed42", 0.0, "PRED 'Termination For Convenience not in provided passages'; gold=either Party may terminate before end of Term."),
    ("doc8_qa11__seed42", 0.0, "PRED 'Change Of Control not in provided passages'; gold=assignment to Affiliate without consent + Dova in change of control."),
    ("doc8_qa12__seed42", 0.0, "PRED 'This Note shall be nonnegotiable...may not be transferred except to permitted transferee' from promissory note; gold=Party shall deliver written notice before assignment."),
    ("doc8_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=Quarterly Minimum Details calculation."),
    ("doc8_qa15__seed42", 0.0, "PRED 'IP Ownership Assignment not in provided passages'; gold=Dova owns all right, title and interest in Product Materials."),
    ("doc8_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 0.0, "PRED 'This Note shall be nonnegotiable' from promissory note; gold=Valeant's rights non-transferable, non-assignable except to Affiliates."),
    ("doc8_qa18__seed42", 0.0, "PRED 'Affiliate License not in provided passages'; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.0, "PRED 'Audit Rights not in provided passages'; gold=Dova has right to audit Valeant's records."),
    ("doc8_qa20__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitations don't apply to indemnification or IP infringement/fraud damages."),
    ("doc8_qa21__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination."),
    ("doc8_qa22__seed42", 0.0, "PRED 'Liquidated Damages not in provided passages'; gold=termination compensation when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=each Party maintain products liability, general commercial, business interruption insurance."),
    ("doc8_qa24__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Valeant shall not impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma strategic licensing agreement (24 entries)
    # doc9 is EKR/PPI — same contract as the promissory note context, so model gets a few hits
    ("doc9_qa0__seed42", 1.0, "PRED 'The contract name is Amended and Restated Strategic Licensing, Distribution and Marketing Agreement' matches gold exactly."),
    ("doc9_qa1__seed42", 0.0, "PRED identifies Maker/Payee from promissory note (EKR/Pacira); gold=F/K/A SKYEPHARMA, INC."),
    ("doc9_qa2__seed42", 0.25, "PRED references 'dated October, 2009' from promissory note text; gold=October 15, 2009; right month+year, missing day."),
    ("doc9_qa3__seed42", 0.0, "PRED 'Effective Date not provided'; gold=August 10, 2007."),
    ("doc9_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 0.0, "PRED 'Renewal Term not in provided passages'; gold=consecutive 2-year periods auto-renewal."),
    ("doc9_qa6__seed42", 0.0, "PRED 'Notice Period not in provided passages'; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 0.75, "PRED 'New York law' technically correct answer (incidentally retrieved from promissory note which also has New York law); gold=New York law."),
    ("doc9_qa8__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product."),
    ("doc9_qa9__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'Termination For Convenience not in provided passages'; gold=after July 1 2015, PPI may terminate with 60 days notice."),
    ("doc9_qa11__seed42", 0.0, "PRED 'Change Of Control not in provided passages'; gold=ceasing to carry on business as trigger."),
    ("doc9_qa12__seed42", 0.0, "PRED 'This Note shall be nonnegotiable, may not be transferred' from promissory note; gold=neither Party shall assign without prior written consent."),
    ("doc9_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED 'IP Ownership Assignment not in provided passages'; gold=EKR transfers NDA and regulatory documentation to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED 'Joint IP Ownership not in provided passages'; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=PPI grants EKR exclusive right and license to market/promote/sell Products in Territory."),
    ("doc9_qa17__seed42", 0.0, "PRED 'This Note shall be nonnegotiable, may not be transferred' from promissory note; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=PPI/EKR Improvements mutually licensed."),
    ("doc9_qa19__seed42", 0.0, "PRED 'Post-Termination Services not in provided passages'; gold=if EKR exercises Step-in Right PPI cooperates at EKR's cost."),
    ("doc9_qa20__seed42", 0.0, "PRED 'Audit Rights not in provided passages'; gold=PPI cannot re-inspect same Calendar Year after completed inspection."),
    ("doc9_qa21__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitation doesn't apply if EKR required to pay excess to third party."),
    ("doc9_qa22__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=each Party shall maintain comprehensive product liability insurance."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__dump-all__batch__"
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
    print(f"cuad dump-all batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
