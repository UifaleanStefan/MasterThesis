"""Phase 1.9 Protocol A — CUAD v4t-canonical batch seed42 (92 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__v4t-canonical__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc3 — web hosting agreement (i-on)
    ("doc3_qa3__seed42", 0.0, "PRED refuses effective date; gold=term commences April 1 1999."),
    ("doc3_qa4__seed42", 0.0, "PRED refuses expiration date; gold=April 1 1999 plus 6 months."),
    ("doc3_qa5__seed42", 0.0, "PRED refuses renewal term; gold=auto-renews 1-month periods."),
    ("doc3_qa6__seed42", 0.0, "PRED refuses notice period to terminate renewal; gold=written notice required to stop 1-month auto-renewal."),
    ("doc3_qa7__seed42", 0.0, "PRED refuses governing law; gold=Florida law."),
    ("doc3_qa8__seed42", 0.0, "PRED refuses termination for convenience; gold=terminate without cause 30 days written notice."),
    ("doc3_qa9__seed42", 0.0, "PRED refuses cap on liability; gold=lost profits cap, no consequential damages."),
    # doc4 — joint filing agreement
    ("doc4_qa0__seed42", 0.0, "PRED cites wrong contract (October 2009 note); gold=JOINT FILING AGREEMENT."),
    ("doc4_qa1__seed42", 0.0, "PRED cites wrong parties (EKR/Pacira); gold=Co-Trustee."),
    ("doc4_qa2__seed42", 0.0, "PRED 'October 2009' wrong; gold=March 27 2020."),
    # doc5 — Adams Golf endorsement agreement
    ("doc5_qa0__seed42", 0.0, "PRED cites wrong contract; gold=ENDORSEMENT AGREEMENT."),
    ("doc5_qa1__seed42", 0.0, "PRED cites wrong parties; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED wrong date 'October'; gold=January 13 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED refuses effective date; gold=term commences September 1 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'fifth anniversary' from wrong contract; gold=September 1 2004 plus years+months per schedule."),
    ("doc5_qa5__seed42", 0.0, "PRED refuses governing law; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED refuses non-compete; gold=Consultant shall not endorse competitive golf products with ADAMS GOLF."),
    ("doc5_qa7__seed42", 0.0, "PRED refuses exclusivity; gold=CONSULTANT exclusively play MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED refuses competitive restriction exception; gold=exceptions to Paragraphs 4A 4B 4C."),
    ("doc5_qa9__seed42", 0.25, "PRED 'nonnegotiable, may not be transferred' has partial anti-assignment content; gold=ADAMS GOLF assignment requires written consent."),
    ("doc5_qa10__seed42", 0.0, "PRED refuses minimum commitment; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED refuses; gold=days per year limited for TV/radio/commercial appearances."),
    ("doc5_qa12__seed42", 0.0, "PRED refuses license grant; gold=CONSULTANT grants exclusive endorsement license to ADAMS GOLF."),
    # doc6 — consulting agreement (Gianluca Rotino)
    ("doc6_qa0__seed42", 0.0, "PRED cites wrong contract; gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.0, "PRED cites wrong party (EKR/Pacira); gold=Gianluca Rotino as Consultant."),
    ("doc6_qa2__seed42", 0.0, "PRED 'October 2009' wrong; gold=July 20 2018."),
    ("doc6_qa3__seed42", 0.0, "PRED refuses effective date; gold=effective July 1 2018."),
    ("doc6_qa4__seed42", 0.0, "PRED refuses expiration; gold=commences Effective Date, continues until terminated."),
    ("doc6_qa5__seed42", 0.0, "PRED refuses governing law; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED refuses exclusivity clause; gold=Consultant best efforts, segregate Company work."),
    ("doc6_qa7__seed42", 0.0, "PRED refuses termination for convenience; gold=either party terminate with prior written notice."),
    ("doc6_qa8__seed42", 0.25, "PRED 'nonnegotiable, may not be transferred' partial anti-assignment content; gold=CONSULTANT rights require written consent of Company."),
    ("doc6_qa9__seed42", 0.0, "PRED refuses revenue/compensation; gold=$400/hr 19hr monthly cap, preapproved expenses."),
    ("doc6_qa10__seed42", 0.0, "PRED refuses IP ownership; gold=Consultant irrevocably assigns all IP to Company."),
    ("doc6_qa11__seed42", 0.0, "PRED refuses license grant; gold=Consultant agrees to license improvements to Company."),
    ("doc6_qa12__seed42", 0.0, "PRED refuses affiliate license; gold=Consultant incorporates inventions with license grant."),
    ("doc6_qa13__seed42", 0.0, "PRED refuses post-termination obligations; gold=deliver all property related to Inventions upon termination."),
    # doc7 — amendment and termination of joint venture (Nissin Kogyo)
    ("doc7_qa0__seed42", 0.0, "PRED cites wrong contract; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT."),
    ("doc7_qa1__seed42", 0.0, "PRED refuses parties; gold=Nissin Kogyo Co. Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED 'October 2009' wrong; gold=October 30 2019."),
    ("doc7_qa3__seed42", 0.0, "PRED refuses effective date; gold=effective upon VNBJ Closing."),
    ("doc7_qa4__seed42", 0.0, "PRED 'fifth anniversary' wrong; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED refuses governing law; gold=Japan law."),
    # doc8 — co-promotion agreement (Dova/Valeant)
    ("doc8_qa0__seed42", 0.0, "PRED cites wrong contract; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.0, "PRED cites wrong party (EKR/Pacira); gold=Valeant as co-promoter."),
    ("doc8_qa2__seed42", 0.0, "PRED 'October 2009' wrong; gold=September 26 2018."),
    ("doc8_qa3__seed42", 0.0, "PRED refuses effective date; gold='Effective Date' meaning set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED 'fifth anniversary' from wrong contract; gold=four-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED refuses governing law; gold=[***] law (redacted in public filing)."),
    ("doc8_qa6__seed42", 0.0, "PRED refuses non-compete; gold=Valeant/Affiliates shall not [***] in Territory other than the Product."),
    ("doc8_qa7__seed42", 0.0, "PRED refuses exclusivity; gold=Dova grants Valeant co-exclusive right (solely with Dova) to Detail and promote Product."),
    ("doc8_qa8__seed42", 0.0, "PRED refuses competitive restriction exception; gold=restrictions shall not apply to [***]."),
    ("doc8_qa9__seed42", 0.0, "PRED refuses no-solicit of employees; gold=neither Valeant nor Dova shall solicit the other's employees."),
    ("doc8_qa10__seed42", 0.0, "PRED refuses termination for convenience; gold=either Party may terminate before end of Term with [***] written notice."),
    ("doc8_qa11__seed42", 0.0, "PRED refuses change of control; gold=either Party may assign to Affiliate without consent; Dova can assign in change of control."),
    ("doc8_qa12__seed42", 0.0, "PRED 'nonnegotiable, may not be transferred' from wrong contract (promissory note); gold=Party shall deliver written notice before assignment."),
    ("doc8_qa13__seed42", 0.0, "PRED refuses revenue sharing; gold=Dova pays Valeant promotion fee based on annual Net Sales (formula specified)."),
    ("doc8_qa14__seed42", 0.0, "PRED refuses minimum commitment; gold=if actual Details < Quarterly Minimum Details, promotion fee calculation adjusted."),
    ("doc8_qa15__seed42", 0.0, "PRED refuses IP ownership; gold=Dova owns all right, title and interest in Product Materials and Product Labeling."),
    ("doc8_qa16__seed42", 0.0, "PRED refuses license grant; gold=Dova grants Valeant co-exclusive right to Detail and promote Products in Territory."),
    ("doc8_qa17__seed42", 0.0, "PRED 'nonnegotiable' from wrong contract; gold=Valeant's rights non-transferable, non-assignable, non-delegable except to Affiliates."),
    ("doc8_qa18__seed42", 0.0, "PRED refuses affiliate license; gold=Valeant grants Dova non-exclusive license to Valeant Property appearing in Product Materials."),
    ("doc8_qa19__seed42", 0.0, "PRED refuses audit rights; gold=Dova has right to audit Valeant's books during business hours with reasonable prior notice."),
    ("doc8_qa20__seed42", 0.0, "PRED refuses uncapped liability; gold=limitations don't apply to third-party indemnification or damages from IP infringement/fraud/willful misconduct."),
    ("doc8_qa21__seed42", 0.0, "PRED refuses cap on liability; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination right."),
    ("doc8_qa22__seed42", 0.0, "PRED refuses liquidated damages; gold=termination compensation applies when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 0.0, "PRED refuses insurance; gold=each Party maintains products liability, general commercial liability, business interruption insurance."),
    ("doc8_qa24__seed42", 0.0, "PRED refuses covenant not to sue; gold=Valeant shall not do anything to impair Dova Trademarks and Copyrights."),
    # doc9 — amended and restated strategic licensing/distribution agreement (EKR/SkyePharma)
    ("doc9_qa0__seed42", 1.0, "PRED 'Amended and Restated Strategic Licensing, Distribution and Marketing Agreement' matches gold exactly."),
    ("doc9_qa1__seed42", 0.25, "PRED identifies EKR Therapeutics and Pacira Pharmaceuticals; gold=F/K/A SKYEPHARMA, INC. (former name of Pacira); partial match on one party."),
    ("doc9_qa2__seed42", 0.75, "PRED 'October 2009' matches gold=October 15 2009 in month+year but missing specific day."),
    ("doc9_qa3__seed42", 0.0, "PRED vague about effective date (references agreement title only); gold=August 10 2007."),
    ("doc9_qa4__seed42", 0.0, "PRED refuses expiration; gold=15 years from Effective Date or last-to-expire licensed patent, whichever is longer."),
    ("doc9_qa5__seed42", 0.0, "PRED refuses renewal term; gold=consecutive 2-year periods auto-renewal."),
    ("doc9_qa6__seed42", 0.0, "PRED refuses notice period; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 0.0, "PRED refuses governing law; gold=New York law."),
    ("doc9_qa8__seed42", 0.0, "PRED refuses non-compete; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product in Territory."),
    ("doc9_qa9__seed42", 0.0, "PRED refuses exclusivity; gold=PPI appoints EKR as exclusive distributor and Authorized Distributor of Record in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED refuses termination for convenience; gold=after July 1 2015 PPI may terminate with 60 days prior written notice to EKR."),
    ("doc9_qa11__seed42", 0.0, "PRED refuses change of control; gold=ceasing to carry on the whole or material part of business as trigger."),
    ("doc9_qa12__seed42", 0.0, "PRED 'nonnegotiable, may not be transferred' from wrong contract; gold=neither Party may assign without prior written consent of other Party."),
    ("doc9_qa13__seed42", 0.0, "PRED refuses revenue sharing; gold=EKR pays PPI royalty equal to $[**] per [**]mg Vial of Product."),
    ("doc9_qa14__seed42", 0.0, "PRED refuses IP ownership; gold=EKR must transfer NDA and regulatory documentation to PPI upon termination (with stated exceptions)."),
    ("doc9_qa15__seed42", 0.0, "PRED refuses joint IP ownership; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR under Agreement."),
    ("doc9_qa16__seed42", 0.0, "PRED refuses license grant; gold=PPI grants EKR exclusive right/license to use, market, promote, sell, distribute and warehouse Products in Territory."),
    ("doc9_qa17__seed42", 0.0, "PRED 'nonnegotiable, may not be transferred' from wrong contract; gold=EKR may appoint sub-distributors provided EKR informs PPI and ensures compliance."),
    ("doc9_qa18__seed42", 0.0, "PRED refuses irrevocable/perpetual license; gold=PPI Improvements owned by PPI licensed to EKR; EKR Improvements owned by EKR licensed to PPI upon PPI-initiated termination."),
    ("doc9_qa19__seed42", 0.0, "PRED refuses post-termination services; gold=if EKR exercises Step-in Right PPI cooperates and EKR reimburses PPI's costs."),
    ("doc9_qa20__seed42", 0.0, "PRED refuses audit rights; gold=after completed inspection PPI cannot re-inspect same Calendar Year records."),
    ("doc9_qa21__seed42", 0.0, "PRED refuses uncapped liability; gold=limitation shall not apply to extent EKR required to pay excess to third party per final judgment."),
    ("doc9_qa22__seed42", 0.0, "PRED refuses cap on liability; gold=same limitation-exception clause as uncapped liability."),
    ("doc9_qa23__seed42", 0.0, "PRED refuses insurance; gold=each Party shall maintain comprehensive product liability, general commercial liability, and business interruption insurance."),
]


def main() -> None:
    assert len(JUDGMENTS) == 92
    qid_prefix = "cuad__v4t-canonical__batch__"
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
    print(f"cuad v4t-canonical batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
