"""Phase 1.9 Protocol A — CUAD v4t-tuned batch seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__v4t-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# v4t-tuned batch: tuned theta in batch mode. Slightly better than v4t-canonical batch
# because tuned memory stores doc8/doc9 more reliably. Still mostly retrieves EKR/PPI
# context for docs 0-7. Mean≈0.125.

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement (19 entries)
    ("doc0_qa0__seed42", 0.0, "PRED 'Document Name not in provided passages'; gold=DISTRIBUTOR AGREEMENT."),
    ("doc0_qa1__seed42", 0.0, "PRED 'Parties not provided in excerpts'; gold=Distributor."),
    ("doc0_qa2__seed42", 0.0, "PRED 'dated as of October, 2009' from EKR/PPI; gold=7th day of September, 1999."),
    ("doc0_qa3__seed42", 0.0, "PRED 'Effective Date means August 10, 2007...extends until four-year anniversary' from EKR/PPI; gold=term commences on date Company delivers last Sample."),
    ("doc0_qa4__seed42", 0.0, "PRED 'expires fifteen (15) years from first Commercial Launch' from EKR/PPI; gold=10 years commencing on date Company delivers last Sample."),
    ("doc0_qa5__seed42", 0.0, "PRED 'renewal term two (2) years...terminated by EKR at end of Initial Term' from EKR/PPI; gold=renewable annually for 1-year terms up to another 10 years."),
    ("doc0_qa6__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=Illinois law."),
    ("doc0_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Distributor shall not purchase Products from source other than Company."),
    ("doc0_qa8__seed42", 0.0, "PRED 'No-Solicit Of Customers not in provided passages'; gold=Distributor shall not interfere with Company's business relations."),
    ("doc0_qa9__seed42", 0.0, "PRED 'neither Valeant nor...soliciting employees' from Dova/Valeant context; gold=Distributor restrictions during Term and 12 months thereafter."),
    ("doc0_qa10__seed42", 0.0, "PRED 'ROFR/ROFO/ROFN not in provided passages'; gold=Distributor right to become exclusive Distributor of other Products."),
    ("doc0_qa11__seed42", 0.5, "PRED 'Consent required from other party for any assignment...except Affiliate or sale/transfer'; gold=no assignment without prior written consent; correct anti-assignment concept with exceptions."),
    ("doc0_qa12__seed42", 0.0, "PRED 'Royalty Cap shall not exceed percentage of net average selling price' from EKR/PPI; gold=Company reserves right to increase/decrease price."),
    ("doc0_qa13__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=$250,000 minimum purchase order."),
    ("doc0_qa14__seed42", 0.0, "PRED 'License Grant not in provided passages for LIMEENERGYCO'; gold=Company appoints Distributor as exclusive distributor."),
    ("doc0_qa15__seed42", 0.0, "PRED 'Post-Termination Services not in provided passages'; gold=Company may repurchase inventory within 30 days."),
    ("doc0_qa16__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=Company unable/unwilling to promptly repair/replace Products."),
    ("doc0_qa17__seed42", 0.25, "PRED 'each Party maintain adequate insurance including products liability and comprehensive general liability'; gold=Company carry product liability insurance naming Distributor; overstates (both parties vs Company only)."),
    ("doc0_qa18__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Distributor not to challenge IP during Term and 3 years."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement (12 entries)
    ("doc1_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from EKR/PPI; gold=Promotion and Distribution Agreement."),
    ("doc1_qa1__seed42", 0.0, "PRED 'Parties not provided'; gold=Distributor."),
    ("doc1_qa2__seed42", 0.0, "PRED 'Effective Date means August 10, 2007...extends until four-year anniversary' from EKR/PPI; gold=1 August 2011."),
    ("doc1_qa3__seed42", 0.0, "PRED 'expires on four (4) year anniversary' from EKR/PPI; gold=earlier of 2 years (31 July 2013) or Maximum Distribution Commitment."),
    ("doc1_qa4__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=English law."),
    ("doc1_qa5__seed42", 0.5, "PRED 'Section 20.4 Change of Control' from EKR/PPI; right Change of Control concept, wrong section from wrong contract."),
    ("doc1_qa6__seed42", 0.5, "PRED 'Consent required from other party, except Affiliate or sale/transfer'; gold=[*] may [*] without prior written consent of [*]; identifies consent requirement."),
    ("doc1_qa7__seed42", 0.0, "PRED 'License Grant not in provided passages for WHITESMOKE'; gold=Google grants Distributor limited [*] license."),
    ("doc1_qa8__seed42", 0.0, "PRED 'Valeant has right to inspect and audit Dova' from wrong contract; gold=Google may audit Distributor's records."),
    ("doc1_qa9__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=each party's total liability limited per Contract Year."),
    ("doc1_qa10__seed42", 0.0, "PRED 'cap on liability stating PPI to EKR shall be limited' from EKR/PPI; gold=each party's total liability limited per Contract Year."),
    ("doc1_qa11__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=Google warrants for [*] period from date of supply."),
    # doc2 — Shenzhen LOHAS supply contract (6 entries)
    ("doc2_qa0__seed42", 0.25, "PRED 'Supply Agreement' different name; gold=SUPPLY CONTRACT."),
    ("doc2_qa1__seed42", 0.0, "PRED '21.1 Relationship of Parties. Nothing in this agreement is deemed to constitute a partnership...' no party names; gold=The seller:."),
    ("doc2_qa2__seed42", 0.0, "PRED 'Expiration Date not in provided passages for LOHAS'; gold=Contract valid for 5 years."),
    ("doc2_qa3__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=People's Republic of China law."),
    ("doc2_qa4__seed42", 0.0, "PRED 'Warranty Duration not in provided passages'; gold=within 7 days after arrival at destination for quality claims."),
    ("doc2_qa5__seed42", 0.0, "PRED 'each Party maintain adequate insurance' from EKR/PPI; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement (10 entries)
    ("doc3_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from EKR/PPI; gold=WEB SITE HOSTING AGREEMENT."),
    ("doc3_qa1__seed42", 0.0, "PRED identifies parties from wrong contract; gold=Centrack International."),
    ("doc3_qa2__seed42", 0.0, "PRED 'Within three (3) days of the Agreement Date, EKR shall pay...' from EKR/PPI; gold=6th day of April, 1999."),
    ("doc3_qa3__seed42", 0.0, "PRED 'Effective Date August 10, 2007...extends until four-year anniversary' from EKR/PPI; gold=term commences April 1, 1999."),
    ("doc3_qa4__seed42", 0.0, "PRED 'expires on four (4) year anniversary' from EKR/PPI; gold=April 1, 1999 plus 6 months."),
    ("doc3_qa5__seed42", 0.0, "PRED 'renewal term two (2) years...terminated by EKR' from EKR/PPI; gold=auto-renews for 1-month periods."),
    ("doc3_qa6__seed42", 0.0, "PRED 'notice period not explicitly stated'; gold=notice required to stop 1-month auto-renewal."),
    ("doc3_qa7__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=Florida law."),
    ("doc3_qa8__seed42", 0.5, "PRED 'party can terminate for convenience upon written notice...section 12.3.1' from Dova/Valeant; correct bilateral termination right, wrong section from wrong contract."),
    ("doc3_qa9__seed42", 0.0, "PRED 'cap on liability...PPI to EKR limited...except death' from EKR/PPI; gold=i-on not liable for lost profits or consequential damages."),
    # doc4 — joint filing agreement (3 entries)
    ("doc4_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from EKR/PPI; gold=JOINT FILING AGREEMENT."),
    ("doc4_qa1__seed42", 0.0, "PRED '21.1 Relationship of Parties...' no party names; gold=Co-Trustee."),
    ("doc4_qa2__seed42", 0.0, "PRED 'Within three (3) days of the Agreement Date, EKR shall pay...' from EKR/PPI; gold=March 27, 2020."),
    # doc5 — Adams Golf endorsement agreement (13 entries)
    ("doc5_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from EKR/PPI; gold=ENDORSEMENT AGREEMENT."),
    ("doc5_qa1__seed42", 0.0, "PRED 'Parties not provided'; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED 'Note is made pursuant to...Amended and Restated Strategic Licensing...' references October 2009; gold=January 13, 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED 'August 10, 2007...extends until four-year anniversary' from EKR/PPI; gold=term commences September 1, 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'expires on four-year anniversary' from EKR/PPI; gold=September 1, 2004 plus [redacted] years."),
    ("doc5_qa5__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED 'restriction on ability to compete in Territory other than Product, Section 2.3.1 Non-Competition' from Dova/Valeant; gold=CONSULTANT not to associate with ADAMS GOLF competitor's Product."),
    ("doc5_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED 'provided that if Agreement is terminated by Dova...' from Dova/Valeant; gold=exception to paragraphs 4A, 4B, 4C."),
    ("doc5_qa9__seed42", 0.0, "PRED 'Consent required from other Party, except Affiliate' from Dova/Valeant; gold=Neither ADAMS GOLF nor CONSULTANT shall assign without consent."),
    ("doc5_qa10__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED 'Volume Restriction found in section 3.5.2...if Dova wishes to conduct speaker programs' from Dova/Valeant; gold=not more than [*****] days per year for appearances."),
    ("doc5_qa12__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=CONSULTANT grants ADAMS GOLF exclusive endorsement license."),
    # doc6 — Kiromic consulting agreement (14 entries)
    ("doc6_qa0__seed42", 0.0, "PRED 'not contain information related to Document Name...KIROMICBIOPHARMA'; gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.0, "PRED 'EKR Therapeutics (Maker) and Pacira Pharmaceuticals (Payee)' from promissory note; gold=Gianluca Rotino."),
    ("doc6_qa2__seed42", 0.0, "PRED 'Agreement Date not provided for KIROMICBIOPHARMA'; gold=July 20, 2018."),
    ("doc6_qa3__seed42", 0.0, "PRED 'Effective Date means August 10, 2007...Effective Date shall have meaning set forth in preamble' from EKR/PPI; gold=effective as of July 1, 2018."),
    ("doc6_qa4__seed42", 0.0, "PRED 'expires fifteen (15) years from first Commercial Launch' from EKR/PPI; gold=commences Effective Date, continues until terminated."),
    ("doc6_qa5__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=Consultant use best efforts to segregate Company work."),
    ("doc6_qa7__seed42", 0.75, "PRED 'party may terminate for convenience at any time upon written notice'; gold=either Consultant or Company may terminate upon prior written notice; correct bilateral termination."),
    ("doc6_qa8__seed42", 0.0, "PRED 'Anti-Assignment not in provided passages'; gold=no assignment without express written consent of Company."),
    ("doc6_qa9__seed42", 0.0, "PRED 'Volume Restriction not in provided passages'; gold=$400/hr 19hr monthly cap."),
    ("doc6_qa10__seed42", 0.0, "PRED 'IP Ownership not in provided passages'; gold=Consultant irrevocably assigns all IP to Company."),
    ("doc6_qa11__seed42", 0.0, "PRED 'License Grant not in provided passages for KIROMICBIOPHARMA'; gold=Consultant grants license if incorporating existing inventions."),
    ("doc6_qa12__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=incorporated-invention license clause."),
    ("doc6_qa13__seed42", 0.0, "PRED 'Post-Termination Services not in provided passages'; gold=deliver all property relating to Inventions upon termination."),
    # doc7 — amendment and termination of joint venture (Veoneer/Nissin Kogyo) (6 entries)
    ("doc7_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing...' from EKR/PPI; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT."),
    ("doc7_qa1__seed42", 0.5, "PRED 'Signature Page to Amendment and Termination of Joint Venture Agreement' mentions the gold title; gold=Nissin Kogyo Co., Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED 'Note is made pursuant to...Amended and Restated Strategic Licensing...' references October 2009; gold=October 30, 2019."),
    ("doc7_qa3__seed42", 0.0, "PRED 'Effective Date August 10, 2007...four-year anniversary' from EKR/PPI; gold=effective upon VNBJ Closing."),
    ("doc7_qa4__seed42", 0.0, "PRED 'expires fifteen (15) years from first Commercial Launch' from EKR/PPI; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement (25 entries)
    ("doc8_qa0__seed42", 0.0, "PRED 'Amended and Restated Strategic Licensing, Distribution and Marketing Agreement' from EKR/PPI; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.25, "PRED 'This Co-Promotion Agreement...is entered into and dated as of September 26, 2018 (the Effective Date)...' identifies agreement but not Valeant specifically; gold=Valeant."),
    ("doc8_qa2__seed42", 1.0, "PRED 'Agreement Date is September 26, 2018' matches gold exactly."),
    ("doc8_qa3__seed42", 0.0, "PRED 'Effective Date means August 10, 2007...Effective Date shall have meaning set forth in preamble' — first part from EKR/PPI; gold=Effective Date set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED 'Expiration Date not provided'; gold=4-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED 'New York law' from EKR/PPI; gold=[***] law."),
    ("doc8_qa6__seed42", 0.5, "PRED 'restriction on ability to compete in Territory other than Product, Section 2.3.1 Non-Competition'; gold=neither Valeant nor its Affiliates shall [***] in Territory; correct non-compete concept, wrong section reference."),
    ("doc8_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa8__seed42", 0.5, "PRED 'Section 2.3.1(a), neither Valeant nor its Affiliates...conditions under which restrictions may not apply'; gold=restrictions shall not apply to [***]; correct competitive restriction exception."),
    ("doc8_qa9__seed42", 0.75, "PRED 'neither Valeant nor Dova (nor any of their respective Affiliates) shall directly or indirectly...' matches gold."),
    ("doc8_qa10__seed42", 0.75, "PRED 'either party has right to terminate Agreement for convenience upon written notice'; gold=either Party shall have right to terminate before end of Term; matches concept."),
    ("doc8_qa11__seed42", 0.25, "PRED 'Change of Control, rights under Agreement may not be assigned without express consent' describes restriction; gold=either Party may assign to Affiliate without consent + Dova in change of control; partial match."),
    ("doc8_qa12__seed42", 0.75, "PRED 'Consent required from other Party, except Affiliate or asset sale'; gold=Party shall deliver written notice before assignment; correct anti-assignment with exceptions."),
    ("doc8_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=Quarterly Minimum Details calculation."),
    ("doc8_qa15__seed42", 0.25, "PRED 'intellectual property created by one party becomes property of counterparty, specifically Valeant...' gets IP ownership concept but wrong direction; gold=Dova owns all right, title and interest in Product Materials."),
    ("doc8_qa16__seed42", 0.25, "PRED 'license granted by Valeant to Dova for use of certain Valeant Property' wrong direction; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 0.0, "PRED 'nonnegotiable and nontransferable, may not be transferred except to permitted transferee' from promissory note; gold=Valeant's rights non-transferable except to Affiliates."),
    ("doc8_qa18__seed42", 0.0, "PRED 'Affiliate License not explicitly mentioned'; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.0, "PRED 'Valeant has right to inspect and audit Dova' wrong direction; gold=Dova has right to audit Valeant."),
    ("doc8_qa20__seed42", 0.5, "PRED 'limitation of liability stated in section 11.4, neither Dova nor Valeant shall be liable for consequential damages'; gold=limitations don't apply to indemnification or IP infringement/fraud damages; partial uncapped liability."),
    ("doc8_qa21__seed42", 0.0, "PRED 'limitation of liability...neither Dova nor Valeant shall be liable for consequential damages' wrong clause; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination."),
    ("doc8_qa22__seed42", 0.0, "PRED 'Liquidated Damages not in provided passages'; gold=termination compensation when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 1.0, "PRED 'each Party maintain adequate insurance including products liability and comprehensive general liability' matches gold."),
    ("doc8_qa24__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Valeant shall not impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma strategic licensing agreement (24 entries)
    ("doc9_qa0__seed42", 0.75, "PRED 'Amended and Restated Strategic Licensing, Distribution and Marketing Agreement dated as of O...' correctly names the contract; gold=AMENDED AND RESTATED STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT."),
    ("doc9_qa1__seed42", 0.25, "PRED 'EKR Therapeutics, Inc. (Maker) and Pacira Pharmaceuticals, Inc. (Payee)' from promissory note; gold=F/K/A SKYEPHARMA, INC.; partial match."),
    ("doc9_qa2__seed42", 1.0, "PRED 'Agreement Date is October 15, 2009' matches gold exactly."),
    ("doc9_qa3__seed42", 0.0, "PRED 'Effective Date defined in the Amended and Restated...made on October 15, 2009' doesn't give August 10, 2007; gold=August 10, 2007."),
    ("doc9_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 0.75, "PRED 'renewal term two (2) years...terminated by EKR at end of Initial Term' matches gold concept."),
    ("doc9_qa6__seed42", 0.0, "PRED 'notice period not specified'; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 1.0, "PRED 'New York law' matches gold=New York law."),
    ("doc9_qa8__seed42", 0.25, "PRED 'restriction on ability of EKR to market Competing Product' confuses PPI/EKR direction; gold=PPI and Affiliates shall not file for Marketing Authorization for Competing Product."),
    ("doc9_qa9__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'EKR may terminate for convenience upon written notice to PPI' wrong party; gold=after July 1 2015, PPI terminate with 60 days notice."),
    ("doc9_qa11__seed42", 0.25, "PRED 'Change of Control, rights may not be assigned without express consent' partial; gold=ceasing to carry on business as trigger."),
    ("doc9_qa12__seed42", 0.0, "PRED 'Note shall be nonnegotiable, may not be transferred' from promissory note; gold=neither Party shall assign without prior written consent."),
    ("doc9_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED 'Ownership of IP found in section 8.1' too vague; gold=EKR transfers NDA to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED 'Joint IP Ownership not in provided passages'; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=PPI grants EKR exclusive right and license to market/promote/sell Products."),
    ("doc9_qa17__seed42", 0.0, "PRED 'Note shall be nonnegotiable, may not be transferred' from promissory note; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=PPI/EKR Improvements mutually licensed."),
    ("doc9_qa19__seed42", 0.5, "PRED 'relevant parts related to Post-Termination Services include provisions regarding EKR's right to Step-in'; gold=if EKR exercises Step-in Right PPI cooperates; partial match."),
    ("doc9_qa20__seed42", 0.25, "PRED 'Yes, contract includes provisions related to Audit Rights'; gold=PPI cannot re-inspect same Calendar Year; confirms audit rights exist."),
    ("doc9_qa21__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitation doesn't apply if EKR required to pay excess to third party."),
    ("doc9_qa22__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 1.0, "PRED 'Yes, each Party maintain comprehensive product liability insurance, general commercial liability, and business interruption insurance' matches gold."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__v4t-tuned__batch__"
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
    print(f"cuad v4t-tuned batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
