"""Phase 1.9 Protocol A — CUAD v4t-tuned online seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__v4t-tuned__online__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# v4t-tuned online: best CUAD performer (mean≈0.4205). Tuned theta in online mode
# correctly retrieves per-document context. Doc3 (i-on hosting) gets near-perfect scores.

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement (19 entries)
    ("doc0_qa0__seed42", 1.0, "PRED 'DISTRIBUTOR AGREEMENT' matches gold exactly."),
    ("doc0_qa1__seed42", 0.0, "PRED 'Parties not explicitly identified in provided passages'; gold=Distributor (general)."),
    ("doc0_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly stated in provided passages'; gold=7th day of September, 1999."),
    ("doc0_qa3__seed42", 0.0, "PRED 'Initial Term means the period from the Effective Date to the First Renewal Date' from wrong context; gold=term commences on date Company delivers last Sample."),
    ("doc0_qa4__seed42", 0.75, "PRED 'term expires 10 years from Effective Date'; gold=10 years commencing on date Company delivers last Sample; correct duration, slight description gap."),
    ("doc0_qa5__seed42", 0.75, "PRED 'renewable for 1-year periods'; gold=renewable annually for 1-year terms up to another 10 years; correct renewal mechanism."),
    ("doc0_qa6__seed42", 1.0, "PRED 'Illinois law' matches gold exactly."),
    ("doc0_qa7__seed42", 0.25, "PRED 'Distributor has exclusive right to purchase Products from Company'; gold=Distributor shall not purchase Products from any other source; gets exclusive-purchase concept."),
    ("doc0_qa8__seed42", 0.0, "PRED 'No-Solicit Of Customers not explicitly stated'; gold=Distributor shall not interfere with Company business relations."),
    ("doc0_qa9__seed42", 0.75, "PRED 'during Term and 12 months, neither party shall solicit employees of other party'; gold=Distributor non-solicit employees during Term and 12 months after; correct scope."),
    ("doc0_qa10__seed42", 0.75, "PRED 'Distributor has right of first refusal on additional Products'; gold=Distributor right to become exclusive Distributor of other Products; correct ROFR concept."),
    ("doc0_qa11__seed42", 1.0, "PRED 'no assignment without prior written consent of other party, except Affiliate' matches gold."),
    ("doc0_qa12__seed42", 0.25, "PRED 'Company may adjust pricing upon advance notice'; gold=Company reserves right to increase/decrease price list; partial match."),
    ("doc0_qa13__seed42", 0.5, "PRED 'minimum purchase commitment...Distributor maintain minimum inventory'; gold=$250,000 minimum purchase order; confirms commitment but not the specific dollar amount."),
    ("doc0_qa14__seed42", 0.0, "PRED 'License Grant not explicitly stated for this contract'; gold=Company appoints Distributor as exclusive distributor."),
    ("doc0_qa15__seed42", 1.0, "PRED 'Company may repurchase all unsold inventory within 30 days at invoice price' matches gold."),
    ("doc0_qa16__seed42", 0.25, "PRED 'limited warranty...Products free from defects in materials and workmanship for 12 months'; gold=Company unable/unwilling to promptly repair/replace; identifies warranty but wrong aspect."),
    ("doc0_qa17__seed42", 1.0, "PRED 'Company shall carry product liability and comprehensive general liability insurance naming Distributor as additional insured' matches gold."),
    ("doc0_qa18__seed42", 1.0, "PRED 'Distributor not to challenge any IP during Term and 3 years thereafter' matches gold."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement (12 entries)
    ("doc1_qa0__seed42", 1.0, "PRED 'Promotion and Distribution Agreement' matches gold exactly."),
    ("doc1_qa1__seed42", 0.0, "PRED 'Parties not explicitly identified'; gold=Distributor."),
    ("doc1_qa2__seed42", 1.0, "PRED 'Effective Date is 1 August 2011' matches gold exactly."),
    ("doc1_qa3__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=earlier of 2 years (31 July 2013) or Maximum Distribution Commitment."),
    ("doc1_qa4__seed42", 1.0, "PRED 'English law' matches gold exactly."),
    ("doc1_qa5__seed42", 0.5, "PRED 'Section 20.4 Change of Control triggers rights'; identifies change of control clause but from wrong section numbering; gold=Change of Control triggers termination rights."),
    ("doc1_qa6__seed42", 0.5, "PRED 'assignment requires prior written consent of other party'; gold=[*] may [*] without prior written consent of [*]; correct anti-assignment concept."),
    ("doc1_qa7__seed42", 0.0, "PRED 'License Grant not in provided passages for WHITESMOKE'; gold=Google grants Distributor limited license."),
    ("doc1_qa8__seed42", 1.0, "PRED 'Google may audit Distributor records once per year' matches gold."),
    ("doc1_qa9__seed42", 0.5, "PRED 'each party limited to liability arising out of Contract Year only'; gold=each party's total liability limited per Contract Year; correct concept."),
    ("doc1_qa10__seed42", 0.75, "PRED 'each party total aggregate liability limited by Contract Year revenue ceiling'; gold=each party's total liability limited per Contract Year; correct concept with details."),
    ("doc1_qa11__seed42", 0.25, "PRED 'warranty duration limited period from date supply'; gold=Google warrants for [*] period from date of supply; correct warranty period concept."),
    # doc2 — Shenzhen LOHAS supply contract (6 entries)
    ("doc2_qa0__seed42", 1.0, "PRED 'SUPPLY CONTRACT' matches gold exactly."),
    ("doc2_qa1__seed42", 0.5, "PRED 'The seller: Shenzhen LOHAS Supply Chain Co., Ltd.'; gold=The seller: (the company name); correct party role, identifies the seller."),
    ("doc2_qa2__seed42", 0.75, "PRED 'Contract valid for 5 years from execution'; gold=Contract valid for 5 years; correct duration."),
    ("doc2_qa3__seed42", 1.0, "PRED 'Chinese law (People's Republic of China)' matches gold."),
    ("doc2_qa4__seed42", 0.0, "PRED 'Warranty Duration not in provided passages for LOHAS'; gold=within 7 days after arrival at destination."),
    ("doc2_qa5__seed42", 0.0, "PRED 'Insurance not explicitly stated'; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement (10 entries)
    ("doc3_qa0__seed42", 1.0, "PRED 'WEB SITE HOSTING AGREEMENT' matches gold exactly."),
    ("doc3_qa1__seed42", 0.75, "PRED 'Centrack International, Inc. (Client)'; gold=Centrack International; correct party identified."),
    ("doc3_qa2__seed42", 1.0, "PRED 'Agreement executed on 6th day of April, 1999' matches gold."),
    ("doc3_qa3__seed42", 0.75, "PRED 'term commences April 1, 1999'; gold=term commences April 1, 1999; exact match."),
    ("doc3_qa4__seed42", 0.75, "PRED 'initial term is six months from April 1, 1999 until September 30, 1999'; gold=April 1, 1999 plus 6 months; correct duration."),
    ("doc3_qa5__seed42", 1.0, "PRED 'Agreement automatically renews for successive one-month periods unless terminated' matches gold."),
    ("doc3_qa6__seed42", 1.0, "PRED 'written notice required to terminate before auto-renewal' matches gold."),
    ("doc3_qa7__seed42", 1.0, "PRED 'Florida law' matches gold exactly."),
    ("doc3_qa8__seed42", 1.0, "PRED 'either party may terminate Agreement upon written notice' matches gold."),
    ("doc3_qa9__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=i-on not liable for lost profits or consequential damages."),
    # doc4 — joint filing agreement (3 entries)
    ("doc4_qa0__seed42", 1.0, "PRED 'JOINT FILING AGREEMENT' matches gold exactly."),
    ("doc4_qa1__seed42", 0.0, "PRED 'Parties not explicitly listed'; gold=Co-Trustee."),
    ("doc4_qa2__seed42", 1.0, "PRED 'Agreement dated March 27, 2020' matches gold."),
    # doc5 — Adams Golf endorsement agreement (13 entries)
    ("doc5_qa0__seed42", 1.0, "PRED 'ENDORSEMENT AGREEMENT' matches gold exactly."),
    ("doc5_qa1__seed42", 0.0, "PRED 'Parties not explicitly identified'; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 1.0, "PRED 'Agreement dated January 13, 2005' matches gold."),
    ("doc5_qa3__seed42", 0.0, "PRED 'term not explicitly stated for Adams Golf agreement'; gold=term commences September 1, 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=September 1, 2004 plus [redacted] years."),
    ("doc5_qa5__seed42", 0.0, "PRED 'Governing Law not explicitly stated in this contract'; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.75, "PRED 'CONSULTANT not to associate with competitor of ADAMS GOLF during Term'; gold=CONSULTANT not to associate with ADAMS GOLF competitor's Product; correct non-compete concept."),
    ("doc5_qa7__seed42", 0.5, "PRED 'CONSULTANT shall exclusively use Adams Golf equipment'; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT; correct exclusivity concept."),
    ("doc5_qa8__seed42", 0.5, "PRED 'exception to exclusivity includes certain events such as Pro-Am or corporate outings'; gold=exception to paragraphs 4A, 4B, 4C; partial match on exclusivity exceptions."),
    ("doc5_qa9__seed42", 0.0, "PRED 'Anti-Assignment not in provided passages'; gold=Neither ADAMS GOLF nor CONSULTANT shall assign without consent."),
    ("doc5_qa10__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.5, "PRED 'CONSULTANT limited to [*] days per year for appearances'; gold=not more than [*****] days per year; correct day-cap concept."),
    ("doc5_qa12__seed42", 1.0, "PRED 'CONSULTANT grants ADAMS GOLF exclusive endorsement license to use name, image, likeness' matches gold."),
    # doc6 — Kiromic consulting agreement (14 entries)
    ("doc6_qa0__seed42", 0.5, "PRED 'CONSULTING AGREEMENT'; gold=CONSULTING AGREEMENT; exact match but answer quality is marginal."),
    ("doc6_qa1__seed42", 0.75, "PRED 'Gianluca Rotino (Consultant)'; gold=Gianluca Rotino; correct party identified."),
    ("doc6_qa2__seed42", 0.25, "PRED 'Agreement Date not explicitly stated in provided passages'; gold=July 20, 2018; fails to retrieve."),
    ("doc6_qa3__seed42", 1.0, "PRED 'effective as of July 1, 2018' matches gold exactly."),
    ("doc6_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages for KIROMICBIOPHARMA'; gold=continues until terminated."),
    ("doc6_qa5__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=Texas law."),
    ("doc6_qa6__seed42", 0.25, "PRED 'Consultant to use best efforts to not use Company information outside of Agreement'; gold=Consultant use best efforts to segregate Company work; partial."),
    ("doc6_qa7__seed42", 0.75, "PRED 'either party may terminate upon written notice to the other'; gold=either Consultant or Company may terminate upon prior written notice; correct."),
    ("doc6_qa8__seed42", 0.0, "PRED 'Anti-Assignment not in provided passages'; gold=no assignment without express written consent of Company."),
    ("doc6_qa9__seed42", 1.0, "PRED '$400/hr, maximum 19 hours per month' matches gold exactly."),
    ("doc6_qa10__seed42", 0.0, "PRED 'IP Ownership not in provided passages'; gold=Consultant irrevocably assigns all IP to Company."),
    ("doc6_qa11__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Consultant grants license if incorporating existing inventions."),
    ("doc6_qa12__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=incorporated-invention license clause."),
    ("doc6_qa13__seed42", 0.25, "PRED 'upon termination, Consultant to deliver all materials' partial; gold=deliver all property relating to Inventions; identifies return obligation."),
    # doc7 — amendment and termination of joint venture (Veoneer/Nissin Kogyo) (6 entries)
    ("doc7_qa0__seed42", 0.5, "PRED 'AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT'; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT; correct."),
    ("doc7_qa1__seed42", 0.75, "PRED 'Nissin Kogyo Co., Ltd.' matches gold exactly."),
    ("doc7_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly stated in provided passages'; gold=October 30, 2019."),
    ("doc7_qa3__seed42", 0.25, "PRED 'Effective Date is upon closing of VNBJ transaction'; gold=effective upon VNBJ Closing; partial match."),
    ("doc7_qa4__seed42", 0.0, "PRED 'Expiration Date upon closing of VNBJ and VNBZ'; gold=effective upon VNBJ/VNBZ Closing; somewhat overlapping."),
    ("doc7_qa5__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement (25 entries)
    ("doc8_qa0__seed42", 0.25, "PRED 'CO-PROMOTION AGREEMENT' (retrieval uncertain); gold=CO-PROMOTION AGREEMENT; correct name but low-confidence retrieval."),
    ("doc8_qa1__seed42", 0.0, "PRED 'Party not clearly identified as Valeant'; gold=Valeant."),
    ("doc8_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly provided'; gold=September 26, 2018."),
    ("doc8_qa3__seed42", 1.0, "PRED 'Effective Date is September 26, 2018' matches gold exactly."),
    ("doc8_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=4-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED 'Governing Law not explicitly stated'; gold=[***] law."),
    ("doc8_qa6__seed42", 0.75, "PRED 'neither Valeant nor its Affiliates shall [***] in Territory' matches gold non-compete clause."),
    ("doc8_qa7__seed42", 0.0, "PRED 'Exclusivity not explicitly stated'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa8__seed42", 0.5, "PRED 'restrictions shall not apply to [***] products previously co-promoted'; gold=restrictions shall not apply to [***]; correct exception concept."),
    ("doc8_qa9__seed42", 0.75, "PRED 'neither Party shall solicit each other's employees during Term and 12 months after'; gold=no solicit employees during Term and 12 months; correct."),
    ("doc8_qa10__seed42", 0.75, "PRED 'either Party shall have right to terminate for convenience before end of Term'; gold=either Party shall have right to terminate before end of Term; matches."),
    ("doc8_qa11__seed42", 0.25, "PRED 'assignment permitted to Affiliate; in change of control, acquiring entity must assume obligations'; gold=Dova may assign in change of control; partial."),
    ("doc8_qa12__seed42", 0.75, "PRED 'assignment requires prior written consent of other Party, except Affiliates'; gold=Party shall deliver written notice before assignment; correct anti-assignment concept."),
    ("doc8_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not explicitly stated'; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED 'Minimum Commitment not explicitly stated'; gold=Quarterly Minimum Details calculation."),
    ("doc8_qa15__seed42", 0.0, "PRED 'IP Ownership not in provided passages'; gold=Dova owns all right, title and interest in Product Materials."),
    ("doc8_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 0.5, "PRED 'Valeant's rights non-transferable'; gold=Valeant's rights non-transferable except to Affiliates; correct concept without exception."),
    ("doc8_qa18__seed42", 0.25, "PRED 'Affiliate License exists'; gold=Valeant grants Dova non-exclusive license to Valeant Property; identifies license without direction."),
    ("doc8_qa19__seed42", 0.25, "PRED 'audit rights mentioned'; gold=Dova has right to audit Valeant; identifies audit rights."),
    ("doc8_qa20__seed42", 0.5, "PRED 'limitations don't apply to indemnification obligations'; gold=limitations don't apply to indemnification or IP infringement/fraud damages; partial."),
    ("doc8_qa21__seed42", 0.0, "PRED 'Sole Remedy not in provided passages'; gold=sole remedy for Section 4.1.2 breach is fee adjustment plus termination."),
    ("doc8_qa22__seed42", 0.0, "PRED 'Liquidated Damages not in provided passages'; gold=termination compensation when Dova terminates."),
    ("doc8_qa23__seed42", 0.0, "PRED 'Insurance not in provided passages for this agreement'; gold=each Party maintain adequate insurance."),
    ("doc8_qa24__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Valeant shall not impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma strategic licensing agreement (24 entries)
    ("doc9_qa0__seed42", 0.75, "PRED 'Amended and Restated Strategic Licensing, Distribution and Marketing Agreement' matches gold."),
    ("doc9_qa1__seed42", 0.25, "PRED 'EKR Therapeutics (Maker) and Pacira Pharmaceuticals (Payee)' from promissory note; gold=F/K/A SKYEPHARMA, INC.; confuses related entities."),
    ("doc9_qa2__seed42", 1.0, "PRED 'Agreement Date is October 15, 2009' matches gold exactly."),
    ("doc9_qa3__seed42", 0.0, "PRED 'Effective Date is the Note Date, October 15, 2009'; gold=August 10, 2007; wrong date retrieved."),
    ("doc9_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 0.75, "PRED 'renewal term two (2) years, terminated by EKR at end of Initial Term' matches gold concept."),
    ("doc9_qa6__seed42", 0.0, "PRED 'Notice Period not in provided passages'; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 1.0, "PRED 'New York law' matches gold exactly."),
    ("doc9_qa8__seed42", 0.25, "PRED 'PPI shall not file for Marketing Authorization for Competing Product in Field'; gold=PPI and Affiliates shall not file for Marketing Authorization; partially correct."),
    ("doc9_qa9__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'Termination after certain date not in provided passages'; gold=after July 1 2015, PPI terminate with 60 days notice."),
    ("doc9_qa11__seed42", 0.25, "PRED 'Change of Control triggers rights' partial; gold=ceasing to carry on business as trigger."),
    ("doc9_qa12__seed42", 0.0, "PRED 'Anti-Assignment not in provided passages'; gold=neither Party shall assign without prior written consent."),
    ("doc9_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED 'IP Ownership not in provided passages'; gold=EKR transfers NDA to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED 'Joint IP Ownership not in provided passages'; gold=Joint Improvements owned jointly."),
    ("doc9_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=PPI grants EKR exclusive right and license to market/promote/sell Products."),
    ("doc9_qa17__seed42", 0.0, "PRED 'Sub-License not in provided passages'; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=PPI/EKR Improvements mutually licensed."),
    ("doc9_qa19__seed42", 0.25, "PRED 'Post-Termination Services includes provisions regarding EKR's right to Step-in'; gold=if EKR exercises Step-in Right PPI cooperates; partial."),
    ("doc9_qa20__seed42", 0.25, "PRED 'Audit Rights confirmed in contract'; gold=PPI cannot re-inspect same Calendar Year; identifies audit rights."),
    ("doc9_qa21__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitation doesn't apply if EKR required to pay excess to third party."),
    ("doc9_qa22__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 1.0, "PRED 'each Party maintain comprehensive product liability insurance, general commercial liability, and business interruption insurance' matches gold."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__v4t-tuned__online__"
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
    print(f"cuad v4t-tuned online: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
