"""Phase 1.9 Protocol A — CUAD v4t-corpus-tuned online seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__v4t-corpus-tuned__online__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# v4t-corpus-tuned online: corpus-tuned theta in online mode. Better than canonical online
# on complex contract clauses (doc0: retrieves insurance, no-solicit-customers correctly).
# Still fails on doc9 (retrieves from wrong contract context). Mean≈0.2576.

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement (19 entries)
    ("doc0_qa0__seed42", 0.0, "PRED lists sections 'Other Agreements, No License, Indemnification'; no document name; gold=DISTRIBUTOR AGREEMENT."),
    ("doc0_qa1__seed42", 0.0, "PRED 'relevant sections regarding Parties not explicitly provided'; gold=Distributor."),
    ("doc0_qa2__seed42", 0.0, "PRED 'term shall be ten (10) years commencing on date Company delivers last Sample'; answers term not Agreement Date; gold=7th day of September, 1999."),
    ("doc0_qa3__seed42", 1.0, "PRED 'term shall be ten (10) years which shall commence on the date upon which the Company delivers to Distributor the last Sample' matches gold exactly."),
    ("doc0_qa4__seed42", 1.0, "PRED 'initial term will expire ten (10) years after date upon which the Company delivers to Distributor the last Sample' matches gold."),
    ("doc0_qa5__seed42", 0.75, "PRED 'renewable on annual basis for 1-year terms for up to another 10 years' matches gold concept."),
    ("doc0_qa6__seed42", 1.0, "PRED 'law governing interpretation is laws of State of Illinois' matches gold."),
    ("doc0_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Distributor shall not purchase Products from source other than Company."),
    ("doc0_qa8__seed42", 1.0, "PRED 'Distributor agrees that it will not interfere with or otherwise disrupt the business relations between the Company or any of its affiliates and any of...' matches gold."),
    ("doc0_qa9__seed42", 0.0, "PRED 'No-Solicit Of Employees not in provided passages'; gold=restriction on soliciting/hiring employees during Term and 12 months."),
    ("doc0_qa10__seed42", 0.0, "PRED 'ROFR/ROFO/ROFN not in provided passages'; gold=Distributor right to become exclusive Distributor of other Products."),
    ("doc0_qa11__seed42", 0.0, "PRED 'Anti-Assignment not explicitly provided'; gold=no assignment without prior written consent of Company."),
    ("doc0_qa12__seed42", 0.5, "PRED 'prices subject to annual adjustments based on Consumer Price Index (CPI)'; gold=Company reserves right to increase/decrease price based on Company-wide changes; price restriction concept correct, CPI detail not in gold."),
    ("doc0_qa13__seed42", 0.5, "PRED 'Distributor must use all commercially reasonable efforts to purchase minimum quantities of the Products'; gold=$250,000 minimum purchase order; confirms minimum commitment, no dollar amount."),
    ("doc0_qa14__seed42", 0.5, "PRED 'license granted by Company to Distributor, allowing them to use name Electric City of Illinois LLC'; gold=Company appoints Distributor as exclusive distributor; identifies license but describes wrong type."),
    ("doc0_qa15__seed42", 0.25, "PRED 'termination shall not relieve either party from obligations which have occurred prior...'; gold=Company may repurchase inventory within 30 days; partial post-termination concept."),
    ("doc0_qa16__seed42", 0.0, "PRED 'does not explicitly state warranty duration'; gold=Company unable/unwilling to promptly repair/replace Products."),
    ("doc0_qa17__seed42", 1.0, "PRED 'Company will carry a reasonable amount of product liability insurance and will name the Distributor as an additional insured under that policy' matches gold exactly."),
    ("doc0_qa18__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Distributor not to challenge IP during Term and 3 years."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement (12 entries)
    ("doc1_qa0__seed42", 1.0, "PRED 'name of the contract is Promotion and Distribution Agreement' matches gold."),
    ("doc1_qa1__seed42", 0.0, "PRED 'relevant sections regarding Parties not explicitly provided'; gold=Distributor."),
    ("doc1_qa2__seed42", 1.0, "PRED 'Effective Date is 1 August 2011' matches gold."),
    ("doc1_qa3__seed42", 0.0, "PRED 'expires 1 August 2021' wrong; gold=earlier of 2 years (31 July 2013) or Maximum Distribution Commitment."),
    ("doc1_qa4__seed42", 1.0, "PRED 'This Agreement is governed by English law' matches gold."),
    ("doc1_qa5__seed42", 0.5, "PRED 'terminate Agreement immediately upon written notice if Change of Control'; gold=notification/termination right on Change of Control; matches concept."),
    ("doc1_qa6__seed42", 0.0, "PRED 'Anti-Assignment not explicitly provided'; gold=[*] may [*] without prior written consent of [*]."),
    ("doc1_qa7__seed42", 0.75, "PRED 'Yes, contract contains license granted by Google to Distributor, allowing them to bundle and distribute certain products under specified conditions'; gold=Google grants Distributor limited [*] license; confirms license and parties."),
    ("doc1_qa8__seed42", 1.0, "PRED 'Yes, Google has right to audit Distributor's relevant records to confirm compliance' matches gold."),
    ("doc1_qa9__seed42", 0.5, "PRED 'Clause 10, which states that Google will...' mentions uncapped liability; gold=total liability limited per Contract Year; partially identifies liability limitation clause."),
    ("doc1_qa10__seed42", 0.0, "PRED 'does not explicitly mention Cap On Liability'; gold=total liability limited per Contract Year."),
    ("doc1_qa11__seed42", 0.0, "PRED 'does not specify warranty duration'; gold=Google warrants for [*] period from date of supply."),
    # doc2 — Shenzhen LOHAS supply contract (6 entries)
    ("doc2_qa0__seed42", 1.0, "PRED 'SUPPLY CONTRACT' matches gold exactly."),
    ("doc2_qa1__seed42", 0.25, "PRED 'Shenzhen LOHAS Supply Chain Management Co., Ltd. as buyer and unnamed seller'; gold=The seller:; partial."),
    ("doc2_qa2__seed42", 0.0, "PRED 'does not specify Expiration Date'; gold=Contract valid for 5 years."),
    ("doc2_qa3__seed42", 0.0, "PRED 'This Agreement is governed by English law' wrong; gold=People's Republic of China law."),
    ("doc2_qa4__seed42", 0.0, "PRED 'does not specify warranty duration'; gold=within 7 days after arrival at destination for quality claims."),
    ("doc2_qa5__seed42", 0.0, "PRED 'does not contain information related to Insurance'; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement (10 entries)
    ("doc3_qa0__seed42", 0.0, "PRED mentions governing law clause, termination provisions but no document name; gold=WEB SITE HOSTING AGREEMENT."),
    ("doc3_qa1__seed42", 0.0, "PRED 'Agreement constitutes entire understanding between parties' — no party names given; gold=Centrack International."),
    ("doc3_qa2__seed42", 0.0, "PRED 'term shall commence upon April 1, 1999' answers term not Agreement Date; gold=6th day of April, 1999."),
    ("doc3_qa3__seed42", 0.0, "PRED 'Effective Date referenced in Clause 9.4 as starting a Contract Year' vague; gold=term commences April 1, 1999."),
    ("doc3_qa4__seed42", 0.75, "PRED 'initial term will expire on October 1, 1999'; gold=April 1, 1999 plus 6 months (=October 1, 1999); correct expiration date."),
    ("doc3_qa5__seed42", 1.0, "PRED 'renewal term is 1-month periods, automatically renews unless either party gives notice' matches gold."),
    ("doc3_qa6__seed42", 0.75, "PRED 'notice period is fifteen (15) days before end of term' matches gold 15-day notice requirement."),
    ("doc3_qa7__seed42", 1.0, "PRED 'governed by laws and judicial decisions of State of Florida' matches gold."),
    ("doc3_qa8__seed42", 1.0, "PRED 'either party may terminate without cause effective upon thirty (30) days written notice' matches gold exactly."),
    ("doc3_qa9__seed42", 0.0, "PRED 'does not explicitly mention Cap On Liability'; gold=i-on not liable for lost profits or consequential damages."),
    # doc4 — joint filing agreement (3 entries)
    ("doc4_qa0__seed42", 0.75, "PRED 'Joint Filing Agreement'; gold=JOINT FILING AGREEMENT; substantively correct."),
    ("doc4_qa1__seed42", 0.0, "PRED 'nothing in agreement constitutes partnership or agency' — no party name; gold=Co-Trustee."),
    ("doc4_qa2__seed42", 0.0, "PRED 'the date first set forth above' no specific date; gold=March 27, 2020."),
    # doc5 — Adams Golf endorsement agreement (13 entries)
    ("doc5_qa0__seed42", 1.0, "PRED 'ENDORSEMENT AGREEMENT' matches gold."),
    ("doc5_qa1__seed42", 0.0, "PRED 'Both parties agree...does not constitute partnership...between ADAM...' mentions ADAMS but doesn't name ADAMS GOLF as party; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly mentioned'; gold=January 13, 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED 'effective as of January 13, 2005' wrong date; gold=term commences September 1, 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED 'initial term will expire six (6) months after April 1, 1999' from wrong contract; gold=September 1, 2004 plus [redacted] years."),
    ("doc5_qa5__seed42", 0.0, "PRED 'governed by laws of State of Florida' wrong; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=CONSULTANT not to associate with competitor's Product."),
    ("doc5_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED 'clauses that define the term of the agreement' vague; gold=exception to paragraphs 4A, 4B, 4C."),
    ("doc5_qa9__seed42", 0.25, "PRED 'Neither party shall have any right to obligate or bind the other party in any manner whatsoever'; gold=Neither ADAMS GOLF nor CONSULTANT shall grant sublicenses; captures bilateral non-granting concept."),
    ("doc5_qa10__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED 'Volume Restriction not explicitly mentioned'; gold=not more than [*****] days per year for appearances."),
    ("doc5_qa12__seed42", 0.0, "PRED 'contract does not contain a license granted by one party to its counterparty'; gold=CONSULTANT grants ADAMS GOLF exclusive endorsement license."),
    # doc6 — Kiromic consulting agreement (14 entries)
    ("doc6_qa0__seed42", 0.75, "PRED 'entire Consulting Agreement should be reviewed'; identifies it as Consulting Agreement; gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.75, "PRED 'Kiromic, Inc. and Gianluca Rotino'; gold=Gianluca Rotino; includes gold name."),
    ("doc6_qa2__seed42", 0.25, "PRED 'effective as of July 1, 2018'; gold=July 20, 2018; correct month+year, wrong day."),
    ("doc6_qa3__seed42", 1.0, "PRED 'This letter agreement will be effective as of July 1, 2018' matches gold exactly."),
    ("doc6_qa4__seed42", 0.0, "PRED 'initial term will expire ten (10) years from July 1, 2018' wrong; gold=commences Effective Date, continues until terminated."),
    ("doc6_qa5__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=Consultant use best efforts to segregate Company work."),
    ("doc6_qa7__seed42", 0.75, "PRED 'either party may terminate without cause upon thirty (30) days written notice'; gold=either Consultant or Company may terminate upon prior written notice; adds 30 days."),
    ("doc6_qa8__seed42", 0.0, "PRED 'Anti-Assignment not in provided passages'; gold=neither Agreement nor any right may be assigned without express written consent of Company."),
    ("doc6_qa9__seed42", 0.0, "PRED 'Volume Restriction not in provided passages'; gold=$400/hr 19hr monthly cap."),
    ("doc6_qa10__seed42", 0.25, "PRED 'Consultant agrees that if...Consultant incorporates into any Invention...'; describes the license scenario not full assignment; gold=Consultant irrevocably assigns all IP to Company."),
    ("doc6_qa11__seed42", 0.5, "PRED 'Yes, contract contains license granted by the Company to use any proprietary subject matter owned by the Consultant that is incorporated'; gold=Consultant grants license if incorporating existing inventions; direction matches."),
    ("doc6_qa12__seed42", 0.75, "PRED 'Yes, irrevocable and perpetual' confirms both characteristics; gold=incorporated-invention license grant is irrevocable and perpetual."),
    ("doc6_qa13__seed42", 0.25, "PRED 'Sections 2, 3, 5(c), 6, 7, 8, 10 shall survive termination'; gold=deliver all property relating to Inventions upon termination; identifies post-termination obligations but not the specific delivery obligation."),
    # doc7 — amendment and termination of joint venture (Veoneer/Nissin Kogyo) (6 entries)
    ("doc7_qa0__seed42", 0.5, "PRED 'Joint Venture Agreement' missing 'AMENDMENT AND TERMINATION OF' prefix; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT."),
    ("doc7_qa1__seed42", 0.0, "PRED 'relevant passage regarding Parties not explicitly provided'; gold=Nissin Kogyo Co., Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly mentioned'; gold=October 30, 2019."),
    ("doc7_qa3__seed42", 0.0, "PRED 'effective as of July 1, 2018' from wrong contract; gold=effective upon VNBJ Closing."),
    ("doc7_qa4__seed42", 0.0, "PRED 'initial term will expire July 1, 2028' wrong; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'English law' wrong; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement (25 entries)
    ("doc8_qa0__seed42", 0.0, "PRED lists Dova commitments but no document name; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.0, "PRED 'Dova Pharmaceuticals Inc. and any other signatory'; gold=Valeant."),
    ("doc8_qa2__seed42", 0.0, "PRED 'Agreement Date not explicitly stated'; gold=September 26, 2018."),
    ("doc8_qa3__seed42", 0.5, "PRED 'effective as of Effective Date and extends until four (4) year anniversary'; gold=Effective Date set forth in preamble; PRED confirms 4-year anniversary context."),
    ("doc8_qa4__seed42", 0.0, "PRED 'expires 10 years from Effective Date which is 1 August 2011' wrong; gold=4-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=[***] law."),
    ("doc8_qa6__seed42", 0.25, "PRED 'neither Dova nor its Affiliates shall, directly or indirectly, engage in certain activities' confuses Dova/Valeant; gold=neither Valeant nor its Affiliates shall [***] in Territory; right concept, wrong party."),
    ("doc8_qa7__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa8__seed42", 0.25, "PRED 'restrictions shall not apply' confirms exception exists; gold=Notwithstanding anything, restrictions shall not apply to [***]."),
    ("doc8_qa9__seed42", 0.0, "PRED 'No-Solicit Of Employees not in provided passages'; gold=neither Valeant nor Dova shall solicit other's employees."),
    ("doc8_qa10__seed42", 0.5, "PRED 'either party may terminate without cause upon 30 days written notice'; gold=either Party shall have right to terminate before end of Term; correct bilateral right."),
    ("doc8_qa11__seed42", 0.5, "PRED 'party may terminate Agreement immediately upon written notice if Change of Control'; gold=either Party may assign to Affiliate without consent + Dova in change of control; related change of control concept."),
    ("doc8_qa12__seed42", 0.75, "PRED 'Consent required from other Party, except assignments to Affiliate'; gold=Party shall deliver written notice before assignment; correct anti-assignment with Affiliate carve-out."),
    ("doc8_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED 'Minimum Commitment not in provided passages'; gold=Quarterly Minimum Details calculation."),
    ("doc8_qa15__seed42", 0.0, "PRED 'Section 8.1.2 should be reviewed' too vague; gold=Dova owns all right, title and interest in Product Materials."),
    ("doc8_qa16__seed42", 0.0, "PRED 'does not explicitly mention License Grant related to Dova to Valeant'; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 1.0, "PRED 'Yes, Valeant's rights and obligations under Section 2.1 are non-transferable, non-assignable, non-delegable, except to Affiliates' matches gold exactly."),
    ("doc8_qa18__seed42", 0.0, "PRED 'does not explicitly mention license grant to licensee'; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.0, "PRED 'Audit Rights not in provided passages'; gold=Dova has right to audit Valeant."),
    ("doc8_qa20__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitations don't apply to indemnification or IP infringement/fraud damages."),
    ("doc8_qa21__seed42", 0.25, "PRED 'relevant passage regarding Cap On Liability is found in Article 11 Indemnification'; identifies section but no specific clause text; gold=sole remedy for breach of Section 4.1.2."),
    ("doc8_qa22__seed42", 0.0, "PRED 'Liquidated Damages not in provided passages'; gold=termination compensation when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=each Party maintain products liability, general commercial, business interruption insurance."),
    ("doc8_qa24__seed42", 0.0, "PRED 'Covenant Not To Sue not in provided passages'; gold=Valeant shall not impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma strategic licensing agreement (24 entries)
    ("doc9_qa0__seed42", 0.5, "PRED 'PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT'; gold=AMENDED AND RESTATED STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT; 'A_R' abbreviation for 'AMENDED AND RESTATED'."),
    ("doc9_qa1__seed42", 0.25, "PRED 'Pacira Pharmaceuticals, Inc. and EKR Therapeutics'; gold=F/K/A SKYEPHARMA, INC.; partial match."),
    ("doc9_qa2__seed42", 0.0, "PRED 'does not provide specific details about Agreement Date'; gold=October 15, 2009."),
    ("doc9_qa3__seed42", 1.0, "PRED 'August 10, 2007' matches gold exactly."),
    ("doc9_qa4__seed42", 0.0, "PRED 'Expiration Date not in provided passages'; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 0.0, "PRED 'Lease Term shall automatically renew at end of each calendar quarter...if Distributor complies' from wrong contract context; gold=consecutive 2-year periods auto-renewal."),
    ("doc9_qa6__seed42", 0.25, "PRED '15 days before end of term' wrong period; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 0.0, "PRED 'Governing Law not in provided passages'; gold=New York law."),
    ("doc9_qa8__seed42", 0.0, "PRED 'Non-Compete not in provided passages'; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product."),
    ("doc9_qa9__seed42", 0.0, "PRED 'Exclusivity not in provided passages'; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'either party may terminate upon 30 days' wrong period and wrong party; gold=after July 1 2015, PPI terminate with 60 days notice."),
    ("doc9_qa11__seed42", 0.5, "PRED 'Change of Control includes...' identifies Change of Control provisions; gold=ceasing to carry on business as trigger; partial match."),
    ("doc9_qa12__seed42", 0.75, "PRED 'Consent required, except in specific cases such as Affiliate' correctly identifies anti-assignment with exceptions; gold=no assignment without prior written consent."),
    ("doc9_qa13__seed42", 0.0, "PRED 'Revenue/Profit Sharing not in provided passages'; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.25, "PRED 'Pacira Pharmaceuticals is owner of or has exclusive rights to all PPI IP'; gold=EKR transfers NDA to PPI upon termination; partial IP ownership concept."),
    ("doc9_qa15__seed42", 0.0, "PRED 'Joint IP Ownership not in provided passages'; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 0.0, "PRED 'License Grant not in provided passages'; gold=PPI grants EKR exclusive right and license to market/promote/sell Products."),
    ("doc9_qa17__seed42", 0.0, "PRED 'Note shall be nonnegotiable, may not be transferred' from promissory note; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED 'Irrevocable Or Perpetual License not in provided passages'; gold=PPI/EKR Improvements mutually licensed."),
    ("doc9_qa19__seed42", 0.0, "PRED 'upon termination all rights and duties cease except post-termination obligations'; gold=if EKR exercises Step-in Right PPI cooperates."),
    ("doc9_qa20__seed42", 0.25, "PRED 'Yes, contract includes provisions for audit rights, allowing audit of counterparty'; gold=PPI cannot re-inspect same Calendar Year; confirms audit rights exist."),
    ("doc9_qa21__seed42", 0.0, "PRED 'Uncapped Liability not in provided passages'; gold=limitation doesn't apply if EKR required to pay excess to third party."),
    ("doc9_qa22__seed42", 0.0, "PRED 'Cap On Liability not in provided passages'; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 0.0, "PRED 'Insurance not in provided passages'; gold=each Party shall maintain comprehensive product liability insurance."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__v4t-corpus-tuned__online__"
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
    print(f"cuad v4t-corpus-tuned online: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
