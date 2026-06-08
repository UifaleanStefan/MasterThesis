"""Phase 1.9 Protocol A — CUAD bm25-corpus batch seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__bm25-corpus__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement
    ("doc0_qa0__seed42", 1.0, "PRED 'DISTRIBUTOR AGREEMENT' matches gold exactly."),
    ("doc0_qa1__seed42", 0.0, "PRED vague about party definitions/roles; gold=Distributor as specific party name."),
    ("doc0_qa2__seed42", 0.0, "PRED cites EKR/PPI NDA context for Agreement Date; gold='7th day of September 1999'."),
    ("doc0_qa3__seed42", 0.0, "PRED 'effective as of Effective Date, extend to 4-year anniversary' from Dova contract; gold=10-year term commencing on date Company delivers last Sample."),
    ("doc0_qa4__seed42", 1.0, "PRED 'initial term expires 10 years from date Company delivers last Sample to Distributor' matches gold."),
    ("doc0_qa5__seed42", 0.75, "PRED 'renewable annually for 1-year terms up to another 10 years, same terms' matches gold annual renewal concept."),
    ("doc0_qa6__seed42", 1.0, "PRED 'construed according to laws of State of Illinois' matches gold exactly."),
    ("doc0_qa7__seed42", 0.0, "PRED refuses exclusivity; gold=Distributor shall not purchase Products from any source other than Company."),
    ("doc0_qa8__seed42", 0.0, "PRED refuses no-solicit of customers; gold=Distributor shall not interfere with Company's business relations."),
    ("doc0_qa9__seed42", 0.0, "PRED refuses no-solicit of employees; gold=restriction on soliciting/hiring employees during Term and 12 months thereafter."),
    ("doc0_qa10__seed42", 0.0, "PRED refuses ROFR; gold=Distributor option to become exclusive Distributor of other Products with written notification."),
    ("doc0_qa11__seed42", 1.0, "PRED 'no assignment without prior written consent of Company' matches gold exactly."),
    ("doc0_qa12__seed42", 0.5, "PRED describes price restriction with CPI adjustments; gold=Company reserves right to increase/decrease price based on Company-wide changes."),
    ("doc0_qa13__seed42", 1.0, "PRED '$250,000 minimum purchase order by first of each month for 12 months to maintain exclusive rights' matches gold."),
    ("doc0_qa14__seed42", 0.75, "PRED 'yes, license granted in Section 2.1 Products License Grant' confirms license; gold=Company grants exclusive distribution rights."),
    ("doc0_qa15__seed42", 0.25, "PRED 'all rights and licenses cease, Distributor must stop reproducing' partial; gold=Company may repurchase inventory within 30 days option."),
    ("doc0_qa16__seed42", 0.0, "PRED '24 months warranty' from wrong contract; gold=warranty performance clause without specified 24-month duration."),
    ("doc0_qa17__seed42", 0.25, "PRED 'each Party maintain adequate insurance including products liability' overstates (both parties); gold=Company will carry product liability insurance."),
    ("doc0_qa18__seed42", 0.0, "PRED refuses covenant not to sue; gold=Distributor agrees not to challenge IP during Term and 3 years thereafter."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement
    ("doc1_qa0__seed42", 1.0, "PRED 'PROMOTION AND DISTRIBUTION AGREEMENT' matches gold (case-insensitive)."),
    ("doc1_qa1__seed42", 0.25, "PRED 'Whitesmoke Inc. and Google Inc.' names both parties; gold=Distributor; partial match."),
    ("doc1_qa2__seed42", 1.0, "PRED 'Effective Date 1 August 2011' matches gold."),
    ("doc1_qa3__seed42", 0.0, "PRED 'expires 1 August 2021' wrong; gold=Term ends earlier of 31 July 2013 or Maximum Distribution Commitment."),
    ("doc1_qa4__seed42", 0.0, "PRED 'governed by laws of State of Illinois' wrong; gold=English law."),
    ("doc1_qa5__seed42", 0.5, "PRED 'Change of Control: immediate termination right + notification requirement'; gold=notification requirement only; PRED adds termination not in gold."),
    ("doc1_qa6__seed42", 0.0, "PRED refuses anti-assignment; gold=[*] may [*] rights/obligations without prior written consent of [*]."),
    ("doc1_qa7__seed42", 0.75, "PRED 'yes, license granted by Google Inc. to Whitesmoke Inc.' confirms license and gets parties right; gold=Google grants Distributor limited [*] license during Term."),
    ("doc1_qa8__seed42", 1.0, "PRED 'yes, audit rights allowing Google to audit Distributor's relevant records' matches gold."),
    ("doc1_qa9__seed42", 0.25, "PRED describes consequential damages exclusion; gold=cap on total liability per Contract Year (different clause)."),
    ("doc1_qa10__seed42", 0.0, "PRED refuses cap; gold=total liability limited per Contract Year."),
    ("doc1_qa11__seed42", 0.0, "PRED '24 months warranty' from wrong contract; gold=Google warrants for [*] period from date of supply."),
    # doc2 — Shenzhen LOHAS supply contract
    ("doc2_qa0__seed42", 0.0, "PRED vague about document; gold=SUPPLY CONTRACT."),
    ("doc2_qa1__seed42", 0.25, "PRED cites seller line ('ADD:...') from contract; gold='The seller:' (asking for seller identity); partial."),
    ("doc2_qa2__seed42", 0.0, "PRED refuses expiration; gold=Contract valid for 5 years."),
    ("doc2_qa3__seed42", 0.0, "PRED 'signed in Hong Kong but doesn't specify law'; gold=People's Republic of China law or UN CISG."),
    ("doc2_qa4__seed42", 0.0, "PRED '24-month warranty' from wrong contract; gold=within 7 days after arrival at destination for quality claims."),
    ("doc2_qa5__seed42", 0.0, "PRED refuses insurance; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement
    ("doc3_qa0__seed42", 1.0, "PRED 'WEB SITE HOSTING AGREEMENT' matches gold exactly."),
    ("doc3_qa1__seed42", 0.0, "PRED vague about parties review; gold=Centrack International."),
    ("doc3_qa2__seed42", 1.0, "PRED 'Agreement Date 6th day of April, 1999' matches gold."),
    ("doc3_qa3__seed42", 0.0, "PRED 'effective after signed by both parties' wrong; gold=term commences April 1 1999."),
    ("doc3_qa4__seed42", 0.0, "PRED '15 years from Commercial Launch' from wrong contract; gold=April 1 1999 + 6 months."),
    ("doc3_qa5__seed42", 0.0, "PRED '2-year renewable periods, EKR/PPI termination' from wrong contract; gold=1-month auto-renewal periods."),
    ("doc3_qa6__seed42", 0.0, "PRED '15 days notice' wrong; gold=notice of intention not to renew 1-month periods."),
    ("doc3_qa7__seed42", 1.0, "PRED 'governed by laws of State of Florida' matches gold."),
    ("doc3_qa8__seed42", 0.5, "PRED 'either Party terminate before end of Term upon written notice' partially correct; gold=either party terminate without cause upon 30 days written notice (adds specific 30-day period)."),
    ("doc3_qa9__seed42", 0.0, "PRED cites EKR/PPI cap clause from wrong contract; gold=i-on not liable for lost profits or consequential damages."),
    # doc4 — joint filing agreement
    ("doc4_qa0__seed42", 1.0, "PRED 'JOINT FILING AGREEMENT' matches gold exactly."),
    ("doc4_qa1__seed42", 0.0, "PRED vague about party review procedures; gold=Co-Trustee."),
    ("doc4_qa2__seed42", 0.0, "PRED references EKR/PPI NDA context; gold=March 27 2020."),
    # doc5 — Adams Golf endorsement agreement
    ("doc5_qa0__seed42", 0.75, "PRED 'ENDORESEMENT AGREEMENT' (typo in spelling); gold=ENDORSEMENT AGREEMENT; substantively correct."),
    ("doc5_qa1__seed42", 0.0, "PRED vague about parties; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED refuses; gold=January 13 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED 'effective as of Effective Date, extend to 4-year anniversary' from Dova contract; gold=term commences September 1 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED '15 years from Commercial Launch or last patent claim' from EKR/PPI; gold=September 1 2004 plus [redacted] period."),
    ("doc5_qa5__seed42", 0.0, "PRED 'governed by laws of State of New York' wrong; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.0, "PRED refuses; gold=CONSULTANT shall not be associated with ADAMS GOLF competitor's Product when endorsing non-competitive product."),
    ("doc5_qa7__seed42", 0.5, "PRED 'CONSULTANT shall not use CONSULTANT'S name unless authorized by ADAMS GOLF'; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT; related exclusivity concept."),
    ("doc5_qa8__seed42", 0.25, "PRED 'unless otherwise authorized at ADAMS GOLF's sole discretion'; gold=exception to paragraphs 4A 4B 4C allowing CONSULTANT to endorse specified product; partial match on authorization exception."),
    ("doc5_qa9__seed42", 0.75, "PRED 'cannot assign without written consent of other Party'; gold=Neither ADAMS GOLF nor CONSULTANT shall assign; anti-assignment concept matches."),
    ("doc5_qa10__seed42", 0.0, "PRED refuses minimum commitment; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED refuses; gold=not more than [*****] days for TV/radio/commercials."),
    ("doc5_qa12__seed42", 1.0, "PRED 'CONSULTANT grants ADAMS GOLF exclusive right and license to use CONSULTANT'S ENDORSEMENT' matches gold exactly."),
    # doc6 — Kiromic consulting agreement
    ("doc6_qa0__seed42", 1.0, "PRED 'Consulting Agreement' matches gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 0.0, "PRED describes no-obligation disclaimer; gold=Gianluca Rotino as specific party."),
    ("doc6_qa2__seed42", 0.0, "PRED refuses; gold=July 20 2018."),
    ("doc6_qa3__seed42", 0.0, "PRED vague about Effective Date being specified but not detailed; gold='effective as of July 1 2018'."),
    ("doc6_qa4__seed42", 0.0, "PRED 'expires 5 years from effective date' wrong; gold=continues until termination."),
    ("doc6_qa5__seed42", 0.0, "PRED 'governed by laws of State of New York' wrong; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED refuses; gold=use best efforts to segregate Consultant's Company work."),
    ("doc6_qa7__seed42", 0.5, "PRED 'either party terminate without cause upon 30 days written notice'; gold=either Consultant or Company terminate upon prior written notice; adds 30 days not in gold."),
    ("doc6_qa8__seed42", 1.0, "PRED 'cannot assign without written consent of other Party'; gold=neither this Agreement nor any right may be assigned without express written consent of Company; matches."),
    ("doc6_qa9__seed42", 0.0, "PRED refuses; gold=$400/hr 19hr monthly cap, preapproved expenses."),
    ("doc6_qa10__seed42", 0.0, "PRED 'other Party shall not acquire ownership rights in Know-How' describes ownership retention; gold=Consultant irrevocably assigns ALL IP to Company."),
    ("doc6_qa11__seed42", 0.0, "PRED refuses; gold=if Consultant incorporates inventions Company gets license."),
    ("doc6_qa12__seed42", 0.75, "PRED 'yes, non-exclusive, royalty-free, perpetual, irrevocable license' has license characteristics; gold=incorporated-invention license clause."),
    ("doc6_qa13__seed42", 0.25, "PRED describes general post-termination obligations surviving; gold=deliver all property relating to Inventions upon termination."),
    # doc7 — Veoneer/Nissin Kogyo joint venture amendment
    ("doc7_qa0__seed42", 0.5, "PRED 'Joint Venture Agreement'; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT; missing prefix."),
    ("doc7_qa1__seed42", 0.0, "PRED vague about parties review; gold=Nissin Kogyo Co., Ltd."),
    ("doc7_qa2__seed42", 0.0, "PRED 'March 7 2016' original JVA date wrong; gold=October 30 2019 (amendment date)."),
    ("doc7_qa3__seed42", 0.25, "PRED 'Effective Date October 30, 2019' cites a date; gold=effective upon VNBJ Closing (event-based trigger)."),
    ("doc7_qa4__seed42", 0.0, "PRED '15 years from Commercial Launch or last patent claim' from EKR/PPI; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'governed by laws of State of New York' wrong; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement
    ("doc8_qa0__seed42", 0.25, "PRED 'Promotion Agreement' misses 'CO-'; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 0.0, "PRED refuses; gold=Valeant."),
    ("doc8_qa2__seed42", 0.0, "PRED '1 August 2011' from wrong contract; gold=September 26 2018."),
    ("doc8_qa3__seed42", 0.0, "PRED vague about Effective Date; gold='Effective Date' meaning set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED '15 years from Commercial Launch' from EKR/PPI; gold=4-year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED 'governed by laws of State of New York' wrong; gold=[***] law (redacted)."),
    ("doc8_qa6__seed42", 0.0, "PRED refuses; gold=Valeant/Affiliates shall not [***] in Territory other than the Product."),
    ("doc8_qa7__seed42", 0.0, "PRED refuses; gold=Dova grants Valeant co-exclusive right to Detail and promote Product."),
    ("doc8_qa8__seed42", 0.0, "PRED refuses competitive restriction exception; gold=Section 2.3.2 restrictions don't apply to [***]."),
    ("doc8_qa9__seed42", 0.0, "PRED refuses; gold=neither Valeant nor Dova shall solicit employees."),
    ("doc8_qa10__seed42", 0.0, "PRED refuses; gold=Either Party terminate with [***] written notice."),
    ("doc8_qa11__seed42", 0.0, "PRED describes Change of Control termination from wrong contract; gold=assignment to Affiliate without consent; Dova assignment in change of control."),
    ("doc8_qa12__seed42", 0.5, "PRED 'cannot assign without written consent except Affiliate or successor'; gold=Party delivers written notice before assignment; correct anti-assignment with partial exceptions."),
    ("doc8_qa13__seed42", 0.0, "PRED refuses; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.5, "PRED 'yes, minimum commitment in Sections 12.2.3 and 6.1' correctly identifies minimum commitment exists; gold=Quarterly Minimum Details calculation."),
    ("doc8_qa15__seed42", 0.0, "PRED 'other Party shall not acquire ownership rights in Know-How'; gold=Dova owns all right title and interest in Product Materials."),
    ("doc8_qa16__seed42", 0.0, "PRED refuses; gold=Dova grants Valeant co-exclusive right to Detail/promote."),
    ("doc8_qa17__seed42", 0.0, "PRED cites Distributor restriction from wrong contract; gold=Valeant's rights non-transferable, non-assignable, non-delegable except to Affiliates."),
    ("doc8_qa18__seed42", 0.0, "PRED refuses; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.25, "PRED 'both Dova and Valeant have audit rights' overstates (both vs Dova only); gold=Dova has right to audit Valeant."),
    ("doc8_qa20__seed42", 0.25, "PRED 'liability limited except negligence; no consequential damages' partial; gold=limitations don't limit indemnification obligations or IP infringement/fraud damages."),
    ("doc8_qa21__seed42", 0.0, "PRED cites general consequential damages cap; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination right."),
    ("doc8_qa22__seed42", 0.0, "PRED refuses; gold=termination compensation applies when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 1.0, "PRED 'each Party maintain comprehensive product liability, general commercial, business interruption insurance' matches gold exactly."),
    ("doc8_qa24__seed42", 0.0, "PRED refuses; gold=Valeant shall not do anything to impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma/Pacira amended strategic licensing agreement
    ("doc9_qa0__seed42", 1.0, "PRED 'Amended and Restated Strategic Licensing, Distribution and Marketing Agreement' matches gold exactly."),
    ("doc9_qa1__seed42", 0.25, "PRED 'PACIRA PHARMACEUTICALS, INC. and EKR THERAPEUTICS, INC.' identifies parties; gold=F/K/A SKYEPHARMA, INC.; partial match."),
    ("doc9_qa2__seed42", 1.0, "PRED 'Agreement Date October 15, 2009' matches gold."),
    ("doc9_qa3__seed42", 0.0, "PRED vague about Effective Date; gold=August 10 2007."),
    ("doc9_qa4__seed42", 0.25, "PRED '15 years from Commercial Launch or last patent claim' vs gold '15 years from Effective Date or last-to-expire licensed patent'; similar structure but wrong starting point."),
    ("doc9_qa5__seed42", 1.0, "PRED 'renewal term 2-year consecutive periods, EKR terminates at end of Initial Term with written notice to PPI' matches gold."),
    ("doc9_qa6__seed42", 0.0, "PRED cites notice periods from wrong contracts (15 days/30 days); gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 1.0, "PRED 'governed by laws of State of New York' matches gold=New York law."),
    ("doc9_qa8__seed42", 0.25, "PRED describes definition of Competing Product; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product; partial match on concept."),
    ("doc9_qa9__seed42", 0.0, "PRED refuses; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.0, "PRED 'either party terminate upon 30 days notice'; gold=only PPI after July 1 2015 with 60 days notice."),
    ("doc9_qa11__seed42", 0.0, "PRED describes Change of Control from wrong contract; gold=ceasing to carry on business as trigger."),
    ("doc9_qa12__seed42", 0.75, "PRED 'consent required, Section 13.2, cannot assign without written consent'; gold=no assignment without prior written consent; matches with section reference."),
    ("doc9_qa13__seed42", 0.0, "PRED refuses; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED 'other Party shall not acquire ownership rights in Know-How'; gold=EKR must transfer NDA to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED refuses; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 1.0, "PRED 'yes, license granted by PPI to EKR to sell/distribute/market Product in Territory' matches gold."),
    ("doc9_qa17__seed42", 0.0, "PRED 'assignment requires express consent of other Party'; gold=EKR may appoint sub-distributors with PPI notification; different clause."),
    ("doc9_qa18__seed42", 0.5, "PRED 'yes, non-exclusive, royalty-free, perpetual, irrevocable license' has license characteristics; gold=PPI/EKR Improvements mutually licensed."),
    ("doc9_qa19__seed42", 0.0, "PRED describes general post-termination effect; gold=if EKR exercises Step-in Right PPI cooperates at EKR's cost."),
    ("doc9_qa20__seed42", 0.25, "PRED 'both parties have audit rights'; gold=PPI cannot re-inspect same Calendar Year after completed inspection; overstates bidirectionality."),
    ("doc9_qa21__seed42", 0.0, "PRED cites 10.9 EKR Liability Limitation from wrong contract; gold=limitation doesn't apply if EKR required to pay excess to third party."),
    ("doc9_qa22__seed42", 0.0, "PRED cites EKR/PPI cap clause from wrong contract; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 1.0, "PRED 'each Party maintain comprehensive product liability, general commercial, business interruption insurance' matches gold exactly."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__bm25-corpus__batch__"
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
    print(f"cuad bm25-corpus batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
