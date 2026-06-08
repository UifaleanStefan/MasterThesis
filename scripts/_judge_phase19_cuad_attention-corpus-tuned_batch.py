"""Phase 1.9 Protocol A — CUAD attention-corpus-tuned batch seed42 (132 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/cuad__attention-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    # doc0 — Lime Energy distributor agreement
    ("doc0_qa0__seed42", 1.0, "PRED 'DISTRIBUTOR AGREEMENT dated September 9 1999 for LIME ENERGY CO.' contains gold=DISTRIBUTOR AGREEMENT."),
    ("doc0_qa1__seed42", 0.0, "PRED describes no-partnership/no-agency relationship; gold=Distributor as specific party name."),
    ("doc0_qa2__seed42", 0.0, "PRED 'September 9 1999' wrong day; gold='7th day of September 1999' (September 7)."),
    ("doc0_qa3__seed42", 0.0, "PRED refuses effective date; gold=term commences upon date Company delivers last Sample."),
    ("doc0_qa4__seed42", 1.0, "PRED 'initial term expires 10 years from date Company delivers last Sample' matches gold."),
    ("doc0_qa5__seed42", 0.75, "PRED 'renewable annually for 1-year terms up to 10 more years' matches gold renewal concept; 10-year cap may be approximate."),
    ("doc0_qa6__seed42", 0.0, "PRED refuses governing law; gold=Illinois law."),
    ("doc0_qa7__seed42", 0.0, "PRED refuses exclusivity; gold=Distributor shall not purchase Products from any source other than Company."),
    ("doc0_qa8__seed42", 0.0, "PRED refuses no-solicit of customers; gold=Distributor shall not interfere with Company's business relations."),
    ("doc0_qa9__seed42", 1.0, "PRED 'restriction on soliciting/hiring employees during Term and 12 months thereafter' matches gold exactly."),
    ("doc0_qa10__seed42", 0.0, "PRED refuses ROFR/ROFO; gold=Distributor option to become exclusive Distributor of other Products with written notification."),
    ("doc0_qa11__seed42", 1.0, "PRED 'consent required from Company for Distributor to assign Agreement' matches gold."),
    ("doc0_qa12__seed42", 0.25, "PRED describes price determined by separate agreement; gold=Company reserves right to increase/decrease price per unit based on Company-wide changes."),
    ("doc0_qa13__seed42", 0.0, "PRED describes delivery failure clause; gold=minimum $250,000 purchase order must be received by Company monthly for 12 months."),
    ("doc0_qa14__seed42", 0.0, "PRED incorrectly states 'no license granted'; gold=Company grants Distributor exclusive distribution rights."),
    ("doc0_qa15__seed42", 0.25, "PRED 'all rights and licenses cease upon expiration/termination' partial; gold=Company may repurchase inventory within 30 days option."),
    ("doc0_qa16__seed42", 0.0, "PRED '24 months after delivery to Distributor's end-user' from wrong contract; gold=warranty performance clause without 24-month duration."),
    ("doc0_qa17__seed42", 0.25, "PRED 'each Party maintain comprehensive product liability...insurance' from wrong contract; gold=Company will carry product liability insurance and name Distributor."),
    ("doc0_qa18__seed42", 0.0, "PRED refuses covenant not to sue; gold=Distributor agrees not to commence IP challenges during Term and 3 years thereafter."),
    # doc1 — WhiteSmoke/Google Promotion and Distribution Agreement
    ("doc1_qa0__seed42", 1.0, "PRED 'Promotion and Distribution Agreement' matches gold exactly."),
    ("doc1_qa1__seed42", 0.25, "PRED 'Whitesmoke Inc. and Google Inc.' names both parties; gold=Distributor (party category); partial match for Whitesmoke as Distributor."),
    ("doc1_qa2__seed42", 1.0, "PRED 'Effective Date 1 August 2011' matches gold=1 August 2011."),
    ("doc1_qa3__seed42", 0.0, "PRED 'expires 1 August 2021' wrong; gold=Term ends earlier of 2 years (31 July 2013) or Maximum Distribution Commitment reached."),
    ("doc1_qa4__seed42", 0.0, "PRED refuses governing law; gold=English law, exclusive jurisdiction of English courts."),
    ("doc1_qa5__seed42", 0.75, "PRED 'Change of Control includes merger/sale/asset transfer, notify in writing before or within [*] after' matches gold notification requirement."),
    ("doc1_qa6__seed42", 0.75, "PRED 'consent required from Company for any assignment without prior written consent of [*]' matches gold anti-assignment concept."),
    ("doc1_qa7__seed42", 0.0, "PRED refuses license grant; gold=Google grants Distributor limited license during Term to use Google Trademarks."),
    ("doc1_qa8__seed42", 1.0, "PRED 'audit rights during Term and period thereafter' matches gold=Google may audit during Term and [*] thereafter."),
    ("doc1_qa9__seed42", 0.25, "PRED notes certain breaches may imply uncapped liability; gold=total liability limited per Contract Year (showing cap, not uncapped)."),
    ("doc1_qa10__seed42", 0.25, "PRED describes consequential damages exclusion; gold=cap on total liability per Contract Year (different clause)."),
    ("doc1_qa11__seed42", 0.0, "PRED '24 months after delivery to Distributor end-user' from wrong contract; gold=Google warrants for [*] from date of supply to Distributor."),
    # doc2 — Shenzhen LOHAS supply contract
    ("doc2_qa0__seed42", 0.5, "PRED 'Supply Agreement'; gold=SUPPLY CONTRACT; similar but different wording."),
    ("doc2_qa1__seed42", 0.0, "PRED identifies buyer (Shenzhen LOHAS); gold='The seller:' (question asks about seller identity)."),
    ("doc2_qa2__seed42", 0.0, "PRED refuses expiration; gold=Contract valid for 5 years."),
    ("doc2_qa3__seed42", 0.0, "PRED refuses governing law; gold=law of People's Republic of China, otherwise UN Convention on CISG."),
    ("doc2_qa4__seed42", 0.0, "PRED '24 months after delivery' from wrong contract; gold=within 7 days after arrival at destination for quality claims."),
    ("doc2_qa5__seed42", 0.0, "PRED describes both-party insurance from wrong contract; gold=Seller to cover 110% invoice value against All Risks and War Risk."),
    # doc3 — i-on web hosting agreement
    ("doc3_qa0__seed42", 1.0, "PRED 'WEB SITE HOSTING AGREEMENT should be reviewed' contains gold name."),
    ("doc3_qa1__seed42", 1.0, "PRED 'Centrack International...and i-on interactive' includes gold=Centrack International."),
    ("doc3_qa2__seed42", 1.0, "PRED 'Agreement Date April 6 1999' matches gold='6th day of April 1999'."),
    ("doc3_qa3__seed42", 1.0, "PRED 'Effective Date April 1 1999' matches gold=term commences April 1 1999."),
    ("doc3_qa4__seed42", 1.0, "PRED 'initial term expires October 1 1999' correct: April 1 + 6 months = October 1."),
    ("doc3_qa5__seed42", 0.0, "PRED 'renewable annually for 1-year terms up to 10 years' wrong; gold=1-month auto-renewal periods."),
    ("doc3_qa6__seed42", 0.25, "PRED '30 days written notice to terminate renewal' adds 30-day period not in gold; gold=notice of intention not to renew 1-month periods."),
    ("doc3_qa7__seed42", 0.0, "PRED 'governed by English law' wrong; gold=Florida law."),
    ("doc3_qa8__seed42", 1.0, "PRED 'either party terminate without cause upon 30 days written notice' matches gold exactly."),
    ("doc3_qa9__seed42", 0.25, "PRED cites Clause 10.8 from wrong contract; concept of no lost profits/consequential damages matches gold."),
    # doc4 — joint filing agreement
    ("doc4_qa0__seed42", 1.0, "PRED 'JOINT FILING AGREEMENT' matches gold exactly."),
    ("doc4_qa1__seed42", 0.0, "PRED describes relationship nature (no partnership/agency); gold=Co-Trustee as specific party."),
    ("doc4_qa2__seed42", 0.0, "PRED 'April 8 2020' wrong; gold=March 27 2020."),
    # doc5 — Adams Golf endorsement agreement
    ("doc5_qa0__seed42", 1.0, "PRED 'ENDORSEMENT AGREEMENT' matches gold exactly."),
    ("doc5_qa1__seed42", 0.0, "PRED describes no-partnership relationship, doesn't name party; gold=ADAMS GOLF."),
    ("doc5_qa2__seed42", 0.0, "PRED refuses; gold=January 13 2005."),
    ("doc5_qa3__seed42", 0.0, "PRED refuses effective date; gold=term commences September 1 2004."),
    ("doc5_qa4__seed42", 0.0, "PRED refuses expiration; gold=September 1 2004 plus [redacted] years and months."),
    ("doc5_qa5__seed42", 0.0, "PRED 'laws of State of Florida' wrong; gold=Kansas law."),
    ("doc5_qa6__seed42", 0.25, "PRED 'Consultant agrees not to participate in competing business formation'; gold=CONSULTANT shall not be associated with ADAMS GOLF competitor's Product when endorsing non-competitive product; related but different restriction."),
    ("doc5_qa7__seed42", 0.0, "PRED vague about conflicting obligations; gold=CONSULTANT shall exclusively play/use MANDATORY PRODUCT."),
    ("doc5_qa8__seed42", 0.0, "PRED cites Section 2.3.1(a) from wrong contract; gold=exception to Paragraphs 4A 4B 4C in Adams Golf agreement."),
    ("doc5_qa9__seed42", 0.0, "PRED cites Dova's assignment clause with exceptions; gold=Neither ADAMS GOLF nor CONSULTANT may assign (no exceptions)."),
    ("doc5_qa10__seed42", 0.0, "PRED refuses; gold=minimum golf events on SPGA/PGA schedule."),
    ("doc5_qa11__seed42", 0.0, "PRED cites Section 3.5.2 from wrong contract (Dova speaker program); gold=not more than [*****] days for TV/radio/commercials."),
    ("doc5_qa12__seed42", 1.0, "PRED 'ADAMS GOLF exclusive right and license to use CONSULTANT'S ENDORSEMENT' matches gold exactly."),
    # doc6 — Kiromic consulting agreement
    ("doc6_qa0__seed42", 1.0, "PRED 'Consulting Agreement' matches gold=CONSULTING AGREEMENT."),
    ("doc6_qa1__seed42", 1.0, "PRED 'Kiromic, Inc. as Company and Gianluca Rotino as Consultant' includes gold=Gianluca Rotino."),
    ("doc6_qa2__seed42", 0.25, "PRED 'effective as of July 1 2018' correct month+year; gold=July 20 2018 (wrong day)."),
    ("doc6_qa3__seed42", 1.0, "PRED 'Effective Date is July 1 2018' matches gold='effective as of July 1 2018'."),
    ("doc6_qa4__seed42", 0.0, "PRED 'expires July 1 2019' wrong; gold=commences Effective Date, continues until termination."),
    ("doc6_qa5__seed42", 0.0, "PRED refuses; gold=Texas law."),
    ("doc6_qa6__seed42", 0.0, "PRED describes no-competing-business-formation; gold=use best efforts to segregate Consultant's Company work."),
    ("doc6_qa7__seed42", 0.75, "PRED 'either party terminate for convenience upon written notice with specified waiting period' matches gold termination concept."),
    ("doc6_qa8__seed42", 0.0, "PRED refuses anti-assignment clause; gold=neither this Agreement nor any right may be assigned without express written consent of Company."),
    ("doc6_qa9__seed42", 0.0, "PRED refuses; gold=$400/hr 19hr monthly cap, preapproved expenses."),
    ("doc6_qa10__seed42", 0.25, "PRED describes power-of-attorney provision; gold=Consultant irrevocably assigns ALL IP to Company (broader obligation)."),
    ("doc6_qa11__seed42", 0.0, "PRED refuses; gold=if Consultant incorporates inventions Company gets license."),
    ("doc6_qa12__seed42", 0.0, "PRED refuses irrevocable license; gold=incorporated-invention license clause exists."),
    ("doc6_qa13__seed42", 0.25, "PRED identifies surviving sections; gold=deliver all property relating to Inventions upon termination or Company's earlier request."),
    # doc7 — Veoneer/Nissin Kogyo joint venture amendment
    ("doc7_qa0__seed42", 0.75, "PRED references both 'Joint Venture Agreement March 7 2016' and 'Amendment and Termination'; gold=AMENDMENT AND TERMINATION OF JOINT VENTURE AGREEMENT; partial match."),
    ("doc7_qa1__seed42", 0.75, "PRED 'Veoneer Parties and Nissin Parties'; gold=Nissin Kogyo Co., Ltd.; PRED includes Nissin Kogyo."),
    ("doc7_qa2__seed42", 0.0, "PRED 'March 7 2016' (original JVA date) wrong; gold=October 30 2019 (amendment date)."),
    ("doc7_qa3__seed42", 0.25, "PRED 'Effective Date October 30 2019' cites a date; gold=effective upon VNBJ Closing (event-based, date may approximate closing)."),
    ("doc7_qa4__seed42", 0.0, "PRED 'expires October 30 2019' from wrong agreement context; gold=effective upon VNBJ/VNBZ Closing."),
    ("doc7_qa5__seed42", 0.0, "PRED 'governed by English law' wrong; gold=Japan law."),
    # doc8 — Dova/Valeant co-promotion agreement
    ("doc8_qa0__seed42", 0.0, "PRED refuses to identify document name; gold=CO-PROMOTION AGREEMENT."),
    ("doc8_qa1__seed42", 1.0, "PRED 'Dova Pharmaceuticals, Inc. and Valeant Pharmaceuticals North America LLC' includes gold=Valeant."),
    ("doc8_qa2__seed42", 1.0, "PRED 'Agreement Date September 26 2018' matches gold=September 26 2018."),
    ("doc8_qa3__seed42", 0.0, "PRED 'Effective Date August 10 2007' from wrong contract (EKR/PPI); gold='Effective Date' meaning set forth in preamble."),
    ("doc8_qa4__seed42", 0.0, "PRED 'expires September 26 2023' wrong calculation; gold=4 year anniversary of Effective Date."),
    ("doc8_qa5__seed42", 0.0, "PRED refuses; gold=[***] law (redacted)."),
    ("doc8_qa6__seed42", 1.0, "PRED 'Valeant and Affiliates shall not, directly or indirectly, compete in Territory other than with the Product' matches gold exactly."),
    ("doc8_qa7__seed42", 0.0, "PRED refuses exclusivity; gold=Dova grants Valeant co-exclusive right to Detail and promote Product."),
    ("doc8_qa8__seed42", 0.25, "PRED cites Section 2.3.1(a); gold cites Section 2.3.2; wrong section but correct competitive-restriction-exception concept."),
    ("doc8_qa9__seed42", 1.0, "PRED 'neither Valeant nor Dova (nor Affiliates) shall directly or indirectly solicit for hire any of the other Party's employees' matches gold exactly."),
    ("doc8_qa10__seed42", 0.25, "PRED 'either party terminate upon 30 days written notice'; gold=Either Party terminate with [***] written notice; correct bilateral right, wrong (explicit) period."),
    ("doc8_qa11__seed42", 0.25, "PRED describes Change of Control notification; gold=assignment to Affiliate without consent + Dova's assignment rights."),
    ("doc8_qa12__seed42", 0.5, "PRED 'consent required...except Affiliate or successor'; gold=Party delivers written notice before assignment; correct concept with partial exceptions."),
    ("doc8_qa13__seed42", 0.0, "PRED refuses; gold=Dova pays Valeant promotion fee based on annual Net Sales."),
    ("doc8_qa14__seed42", 0.0, "PRED refuses; gold=Quarterly Minimum Details calculation/adjustment."),
    ("doc8_qa15__seed42", 0.75, "PRED 'Dova shall have sole and exclusive rights to all IP related to Product'; gold='Dova owns all right, title and interest in Product Materials'; close match."),
    ("doc8_qa16__seed42", 0.0, "PRED 'Valeant grants Dova a license'; gold=Dova grants Valeant co-exclusive right (parties reversed in PRED)."),
    ("doc8_qa17__seed42", 0.5, "PRED 'Agreement may not be assigned without written consent of other Party'; gold=Valeant's rights non-transferable, non-assignable, non-delegable except to Affiliates; correct but misses Affiliate carve-out."),
    ("doc8_qa18__seed42", 0.0, "PRED describes Distributor restrictions from wrong contract; gold=Valeant grants Dova non-exclusive license to Valeant Property."),
    ("doc8_qa19__seed42", 0.75, "PRED 'yes, contract includes audit rights allowing a party to audit books and records'; gold=Dova has audit right; concept matches."),
    ("doc8_qa20__seed42", 0.75, "PRED 'Section 11.4 no consequential damages, does not limit indemnification for third-party claims'; gold=limitations don't limit indemnification or IP infringement/fraud damages; partial match."),
    ("doc8_qa21__seed42", 0.0, "PRED describes general liability exclusion; gold=sole remedy for breach of Section 4.1.2 is fee adjustment plus termination right."),
    ("doc8_qa22__seed42", 0.0, "PRED refuses; gold=termination compensation applies when Dova terminates per Section 12.3.1."),
    ("doc8_qa23__seed42", 1.0, "PRED 'each Party maintain adequate insurance including products liability and comprehensive general liability' matches gold exactly."),
    ("doc8_qa24__seed42", 0.0, "PRED refuses; gold=Valeant shall not do anything to impair Dova Trademarks and Copyrights."),
    # doc9 — EKR/SkyePharma/Pacira amended strategic licensing agreement
    ("doc9_qa0__seed42", 0.5, "PRED 'PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING...' abbreviated title; gold=AMENDED AND RESTATED STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT; partial."),
    ("doc9_qa1__seed42", 1.0, "PRED 'PACIRA PHARMACEUTICALS, INC. (F/K/A SKYEPHARMA, INC.) and EKR THERAPEUTICS, INC.' includes gold=F/K/A SKYEPHARMA, INC."),
    ("doc9_qa2__seed42", 1.0, "PRED 'Agreement Date October 15 2009' matches gold=October 15 2009."),
    ("doc9_qa3__seed42", 0.0, "PRED vague about effective date (says 'October 15 2009 agreement context'); gold=August 10 2007."),
    ("doc9_qa4__seed42", 0.0, "PRED 'expires October 15 2014' wrong; gold=15 years from Effective Date or last-to-expire licensed patent."),
    ("doc9_qa5__seed42", 1.0, "PRED 'renewal term is 2-year consecutive periods, terminable by EKR at end of Initial Term with 180 days notice' matches gold=automatically renew for consecutive 2-year periods."),
    ("doc9_qa6__seed42", 0.0, "PRED '15 days notice' wrong; gold=180 days prior written notice before end of Initial Term."),
    ("doc9_qa7__seed42", 0.0, "PRED refuses; gold=New York law."),
    ("doc9_qa8__seed42", 0.0, "PRED describes EKR's restriction; gold=PPI/Affiliates shall not file for Marketing Authorization for Competing Product (different party restricted)."),
    ("doc9_qa9__seed42", 0.0, "PRED refuses; gold=PPI appoints EKR as exclusive distributor in Field in Territory."),
    ("doc9_qa10__seed42", 0.25, "PRED 'either party terminate for convenience upon written notice' partial; gold=only PPI after July 1 2015 with 60 days notice (not either party)."),
    ("doc9_qa11__seed42", 0.0, "PRED describes Change of Control notification from different contract; gold=ceasing to carry on business as trigger."),
    ("doc9_qa12__seed42", 0.75, "PRED 'consent required...except Affiliate or successor in interest'; gold=no assignment without prior written consent; correct with exceptions."),
    ("doc9_qa13__seed42", 0.0, "PRED refuses; gold=EKR pays PPI royalty per [**]mg Vial."),
    ("doc9_qa14__seed42", 0.0, "PRED references Section 8.1 Ownership vaguely; gold=EKR must transfer NDA and regulatory documentation to PPI upon termination."),
    ("doc9_qa15__seed42", 0.0, "PRED refuses; gold=Joint Improvements owned jointly, PPI's interest licensed to EKR."),
    ("doc9_qa16__seed42", 0.25, "PRED 'yes, contract contains a license' confirms existence; gold=PPI grants EKR exclusive right/license to market/promote/sell Products."),
    ("doc9_qa17__seed42", 0.0, "PRED cites Distributor restriction from wrong contract; gold=EKR may appoint sub-distributors with PPI notification."),
    ("doc9_qa18__seed42", 0.0, "PRED refuses; gold=PPI Improvements licensed to EKR; EKR Improvements licensed to PPI upon PPI-initiated termination."),
    ("doc9_qa19__seed42", 0.0, "PRED identifies EKR's obligation to cease sales; gold=if EKR exercises Step-in Right PPI cooperates at EKR's cost."),
    ("doc9_qa20__seed42", 0.25, "PRED 'yes audit rights exist' partial; gold=PPI cannot re-inspect same Calendar Year after completed inspection."),
    ("doc9_qa21__seed42", 0.0, "PRED cites IP infringement liability from different contract; gold=limitation doesn't apply if EKR required to pay excess to third party per final judgment."),
    ("doc9_qa22__seed42", 0.0, "PRED refuses; gold=same limitation-exception clause."),
    ("doc9_qa23__seed42", 1.0, "PRED 'each Party maintain comprehensive product liability, general commercial, business interruption insurance' matches gold exactly."),
]


def main() -> None:
    assert len(JUDGMENTS) == 132
    qid_prefix = "cuad__attention-corpus-tuned__batch__"
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
    print(f"cuad attention-corpus-tuned batch: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
