"""
DocumentQA — sequential document reading + multi-hop question answering.

This is the flagship real-world benchmark. The agent reads a long document
one paragraph at a time ("steps") and must answer multi-hop questions after reading.

Memory pressure:
  - The document is read in one pass — the agent cannot re-read.
  - Questions require connecting facts spread across many paragraphs.
  - Context window overflow: a 50-page document has ~25,000 tokens; GPT-4o-mini
    can handle this in context, but doing so costs ~$0.004/document. With selective
    memory, the agent can answer correctly using only the relevant paragraphs.

Design:
  - Document: structured as (title, paragraphs, QA pairs).
  - Step = read one paragraph (observation = paragraph text).
  - After all paragraphs: Q&A phase begins (observation = question).
  - Reward: BERTScore or exact-match against ground truth answer.

Built-in document library (no external API needed for documents):
  - FANTASY_LORE: a 40-paragraph fantasy world lore document with 10 QA pairs.
  - MYSTERY_CASE: a detective case file with 30 paragraphs and 8 QA pairs.
  - SCIENCE_FACTS: a science overview with 25 paragraphs and 6 QA pairs.

For real thesis experiments, GPT-4o generates additional documents and QA pairs.

Scoring:
  - Without LLM: exact_match score (1.0 if answer string in predicted answer).
  - With LLM: GPT-4o-mini scores the answer (0–1 based on semantic similarity).
  - partial_score = n_correct / n_questions.

Usage:
    env = DocumentQA(document_name="fantasy_lore", seed=0)
    obs = env.reset()          # returns first paragraph
    while not env.done:
        obs, done, success = env.step(action)   # action = memory retrieval choice
    # After reading, query phase:
    # observe question, agent answers via LLM, score recorded.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

# ------------------------------------------------------------------
# Built-in document library
# ------------------------------------------------------------------

FANTASY_LORE = {
    "title": "The Lore of Eldenmoor",
    "paragraphs": [
        "Eldenmoor is an ancient kingdom situated between the Ironback Mountains and the Silverleaf Forest. It was founded 800 years ago by Queen Mira the Wise after the Great Sundering.",
        "Queen Mira commissioned the construction of three great towers: the Tower of Dawn in the east, the Tower of Dusk in the west, and the Tower of Midnight at the kingdom's center.",
        "The Tower of Dawn houses the Sunstone, a magical artifact that controls the weather. Only descendants of Queen Mira may touch it without being burned.",
        "The Silver Key is kept in the Tower of Dusk. It is said to unlock the Vault of Ages, which contains the kingdom's most powerful magical artifacts.",
        "The Tower of Midnight is home to the Grand Library, containing every written work ever produced in Eldenmoor. The librarian Aldred has served there for 200 years.",
        "Three centuries after Mira's reign, King Valorin the Reckless attempted to use the Sunstone to end a drought. He was burned and died. His son, Valorin II, swore never to touch it.",
        "The Vault of Ages was last opened 150 years ago by Archmage Selen. She retrieved the Binding Chains, used to imprison the demon Xareth beneath the Ironback Mountains.",
        "Xareth the demon was defeated by the hero Jorin in the Second Age. Jorin used a blade forged from starfall iron, which he found in the ruins of Old Eldenmoor.",
        "Old Eldenmoor lies beneath the Silverleaf Forest, abandoned after the Great Sundering. Ruins contain the original crown of the first king, which is cursed.",
        "The Great Sundering occurred when the magical ley lines beneath Eldenmoor collapsed, splitting the original kingdom into three. Queen Mira reunited them.",
        "Aldred the librarian is secretly the last member of the Order of the Pale Moon, a secret society that has protected the kingdom from magical threats for 500 years.",
        "The Order of the Pale Moon possesses a Shadow Key that opens a hidden chamber in the Tower of Midnight. The chamber contains the Mirror of Truth.",
        "The Mirror of Truth shows anyone who gazes into it their greatest regret. King Valorin II gazed into it and saw his father's death. He wept for three days.",
        "The Ironback Mountains are home to the Stonekin, dwarven craftspeople who forge all metal in Eldenmoor. They are led by Elder Gruk, who is 400 years old.",
        "Elder Gruk knows the location of the original starfall iron deposit used to forge Jorin's blade. He refuses to share it, saying the world is not ready.",
        "The Silverleaf Forest is inhabited by the Sylvan, elven guardians of the ancient ruins. They permit no outsiders near Old Eldenmoor.",
        "The Sylvan leader, High Warden Elyss, is Queen Mira's granddaughter. She is 600 years old and remembers the Great Sundering firsthand.",
        "High Warden Elyss guards the Original Crown, preventing anyone from wearing it. The crown grants absolute authority over Eldenmoor but curses its wearer with madness.",
        "The current ruler of Eldenmoor is King Aldric III. He is the 23rd monarch in Mira's lineage and has never visited the Tower of Dawn.",
        "King Aldric III is searching for the Silver Key, believing it can open not just the Vault of Ages but also a personal vault left by his ancestor Valorin II.",
        "Valorin II's personal vault is hidden in the foundations of the palace. It contains his personal journals, detailing what he saw in the Mirror of Truth.",
        "The journals of Valorin II reveal that the Mirror of Truth showed him not just his father's death but a vision: Xareth will escape in 200 years if the Binding Chains are not renewed.",
        "The Binding Chains weaken over time. Archmage Selen renewed them 150 years ago. They are due for renewal again in 50 years.",
        "To renew the Binding Chains, one must: retrieve them from the Vault of Ages (using the Silver Key), take them to the Ironback Mountains, and have the Stonekin reforge them.",
        "Elder Gruk will only reforge the Binding Chains if given a gift of starfall iron. The only remaining starfall iron is in Jorin's original blade, kept in the Grand Library.",
        "Jorin's blade is displayed in a locked case in the Grand Library. Only Aldred has the key to the case. He will only give it to a descendant of Jorin.",
        "Jorin's lineage survived in secret. His last known descendant is a farmer named Petra who lives in the village of Mosswater, south of Eldenmoor.",
        "Petra does not know of her heritage. She was told by her grandmother that their family 'had an old sword' but thought it was a metaphor.",
        "To save Eldenmoor from Xareth: find Petra → prove her heritage → get blade from Aldred → give to Gruk → get renewed chains → seal Xareth.",
        "The path requires: Silver Key (Tower of Dusk) → Vault of Ages → Binding Chains → Ironback Mountains. Additionally: Petra → Grand Library → Jorin's Blade → Elder Gruk.",
        "The Shadow Key held by Aldred (the Pale Moon librarian) also opens the Grand Library's restricted section, which contains the Jorin family records proving Petra's heritage.",
        "High Warden Elyss knows Petra's heritage independently, having witnessed Jorin's marriage 800 years ago. She will verify Petra's claim without the library records.",
        "King Aldric III, if informed about Xareth, will immediately provide the Silver Key to whoever takes on the quest, despite its personal significance to him.",
        "The Tower of Dusk is guarded by the Dusk Knights, who will only admit those with a royal seal from King Aldric III.",
        "The Dusk Knights' captain, Sera, is actually a spy for a foreign kingdom. She intends to steal the Silver Key once it is retrieved.",
        "Sera's true allegiance is to the Kingdom of Ashfall, which seeks the Silver Key to plunder the Vault of Ages before Eldenmoor can renew the Binding Chains.",
        "Ashfall's king, Maldrek, believes that freeing Xareth will give him power over the demon. He is wrong; Xareth serves no mortal.",
        "The only way to stop Ashfall is to expose Sera. Aldred knows Sera is a spy — he has seen it in the Mirror of Truth — but has not spoken because it is not yet time.",
        "The resolution: King Aldric → Silver Key → Vault → Chains → Gruk → Petra → Blade → Gruk reforges → renew Xareth's binding. Expose Sera at the Dusk Tower.",
        "The prophecy of the Pale Moon says: 'When the chains grow thin and the mirror weeps, the farmer's blood shall save what kings cannot.' This refers to Petra.",
    ],
    "qa_pairs": [
        {
            "question": "Who is the last known descendant of Jorin, the hero who defeated Xareth?",
            "answer": "Petra, a farmer living in the village of Mosswater, south of Eldenmoor.",
            "relevant_paragraphs": [8, 26, 27, 28],
        },
        {
            "question": "What must be done to renew the Binding Chains that imprison Xareth?",
            "answer": "Retrieve the chains from the Vault of Ages using the Silver Key, take them to the Ironback Mountains, and have the Stonekin reforge them using starfall iron from Jorin's blade.",
            "relevant_paragraphs": [23, 24, 25],
        },
        {
            "question": "Who guards the Original Crown and why?",
            "answer": "High Warden Elyss, Queen Mira's granddaughter, guards the Original Crown to prevent anyone from wearing it because it grants absolute authority but curses its wearer with madness.",
            "relevant_paragraphs": [16, 17, 18],
        },
        {
            "question": "What did King Valorin II see in the Mirror of Truth?",
            "answer": "He saw his father's death and a vision that Xareth will escape in 200 years if the Binding Chains are not renewed.",
            "relevant_paragraphs": [12, 21, 22],
        },
        {
            "question": "Who is the spy in the Dusk Knights and who do they serve?",
            "answer": "Sera, the captain of the Dusk Knights, is a spy for the Kingdom of Ashfall and its king, Maldrek.",
            "relevant_paragraphs": [34, 35, 36],
        },
        {
            "question": "What is the Shadow Key and who possesses it?",
            "answer": "The Shadow Key is held by Aldred the librarian, who is secretly the last member of the Order of the Pale Moon. It opens a hidden chamber in the Tower of Midnight containing the Mirror of Truth.",
            "relevant_paragraphs": [10, 11, 12],
        },
        {
            "question": "How long ago were the Binding Chains last renewed and when must they be renewed again?",
            "answer": "The Binding Chains were last renewed 150 years ago by Archmage Selen. They must be renewed again in 50 years.",
            "relevant_paragraphs": [6, 7, 22, 23],
        },
        {
            "question": "What gift does Elder Gruk require before he will reforge the Binding Chains?",
            "answer": "Elder Gruk requires a gift of starfall iron, which is the material in Jorin's original blade kept in the Grand Library.",
            "relevant_paragraphs": [24, 25],
        },
    ],
}

MYSTERY_CASE = {
    "title": "The Ashford Mansion Mystery",
    "paragraphs": [
        "At 11:47 PM on a stormy Thursday, the butler of Ashford Mansion, Mr. Harwick, discovered Lord Ashford dead in the study. The cause of death: poisoning.",
        "Lord Ashford had been attending a dinner party with five guests: Lady Pemberton, Dr. Crane, the artist Viola, his nephew Gerald, and his business partner Simmons.",
        "Dr. Crane was the last known person to speak with Lord Ashford at 10:30 PM. He claims they discussed the estate's finances.",
        "Lady Pemberton arrived at the mansion at 7:00 PM, an hour earlier than invited. She was seen near the study at 8:15 PM by the cook.",
        "Viola the artist had been commissioned to paint Lord Ashford's portrait. She was in the art room from 9:00 PM until the discovery.",
        "Gerald, the nephew, stands to inherit the estate. He was overheard arguing with Lord Ashford about the will at 6:00 PM.",
        "Simmons, the business partner, claims he was on the phone in the library from 9:30 PM to 11:00 PM. Phone records confirm a 45-minute call ending at 10:15 PM.",
        "The poison used was belladonna, which takes 2-4 hours to cause death. Lord Ashford died at approximately 11:45 PM, placing the poisoning between 7:45 PM and 9:45 PM.",
        "A glass of brandy found in the study tested positive for belladonna. The brandy decanter was from a locked cabinet that only Lord Ashford and Mr. Harwick had keys to.",
        "Mr. Harwick claims he never opened the locked cabinet that evening. However, his fingerprints were found on the inner handle of the cabinet.",
        "Lady Pemberton is a trained herbalist with extensive knowledge of plant-based poisons. She grows belladonna in her garden, which police confirmed.",
        "Dr. Crane prescribed Lord Ashford medication that reacts badly with belladonna, effectively doubling its lethality. He knew this.",
        "Gerald recently discovered that Lord Ashford had changed his will two weeks ago, leaving the estate to a charity rather than Gerald. Gerald did not know the new beneficiary.",
        "Viola overheard Lady Pemberton saying 'he deserves what's coming' on the phone at 8:00 PM. Viola did not come forward until questioned.",
        "The cook saw Lady Pemberton near the study with a small vial in her hand at 8:15 PM. The cook assumed it was perfume.",
        "A search of Lady Pemberton's room found dried belladonna leaves and a mortar and pestle with traces of the poison.",
        "Simmons lied about the phone call duration. Records show it ended at 10:15 PM, but witnesses saw him in the library until 11:00 PM.",
        "The missing 45 minutes of Simmons's alibi cannot be accounted for. He refuses to explain where he was.",
        "Lord Ashford's new will left the estate to a cancer research charity in memory of his late wife, who died of cancer two years ago.",
        "Lady Pemberton had loaned Lord Ashford £50,000 five years ago. According to recently discovered documents, the loan was contingent on marriage — if Ashford married anyone else, the loan became due immediately.",
        "Lord Ashford had recently become engaged to a woman named Clara. The engagement was announced the day of the dinner party.",
        "The engagement to Clara would have triggered the repayment clause in Lady Pemberton's loan agreement. She would have lost £50,000.",
        "Conclusion: Lady Pemberton poisoned the brandy between 8:00 PM and 8:30 PM using belladonna from her own garden. Motive: the engagement triggered a £50,000 debt.",
        "Dr. Crane is complicit but not guilty of murder: he knew Ashford's medication would react with belladonna but said nothing, hoping Ashford's death would prevent a business deal that would ruin Crane.",
        "Gerald is innocent. His argument was about a minor allowance dispute unrelated to the will.",
        "Simmons's missing time was spent meeting with a journalist about Ashford's financial irregularities. He feared being implicated and kept quiet.",
        "Viola is an innocent witness who delayed coming forward due to fear of involvement.",
        "Mr. Harwick opened the cabinet to retrieve a glass for Lord Ashford earlier in the evening, before the poison was added. He is innocent.",
        "The case is solved: Lady Pemberton is charged with murder. Dr. Crane is charged with accessory after the fact for withholding knowledge.",
    ],
    "qa_pairs": [
        {
            "question": "Who poisoned Lord Ashford and what was their motive?",
            "answer": "Lady Pemberton poisoned Lord Ashford. Her motive was that his engagement to Clara triggered a loan repayment clause: Ashford would owe her £50,000 if he married anyone else.",
            "relevant_paragraphs": [3, 10, 14, 19, 20, 21, 22],
        },
        {
            "question": "When was the poison administered, based on the timeline?",
            "answer": "Between 7:45 PM and 9:45 PM (2-4 hours before death at 11:45 PM), most likely between 8:00 PM and 8:30 PM when Lady Pemberton was seen near the study.",
            "relevant_paragraphs": [7, 14],
        },
        {
            "question": "Why is Dr. Crane considered complicit but not guilty of murder?",
            "answer": "Dr. Crane knew that Ashford's medication reacted badly with belladonna but said nothing, hoping Ashford's death would prevent a business deal that would ruin Crane.",
            "relevant_paragraphs": [11, 23],
        },
        {
            "question": "What was Simmons doing during his unaccounted 45 minutes?",
            "answer": "Simmons was meeting with a journalist about Ashford's financial irregularities. He kept quiet out of fear of being implicated.",
            "relevant_paragraphs": [6, 16, 17, 25],
        },
        {
            "question": "Who does Lord Ashford's new will leave the estate to, and why?",
            "answer": "The estate was left to a cancer research charity in memory of Ashford's late wife, who died of cancer two years ago.",
            "relevant_paragraphs": [12, 18],
        },
    ],
}

CORPORATE_MEMO = {
    "title": "Project Northstar — Acquisition Strategy",
    "paragraphs": [
        "Helix Industries is preparing a hostile acquisition of Vantage Robotics. The deal is codenamed Project Northstar inside Helix.",
        "Helix's CEO, Marina Voss, took the helm in 2023 after the previous CEO Daniel Krane was forced out following an accounting scandal.",
        "Vantage Robotics, founded by engineer Anil Sahota in 2018, holds five patents on warehouse picking arms — assets that Helix lacks internally.",
        "Anil Sahota owns 41% of Vantage and serves as its CTO. He has publicly opposed every previous merger overture from larger firms.",
        "Vantage's CEO is Joanna Reyes, a former McKinsey partner brought in to professionalize operations after a Series C in 2022.",
        "Joanna Reyes holds 6% of Vantage and has expressed openness to acquisition in private conversations with Helix's banker Konrad Lin.",
        "Konrad Lin works at Sterling Wickfield, the boutique investment bank Helix retained in January. Lin reports directly to Marina Voss.",
        "Sterling Wickfield's M&A track record includes 18 completed tech deals between 2020 and 2025, with a median premium of 27%.",
        "The proposed Northstar offer is $14 per share — a 31% premium over Vantage's 90-day VWAP of $10.69.",
        "Vantage's outstanding shares total 88 million, making the headline equity value $1.232 billion at the proposed offer.",
        "Helix's balance sheet has $410 million in cash and the remainder of the deal would be financed via $850 million in senior debt from Citadel Capital.",
        "Citadel Capital's credit committee meets on the third Wednesday of every month. The next meeting is May 21st.",
        "Helix's general counsel, Priya Nair, has flagged that the acquisition may trigger an HSR Act review at the FTC because both firms operate in U.S. logistics automation.",
        "Priya Nair previously led FTC defense at Cleary Gottlieb. She estimates a 60-70% probability the FTC will issue a Second Request for documents.",
        "A Second Request would extend the closing timeline by 6-9 months. Helix's debt commitment from Citadel expires after 12 months from signing.",
        "If the FTC review lasts longer than the debt commitment window, Helix would need to renegotiate financing or walk away.",
        "Vantage's largest customer is GoodieBox, which represents 38% of Vantage's 2024 revenue. GoodieBox's CEO Ramona Vela has expressed concern about supplier consolidation.",
        "If GoodieBox terminates its master services agreement post-acquisition, Vantage would lose roughly $112 million in annual recurring revenue.",
        "GoodieBox's MSA contains a 'change of control' clause permitting termination on 90 days' notice if Vantage is acquired by a competitor in U.S. logistics automation.",
        "Marina Voss has held two informal dinners with Ramona Vela in March, exploring whether Helix could secure a successor MSA pre-close.",
        "Ramona Vela's stated condition is a 5-year price lock and exclusivity from any Helix-aligned competitor — terms Helix's CFO David Ling estimates would cost $40-60 million NPV.",
        "The Northstar bid book identifies three contingency scenarios: Plan A (friendly approach, 90% confidence), Plan B (tender offer, 60% confidence), Plan C (proxy contest, 35% confidence).",
        "Plan B requires Helix to win >50% of voting shares directly. Anil Sahota's 41% stake is the key blocker; Helix must convert him or buy out non-Sahota shareholders.",
        "Plan C — a proxy contest to replace Vantage's board — would take 12-18 months and is considered a last resort by Voss.",
        "Marina Voss has authorized confidential outreach to Anil Sahota via mutual contact Theo Marsh, an early Vantage investor who sold his stake in 2023.",
        "Theo Marsh has agreed to facilitate a meeting if and only if Helix commits in writing that Sahota retains the CTO role and creative control of the patent roadmap.",
        "Sahota's known objection to merger overtures has historically been loss of control — he has rejected three prior bids over governance terms, not price.",
        "Helix is also evaluating an alternative target, Argo Mechatronics, with overlapping technology and a more cooperative CEO. Argo's headline price would be $940M.",
        "Argo lacks Vantage's GoodieBox relationship but has stronger European footprint. CFO David Ling considers Argo the safer Plan B if Northstar fails.",
        "The board sub-committee reviewing Northstar must vote by April 30th. As of April 25th, four of seven members have signaled support; two are undecided; one (Lena Ortiz) is publicly skeptical.",
    ],
    "qa_pairs": [
        {
            "question": "Why does Helix want to acquire Vantage Robotics specifically, and what is the codename of the deal?",
            "answer": "Helix wants Vantage's five warehouse picking arm patents that Helix lacks internally. The deal codename is Project Northstar.",
            "relevant_paragraphs": [0, 2],
        },
        {
            "question": "Who is the largest individual shareholder of Vantage and what is his historical view on mergers?",
            "answer": "Anil Sahota, the founder and CTO, owns 41% and has publicly opposed every previous merger overture, primarily over governance and control issues rather than price.",
            "relevant_paragraphs": [3, 26],
        },
        {
            "question": "What financing does Helix have lined up and what is the deadline risk?",
            "answer": "$410M cash plus $850M senior debt from Citadel Capital, but the debt commitment expires after 12 months from signing — risky if the FTC issues a Second Request that extends closing by 6-9 months.",
            "relevant_paragraphs": [10, 14, 15],
        },
        {
            "question": "Why does the GoodieBox relationship matter to the deal?",
            "answer": "GoodieBox is Vantage's largest customer at 38% of revenue with a change-of-control termination clause; losing it post-acquisition would cost ~$112M ARR. Helix is preemptively negotiating with GoodieBox CEO Ramona Vela.",
            "relevant_paragraphs": [16, 17, 18, 19],
        },
        {
            "question": "Who is mediating the outreach to Anil Sahota and what condition has he set?",
            "answer": "Theo Marsh, an early Vantage investor who sold in 2023, will facilitate the meeting only if Helix commits in writing that Sahota keeps the CTO role and creative control of the patent roadmap.",
            "relevant_paragraphs": [24, 25],
        },
        {
            "question": "What legal review is expected and why?",
            "answer": "An HSR Act review at the FTC, because both Helix and Vantage operate in U.S. logistics automation. General counsel Priya Nair estimates a 60-70% probability of a Second Request.",
            "relevant_paragraphs": [12, 13],
        },
        {
            "question": "What are the three contingency plans for Northstar and which is considered last resort?",
            "answer": "Plan A is a friendly approach (90% confidence), Plan B is a tender offer (60%), Plan C is a proxy contest (35%). Plan C — proxy contest to replace the board — is the last resort.",
            "relevant_paragraphs": [21, 23],
        },
        {
            "question": "What is the alternative acquisition target and what trade-off does it represent?",
            "answer": "Argo Mechatronics at ~$940M. Argo lacks Vantage's GoodieBox relationship but has a stronger European footprint and a more cooperative CEO. CFO David Ling considers Argo the safer fallback if Northstar fails.",
            "relevant_paragraphs": [27, 28],
        },
    ],
}

SOAP_OPERA = {
    "title": "Crestwood Hollow — Family Saga",
    "paragraphs": [
        "Eleanor Crestwood is the matriarch of the Crestwood family and the owner of the Crestwood Hollow estate, valued at $40 million.",
        "Eleanor has three children: Charles (eldest, 52), Diana (49), and the youngest, Marcus (44).",
        "Charles Crestwood manages the family vineyard. He is married to Beatrice, but is secretly having an affair with the estate's architect, Felicity Tran.",
        "Beatrice does not know about the affair, but the housekeeper Margaret has photographic evidence and is using it to extort Charles for $50,000 a year.",
        "Diana Crestwood is a divorce attorney. She divorced her husband Roman last year after discovering he had fathered a child with her sister-in-law Beatrice eight years ago.",
        "The child of Roman and Beatrice is named Olivia, who Beatrice has raised believing Charles is the father. Olivia is now 8 years old.",
        "Marcus Crestwood is a recovering addict who runs the family charity. He stole $200,000 from the charity in 2023 to pay gambling debts but replaced the funds before discovery.",
        "Eleanor is dying of late-stage pancreatic cancer. Her doctor, Dr. Renato Vega, gave her three months to live as of January.",
        "Eleanor has not yet updated her will. Currently it splits the estate equally three ways: Charles, Diana, Marcus.",
        "Eleanor's lawyer, Bertram Cassidy, has been pressuring her to update the will to exclude Marcus due to his addiction history. Eleanor has resisted out of love.",
        "Bertram Cassidy is secretly in love with Diana and would benefit professionally if Marcus's share went to her instead.",
        "Felicity Tran (Charles's mistress) is pregnant. The child would be Charles's second secret child if Olivia's true paternity remains hidden.",
        "Beatrice has begun to suspect the affair after finding Felicity's perfume on Charles's coat in March.",
        "Roman, Diana's ex-husband, has filed a paternity claim demanding visitation rights with Olivia. The case is sealed but the family has been notified.",
        "If Roman wins the paternity claim, Charles would learn for the first time that Olivia is not his daughter.",
        "Margaret the housekeeper is also Roman's cousin and has been secretly feeding him information about the family for years.",
        "Margaret's $50,000 extortion of Charles funds the legal fees for Roman's paternity case.",
        "Marcus has fallen back into gambling and now owes $400,000 to a loan shark named Vincent Po. Vincent has threatened Eleanor's safety unless Marcus pays.",
        "Marcus has approached Eleanor's nurse, Yvette, asking her to administer a higher morphine dose to accelerate Eleanor's death so he can inherit sooner.",
        "Yvette refused but did not report Marcus to the police. She has, however, told Diana in confidence.",
        "Diana is now considering whether to tell Eleanor about Marcus's request, knowing it will likely cause Eleanor to cut Marcus from the will entirely.",
        "If Marcus is cut from the will, his $13.3M share would be split between Charles and Diana, giving each $20M instead of $13.3M.",
        "Felicity Tran's pregnancy is six weeks along. She has not told Charles. She is considering raising the child alone in Lisbon, where her sister lives.",
        "Bertram Cassidy has discovered the gambling debt and approached Vincent Po offering to pay $200,000 for evidence Marcus committed the murder attempt — Yvette's testimony.",
        "Vincent Po is willing to provide the evidence in exchange but only if Marcus's debt is wiped clean — i.e., Bertram or Diana must pay $400,000.",
        "Diana has agreed in principle to pay Vincent. She has not yet told Bertram. She also has not told Charles.",
        "Eleanor, sensing tension, has called all three children to the estate for a Thanksgiving reconciliation dinner — the first time the family has gathered in two years.",
        "At the dinner, Beatrice plans to confront Charles publicly about the affair using the photos she stole from Margaret's office.",
        "Roman has indicated he will arrive at the dinner uninvited, claiming his right to see Olivia. The estate's security has been alerted.",
        "The dinner is on November 28th. Eleanor's doctor Dr. Renato Vega has indicated Eleanor will not survive past mid-December.",
    ],
    "qa_pairs": [
        {
            "question": "Who is the true biological father of Olivia, and what does Charles believe?",
            "answer": "Roman, Diana's ex-husband, is Olivia's biological father (he fathered her with Beatrice eight years ago). Charles believes he is Olivia's father.",
            "relevant_paragraphs": [4, 5],
        },
        {
            "question": "What is Margaret the housekeeper extorting Charles for, and what is she doing with the money?",
            "answer": "Margaret is extorting Charles for $50,000 a year using photographic evidence of his affair with Felicity Tran. She is using the money to fund her cousin Roman's sealed paternity case for visitation with Olivia.",
            "relevant_paragraphs": [3, 15, 16],
        },
        {
            "question": "What did Marcus ask Yvette the nurse to do, and who knows?",
            "answer": "Marcus asked Yvette to administer a higher morphine dose to accelerate Eleanor's death so he could inherit sooner. Yvette refused and told Diana in confidence; Eleanor and the police do not yet know.",
            "relevant_paragraphs": [18, 19, 20],
        },
        {
            "question": "How would Diana benefit financially if Marcus is cut from the will?",
            "answer": "Marcus's $13.3M share would split between Charles and Diana, giving each $20M instead of the original $13.3M.",
            "relevant_paragraphs": [21],
        },
        {
            "question": "What is Bertram Cassidy's hidden motive in pressuring Eleanor about the will?",
            "answer": "Bertram is secretly in love with Diana and would benefit professionally if Marcus's share went to her instead.",
            "relevant_paragraphs": [10],
        },
        {
            "question": "Why is Vincent Po threatening the family, and how is the threat being addressed?",
            "answer": "Vincent Po is owed $400,000 by Marcus for gambling debts. Diana has agreed in principle to pay Vincent in exchange for Yvette's testimony about Marcus's murder request, brokered by Bertram Cassidy.",
            "relevant_paragraphs": [17, 23, 24, 25],
        },
        {
            "question": "What does Felicity Tran know that Charles does not, and where is she considering going?",
            "answer": "Felicity is six weeks pregnant with Charles's child and has not told him. She is considering raising the child alone in Lisbon where her sister lives.",
            "relevant_paragraphs": [11, 22],
        },
        {
            "question": "What three confrontations are expected to happen at the November 28th Thanksgiving dinner?",
            "answer": "Beatrice will publicly confront Charles about the affair using stolen photos; Roman will arrive uninvited claiming visitation with Olivia; and the family will be gathered for the first time in two years with Eleanor expected to not survive past mid-December.",
            "relevant_paragraphs": [27, 28, 29],
        },
    ],
}

SCIENCE_SURVEY = {
    "title": "Detecting Exoplanets — A Methods Survey",
    "paragraphs": [
        "Exoplanets are planets that orbit stars other than our Sun. As of 2025, more than 5,800 exoplanets have been confirmed.",
        "The first widely-accepted exoplanet detection was 51 Pegasi b in 1995, by Mayor and Queloz, who later won the 2019 Nobel Prize in Physics for the discovery.",
        "51 Pegasi b is a hot Jupiter — a gas giant orbiting very close to its star, completing one orbit in 4.2 days. The detection used the radial velocity method.",
        "The radial velocity method measures the Doppler shift of a star's spectrum caused by the gravitational tug of an orbiting planet. It works best for massive, close-in planets.",
        "The transit method detects the small periodic dimming of a star's light when a planet passes in front of it. NASA's Kepler space telescope, launched in 2009, used this method.",
        "Kepler discovered roughly 2,800 confirmed exoplanets before its mission ended in 2018, including the first Earth-sized planets in the habitable zone of their stars.",
        "Kepler's most famous discovery is Kepler-186f, an Earth-sized planet in the habitable zone of a red dwarf star 500 light years away. It was announced in 2014.",
        "The TESS (Transiting Exoplanet Survey Satellite) mission, launched in 2018, succeeded Kepler. TESS targets nearby bright stars across the entire sky.",
        "TESS has discovered more than 7,000 candidate exoplanets and more than 400 confirmed ones as of mid-2025.",
        "Direct imaging — actually photographing a planet next to its star — is the rarest detection method due to the extreme contrast: a star is roughly 10 billion times brighter than its planet.",
        "Direct imaging works best for young, hot, gas-giant planets far from their star. The first directly imaged exoplanet was Beta Pictoris b in 2008.",
        "Beta Pictoris b is roughly 11 Jupiter masses and orbits at 9 AU from its host star, comparable to Saturn's distance from the Sun.",
        "Gravitational microlensing is the fourth major method: when a foreground star with a planet passes in front of a background star, the planet's gravity briefly amplifies the background star's light.",
        "Microlensing is uniquely sensitive to planets at large orbital distances and to free-floating planets. The OGLE survey at Las Campanas, Chile has been the most prolific microlensing observatory.",
        "OGLE discovered the first free-floating planet candidate, OGLE-2016-BLG-1928, an object roughly Earth-mass with no detectable host star.",
        "The Roman Space Telescope, scheduled for launch in 2027, is designed to extend microlensing surveys to much fainter background stars and detect thousands more free-floating planets.",
        "Astrometry — measuring the tiny side-to-side motion of a star caused by an orbiting planet — has historically been the least productive method, with few confirmed detections before Gaia.",
        "ESA's Gaia mission, launched in 2013, measures the positions of more than 1.8 billion stars to micro-arcsecond precision and is expected to detect tens of thousands of long-period exoplanets via astrometry.",
        "Gaia's first dedicated exoplanet catalog, released in 2023, included Gaia-ASTRO-1, the first exoplanet whose mass was directly measured by astrometry alone.",
        "The James Webb Space Telescope (JWST), launched in 2021, does not detect new exoplanets directly but characterizes their atmospheres via transmission spectroscopy.",
        "JWST's most cited atmospheric result is the detection of carbon dioxide in the atmosphere of WASP-39b, a hot Saturn 700 light years away, announced in 2022.",
        "The discovery of biosignatures — atmospheric markers like oxygen, methane, or dimethyl sulfide that suggest life — is the long-term scientific motivation for exoplanet characterization.",
        "In 2023, JWST tentatively detected dimethyl sulfide in the atmosphere of K2-18b, a sub-Neptune in the habitable zone of a red dwarf 124 light years away. The detection has been contested.",
        "K2-18b's discovery was originally made by the Kepler K2 extended mission in 2015, after Kepler's primary mission failed due to a reaction-wheel breakdown.",
        "The Kepler K2 mission used pressure from sunlight on its solar panels to stabilize pointing after losing reaction wheels, an engineering improvisation by mission scientist Steve Howell.",
        "Future ground-based facilities include the Extremely Large Telescope (ELT), under construction in Chile, with first light expected in 2028. Its 39-meter mirror will enable direct imaging of Earth-like planets.",
        "ELT will be operated by the European Southern Observatory (ESO). Its construction was approved in 2014 and is expected to cost approximately €1.5 billion.",
        "The Habitable Worlds Observatory (HWO), the proposed successor to JWST, is targeted for launch in the 2040s with a primary mission to characterize 25 Earth-like exoplanet atmospheres.",
        "HWO's design is informed by lessons from JWST's coronagraph, which suppresses starlight by a factor of 10^7 — insufficient to image true Earth analogs but a key step toward HWO's required 10^10 suppression.",
        "The next decade of exoplanet science will be defined by the Roman Telescope, ELT, and HWO design phase, with the first definitive biosignature detection considered possible by the late 2030s.",
    ],
    "qa_pairs": [
        {
            "question": "Who discovered the first widely-accepted exoplanet, what was it, and what method was used?",
            "answer": "Mayor and Queloz discovered 51 Pegasi b in 1995, a hot Jupiter orbiting in 4.2 days, using the radial velocity method. They won the 2019 Nobel Prize in Physics for it.",
            "relevant_paragraphs": [1, 2, 3],
        },
        {
            "question": "Which space telescope discovered the most exoplanets via the transit method, and what is its most famous Earth-sized discovery?",
            "answer": "NASA's Kepler space telescope (launched 2009) discovered ~2,800 confirmed exoplanets via the transit method. Its most famous Earth-sized discovery is Kepler-186f, in the habitable zone of a red dwarf 500 light years away, announced in 2014.",
            "relevant_paragraphs": [4, 5, 6],
        },
        {
            "question": "What was the first directly imaged exoplanet, and why is direct imaging difficult?",
            "answer": "The first directly imaged exoplanet was Beta Pictoris b in 2008. Direct imaging is difficult because a star is roughly 10 billion times brighter than its planet.",
            "relevant_paragraphs": [9, 10],
        },
        {
            "question": "What is gravitational microlensing uniquely sensitive to, and which observatory has been most prolific?",
            "answer": "Microlensing is uniquely sensitive to planets at large orbital distances and to free-floating planets. The OGLE survey at Las Campanas, Chile has been the most prolific.",
            "relevant_paragraphs": [12, 13],
        },
        {
            "question": "What atmospheric molecule did JWST detect in WASP-39b, and what controversial detection did it make in K2-18b?",
            "answer": "JWST detected carbon dioxide in WASP-39b's atmosphere (announced 2022) and tentatively detected dimethyl sulfide in K2-18b's atmosphere (2023, contested).",
            "relevant_paragraphs": [20, 22],
        },
        {
            "question": "How did Kepler continue operating after its reaction-wheel breakdown, and who was responsible?",
            "answer": "The Kepler K2 mission used pressure from sunlight on the solar panels to stabilize pointing after losing reaction wheels. The improvisation was by mission scientist Steve Howell.",
            "relevant_paragraphs": [23, 24],
        },
        {
            "question": "What does the Roman Space Telescope aim to do for exoplanet science, and when does it launch?",
            "answer": "Roman, scheduled for launch in 2027, is designed to extend microlensing surveys to much fainter background stars and detect thousands more free-floating planets.",
            "relevant_paragraphs": [15],
        },
        {
            "question": "What is the proposed successor to JWST and what is its primary mission?",
            "answer": "The Habitable Worlds Observatory (HWO), targeted for launch in the 2040s, with a primary mission to characterize 25 Earth-like exoplanet atmospheres. Its starlight suppression target is 10^10.",
            "relevant_paragraphs": [27, 28],
        },
    ],
}

_DOCUMENTS = {
    "fantasy_lore": FANTASY_LORE,
    "mystery_case": MYSTERY_CASE,
    "corporate_memo": CORPORATE_MEMO,
    "soap_opera": SOAP_OPERA,
    "science_survey": SCIENCE_SURVEY,
}


class DocumentQA:
    """
    Sequential document reading environment with multi-hop QA evaluation.

    Reading phase: each step returns the next paragraph.
    QA phase: after all paragraphs, agent answers questions.
    Reward = fraction of questions answered correctly.

    Compatible with both rule-based (trivial) and LLM agents.
    """

    def __init__(
        self,
        document_name: str = "fantasy_lore",
        seed: int = 0,
        question_shuffle: bool = True,
        score_fn=None,
    ) -> None:
        """
        Parameters
        ----------
        score_fn : callable(predicted: str, ground_truth: str) -> float, optional
            Custom scorer overriding the default keyword-overlap heuristic.
            Use ``evaluation.document_qa_llm_judge.llm_judge_score`` to grade
            answers with GPT-4o-mini (gracefully falls back to the heuristic
            when ``OPENAI_API_KEY`` is unset, so this can be wired in
            production code without forcing API calls in CI).
        """
        if document_name not in _DOCUMENTS:
            raise ValueError(f"Unknown document: {document_name}. Choose from {list(_DOCUMENTS)}")

        doc = _DOCUMENTS[document_name]
        self._title = doc["title"]
        self._paragraphs: list[str] = doc["paragraphs"]
        self._qa_pairs: list[dict] = doc["qa_pairs"]
        self._score_fn = score_fn

        self._rng = random.Random(seed)
        if question_shuffle:
            self._qa_pairs = self._qa_pairs.copy()
            self._rng.shuffle(self._qa_pairs)

        self._para_idx = 0
        self._qa_idx = 0
        self._phase = "reading"   # "reading" or "qa"
        self._answers: list[str] = []
        self._scores: list[float] = []
        self.done = False
        self.success = False

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        self._para_idx = 0
        self._qa_idx = 0
        self._phase = "reading"
        self._answers = []
        self._scores = []
        self.done = False
        self.success = False
        return f"[Document: {self._title}]\n{self._paragraphs[0]}"

    def step(self, action: str) -> tuple[str, bool, bool]:
        """
        In reading phase: action is ignored, next paragraph is returned.
        In QA phase: action is treated as the answer to the current question.
        """
        if self.done:
            return "Episode done.", True, self.success

        if self._phase == "reading":
            self._para_idx += 1
            if self._para_idx >= len(self._paragraphs):
                self._phase = "qa"
                return self._current_question(), False, False
            return self._paragraphs[self._para_idx], False, False

        elif self._phase == "qa":
            # Score current answer
            answer = str(action)
            score = self._score_answer(answer, self._qa_idx)
            self._answers.append(answer)
            self._scores.append(score)

            self._qa_idx += 1
            if self._qa_idx >= len(self._qa_pairs):
                self.done = True
                self.success = self.partial_score >= 0.5
                return f"QA complete. Score: {self.partial_score:.2f}", True, self.success

            return self._current_question(), False, False

        return "Done.", True, self.success

    def get_actions(self) -> list[str]:
        if self._phase == "reading":
            return ["next"]
        return ["(free text answer)"]

    @property
    def partial_score(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._qa_pairs)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def current_question(self) -> str | None:
        if self._phase == "qa" and self._qa_idx < len(self._qa_pairs):
            return self._qa_pairs[self._qa_idx]["question"]
        return None

    @property
    def hint_observations(self) -> list[str]:
        """Key fact sentences from high-relevant paragraphs (for retrieval_precision)."""
        hints = []
        for qa in self._qa_pairs:
            for pidx in qa.get("relevant_paragraphs", []):
                if 0 <= pidx < len(self._paragraphs):
                    hints.append(self._paragraphs[pidx][:100])
        return hints

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_question(self) -> str:
        if self._qa_idx < len(self._qa_pairs):
            q = self._qa_pairs[self._qa_idx]["question"]
            return f"[QUESTION {self._qa_idx + 1}/{len(self._qa_pairs)}] {q}"
        return "All questions answered."

    def _score_answer(self, predicted: str, qa_idx: int) -> float:
        """
        Score predicted answer against ground truth.

        Default: substring matching on 4+ letter nouns from the ground truth.
        Override via ``score_fn`` constructor parameter (e.g. LLM-judge).
        """
        if qa_idx >= len(self._qa_pairs):
            return 0.0
        ground_truth_raw = self._qa_pairs[qa_idx]["answer"]

        # When a custom score_fn was supplied, delegate to it (with the
        # original-case strings, in case the scorer is sensitive to casing).
        if self._score_fn is not None:
            try:
                score = self._score_fn(predicted, ground_truth_raw)
                return float(max(0.0, min(1.0, score)))
            except Exception as exc:
                # Never let a scorer error tear down the episode loop.
                print(f"[DocumentQA] custom score_fn failed: {exc} — keyword fallback")

        ground_truth = ground_truth_raw.lower()
        predicted = predicted.lower()

        # Extract key noun phrases from ground truth
        words = re.findall(r'\b[a-z]{4,}\b', ground_truth)
        key_words = [w for w in words if w not in {"that", "this", "from", "with", "have", "been", "they", "their", "were"}]

        if not key_words:
            return 1.0 if ground_truth[:30] in predicted else 0.0

        # Partial credit: fraction of key words found in predicted answer
        matches = sum(1 for w in key_words if w in predicted)
        return matches / len(key_words)
