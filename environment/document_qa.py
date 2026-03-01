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

_DOCUMENTS = {
    "fantasy_lore": FANTASY_LORE,
    "mystery_case": MYSTERY_CASE,
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
    ) -> None:
        if document_name not in _DOCUMENTS:
            raise ValueError(f"Unknown document: {document_name}. Choose from {list(_DOCUMENTS)}")

        doc = _DOCUMENTS[document_name]
        self._title = doc["title"]
        self._paragraphs: list[str] = doc["paragraphs"]
        self._qa_pairs: list[dict] = doc["qa_pairs"]

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
        Uses simple substring matching (exact match of key facts).
        Can be upgraded to BERTScore or LLM-judge in full experiments.
        """
        if qa_idx >= len(self._qa_pairs):
            return 0.0
        ground_truth = self._qa_pairs[qa_idx]["answer"].lower()
        predicted = predicted.lower()

        # Extract key noun phrases from ground truth
        words = re.findall(r'\b[a-z]{4,}\b', ground_truth)
        key_words = [w for w in words if w not in {"that", "this", "from", "with", "have", "been", "they", "their", "were"}]

        if not key_words:
            return 1.0 if ground_truth[:30] in predicted else 0.0

        # Partial credit: fraction of key words found in predicted answer
        matches = sum(1 for w in key_words if w in predicted)
        return matches / len(key_words)
