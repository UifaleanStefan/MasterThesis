"""Phase 1.9 — NarrativeQA Protocol A (online + batch) judge: all 6 configs.

Cells (12 total, 10 entries each = 120 entries):
  narrativeqa__{cfg}__{mode}__seed42
  where cfg in {v4t-canonical, v4t-tuned, v4t-corpus-tuned,
                attention-corpus-tuned, bm25-corpus, dump-all}
  and mode in {online, batch}

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Standard 5-point rubric

Benchmark: NarrativeQA (Gutenberg plays, 10 docs = "Cynthia's Revels" by Ben Jonson
and related texts). All 10 docs are from the same work. Each doc provides 1 online
QA question + 1 batch QA question.

Key calibration findings:
  - bm25-corpus dominates: correct on 9/10 questions online, 7/10 batch
  - v4t configs (canonical/tuned/corpus-tuned): all use canonical θ (per-doc and
    corpus tuning both fail for NarrativeQA — no gold QAs available during tuning)
    → identical predictions; correct on ~3.5/10 questions
  - attention-corpus-tuned: uniquely gets doc0/doc1 correct but misses on doc2/doc5;
    many refusals from doc6 onward
  - dump-all: similar to attention; correct on doc2/doc3/doc5 online but many refusals

NarrativeQA scoring notes:
  - Gold answers provided as arrays with multiple valid phrasings (ALL CAPS + mixed)
  - "Moon" as answer to "what name was Cynthia known by?" → 0.5 (correct deity,
    wrong name; gold is "The goddess Diana")
  - Refusals score 0.0 (should be able to answer given the ingested paragraphs)
  - "sentence of banishment" for doc6 → 0.25 (gives a sentence but wrong content)

All 6 configs use same question bank (same 10 docs); differences are retrieval quality.

NarrativeQA tuning status:
  - v4t-tuned: uses per-doc tuned θ → falls back to canonical (no gold QAs in tuning)
  - v4t-corpus-tuned: uses corpus-mode tuned θ → falls back to canonical
  - Hence v4t-canonical, v4t-tuned, v4t-corpus-tuned are functionally identical

Scoring rules (Protocol A, standard 5-point rubric):
  0.0 — wrong or refusal
  0.25 — loosely related (mentions a plausible character/type but wrong specifics)
  0.5 — partial (correct entity but wrong name/attribute)
  0.75 — close (right answer, minor error)
  1.0 — exact / fully correct
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"

CONFIGS = [
    "v4t-canonical",
    "v4t-tuned",
    "v4t-corpus-tuned",
    "attention-corpus-tuned",
    "bm25-corpus",
    "dump-all",
]

MODES = ["online", "batch"]

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    "do not have", "don't have", "context provided", "provided context",
    "passages provided", "provided passages", "not mentioned", "not provided",
    "not available", "insufficient", "no information", "cannot determine",
    "not specify", "no mention", "cannot find", "does not contain",
    "not found", "i'm sorry", "unable to", "no context", "not enough",
    "apologies", "not explicitly", "not clear", "not specified",
    "passages do not", "does not provide", "do not provide",
    "cannot be determined", "no relevant", "not discussed",
    "no specific", "cannot answer", "not contain", "no detail",
    "information is not", "there is no", "is not mentioned",
    "are not mentioned", "is not provided", "are not provided",
    "there are no", "not include", "not included", "do not contain",
    "not be determined", "not be found", "without more", "lacks",
    "no passage", "i do not see", "not see", "not support",
    "unanswerable", "unspecified", "no answer", "the relevant passages",
    "the document passages", "document does not", "documents do not",
]


def is_refusal(pred: str) -> bool:
    p = pred.strip().lower()
    if not p:
        return True
    return any(pat in p for pat in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# JUDGMENTS: keyed by "{cfg}__{mode}__{doc_qa}" → (score, rationale)
# Only non-zero scores are listed; everything else defaults to 0.0.
# ---------------------------------------------------------------------------

# Questions and gold answers (for reference in rationales):
# doc0: WHO NORMALLY DELIVERS THE OPENING PROLOGUE?
#       gold=["THE ACTOR WEARING THE BLACK CLOAK", "The actor in the black cloak"]
# doc1: WHAT NAME WAS CYNTHIA MORE FAMOUSLY KNOWN BY?
#       gold=["THE GODDESS DIANA", "The goddess Diana"]
# doc2: WHO DOES ECHO WEEP FOR?
#       gold=["NARCISSUS", "Narcissus"]
# doc3: WHAT DOES A DRINK FROM NARCISSUS'S SPRING CAUSE THE DRINKER TO DO?
#       gold=["FALL IN LOVE WITH THEMSELVES", "Grow dotingly enamored with themselves"]
# doc4: IN WHAT VALLEY DID THE SOLEMN REVELS OF CYNTHIA TAKE PLACE?
#       gold=["GARGAPHIE IN GREECE", "Gargaphie"]
# doc5: WHAT DID THE SYMBOLIC VICES DISGUISE THEMSELVES TO BE?
#       gold=["VIRTUES", "Virtues"]
# doc6: WHAT SENTENCE DID CYNTHIA GIVE TO THE SYMBOLIC VICES?
#       gold=["TO BATHE IN THE SPRING OF HELICON", "Make reperations and purify themselves."]
# doc7: WHO DRANK FROM THE SPRING AT CYNTHIA'S REVELS?
#       gold=["ALL THE COURTIERS AND LADIES WHO ATTENDED CYNTHIA'S REVELS",
#             "The courtiers and ladies that are there and Asotus"]
# doc8: HOW MANY PHASES DID THE COURT COMPLIMENT COMPETITION HAVE?
#       gold=["4 PHASES", "2"]
# doc9: WHO CHALLENGES THE COURTIERS TO COURT COMPLIMENT COMPETITION?
#       gold=["ASOTUS", "Asotus"]

JUDGMENTS: dict[str, tuple[float, str]] = {
    # ═══════════════════════════════════════════════════════════════════════
    # bm25-corpus — best config; retrieves specific textual passages well
    # ═══════════════════════════════════════════════════════════════════════
    "bm25-corpus__online__doc0_qa0": (1.0,
        "Pred: 'The actor who wears the black cloak.' EXACT: gold='The actor in the "
        "black cloak'. Score 1.0."),
    "bm25-corpus__batch__doc0_qa0": (1.0,
        "Pred: 'The actor who delivers the prologue is usually the one who wears the "
        "black cloak.' EXACT: contains 'wears the black cloak'. Score 1.0."),
    "bm25-corpus__online__doc1_qa0": (1.0,
        "Pred: 'Diana' EXACT: gold='The goddess Diana', Diana is explicitly named. "
        "Score 1.0."),
    "bm25-corpus__batch__doc1_qa0": (0.5,
        "Pred: 'The Moon.' Gold='The goddess Diana'. Moon = correct deity/aspect "
        "but wrong name — Cynthia/Diana/Moon are aspects of same goddess but gold "
        "asks specifically for the name Diana. Score 0.5."),
    "bm25-corpus__online__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "bm25-corpus__batch__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "bm25-corpus__online__doc3_qa0": (1.0,
        "Pred: '...drinkers to \"Grow dotingly enamored of themselves.\"' EXACT: "
        "matches gold 'Grow dotingly enamored with themselves'. Score 1.0."),
    "bm25-corpus__batch__doc3_qa0": (1.0,
        "Pred: '...drinkers to \"Grow dotingly enamored of themselves.\"' EXACT. "
        "Score 1.0."),
    "bm25-corpus__online__doc4_qa0": (1.0,
        "Pred: 'The solemn revels of Cynthia took place in the valley of Gargaphie "
        "in Greece.' EXACT: gold='GARGAPHIE IN GREECE'. Score 1.0."),
    # bm25-corpus__batch__doc4_qa0: 0.0 (Shadow of Death — wrong valley)
    "bm25-corpus__online__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices masqueraded as virtues.' EXACT: gold='VIRTUES'. "
        "Score 1.0."),
    "bm25-corpus__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices masqueraded as virtues.' EXACT. Score 1.0."),
    "bm25-corpus__online__doc6_qa0": (1.0,
        "Pred: 'Cynthia sentenced them to make reparation and to purify themselves "
        "by bathing in the spring at Mount Helicon.' EXACT: gold='TO BATHE IN THE "
        "SPRING OF HELICON'. Score 1.0."),
    "bm25-corpus__batch__doc6_qa0": (1.0,
        "Pred: 'Cynthia sentenced them...to purify themselves by bathing in the "
        "spring at Mount Helicon.' EXACT. Score 1.0."),
    "bm25-corpus__online__doc7_qa0": (1.0,
        "Pred: 'The courtiers and ladies assembled for Cynthia's revels, as well as "
        "Asotus...' EXACT: gold='The courtiers and ladies that are there and Asotus'. "
        "Score 1.0."),
    "bm25-corpus__batch__doc7_qa0": (1.0,
        "Pred: 'The courtiers and ladies assembled for Cynthia's revels, as well as "
        "Asotus...' EXACT. Score 1.0."),
    "bm25-corpus__online__doc8_qa0": (1.0,
        "Pred: 'The competition of \"court compliment\" had four phases.' EXACT: "
        "gold='4 PHASES'. Score 1.0."),
    "bm25-corpus__batch__doc8_qa0": (1.0,
        "Pred: 'The competition of \"court compliment\" had four phases.' EXACT. "
        "Score 1.0."),
    "bm25-corpus__online__doc9_qa0": (1.0,
        "Pred: 'Asotus, a foolish spendthrift, challenges the courtiers...' EXACT: "
        "gold='ASOTUS'. Score 1.0."),
    "bm25-corpus__batch__doc9_qa0": (1.0,
        "Pred: 'Asotus, a foolish spendthrift, challenges the courtiers...' EXACT. "
        "Score 1.0."),

    # ═══════════════════════════════════════════════════════════════════════
    # attention-corpus-tuned — good on doc0/doc1 online; many refusals later
    # ═══════════════════════════════════════════════════════════════════════
    "attention-corpus-tuned__online__doc0_qa0": (1.0,
        "Pred: 'The actor who wears the black cloak usually delivers the opening "
        "prologue.' EXACT: gold='The actor in the black cloak'. Score 1.0."),
    # attention-corpus-tuned__batch__doc0_qa0: 0.0 (says Chorus — wrong)
    "attention-corpus-tuned__online__doc1_qa0": (1.0,
        "Pred: 'Cynthia was more famously known as Diana.' EXACT: gold='The goddess "
        "Diana', Diana explicitly named. Score 1.0."),
    "attention-corpus-tuned__batch__doc1_qa0": (0.5,
        "Pred: '...more famously known as the Moon.' Gold='The goddess Diana'. "
        "Moon = correct deity but wrong name. Score 0.5."),
    # attention-corpus-tuned__online__doc2_qa0: 0.0 (Niobe — wrong)
    # attention-corpus-tuned__batch__doc2_qa0: 0.0 (refusal)
    "attention-corpus-tuned__online__doc3_qa0": (1.0,
        "Pred: '...causes the drinkers to \"Grow dotingly enamored of themselves.\"' "
        "EXACT. Score 1.0."),
    "attention-corpus-tuned__batch__doc3_qa0": (1.0,
        "Pred: '...causes the drinker to fall in love with their own reflection.' "
        "EXACT: matches 'FALL IN LOVE WITH THEMSELVES'. Score 1.0."),
    # doc4: 0.0 (Thames) for both modes
    # attention-corpus-tuned__online__doc5_qa0: 0.0 (buffoon)
    "attention-corpus-tuned__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    # doc6: 0.0 (refusal) for both modes
    # doc7: 0.0 (refusal) for both modes
    # doc8: 0.0 (refusal) for both modes
    # doc9: 0.0 (PHA.) for both modes

    # ═══════════════════════════════════════════════════════════════════════
    # dump-all — partial retrieval; many refusals from doc6 onward
    # ═══════════════════════════════════════════════════════════════════════
    # dump-all__online__doc0_qa0: 0.0 (Chorus)
    # dump-all__batch__doc0_qa0: 0.0 (Prologue character)
    "dump-all__online__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Gold='The goddess Diana'. Moon = correct "
        "deity but wrong name. Score 0.5."),
    "dump-all__batch__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Gold='The goddess Diana'. Score 0.5."),
    "dump-all__online__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "dump-all__batch__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "dump-all__online__doc3_qa0": (1.0,
        "Pred: '...causes the drinker to fall in love with their own reflection.' "
        "EXACT: matches 'FALL IN LOVE WITH THEMSELVES'. Score 1.0."),
    # dump-all__batch__doc3_qa0: 0.0 (refusal)
    # doc4: 0.0 for both (wrong valley)
    "dump-all__online__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "dump-all__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    # doc6-doc9: 0.0 (refusals or PHA.) for both modes

    # ═══════════════════════════════════════════════════════════════════════
    # v4t-canonical (+ v4t-tuned + v4t-corpus-tuned = all identical θ due to
    # tuning fallback to canonical for NarrativeQA)
    # ═══════════════════════════════════════════════════════════════════════
    "v4t-canonical__online__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Gold='The goddess Diana'. Moon=same deity, "
        "wrong name. Score 0.5."),
    "v4t-canonical__batch__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Gold='The goddess Diana'. Score 0.5."),
    "v4t-canonical__online__doc0_qa0": (0.25,
        "Pred: '...by the author or a character designated for that purpose.' "
        "Vague — mentions 'character designated' but not the black cloak actor. "
        "Loosely related. Score 0.25."),
    "v4t-canonical__batch__doc0_qa0": (0.25,
        "Pred: '...by the author or a character designated for that purpose.' "
        "Same vague response. Score 0.25."),
    "v4t-canonical__online__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-canonical__batch__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-canonical__online__doc3_qa0": (1.0,
        "Pred: '...fall in love with their own reflection.' EXACT: matches "
        "'FALL IN LOVE WITH THEMSELVES'. Score 1.0."),
    "v4t-canonical__batch__doc3_qa0": (1.0,
        "Pred: '...fall in love with their own reflection.' EXACT. Score 1.0."),
    # doc4: 0.0 (Thames)
    "v4t-canonical__online__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-canonical__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-canonical__online__doc6_qa0": (0.25,
        "Pred: 'Cynthia gave the symbolic vices a sentence of banishment.' "
        "Gives a sentence but wrong content — gold is 'TO BATHE IN THE SPRING OF "
        "HELICON', not banishment. Partially correct (gives a sentence). Score 0.25."),
    "v4t-canonical__batch__doc6_qa0": (0.25,
        "Pred: 'Cynthia gave the symbolic vices a sentence of banishment.' "
        "Wrong content (banishment not Helicon spring). Score 0.25."),
    # doc7: 0.0 (refusal)
    # doc8: 0.0 (three phases — wrong number)
    # doc9: 0.0 (Cynthia — wrong person)

    # v4t-tuned: identical predictions to v4t-canonical (same fallback θ)
    "v4t-tuned__online__doc0_qa0": (0.0,
        "Pred: '...by the character known as the \"Prologue.\"' Wrong — gold is the "
        "black cloak actor, not the Prologue character. Score 0.0."),
    "v4t-tuned__batch__doc0_qa0": (0.25,
        "Pred: '...by the author or a character designated for that purpose.' "
        "Vague; mentions character but not black cloak. Score 0.25."),
    "v4t-tuned__online__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Score 0.5."),
    "v4t-tuned__batch__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Score 0.5."),
    "v4t-tuned__online__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-tuned__batch__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-tuned__online__doc3_qa0": (1.0,
        "Pred: '...fall in love with their own reflection.' EXACT. Score 1.0."),
    "v4t-tuned__batch__doc3_qa0": (1.0,
        "Pred: '...fall in love with their own reflection.' EXACT. Score 1.0."),
    # doc4: 0.0 (Thames)
    "v4t-tuned__online__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-tuned__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-tuned__online__doc6_qa0": (0.25,
        "Pred: '...sentence of banishment.' Wrong content; gold is Helicon spring. "
        "Score 0.25."),
    "v4t-tuned__batch__doc6_qa0": (0.25,
        "Pred: '...sentence of banishment.' Score 0.25."),
    # doc7: 0.0 (refusal)
    # doc8: 0.0 (three phases)
    # doc9: 0.0 (Cynthia)

    # v4t-corpus-tuned: same canonical θ fallback; mix of online=Prologue / batch=author
    "v4t-corpus-tuned__online__doc0_qa0": (0.0,
        "Pred: '...by the character known as the \"Prologue.\"' Wrong — gold is the "
        "black cloak actor, not the Prologue character. Score 0.0."),
    "v4t-corpus-tuned__batch__doc0_qa0": (0.25,
        "Pred: '...by the author or a character designated for that purpose.' "
        "Vague; mentions character but not black cloak. Score 0.25."),
    "v4t-corpus-tuned__online__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Score 0.5."),
    "v4t-corpus-tuned__batch__doc1_qa0": (0.5,
        "Pred: '...known as the Moon.' Score 0.5."),
    "v4t-corpus-tuned__online__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-corpus-tuned__batch__doc2_qa0": (1.0,
        "Pred: 'Echo weeps for Narcissus.' EXACT. Score 1.0."),
    "v4t-corpus-tuned__online__doc3_qa0": (1.0,
        "Pred: '...fall in love with their own reflection.' EXACT. Score 1.0."),
    "v4t-corpus-tuned__batch__doc3_qa0": (1.0,
        "Pred: '...fall in love with themselves.' EXACT. Score 1.0."),
    # doc4: 0.0 (Thames)
    "v4t-corpus-tuned__online__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-corpus-tuned__batch__doc5_qa0": (1.0,
        "Pred: 'The symbolic vices disguised themselves as virtues.' EXACT. "
        "Score 1.0."),
    "v4t-corpus-tuned__online__doc6_qa0": (0.25,
        "Pred: '...sentence of banishment.' Wrong content; gold is Helicon spring. "
        "Score 0.25."),
    "v4t-corpus-tuned__batch__doc6_qa0": (0.25,
        "Pred: '...sentence of banishment.' Score 0.25."),
    # doc7: 0.0 (refusal)
    # doc8: 0.0 (three phases)
    # doc9: 0.0 (Cynthia)
}


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------
def score_entry(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    cfg = r["config"]
    mode = r["mode"]
    doc_qa = r["qid"].split("__")[-2]

    if is_refusal(pred):
        return 0.0, "Refusal — prediction contains no-information pattern. Score 0.0."

    key = f"{cfg}__{mode}__{doc_qa}"
    if key in JUDGMENTS:
        s, rat = JUDGMENTS[key]
        if s == 0.0:
            return 0.0, rat
        return s, rat

    return 0.0, "Wrong answer — prediction does not match gold answer. Score 0.0."


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_judgments(queue_path: Path, results_path: Path) -> None:
    if not queue_path.exists():
        print(f"  [SKIP] queue not found: {queue_path}")
        return

    rows = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    existing: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line)["qid"])

    new_records = []
    for r in rows:
        if r["qid"] in existing:
            continue
        score, rationale = score_entry(r)
        new_records.append({
            "qid": r["qid"],
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
        })

    if new_records:
        with results_path.open("a", encoding="utf-8") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec) + "\n")

    print(f"  {queue_path.parent.name}: {len(rows)} queued, "
          f"{len(new_records)} written, {len(existing)} already existed")


if __name__ == "__main__":
    print("Judging NarrativeQA Protocol A — all 6 configs (online + batch) ...")
    for cfg in CONFIGS:
        for mode in MODES:
            cell_name = f"narrativeqa__{cfg}__{mode}__seed42"
            cell_dir = JQ / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            write_judgments(cell_dir / "queue.jsonl", cell_dir / "results.jsonl")
    print("Done.")
