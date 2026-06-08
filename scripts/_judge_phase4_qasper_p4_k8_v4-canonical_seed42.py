"""Phase 4 cross-vendor finishing — QASPER p4_k8 v4-canonical seed=42 (27 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k8__qasper__v4-canonical__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q001", 0.0, "PRED hallucinated POS/coreference methods; gold unanswerable."),
    ("q002", 0.0, "PRED 'N400/P600, post-N400/behavioral' WRONG; gold=ELAN/LAN and PNP ERP."),
    ("q003", 0.25, "PRED 'CoNLL, TAC, ACE, AQUAINT' partial; gold=CoNLL-YAGO specifically."),
    ("q010", 0.25, "PRED 'Czech, French, Italian, Indonesian' partial; gold=UD1.2 full 16-language list."),
    ("q012", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q013", 0.75, "PRED 'diagnosis, patient movement, procedures, symptoms, demographics, vitals, medication' matches most gold topics."),
    ("q014", 0.75, "PRED 'automatic evaluations vs human segmentation subcorpus' partial match to gold ANNODIS comparison."),
    ("q018", 0.25, "PRED 'in the QA+QG framework' partial; gold specifies scoring via semantic relevance."),
    ("q020", 0.0, "PRED 'higher kappa, above-chance accuracy' WRONG; gold=tSNE."),
    ("q024", 0.0, "PRED 'separate optimizers' WRONG; gold=extra BERT position embeddings."),
    ("q030", 0.0, "PRED 'may not easily generalize' implies No; gold='Yes'."),
    ("q041", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q050", 1.0, "PRED 'CJFA encoder' matches gold."),
    ("q051", 0.0, "PRED refuses; gold=pivot-based translation."),
    ("q058", 0.0, "PRED 'pairwise coherence, recipe score' WRONG; gold has 9 specific metrics."),
    ("q062", 0.0, "PRED 'original dataset + artificially generated' WRONG; gold=Rei2016 error detection."),
    ("q063", 0.25, "PRED 'wide margin on informativeness' vague; gold has specific per-dataset numbers."),
    ("q069", 0.0, "PRED 'Yes, MTL' but gold='No' — Y/N flip."),
    ("q075", 0.75, "PRED 'Back Translation and Mix-Source' includes gold=Back Translation."),
    ("q083", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q084", 0.0, "PRED hallucinated WER; gold unanswerable."),
    ("q089", 0.0, "PRED 'WER, acceptability, correction' doesn't include BLEU; gold=BLEU."),
    ("q090", 0.75, "PRED 'spaCy, Char-biLSTM, Stanford NER' includes gold Stanford NER."),
    ("q091", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q092", 0.0, "PRED 'Mwmote, Eusboost' WRONG; gold=MLP."),
    ("q093", 0.75, "PRED 'METEOR, ROUGE-L, BLEU-1' includes gold BLEU-1."),
    ("q098", 0.0, "PRED 'best=implicit, worst=explicit' WRONG direction; gold=explicit discourse relations."),
]


def main() -> None:
    assert len(JUDGMENTS) == 27
    qid_prefix = "p4_k8__qasper__v4-canonical__seed42__"
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
    print(f"qasper p4_k8 v4-canonical seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
