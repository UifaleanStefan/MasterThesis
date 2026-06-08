"""Phase 4 cross-vendor finishing — QASPER p4_k4 v4-canonical seed=42 (76 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k4__qasper__v4-canonical__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED refuses; gold=SVMs."),
    ("q001", 0.75, "PRED 'passages do not provide' implicit match to gold unanswerable."),
    ("q002", 0.0, "PRED refuses; gold=ELAN/LAN and PNP ERP."),
    ("q003", 0.25, "PRED 'CoNLL testa' partial; gold=CoNLL-YAGO specifically."),
    ("q004", 1.0, "PRED 'Yes, SAN-based' matches gold='Yes'."),
    ("q005", 0.75, "PRED 'does not explicitly address' implicit match to gold unanswerable."),
    ("q006", 0.0, "PRED 'slightly better than Lead-3' hallucinated; gold unanswerable."),
    ("q007", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q010", 0.25, "PRED 'Slavic + English' partial; gold=UD1.2 full 16-language list."),
    ("q011", 0.0, "PRED refuses; gold=precision, recall, F1, accuracy."),
    ("q012", 0.0, "PRED 'No' hallucinated; gold unanswerable."),
    ("q013", 0.25, "PRED 'diagnosis history, symptoms/signs' partial; gold has 10 categories."),
    ("q014", 0.75, "PRED 'precision, recall, F1' matches gold metrics."),
    ("q015", 0.0, "PRED refuses; gold=0.8% F1."),
    ("q017", 0.0, "PRED refuses; gold=Loaded language."),
    ("q018", 0.25, "PRED 'in the QA+QG framework' partial; gold specifies scoring via semantic relevance."),
    ("q019", 0.0, "PRED refuses; gold=14 times."),
    ("q020", 0.0, "PRED 'accuracy improvement, inter-subject variability' WRONG; gold=tSNE."),
    ("q024", 0.0, "PRED refuses; gold=extra BERT position embeddings."),
    ("q026", 0.0, "PRED refuses; gold=ceccarelli2013learning dataset from CoNLL 2003."),
    ("q027", 0.0, "PRED 'No, worsens' but gold='Yes' — Y/N flip."),
    ("q028", 0.0, "PRED refuses; gold=MaxEnt to SVMs."),
    ("q029", 0.5, "PRED 'irony accuracy, word repetition' partial; gold=Irony Classifier specifically."),
    ("q030", 0.0, "PRED 'may not generalize' implies No; gold='Yes'."),
    ("q031", 0.0, "PRED refuses; gold=German-English."),
    ("q032", 0.0, "PRED 'leveled preprocessing' vague; gold=raw text."),
    ("q034", 0.0, "PRED 'advanced methods lacking contextual info' vague; gold=BIBREF9."),
    ("q037", 0.5, "PRED covers race part (African American/European American) but misses gender (female higher anger/joy/valence, male higher fear)."),
    ("q038", 0.0, "PRED 'loss function, validation' doesn't mention MAP; gold=MAP only."),
    ("q041", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q042", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q043", 0.0, "PRED refuses; gold=average unique predictions."),
    ("q044", 0.0, "PRED refuses; gold=BIBREF26."),
    ("q045", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q046", 0.0, "PRED 'Turkish 74.36%' WRONG; gold=Russian."),
    ("q047", 0.0, "PRED completely wrong topics; gold=politics, business, science, AskReddit."),
    ("q048", 0.0, "PRED hallucinated Jasper baselines; gold unanswerable."),
    ("q050", 1.0, "PRED 'CJFA encoder' matches gold."),
    ("q051", 0.0, "PRED refuses; gold=pivot-based translation."),
    ("q052", 0.0, "PRED 'grammaticality judgment, PTB vocabulary' WRONG; gold=INLINEFORM0 scores."),
    ("q053", 0.0, "PRED refuses; gold=Spearman correlation + entailment datasets."),
    ("q054", 0.75, "PRED 'passages do not mention' implicit match to gold='No'."),
    ("q055", 0.25, "PRED 'EI-Reg-anger' partial; gold=EI-Reg, EI-Oc, V-Reg, V-Oc (all four)."),
    ("q058", 0.0, "PRED 'coherence and recipe-name goal' WRONG; gold has 9 specific metrics."),
    ("q061", 0.0, "PRED refuses; gold=BOW-LR, BOW-RF, TFIDF-RF, TextCNN, C-TextCNN."),
    ("q062", 0.0, "PRED 'Felice2014a' WRONG; gold=Rei2016 error detection."),
    ("q063", 0.0, "PRED refuses; gold has specific per-dataset Rouge/Meteor numbers."),
    ("q064", 0.75, "PRED 'passages do not provide' implicit match to gold unanswerable."),
    ("q065", 0.0, "PRED refuses; gold=Transformer architecture."),
    ("q067", 0.75, "PRED 'does not contain info' implicit match to gold unanswerable."),
    ("q068", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes, discontinuous entities, multi-task' but gold='No' — Y/N flip."),
    ("q070", 0.0, "PRED 'determine text from audio' WRONG; gold=text transcription."),
    ("q071", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q072", 0.0, "PRED refuses; gold=CNN."),
    ("q073", 0.75, "PRED 'passages do not indicate' implicit match to gold='No'."),
    ("q074", 0.0, "PRED refuses; gold=Reuters-8 without stop words."),
    ("q075", 0.0, "PRED refuses; gold=Back Translation."),
    ("q076", 0.0, "PRED refuses; gold=SGD, naive bayes, decision tree."),
    ("q080", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q081", 0.0, "PRED 'CNN' WRONG; gold=Bi-LSTM-CRF."),
    ("q083", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q084", 0.0, "PRED hallucinated WER; gold unanswerable."),
    ("q085", 0.75, "PRED 'LORELEI + Switchboard' includes gold Switchboard."),
    ("q087", 0.0, "PRED 'no quantitative evaluation' WRONG; gold has specific F1 scores."),
    ("q088", 0.0, "PRED 'two-staged CNN, attention value' hallucinated; gold unanswerable."),
    ("q089", 0.0, "PRED refuses; gold=BLEU."),
    ("q090", 0.0, "PRED 'spaCy, Char-biLSTM' doesn't include Stanford NER; gold=Stanford NER."),
    ("q092", 0.0, "PRED refuses; gold=MLP."),
    ("q093", 0.0, "PRED 'three generative metrics' vague; gold=BLEU-1."),
    ("q094", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q095", 0.0, "PRED 'contextual embeddings, LAT focus word' WRONG; gold=always answer YES baseline."),
    ("q096", 0.0, "PRED 'creates huge labeled dataset' hallucinated; gold unanswerable."),
    ("q097", 0.25, "PRED 'prepared and spontaneous speech' partial; gold=ASR."),
    ("q098", 0.0, "PRED refuses; gold=explicit discourse relations."),
    ("q099", 0.0, "PRED 'QT21-HimL-SysComb, AMU-UEDIN' WRONG; gold=attentional encoder-decoder."),
]


def main() -> None:
    assert len(JUDGMENTS) == 76
    qid_prefix = "p4_k4__qasper__v4-canonical__seed42__"
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
    print(f"qasper p4_k4 v4-canonical seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
