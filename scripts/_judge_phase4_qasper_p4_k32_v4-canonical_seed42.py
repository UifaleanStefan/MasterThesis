"""Phase 4 cross-vendor finishing — QASPER p4_k32 v4-canonical seed=42 (58 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k32__qasper__v4-canonical__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.75, "PRED 'SVMs and LRs' includes gold=SVMs; LR extra."),
    ("q001", 0.0, "PRED hallucinated image categorization; gold unanswerable."),
    ("q002", 0.0, "PRED 'N400/P600, EPNP/P600' WRONG; gold=ELAN/LAN and PNP ERP."),
    ("q003", 0.25, "PRED 'CoNLL, TAC, ACE...' partial; gold=CoNLL-YAGO specifically."),
    ("q004", 0.0, "PRED refuses; gold='Yes'."),
    ("q006", 0.0, "PRED hallucinated ROGUE scores; gold unanswerable."),
    ("q008", 0.0, "PRED 'Yes English only' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED refuses; gold='Yes'."),
    ("q010", 1.0, "PRED 'UD1.2 corpora 16 languages full list' matches gold exactly."),
    ("q011", 0.0, "PRED 'response times, CPU, disk' WRONG; gold=precision, recall, F1, accuracy."),
    ("q012", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q014", 0.75, "PRED 'F-score, precision, recall' matches gold's evaluation metrics."),
    ("q015", 0.25, "PRED gives numbers but not the 0.8% F1 improvement figure."),
    ("q016", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold RNN-based NMT."),
    ("q018", 0.5, "PRED 'generate questions from candidate answer paths' partial; gold has scoring detail."),
    ("q019", 0.0, "PRED refuses; gold=14 times larger."),
    ("q020", 0.0, "PRED 'higher kappa value' WRONG; gold=tSNE plot."),
    ("q022", 0.5, "PRED 'CBOW, PV-DM, GloVe, EqEmb'; gold has b-emb not EqEmb — 3/4 match."),
    ("q024", 0.0, "PRED 'separate optimizers' WRONG; gold=extra BERT position embeddings."),
    ("q025", 0.5, "PRED 'not provide info' implicit match to gold=No data, pretrained model."),
    ("q029", 0.5, "PRED 'irony accuracy, sentiment, content' partial; gold=Irony Classifier specifically."),
    ("q030", 0.0, "PRED implies No/limitations; gold='Yes'."),
    ("q032", 0.0, "PRED 'level 3, leveled preprocessing' vague WRONG; gold=raw text."),
    ("q033", 0.25, "PRED 'soft attention, unnecessary noise' different from gold=ambiguous/common entities."),
    ("q034", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold BIBREF9."),
    ("q038", 0.5, "PRED includes MAP among metrics; gold specifies MAP only — over-answers."),
    ("q042", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q043", 0.25, "PRED 'recall on absent portion + R@10/50' partial; gold=average unique predictions."),
    ("q044", 0.0, "PRED describes dataset not BIBREF26; gold=BIBREF26."),
    ("q045", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q047", 0.5, "PRED 'politics, science, technology, IAmA, AskReddit' partial; gold=politics, business, science, AskReddit."),
    ("q051", 0.0, "PRED '1:1 ratio monolingual/parallel' WRONG; gold=pivot-based translation."),
    ("q053", 0.75, "PRED 'entailment datasets, word similarity datasets, SCWS' matches gold experiments."),
    ("q054", 1.0, "PRED 'No' matches gold='No'."),
    ("q055", 1.0, "PRED 'EI-Reg, EI-Oc, V-Reg, V-Oc' exact match."),
    ("q058", 0.5, "PRED 'UMA, MRR, BLEU-1, BLEU-4' partial; gold has 9 metrics including BPE PPL, ROUGE-L, D-1, D-2, PP."),
    ("q062", 0.0, "PRED 'previous results, cross-entropy' WRONG; gold=Rei2016 error detection system."),
    ("q063", 0.25, "PRED 'wide margin, mixed ROUGE-L' vague; gold has specific per-dataset numbers."),
    ("q064", 0.75, "PRED 'does not provide info' implicit match to gold unanswerable."),
    ("q065", 0.0, "PRED 'neural MT + Kaldi' WRONG; gold=Transformer architecture."),
    ("q068", 0.0, "PRED hallucinated GloVe/LSTM baseline; gold unanswerable."),
    ("q070", 1.0, "PRED 'datasets with transcribed text' matches gold='text transcription'."),
    ("q072", 0.0, "PRED 'encoder with five decoders' WRONG; gold=CNN."),
    ("q073", 1.0, "PRED 'No, variety of topics' matches gold='No'."),
    ("q075", 0.75, "PRED 'Data Augmentation and Back Translation' includes gold=Back Translation."),
    ("q076", 0.0, "PRED gives offensive-comment stats WRONG; gold=SGD, naive bayes, decision tree."),
    ("q080", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q082", 0.5, "PRED includes Stanford NER, Babelfy, DBpedia Spotlight from gold list; missing FOX, LingPipe, NERD-ML, TagMe."),
    ("q083", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q084", 0.0, "PRED hallucinated WER; gold unanswerable."),
    ("q085", 0.75, "PRED 'DARPA LORELEI + Switchboard' includes gold Switchboard Telephone Speech Corpus."),
    ("q090", 0.75, "PRED 'recurrent CRF, spaCy, Stanford NER' includes gold Stanford NER."),
    ("q091", 0.0, "PRED refuses; gold='Yes'."),
    ("q092", 0.0, "PRED 'Mwmote, Eusboost, autoencoders' doesn't mention MLP; gold=MLP."),
    ("q094", 0.0, "PRED 'No, multiple datasets' hallucinated; gold unanswerable."),
    ("q096", 0.0, "PRED 'book corpus' hallucinated; gold unanswerable."),
    ("q097", 0.25, "PRED 'prepared and spontaneous speech' partial; gold=ASR specifically."),
    ("q099", 0.0, "PRED 'only trained on parallel data' WRONG; gold=attentional encoder-decoder networks."),
]


def main() -> None:
    assert len(JUDGMENTS) == 58
    qid_prefix = "p4_k32__qasper__v4-canonical__seed42__"
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
    print(f"qasper p4_k32 v4-canonical seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
