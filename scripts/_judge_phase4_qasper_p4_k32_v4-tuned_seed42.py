"""Phase 4 cross-vendor finishing — QASPER p4_k32 v4-tuned seed=42 (58 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k32__qasper__v4-tuned__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.75, "PRED 'SVMs and LRs' includes gold=SVMs; LR extra."),
    ("q001", 0.0, "PRED hallucinated image categorization; gold unanswerable."),
    ("q004", 0.0, "PRED refuses; gold='Yes'."),
    ("q007", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q009", 0.0, "PRED 'No, Chinese and English' WRONG; gold='Yes'."),
    ("q010", 0.25, "PRED lists partial subset of 16 languages, not full UD1.2 list."),
    ("q011", 0.0, "PRED 'response time, CPU, disk' WRONG; gold=precision, recall, F1, accuracy."),
    ("q012", 0.0, "PRED 'Yes SQuAD 1.1 English' hallucinated; gold unanswerable."),
    ("q013", 0.75, "PRED 'diagnosis, patient movement, procedures, symptoms, vitals, medication' matches most gold topics."),
    ("q014", 0.75, "PRED 'standard protocol, precision, recall, F-score' matches gold metrics."),
    ("q015", 0.0, "PRED '84.9 F1, 23.7 point improvement' WRONG numbers; gold=0.8% F1."),
    ("q016", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold RNN-based NMT."),
    ("q017", 0.0, "PRED refuses; gold=Loaded language."),
    ("q018", 0.25, "PRED 'in the framework for MRC' partial; gold has scoring-specific detail."),
    ("q019", 0.25, "PRED '250M vs 20M tokens' approximates 14× but doesn't say 14× explicitly."),
    ("q020", 0.0, "PRED 'accuracy improvement, above-chance' WRONG; gold=tSNE plot."),
    ("q022", 0.5, "PRED 'CBOW, PV-DM, GloVe, EqEmb'; gold has b-emb not EqEmb — 3/4 overlap."),
    ("q024", 0.0, "PRED 'relies less on shallow position' WRONG; gold=extra BERT position embeddings."),
    ("q025", 0.0, "PRED 'only uses parallel data' WRONG; gold=No data, pretrained."),
    ("q028", 0.0, "PRED 'BERT from Khandelwal' WRONG; gold=MaxEnt to SVMs."),
    ("q029", 0.5, "PRED 'irony, sentiment, content' partial; gold=Irony Classifier specifically."),
    ("q030", 0.0, "PRED 'may not easily generalize' implies No; gold='Yes'."),
    ("q033", 0.25, "PRED 'not explored for summarization' different from gold=ambiguous/common entities."),
    ("q034", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold BIBREF9."),
    ("q037", 0.5, "PRED covers race part of gold but missing gender (females higher anger/joy/valence) component."),
    ("q038", 0.5, "PRED includes MAP among multiple metrics; gold=only MAP — over-answers."),
    ("q041", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q043", 0.0, "PRED 'two metrics, quantitative and qualitative' vague; gold=average unique predictions."),
    ("q044", 0.0, "PRED refuses; gold=BIBREF26."),
    ("q047", 0.0, "PRED 'Libertarianism, ronpaul, ukpolitics, guns...' WRONG; gold=politics, business, science, AskReddit."),
    ("q048", 0.0, "PRED hallucinated Hub5'00 baselines; gold unanswerable."),
    ("q050", 1.0, "PRED 'CJFA encoder' matches gold."),
    ("q051", 0.0, "PRED 'base NMT model VII' WRONG; gold=pivot-based translation."),
    ("q053", 0.25, "PRED 'qualitative nearest-neighbour results' doesn't match gold=Spearman correlation + entailment datasets."),
    ("q054", 1.0, "PRED 'No' matches gold='No'."),
    ("q058", 0.25, "PRED 'UMA, MRR, coherence, entailment' partial; gold has 9 metrics including BPE PPL, BLEU, ROUGE."),
    ("q060", 1.0, "PRED 'No' matches gold='No'."),
    ("q062", 0.0, "PRED 'Felice2014a' WRONG; gold=Rei2016 error detection."),
    ("q063", 0.25, "PRED 'state-of-the-art, pronounced for longer docs' vague; gold has specific per-dataset Rouge numbers."),
    ("q065", 1.0, "PRED 'Transformer architecture' matches gold."),
    ("q068", 0.0, "PRED hallucinated GloVe/LSTM baseline; gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes, RE task' but gold='No' — Y/N flip."),
    ("q071", 1.0, "PRED 'Yes, GluonCV/NLP models' matches gold='Yes'."),
    ("q072", 0.0, "PRED refuses; gold=CNN."),
    ("q074", 0.75, "PRED 'Reuters-8 database' matches gold=Reuters-8 dataset (stop words detail missing)."),
    ("q075", 0.75, "PRED 'VNBPE, Back Translation, Mix-Source' includes gold=Back Translation."),
    ("q076", 0.0, "PRED gives offensive-comment stats WRONG; gold=SGD, naive bayes, decision tree."),
    ("q079", 0.0, "PRED refuses; gold=four MT tasks DE-EN, JA-EN, RO-EN, EN-DE."),
    ("q082", 0.0, "PRED 'GENIA chunker, DATEXIS-NER' WRONG; gold=Babelfy, DBpedia, Entityclassifier, FOX, LingPipe, NERD-ML, Stanford NER, TagMe."),
    ("q083", 0.25, "PRED 'lack of powerful tool, large corpus needed' partial; gold=ambiguous words."),
    ("q084", 0.0, "PRED hallucinated WER numbers; gold unanswerable."),
    ("q085", 0.75, "PRED 'LORELEI + Switchboard' includes gold Switchboard Telephone Speech Corpus."),
    ("q089", 0.75, "PRED 'usability, WER, BLEU' includes gold=BLEU."),
    ("q090", 0.75, "PRED 'recurrent CRF, spaCy, Stanford NER' includes gold Stanford NER."),
    ("q091", 0.0, "PRED 'unclear'; gold='Yes' — refusal wrong."),
    ("q096", 0.0, "PRED 'PPDB' hallucinated; gold unanswerable."),
    ("q097", 0.25, "PRED 'prepared and spontaneous speech' partial; gold=ASR specifically."),
    ("q098", 0.5, "PRED 'Comparison, Temporal best; Implicit worst' partial; gold=explicit discourse relations distinction."),
]


def main() -> None:
    assert len(JUDGMENTS) == 58
    qid_prefix = "p4_k32__qasper__v4-tuned__seed42__"
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
    print(f"qasper p4_k32 v4-tuned seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
