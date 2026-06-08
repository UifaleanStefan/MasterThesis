"""Phase 4 cross-vendor finishing — QASPER p4_k4 v4-tuned seed=42 (66 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k4__qasper__v4-tuned__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'retrained on train+dev union' WRONG; gold=SVMs."),
    ("q001", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q003", 0.25, "PRED 'TAC2010, WW, ACE2004, CoNLL, AQUAINT' partial; gold=CoNLL-YAGO specifically."),
    ("q004", 1.0, "PRED 'Yes, SAN' matches gold='Yes'."),
    ("q005", 0.75, "PRED 'does not provide info' implicit match to gold unanswerable."),
    ("q006", 0.0, "PRED 'outperformed previous summary graph methods' hallucinated; gold unanswerable."),
    ("q007", 0.0, "PRED 'passages do not specify'; gold='Yes' — refusal wrong."),
    ("q010", 0.25, "PRED 'Czech, French, Italian, Indonesian' partial; gold=full UD1.2 16-language list."),
    ("q011", 0.0, "PRED 'correctness, response time, computing resources' WRONG; gold=precision, recall, F1, accuracy."),
    ("q012", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q013", 0.25, "PRED 'diagnosis history, symptoms/signs' partial; gold has 10 categories."),
    ("q014", 0.5, "PRED 'automatic evaluations, simple protocol' partial; gold specifies F-score vs ANNODIS annotators."),
    ("q015", 0.0, "PRED refuses; gold=0.8% F1."),
    ("q017", 0.0, "PRED refuses; gold=Loaded language."),
    ("q018", 0.25, "PRED 'in the QA+QG framework' partial; gold specifies scoring via semantic relevance."),
    ("q019", 0.25, "PRED '270M vs 20M tokens' approximates 14× without stating it explicitly."),
    ("q020", 0.0, "PRED 'novel mixed DNN, improved classification' WRONG; gold=tSNE."),
    ("q022", 1.0, "PRED 'b-emb, CBOW, PV-DM, GloVe' exact match to gold."),
    ("q024", 0.0, "PRED 'relies less on shallow position' WRONG; gold=extra BERT position embeddings."),
    ("q025", 0.0, "PRED 'limited amount, Hindi' WRONG; gold=No data, pretrained model."),
    ("q026", 0.75, "PRED 'from CoNLL 2003 data' partial match to gold ceccarelli2013learning from CoNLL 2003."),
    ("q027", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q028", 0.0, "PRED 'BERT from Khandelwal' WRONG; gold=MaxEnt to SVMs."),
    ("q029", 0.0, "PRED 'Additional experiments' vague; gold=Irony Classifier specifically."),
    ("q030", 0.0, "PRED 'may not easily generalize' implies No; gold='Yes'."),
    ("q034", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold BIBREF9."),
    ("q035", 0.75, "PRED 'De-En, Ja-En, Ro-En' includes gold De-En."),
    ("q038", 0.0, "PRED 'loss function, validation' doesn't mention MAP; gold=MAP only."),
    ("q040", 0.25, "PRED 'F1 0.92 intent, 0.97 slot' different from gold=2.3-6.8 F1 intent, 0.8-3.5 slot."),
    ("q041", 0.0, "PRED refuses; gold='Yes'."),
    ("q043", 0.0, "PRED refuses; gold=average unique predictions."),
    ("q044", 0.0, "PRED 'task-specific architecture' doesn't cite BIBREF26; gold=BIBREF26."),
    ("q046", 0.0, "PRED 'Turkish 74.36%' WRONG; gold=Russian."),
    ("q047", 0.0, "PRED 'climate change, abortion, world news' WRONG; gold=politics, business, science, AskReddit."),
    ("q048", 0.0, "PRED 'other models trained on same data' hallucinated; gold unanswerable."),
    ("q050", 0.0, "PRED refuses; gold=CJFA encoder."),
    ("q051", 0.0, "PRED 'base NMT model VII' WRONG; gold=pivot-based translation."),
    ("q052", 0.0, "PRED 'grammaticality judgment, PTB vocabulary' WRONG; gold=INLINEFORM0 scores."),
    ("q053", 0.0, "PRED refuses; gold=Spearman correlation + entailment datasets."),
    ("q054", 0.75, "PRED 'does not mention' implicit match to gold='No'."),
    ("q056", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q058", 0.0, "PRED 'coherence, semantic plausibility' WRONG; gold has 9 specific metrics."),
    ("q061", 0.0, "PRED refuses; gold=BOW-LR, BOW-RF, TFIDF-RF, TextCNN, C-TextCNN."),
    ("q065", 0.0, "PRED refuses; gold=Transformer architecture."),
    ("q068", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q069", 0.75, "PRED 'does not indicate' implicit match to gold='No'."),
    ("q071", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q073", 0.75, "PRED 'does not indicate' implicit match to gold='No'."),
    ("q075", 0.0, "PRED 'advanced scalable methods' vague; gold=Back Translation."),
    ("q076", 0.0, "PRED refuses; gold=SGD, naive bayes, decision tree."),
    ("q077", 0.0, "PRED '-PMI and +PPMI' WRONG; gold=clipped PMI, NNEGPMI."),
    ("q079", 0.0, "PRED refuses; gold=four MT tasks DE-EN, JA-EN, RO-EN, EN-DE."),
    ("q080", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q082", 0.0, "PRED 'biLSTM F1 84-94%' WRONG; gold=list of 8 named NER systems."),
    ("q084", 0.0, "PRED hallucinated WER; gold unanswerable."),
    ("q085", 0.75, "PRED 'LORELEI, Switchboard, Uzbek, Mandarin' includes gold Switchboard."),
    ("q088", 0.0, "PRED 'two-staged CNN, attention value' hallucinated; gold unanswerable."),
    ("q089", 0.0, "PRED 'WER, acceptability, editing' doesn't include BLEU; gold=BLEU."),
    ("q090", 0.0, "PRED 'spaCy, Char-biLSTM' doesn't include Stanford NER; gold=Stanford NER."),
    ("q091", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q092", 1.0, "PRED 'MLP' matches gold=MLP."),
    ("q094", 0.0, "PRED 'No, not only English' hallucinated; gold unanswerable."),
    ("q096", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q097", 0.25, "PRED 'WER for Punctual and spontaneous speech' partial; gold=ASR."),
    ("q098", 0.5, "PRED 'Explicit-Temporal best, Implicit-Comparison worst' partial; gold=explicit discourse relations."),
    ("q099", 0.0, "PRED refuses; gold=attentional encoder-decoder networks."),
]


def main() -> None:
    assert len(JUDGMENTS) == 66
    qid_prefix = "p4_k4__qasper__v4-tuned__seed42__"
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
    print(f"qasper p4_k4 v4-tuned seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
