"""Phase 4 cross-vendor finishing — QASPER p4_main flat-50 seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__flat-50__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'CNN' WRONG; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED 'No Freebase WordNet' hallucinated; gold unanswerable."),
    ("q002", 0.5, "PRED 'no improvements + devastating linear mapping' partial."),
    ("q003", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q004", 0.0, "PRED refuses; gold=F-score."),
    ("q005", 0.25, "PRED 'optimal combinations word2vec' partial."),
    ("q006", 0.0, "PRED 'No out-of-domain Wikipedia' but gold='Yes' — Y/N flip."),
    ("q007", 0.5, "PRED 'careful collection out-of-scope' partial."),
    ("q008", 0.0, "PRED 'No does not optimize' hallucinated; gold unanswerable."),
    ("q009", 0.25, "PRED 'non-expert individuals sentiment' partial."),
    ("q010", 0.25, "PRED 'CAS-LSTM sentence modeling' partial; gold=Stacked LSTMs."),
    ("q011", 0.0, "PRED refuses; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.0, "PRED refuses; gold=OSG crawling."),
    ("q014", 0.0, "PRED refuses; gold=Loaded language."),
    ("q015", 0.0, "PRED 'derived from corpus data' vague; gold=antonym/synonym pairs."),
    ("q016", 0.0, "PRED 'Google Translate dictionary' hallucinated; gold unanswerable."),
    ("q017", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q018", 1.0, "PRED 'improvement increases + 6 acc on 100:1000' matches gold."),
    ("q019", 1.0, "PRED 'do not provide specific info' matches gold unanswerable."),
    ("q020", 1.0, "PRED 'novel CNN + NIN + Bi-LSTM + LSTM enc-dec' matches gold."),
    ("q021", 1.0, "PRED 'Yes SAN attention' matches gold='Yes'."),
    ("q022", 1.0, "PRED 'do not specify quantifiable' matches gold unanswerable."),
    ("q023", 0.75, "PRED 'Profile Name + Username + Description + Image + Location' includes gold."),
    ("q024", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q025", 0.75, "PRED 'considerable improvement vs baselines' matches gold."),
    ("q026", 1.0, "PRED 'Yes word segmentation + spelling + rare' matches gold='Yes'."),
    ("q027", 0.0, "PRED 'demonstrate by sp' fragment; gold=tSNE."),
    ("q028", 0.25, "PRED 'higher or comparable' vague."),
    ("q029", 0.75, "PRED 'Over 20,000 blog users' approximates gold 22,880."),
    ("q030", 1.0, "PRED 'do not specify downstream tasks' matches gold unanswerable."),
    ("q031", 0.0, "PRED 'linear + SVM + BiLSTM' WRONG; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'turn sizes 3 and 5' WRONG; gold=two previous turns."),
    ("q033", 0.0, "PRED 'meaning preservation + fluency' hallucinated; gold unanswerable."),
    ("q034", 0.25, "PRED 'both agents pass bottleneck' partial."),
    ("q035", 0.25, "PRED '600 articles 68.1%' partial different angle."),
    ("q036", 1.0, "PRED 'Yes inspect' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and F1' includes gold Exact Match."),
    ("q038", 0.25, "PRED 'CNN-RNN + seq2seq' partial; no actor-critic mention."),
    ("q039", 0.0, "PRED 'general framework abstractive extractive' WRONG."),
    ("q040", 1.0, "PRED 'mainstream + disinformation news' matches gold."),
    ("q041", 0.0, "PRED refuses; gold=German-English."),
    ("q042", 0.5, "PRED 'responsiveness to input + unique predictions' partial."),
    ("q043", 0.0, "PRED 'BERT BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q044", 0.0, "PRED refuses; gold=BLEU-1."),
    ("q045", 1.0, "PRED 'No focus NER medical' matches gold='No'."),
    ("q046", 0.75, "PRED 'do not mention transformer' implicit match to gold='No'."),
    ("q047", 0.5, "PRED 'papers about COVID-19' partial CORD-19 description."),
    ("q048", 0.5, "PRED 'LAN+P600 + ELAN+P600' alternate answer."),
    ("q049", 0.0, "PRED refuses; gold='No'."),
    ("q050", 0.0, "PRED refuses; gold='Yes'."),
    ("q051", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q052", 0.25, "PRED 'morphologically rich Slavic + English' partial."),
    ("q053", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q054", 0.0, "PRED 'outperformed for summary graphs' hallucinated; gold unanswerable."),
    ("q055", 1.0, "PRED 'HMM + conv + recurrent + FC' matches gold CNN-DNN-BLSTM-HMM."),
    ("q056", 1.0, "PRED 'Further constraining prevent inaccuracies' matches gold exactly."),
    ("q057", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q058", 1.0, "PRED 'RoBERTa' matches gold."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% + 0.3' matches gold exactly."),
    ("q060", 0.25, "PRED 'standard benchmark commonsense' partial."),
    ("q061", 0.75, "PRED 'diagnosis + patient movement + symptoms + vitals + medication + procedures' matches most gold topics."),
    ("q062", 0.75, "PRED 'CDA + proposed loss debiasing' includes gold CDA."),
    ("q063", 1.0, "PRED 'transcribed text' matches gold."),
    ("q064", 0.5, "PRED 'four Spanish subtasks' partial (count matches but no specific names)."),
    ("q065", 0.0, "PRED 'Yes votes' but gold='No' — Y/N flip."),
    ("q066", 0.5, "PRED 'do not specify amount' implicit match."),
    ("q067", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q068", 0.0, "PRED refuses; gold='Yes'."),
    ("q069", 0.0, "PRED 'No method does not help' but gold='Yes' — Y/N flip."),
    ("q070", 0.75, "PRED 'images + paragraphs + lines + words' includes gold paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'behavioral predictors dogmatism' WRONG topics."),
    ("q073", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT + 20newsgroups + Fisher' includes gold CSAT."),
    ("q075", 0.75, "PRED 'SVMs + neural networks' includes gold linear SVM."),
    ("q076", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q079", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q080", 1.0, "PRED '0.94 + 0.94 + 0.95 + 98%' matches gold exactly."),
    ("q081", 0.25, "PRED 'lack of powerful Vietnamese tool' partial; gold=ambiguous words."),
    ("q082", 0.0, "PRED 'bidirectional LSTMs + trigram' missing SOTA tool names."),
    ("q083", 0.75, "PRED 'do not mention error analysis' implicit match to gold='No'."),
    ("q084", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB and Yelp' matches gold."),
    ("q086", 0.75, "PRED '14% size' matches gold's 1000 hours concept."),
    ("q087", 0.5, "PRED 'do not mention deep learning' implicit match to gold='No'."),
    ("q088", 0.0, "PRED 'Five evaluation characters' WRONG; gold=45,821."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'five datasets CoNLL testa + TAC2010' partial; gold=CoNLL-YAGO."),
    ("q091", 0.5, "PRED 'transformation + irony generation' partial."),
    ("q092", 0.0, "PRED 'Yes unsupervised LDA' but gold='No' — Y/N flip."),
    ("q093", 1.0, "PRED 'CPMI + NNEGPMI' matches gold."),
    ("q094", 0.25, "PRED 'Multi-SimLex community website' partial."),
    ("q095", 0.0, "PRED 'multilingual speech-to-text 11 languages' WRONG; gold=no specific domain."),
    ("q096", 0.0, "PRED 'human transcripts + segmentation' WRONG; gold=QA PGNet."),
    ("q097", 0.75, "PRED 'do not indicate comparison' implicit match to gold='No'."),
    ("q098", 0.25, "PRED 'UPC + UNC categorized' partial count only."),
    ("q099", 0.0, "PRED refuses; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_tierb__qasper__flat-50__seed7__"
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
                skipped += 1
                continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_tierb flat-50 seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
