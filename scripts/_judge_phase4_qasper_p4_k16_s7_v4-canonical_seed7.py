"""Phase 4 cross-vendor finishing — QASPER p4_k16_s7 v4-canonical seed=7 (100 entries).

Note: predictions are identical to p4_k16 v4-canonical seed=7 (same retrieval pool).
Scoring matches that template.
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16_s7__qasper__v4-canonical__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'novel DL n-gram + distant rep' WRONG; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED 'No Freebase WordNet' hallucinated; gold unanswerable."),
    ("q002", 0.5, "PRED 'no improvements + devastating linear mapping' partial."),
    ("q003", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q004", 0.0, "PRED refuses; gold=F-score."),
    ("q005", 0.25, "PRED 'Table TABREF2 most hyperparameters' partial."),
    ("q006", 1.0, "PRED 'Yes SemEval-2016 Twitter English' matches gold='Yes'."),
    ("q007", 0.75, "PRED 'bootstrapped crowdsourcing' matches gold."),
    ("q008", 0.0, "PRED 'No does not optimize' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'analyzing tweets sentiment NER' WRONG."),
    ("q010", 1.0, "PRED 'plain stacked LSTMs + peephole + variants' matches gold."),
    ("q011", 0.5, "PRED 'LSTM-CNN' partial; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.75, "PRED 'extracting threads OSG' matches gold."),
    ("q014", 0.0, "PRED refuses; gold=Loaded language."),
    ("q015", 0.0, "PRED 'same as Overall Results' vague."),
    ("q016", 0.0, "PRED 'bilingual dict Google Translate' hallucinated."),
    ("q017", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q018", 1.0, "PRED '6 + over 20 acc' matches gold's specific imbalance figures."),
    ("q019", 1.0, "PRED 'do not provide specific info' matches gold unanswerable."),
    ("q020", 1.0, "PRED 'four conv + two NIN + 3 Bi-LSTM 256 hidden' matches gold detailed."),
    ("q021", 0.0, "PRED refuses; gold='Yes'."),
    ("q022", 1.0, "PRED 'do not specify exact improvement' matches gold unanswerable."),
    ("q023", 0.75, "PRED 'display names + descriptions + usernames' includes gold username."),
    ("q024", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q025", 0.0, "PRED 'improvement limited + drops' WRONG; gold=AutoJudge outperforms."),
    ("q026", 1.0, "PRED 'Yes word segmentation + spelling + rare/frequent' matches gold='Yes'."),
    ("q027", 0.0, "PRED 'demonstrate by' fragment; gold=tSNE."),
    ("q028", 0.25, "PRED 'higher or comparable' vague."),
    ("q029", 1.0, "PRED '22,880 users' exact match."),
    ("q030", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q031", 0.0, "PRED 'linear + RNN-based' WRONG; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'max turn length 160' WRONG; gold=two previous turns."),
    ("q033", 1.0, "PRED 'do not provide details' matches gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C outperform baseline' partial; gold has specific reward numbers."),
    ("q035", 0.5, "PRED 'consistent F1 + improved recall' partial; gold has specific 3.5 F1."),
    ("q036", 1.0, "PRED 'Yes inspect' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM + F1' includes gold Exact Match."),
    ("q038", 1.0, "PRED 'actor-critic + 3 parallel CNNs' matches gold actor-critic."),
    ("q039", 0.0, "PRED 'separate optimizers' WRONG; gold=BERT 512+positions."),
    ("q040", 1.0, "PRED 'Disinformation and mainstream news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match."),
    ("q042", 0.0, "PRED 'ability to maintain perf adversarial' WRONG; gold=distinct word recognitions."),
    ("q043", 0.0, "PRED refuses; gold=MaxEnt + SVMs."),
    ("q044", 0.75, "PRED 'METEOR + ROUGE-L + BLEU-1' includes gold BLEU-1."),
    ("q045", 1.0, "PRED 'No focus NER medical' matches gold='No'."),
    ("q046", 0.75, "PRED 'do not mention transformer' implicit match to gold='No'."),
    ("q047", 0.0, "PRED '3000 negative samples' WRONG; gold has CORD-19 specific details."),
    ("q048", 0.5, "PRED 'N400 + P600 + EPNP + P600' partial."),
    ("q049", 0.0, "PRED refuses; gold='No'."),
    ("q050", 0.0, "PRED refuses; gold='Yes'."),
    ("q051", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q052", 1.0, "PRED 'UD1.2 corpora 16 languages full list' matches gold."),
    ("q053", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q054", 0.0, "PRED 'low ROGUE-2 ROGUE-L' hallucinated; gold unanswerable."),
    ("q055", 1.0, "PRED 'HMM + conv + recurrent + FC layers' matches gold CNN-DNN-BLSTM-HMM."),
    ("q056", 1.0, "PRED 'Further constraining model data structure prevent inaccuracies' matches gold exactly."),
    ("q057", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q058", 1.0, "PRED 'previous SOTA based on RoBERTa' matches gold='RoBERTa'."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% + 0.3%' matches gold exactly."),
    ("q060", 0.0, "PRED 'standard benchmark + reliability' WRONG; gold has specific stats."),
    ("q061", 0.75, "PRED 'diagnosis + patient movement + symptoms + vitals + medication' matches most gold topics."),
    ("q062", 0.75, "PRED 'CDA and REG' includes gold CDA."),
    ("q063", 0.5, "PRED 'transcribed text + ASR' partial; gold=text transcription only."),
    ("q064", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold."),
    ("q065", 0.0, "PRED 'Yes votes' but gold='No' — Y/N flip."),
    ("q066", 0.0, "PRED refuses; gold=No data (pretrained)."),
    ("q067", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q068", 0.0, "PRED refuses; gold='Yes'."),
    ("q069", 0.0, "PRED refuses; gold='Yes'."),
    ("q070", 0.75, "PRED 'page + paragraph + line segmentation' includes gold paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.5, "PRED 'politics + science + tech + Libertarianism' partial includes politics + science."),
    ("q073", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT + Fisher + 20newsgroups' includes gold CSAT."),
    ("q075", 0.75, "PRED 'linear SVM + BiLSTM + CNN' includes gold linear SVM."),
    ("q076", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q079", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q080", 1.0, "PRED '0.94 + 0.94 + 0.95 + 98%' matches gold's specific F1 scores."),
    ("q081", 0.75, "PRED 'ambiguous + unknown words' includes gold ambiguous words."),
    ("q082", 0.75, "PRED 'Stanford NER + DATEXIS-NER' includes gold Stanford NER."),
    ("q083", 0.75, "PRED 'do not mention error analysis' matches gold='No' implicitly."),
    ("q084", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB and Yelp' matches gold."),
    ("q086", 1.0, "PRED '14% = 1000 hours' matches gold."),
    ("q087", 0.5, "PRED 'do not mention deep learning' implicit match to gold='No'."),
    ("q088", 0.0, "PRED 'Five characters' WRONG; gold=45,821."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'CoNLL + TAC + ACE + AQUAINT + WW' partial; gold=CoNLL-YAGO."),
    ("q091", 0.5, "PRED 'irony accuracy + sentiment + content + improper' partial."),
    ("q092", 1.0, "PRED 'No not unsupervised' matches gold='No'."),
    ("q093", 0.75, "PRED 'NPMI + NNEGPMI + CPMI' includes gold."),
    ("q094", 0.75, "PRED '1,888 pairs + 0-6 integer scores' matches gold's score guideline."),
    ("q095", 0.0, "PRED 'multilingual ST + ASR' WRONG; gold=no specific domain."),
    ("q096", 0.0, "PRED 'not explicitly mentioned' refusal; gold=QA PGNet."),
    ("q097", 1.0, "PRED 'No do not compare' matches gold='No'."),
    ("q098", 0.25, "PRED 'Trump + Bachchan + Modi' partial but mostly wrong celebrity list."),
    ("q099", 0.0, "PRED 'SNLI + Multi-NLI classification + regression' WRONG; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_k16_s7__qasper__v4-canonical__seed7__"
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
    print(f"qasper p4_k16_s7 v4-canonical seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
