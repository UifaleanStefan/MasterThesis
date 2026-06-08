"""Phase 4 cross-vendor finishing — QASPER p4_main v4-canonical seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__v4-canonical__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'n-gram + self-attention' WRONG; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED 'multilingual data' hallucinated; gold unanswerable."),
    ("q002", 0.5, "PRED 'no improvements + devastating effects' partial."),
    ("q003", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q004", 0.0, "PRED 'Chi squared + Info Gain' WRONG; gold=F-score."),
    ("q005", 0.5, "PRED 'dimensions + model combos' partial; gold has full hyperparam list."),
    ("q006", 0.0, "PRED refuses; gold='Yes'."),
    ("q007", 0.75, "PRED 'bootstrapped using crowdsourcing' matches gold."),
    ("q008", 0.0, "PRED 'No does not optimize' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'analyze tweets' WRONG; gold=Mechanical Turk."),
    ("q010", 0.5, "PRED 'CAS-LSTM + variants' partial; gold=Stacked LSTMs."),
    ("q011", 0.5, "PRED 'LSTM-CNN' partial; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.0, "PRED refuses; gold=crawling OSG web forum."),
    ("q014", 0.0, "PRED refuses; gold=Loaded language."),
    ("q015", 0.0, "PRED 'Overall Results section' vague; gold=antonym/synonym pairs."),
    ("q016", 0.0, "PRED 'bilingual dictionary' hallucinated; gold unanswerable."),
    ("q017", 0.0, "PRED 'WebQSP' WRONG; gold=SimpleQuestions."),
    ("q018", 1.0, "PRED '6 acc + 20 acc on imbalance' matches gold."),
    ("q019", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q020", 0.0, "PRED 'Attention Encoder-Decoder' WRONG; gold=CNN+NIN+BLSTM."),
    ("q021", 1.0, "PRED 'Yes SAN attention' matches gold='Yes'."),
    ("q022", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q023", 0.0, "PRED refuses; gold=username."),
    ("q024", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q025", 0.0, "PRED 'improvement quite limited' WRONG; gold=AutoJudge outperforms."),
    ("q026", 0.5, "PRED 'implies challenges' partial; gold='Yes'."),
    ("q027", 0.0, "PRED 'kappa' WRONG; gold=tSNE plots."),
    ("q028", 0.25, "PRED 'higher or comparable' vague; gold=specific F1 scores."),
    ("q029", 1.0, "PRED '22,880 users' exact match."),
    ("q030", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q031", 0.0, "PRED 'CEI Only + pipeline' WRONG; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'max turn length 160' WRONG; gold=two previous turns."),
    ("q033", 1.0, "PRED 'do not provide details' matches gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C outperform A2C' partial; gold has specific reward numbers."),
    ("q035", 0.5, "PRED 'increases F1 + recall' partial; gold has specific 3.5 F1 number."),
    ("q036", 1.0, "PRED 'Yes inspect' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and F1' includes gold Exact Match."),
    ("q038", 0.0, "PRED 'seq2seq + pointer' WRONG; gold=actor-critic."),
    ("q039", 0.0, "PRED 'separate optimizers' WRONG; gold=BERT max 512 + position embeddings."),
    ("q040", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match."),
    ("q042", 0.5, "PRED 'responsiveness to OOV' partial; gold=distinct word recognitions."),
    ("q043", 0.0, "PRED refuses; gold=MaxEnt + SVMs."),
    ("q044", 0.75, "PRED 'METEOR + ROUGE-L + BLEU-1' includes gold BLEU-1."),
    ("q045", 0.0, "PRED 'Yes MTL' but gold='No' — Y/N flip."),
    ("q046", 0.75, "PRED 'do not mention' matches gold='No' implicitly."),
    ("q047", 0.0, "PRED 'do not contain info'; gold=CORD-19 specific dataset description."),
    ("q048", 0.5, "PRED 'N400 + P600' partial match to gold's broken answer."),
    ("q049", 0.0, "PRED refuses; gold='No'."),
    ("q050", 0.0, "PRED 'No multiple pairs' but gold='Yes' — Y/N flip."),
    ("q051", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q052", 0.5, "PRED 'Czech French Italian Indonesian English' partial (5 of 16)."),
    ("q053", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q054", 0.0, "PRED 'SummaRunNer Lead-3' hallucinated; gold unanswerable."),
    ("q055", 0.75, "PRED 'HMM + conv + recurrent' partial; gold=CNN-DNN-BLSTM-HMM."),
    ("q056", 1.0, "PRED 'further constrain model on data structure' matches gold exactly."),
    ("q057", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q058", 1.0, "PRED 'RoBERTa' matches gold='RoBERTa'."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% + 0.3' matches gold exactly."),
    ("q060", 0.0, "PRED 'standard benchmark + reliability' WRONG; gold has specific stats."),
    ("q061", 0.75, "PRED 'diagnosis history + symptoms + vitals/labs + medication' includes many gold topics."),
    ("q062", 0.75, "PRED 'CDA + proposed method' includes gold CDA."),
    ("q063", 0.5, "PRED 'transcribed text + ASR' partial; gold=text transcription only."),
    ("q064", 1.0, "PRED 'EI-Reg EI-Oc V-Reg V-Oc' matches gold exactly."),
    ("q065", 0.0, "PRED 'Yes votes' but gold='No' — Y/N flip."),
    ("q066", 0.5, "PRED 'do not specify amount' partial unanswerable behavior."),
    ("q067", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q068", 1.0, "PRED 'Yes specific domain' matches gold='Yes'."),
    ("q069", 0.0, "PRED refuses; gold='Yes'."),
    ("q070", 0.75, "PRED 'page + paragraph + line segmentation' includes gold paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'Libertarianism + Anarcho + ronpaul' WRONG topics."),
    ("q073", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT Fisher 20newsgroups' includes gold CSAT."),
    ("q075", 0.0, "PRED 'CNN BiLSTM' WRONG; gold=linear SVM."),
    ("q076", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'do not specify number' matches gold unanswerable."),
    ("q079", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q080", 1.0, "PRED '0.94 + 0.94 + 0.95 + 98% acc' matches gold exactly."),
    ("q081", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q082", 0.0, "PRED 'F1 84-94%' WRONG; gold lists specific SOTA tools."),
    ("q083", 0.75, "PRED 'do not mention' matches gold='No' implicitly."),
    ("q084", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB + Yelp' includes gold."),
    ("q086", 1.0, "PRED '14% = 1000 hours' matches gold."),
    ("q087", 0.5, "PRED 'do not mention deep learning' implicitly matches gold='No'."),
    ("q088", 0.0, "PRED 'Five characters' WRONG; gold=45,821."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'CoNLL TAC2010 ACE2004 WW AQUAINT' partial; gold=CoNLL-YAGO."),
    ("q091", 0.0, "PRED 'improper word generation + style transfer' WRONG; gold=Irony Classifier."),
    ("q092", 1.0, "PRED 'No not unsupervised' matches gold='No'."),
    ("q093", 0.75, "PRED 'PPMI CPMI NNEGPMI' includes gold."),
    ("q094", 0.5, "PRED 'detailed protocol' partial; gold has specific score guideline."),
    ("q095", 0.0, "PRED 'multilingual ST and MT' WRONG; gold=no specific domain."),
    ("q096", 0.0, "PRED 'ASR transcripts' WRONG; gold=QA PGNet."),
    ("q097", 0.5, "PRED 'do not explicitly mention' implicitly matches gold='No'."),
    ("q098", 0.0, "PRED 'UPC + UNC' WRONG; gold lists specific celebrities."),
    ("q099", 0.0, "PRED refuses; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__v4-canonical__seed7__"
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
    print(f"qasper p4_main v4-canonical seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
