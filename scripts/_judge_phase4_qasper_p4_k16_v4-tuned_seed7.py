"""Phase 4 cross-vendor finishing — QASPER p4_k16 v4-tuned seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16__qasper__v4-tuned__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'n-gram + self-attention' WRONG; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED 'No Freebase WordNet' hallucinated; gold unanswerable."),
    ("q002", 0.25, "PRED 'fine-tuned untranslated better' partial."),
    ("q003", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q004", 0.75, "PRED 'F-score and Kappa' includes gold F-score."),
    ("q005", 0.5, "PRED 'window + dimensions + skip-gram softmax' partial."),
    ("q006", 1.0, "PRED 'Yes English datasets' matches gold='Yes'."),
    ("q007", 1.0, "PRED 'crowdsourcing + out-of-scope queries + scoping' matches gold."),
    ("q008", 0.0, "PRED 'No does not optimize' hallucinated; gold unanswerable."),
    ("q009", 0.5, "PRED 'individuals in US BIBREF22' partial."),
    ("q010", 1.0, "PRED 'plain stacked LSTMs + variants' matches gold."),
    ("q011", 0.5, "PRED 'LSTM-CNN' partial; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 1.0, "PRED 'crawling + pre-processing OSG' matches gold exactly."),
    ("q014", 0.0, "PRED lists techniques but no 'Loaded language'."),
    ("q015", 1.0, "PRED 'word pairs antonym/synonym' matches gold."),
    ("q016", 0.0, "PRED 'pre-ordering rules' hallucinated; gold unanswerable."),
    ("q017", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q018", 0.5, "PRED 'augmentation + weighting greatly improves' partial."),
    ("q019", 1.0, "PRED 'do not provide specific info' matches gold unanswerable."),
    ("q020", 1.0, "PRED 'conv + NIN + deep bidirectional LSTM' matches gold."),
    ("q021", 1.0, "PRED 'Yes use attention' matches gold='Yes'."),
    ("q022", 1.0, "PRED 'do not provide quantitative' matches gold unanswerable."),
    ("q023", 0.75, "PRED 'username + display + image + location + description' includes gold."),
    ("q024", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q025", 0.75, "PRED 'considerable improvement vs baselines' matches gold."),
    ("q026", 1.0, "PRED 'Yes challenges exist' matches gold='Yes'."),
    ("q027", 0.0, "PRED 'demonstrate by sh' fragment; gold=tSNE."),
    ("q028", 0.25, "PRED 'higher or comparable' vague."),
    ("q029", 1.0, "PRED '22,880 users' exact match."),
    ("q030", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q031", 1.0, "PRED 'state-of-the-art PDTB taggers' exact match."),
    ("q032", 0.0, "PRED 'varying dialog turn sizes 3 and 5' WRONG; gold=two previous turns."),
    ("q033", 0.0, "PRED 'evaluators score +1' hallucinated; gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C outperform A2C + sample efficient' partial."),
    ("q035", 0.5, "PRED 'expert annotations superior + F1 higher' partial."),
    ("q036", 1.0, "PRED 'Yes inspect' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and F1' includes gold Exact Match."),
    ("q038", 0.5, "PRED 'CNN-RNN + seq2seq global attention' partial; no actor-critic."),
    ("q039", 0.5, "PRED 'pretrained Bert document-level' partial."),
    ("q040", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match."),
    ("q042", 1.0, "PRED 'expected unique outputs adversarial perturbations' matches gold."),
    ("q043", 0.0, "PRED 'BERT from BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q044", 0.75, "PRED 'BLEU-1 + Meteor + Rouge-L' includes gold BLEU-1."),
    ("q045", 0.0, "PRED 'Yes MTL' but gold='No' — Y/N flip."),
    ("q046", 0.75, "PRED 'do not mention transformer' matches gold='No' implicitly."),
    ("q047", 1.0, "PRED 'over 45,000 articles + COVID-19' matches gold exactly."),
    ("q048", 0.5, "PRED 'LAN+P600 + ELAN+P600' alternate answer to gold's pairs."),
    ("q049", 1.0, "PRED 'No important words better translation' matches gold='No'."),
    ("q050", 0.0, "PRED 'No Chinese-to-English' but gold='Yes' — Y/N flip."),
    ("q051", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q052", 1.0, "PRED 'UD1.2 16 languages full list' matches gold."),
    ("q053", 0.0, "PRED 'grammaticality + perplexity + F1 length' WRONG; gold=INLINEFORM0."),
    ("q054", 0.0, "PRED 'outperformed previous AMR' hallucinated; gold unanswerable."),
    ("q055", 1.0, "PRED 'HMM + conv + recurrent + FC' matches gold CNN-DNN-BLSTM-HMM."),
    ("q056", 1.0, "PRED 'Further constraining model data structure prevent inaccuracies' matches gold."),
    ("q057", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q058", 1.0, "PRED 'previous SOTA based on RoBERTa' matches gold='RoBERTa'."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% + 0.3' matches gold exactly."),
    ("q060", 0.0, "PRED '13,939 questions + 3,827 commonsense + training split' WRONG; gold has different stats."),
    ("q061", 0.75, "PRED 'HPI demographics + diagnosis + symptoms/signs' matches most gold topics."),
    ("q062", 0.75, "PRED 'CDA and REG' includes gold CDA."),
    ("q063", 0.5, "PRED 'transcribed text + ASR' partial; gold=text transcription."),
    ("q064", 0.75, "PRED 'anger EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold (anger qualifier added)."),
    ("q065", 1.0, "PRED 'No combining votes + speeches' matches gold='No'."),
    ("q066", 0.25, "PRED 'monolingual + raw Wikipedia + parallel' partial; gold=No data (pretrained)."),
    ("q067", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q068", 1.0, "PRED 'Yes ShapeWorld' matches gold='Yes'."),
    ("q069", 0.0, "PRED 'No shallow syntactic no benefit' but gold='Yes' — Y/N flip."),
    ("q070", 0.75, "PRED 'page + paragraph + line + structure' includes gold paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'Libertarianism + Anarcho + ronpaul' WRONG topics."),
    ("q073", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT + 20newsgroups + Fisher' includes gold CSAT."),
    ("q075", 0.0, "PRED 'CNN BiLSTM' WRONG; gold=linear SVM."),
    ("q076", 1.0, "PRED 'not explicitly mentioned' matches gold unanswerable."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 0.0, "PRED '700' hallucinated; gold unanswerable."),
    ("q079", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q080", 1.0, "PRED '0.94 + 0.94 + 0.95 + 98%' matches gold exactly."),
    ("q081", 0.25, "PRED 'lack of powerful Vietnamese tool' partial; gold=ambiguous words."),
    ("q082", 0.0, "PRED 'F1 84-94% + LSTMs' missing SOTA tool names."),
    ("q083", 0.75, "PRED 'do not mention error analysis' matches gold='No' implicitly."),
    ("q084", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB and Yelp' matches gold."),
    ("q086", 0.75, "PRED '14% size of training' matches gold's 1000 hours concept."),
    ("q087", 0.5, "PRED 'do not specify deep learning' implicit match to gold='No'."),
    ("q088", 0.0, "PRED '327 different characters' WRONG; gold=45,821."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'TAC2010 + WW + ACE + AQUAINT + CoNLL' partial; gold=CoNLL-YAGO."),
    ("q091", 0.5, "PRED 'transformation ironic to non-ironic + word repetition' partial."),
    ("q092", 0.0, "PRED 'Yes unsupervised LDA' but gold='No' — Y/N flip."),
    ("q093", 1.0, "PRED 'CPMI + NNEGPMI' matches gold."),
    ("q094", 0.5, "PRED '145 annotators 1,888 pairs' partial; gold has specific score guideline."),
    ("q095", 0.0, "PRED 'ASR + MT + ST' WRONG; gold=no specific domain."),
    ("q096", 0.0, "PRED 'Nearest Number Dosage' WRONG; gold=QA PGNet."),
    ("q097", 0.75, "PRED 'do not mention comparison' matches gold='No' implicitly."),
    ("q098", 0.25, "PRED '15 celebrities various domains' partial count only."),
    ("q099", 0.25, "PRED 'seven SentEval tasks' vague; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_k16__qasper__v4-tuned__seed7__"
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
    print(f"qasper p4_k16 v4-tuned seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
