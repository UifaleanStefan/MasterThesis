"""Phase 4 cross-vendor finishing — QASPER p4_main attention-tuned seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__attention-tuned__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.25, "PRED 'novel DL model Thai segmentation' partial; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED 'Results and Analysis' section header."),
    ("q002", 0.0, "PRED 'Reading Comprehension central task' fragment."),
    ("q003", 0.0, "PRED 'Testing Generalizability' section header."),
    ("q004", 0.0, "PRED 'I do not know'; gold=F-score."),
    ("q005", 0.5, "PRED 'hyper-param combinations word2vec' partial."),
    ("q006", 0.5, "PRED '36M English tweets' partially matches gold='Yes only English'."),
    ("q007", 0.0, "PRED 'Dataset' section header."),
    ("q008", 0.25, "PRED 'balancing data would be helpful' partial."),
    ("q009", 0.0, "PRED 'social media Twitter' WRONG; gold=Mechanical Turk."),
    ("q010", 0.0, "PRED 'I do not know'; gold=Stacked LSTMs."),
    ("q011", 0.0, "PRED 'I do not know'; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED 'Wikipedia entity pages' WRONG."),
    ("q013", 0.0, "PRED 'I do not know'."),
    ("q014", 0.0, "PRED 'Propaganda Techniques' section header."),
    ("q015", 0.0, "PRED 'Dataset' section header."),
    ("q016", 0.0, "PRED 'CFILT-preorder' hallucinated."),
    ("q017", 0.0, "PRED 'HR-BiLSTM WebQSP' WRONG; gold=SimpleQuestions."),
    ("q018", 0.25, "PRED 'data weighting improves' partial."),
    ("q019", 0.0, "PRED 'few-fact entities' fragment."),
    ("q020", 0.25, "PRED 'encoder-decoder neural network' vague."),
    ("q021", 0.0, "PRED 'Memory Generation Layer' fragment."),
    ("q022", 0.0, "PRED 'Attention Vectors' section header."),
    ("q023", 0.0, "PRED 'Profile Attribute Analysis' section header."),
    ("q024", 0.0, "PRED 'twitter public-opinion' fragment."),
    ("q025", 0.0, "PRED 'Dataset Construction for Evaluation' section header."),
    ("q026", 0.25, "PRED 'Table 4 + 3 datasets' partial."),
    ("q027", 0.0, "PRED 'production of articulatory speech' fragment."),
    ("q028", 0.0, "PRED 'Models' fragment."),
    ("q029", 0.5, "PRED 'Table TABREF1 users in each category 22,880' partial."),
    ("q030", 0.0, "PRED 'active learning batch size' fragment."),
    ("q031", 0.0, "PRED 'CEI task DA AVG LSTM' WRONG."),
    ("q032", 0.0, "PRED 'compare contextual to DACL' fragment."),
    ("q033", 0.0, "PRED 'style transfer evaluation' fragment."),
    ("q034", 0.25, "PRED 'KG-A2C knowledge graph state-rep' partial."),
    ("q035", 0.25, "PRED 'Rows 3 and 4 crowd annotation' partial."),
    ("q036", 0.5, "PRED 'image-aided model + modality attention' partial Yes."),
    ("q037", 0.75, "PRED 'EM + Macro F1' includes gold Exact Match."),
    ("q038", 0.0, "PRED 'Prose for a Painting title' fragment."),
    ("q039", 0.0, "PRED 'learning rate hyperparams' WRONG."),
    ("q040", 0.75, "PRED 'country-independent two news domains' matches gold concept."),
    ("q041", 0.0, "PRED 'Character-level NMT' WRONG; gold=German-English."),
    ("q042", 0.0, "PRED 'Understanding Model Sensitivity' section header."),
    ("q043", 0.0, "PRED 'CoNLL-2010' WRONG."),
    ("q044", 0.0, "PRED 'Task and Evaluation' section header."),
    ("q045", 0.0, "PRED 'I do not know'; gold='No'."),
    ("q046", 0.25, "PRED 'I do not know' partial match to gold='No' (no analysis)."),
    ("q047", 0.0, "PRED 'Dataset' fragment."),
    ("q048", 0.0, "PRED 'ERP component discussion' fragment."),
    ("q049", 0.5, "PRED 'necessity word importance' partial Yes."),
    ("q050", 0.0, "PRED 'Dependency Word Pair Features' section header."),
    ("q051", 0.0, "PRED 'NeuronBlocks CoNLL-2003' hallucinated."),
    ("q052", 0.5, "PRED 'Plank bi-LSTM MarMoT 3 of 16 Czech French Italian' partial."),
    ("q053", 0.0, "PRED 'modify dataset BIBREF56' fragment."),
    ("q054", 0.0, "PRED 'baseline models for each step' fragment."),
    ("q055", 0.0, "PRED 'I do not know'; gold=CNN-DNN-BLSTM-HMM."),
    ("q056", 0.0, "PRED 'Conclusion and future work' section header."),
    ("q057", 0.5, "PRED 'I do not know' partial match to gold unanswerable."),
    ("q058", 0.75, "PRED 'previous encoder-only RoBERTa' matches gold='RoBERTa'."),
    ("q059", 0.0, "PRED 'Table TABREF8 baseline HUMAN' fragment."),
    ("q060", 0.0, "PRED 'Data Statistics' section header."),
    ("q061", 0.5, "PRED 'classifier for HPI demographics + diagnosis + symptoms' partial."),
    ("q062", 0.0, "PRED 'Existing Approaches' section header."),
    ("q063", 0.5, "PRED 'IEMOCAP transcripts' partial."),
    ("q064", 0.0, "PRED 'I do not know'; gold=EI-Reg etc."),
    ("q065", 0.0, "PRED 'UN roll call' WRONG."),
    ("q066", 0.0, "PRED 'linguistic knowledge transfer' section header."),
    ("q067", 0.0, "PRED 'Summary selection' fragment."),
    ("q068", 0.0, "PRED 'Failure to learn spatial' WRONG."),
    ("q069", 0.0, "PRED 'Downstream Task Sentiment' section header."),
    ("q070", 0.5, "PRED 'text structure + typography + images' partial."),
    ("q071", 0.0, "PRED 'INLINEFORM' garbled."),
    ("q072", 0.25, "PRED 'classify dogmatic posts Reddit' partial."),
    ("q073", 0.0, "PRED 'Results' fragment."),
    ("q074", 0.25, "PRED '3 different datasets' vague."),
    ("q075", 0.0, "PRED 'offensive target identification' fragment."),
    ("q076", 0.0, "PRED 'data noise patterns' fragment."),
    ("q077", 0.0, "PRED 'word reps embeddings' fragment."),
    ("q078", 0.0, "PRED 'Human evaluation' section header."),
    ("q079", 0.0, "PRED 'Consistent Language' section header."),
    ("q080", 0.5, "PRED 'DNN + transfer learning + previous F1 Wikipedia' partial."),
    ("q081", 0.25, "PRED 'review SOTA Vietnamese' general topic match."),
    ("q082", 0.0, "PRED 'I do not know'; gold lists specific SOTA tools."),
    ("q083", 0.0, "PRED 'comparative analysis sub-task A' fragment."),
    ("q084", 0.0, "PRED 'second experiment consumer health' fragment."),
    ("q085", 0.0, "PRED 'Experiments Datasets' section header."),
    ("q086", 0.0, "PRED 'layer-wise training' WRONG."),
    ("q087", 0.25, "PRED 'full text processing pipeline' partial."),
    ("q088", 0.0, "PRED 'TV Tropes' WRONG."),
    ("q089", 0.0, "PRED 'I do not know'; gold='Yes'."),
    ("q090", 0.0, "PRED 'Baselines and Datasets' section header."),
    ("q091", 0.0, "PRED 'Additional Experiments' section header."),
    ("q092", 0.0, "PRED paper title fragment."),
    ("q093", 0.0, "PRED 'I do not know'; gold=clipped PMI + NNEGPMI."),
    ("q094", 0.5, "PRED '145 annotators 1,888 pairs' partial."),
    ("q095", 0.25, "PRED 'CoVoST multilingual ST' WRONG; gold=no specific domain."),
    ("q096", 0.25, "PRED 'two naive baselines Dosage Frequency' partial."),
    ("q097", 0.25, "PRED 'dataset opportunities + explanations' partial."),
    ("q098", 0.0, "PRED 'celebrity Twitter PR agencies' vague."),
    ("q099", 0.25, "PRED 'SBERT seven SentEval' vague."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__attention-tuned__seed7__"
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
    print(f"qasper p4_main attention-tuned seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
