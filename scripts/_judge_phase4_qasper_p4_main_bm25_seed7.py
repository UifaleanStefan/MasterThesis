"""Phase 4 cross-vendor finishing — QASPER p4_main bm25 seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__bm25__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Pattern: bm25 on QASPER returns relevant passages but predictions are
# text fragments / context excerpts rather than direct answers.
JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'English n-gram drops F1' fragment."),
    ("q001", 0.0, "PRED 'Chatbots long history' fragment; gold unanswerable but PRED is wrong context."),
    ("q002", 0.0, "PRED title fragment."),
    ("q003", 0.0, "PRED 'state-of-the-art lacks contextual' WRONG; gold=BIBREF9."),
    ("q004", 0.0, "PRED 'Table TABREF17' fragment."),
    ("q005", 0.0, "PRED 'words less than 5 times' wrong context."),
    ("q006", 0.25, "PRED 'three NLP tasks twitter' implies English partial."),
    ("q007", 0.0, "PRED 'Table TABREF14 BERT' fragment."),
    ("q008", 0.0, "PRED 'Firat2016 attention-based NMT' hallucinated."),
    ("q009", 0.0, "PRED 'BIBREF2 decision tree' WRONG; gold=Mechanical Turk."),
    ("q010", 0.75, "PRED 'common practice to stack multiple RNN layers' matches gold's Stacked LSTMs."),
    ("q011", 0.25, "PRED 'output confidence + LSTM-CNN' partial."),
    ("q012", 0.0, "PRED 'AEP placement task' fragment."),
    ("q013", 0.5, "PRED 'OSG and Twitter social media datasets' mentions OSG."),
    ("q014", 0.0, "PRED '18 Straw man' partial but no 'Loaded language'."),
    ("q015", 0.0, "PRED 'distance feature vs direction' WRONG."),
    ("q016", 0.0, "PRED 'standard RNN Bi-LSTM' hallucinated."),
    ("q017", 0.0, "PRED 'STAGG multiple relation detectors' WRONG."),
    ("q018", 0.25, "PRED 'data weighting greatly improves' partial."),
    ("q019", 0.0, "PRED 'gating mechanism' hallucinated."),
    ("q020", 0.0, "PRED 'raw waveform + grapheme' WRONG architecture."),
    ("q021", 0.0, "PRED 'INLINEFORM transforms' fragment."),
    ("q022", 0.0, "PRED 'Bahdanau attention' hallucinated."),
    ("q023", 0.75, "PRED '5 major profile attributes: username + display + image + location + description' includes gold username."),
    ("q024", 0.0, "PRED 'Stopwords definition' WRONG."),
    ("q025", 0.0, "PRED 'Name Replacement Plaintiff' fragment."),
    ("q026", 0.25, "PRED 'Table 4 compares + 3 datasets' partial."),
    ("q027", 0.0, "PRED title fragment."),
    ("q028", 0.25, "PRED 'attention mechanism Chinese word segmentation' partial."),
    ("q029", 0.0, "PRED 'top ranked words industries' WRONG."),
    ("q030", 0.0, "PRED 'active learning reduces costs' hallucinated."),
    ("q031", 0.25, "PRED 'feature ablation test causality' partial."),
    ("q032", 0.0, "PRED 'compare contextual to DACL' fragment."),
    ("q033", 0.0, "PRED 'model copes with some errors' fragment."),
    ("q034", 0.0, "PRED title fragment."),
    ("q035", 0.25, "PRED 'Rows 3 and 4 crowd annotation' fragment."),
    ("q036", 0.5, "PRED 'modality attention unified representation' partial."),
    ("q037", 0.75, "PRED 'EM and Macro F1' includes gold Exact Match."),
    ("q038", 0.0, "PRED title fragment."),
    ("q039", 0.0, "PRED '226,711 news articles' WRONG."),
    ("q040", 0.75, "PRED 'country-independent two news domains' matches gold concept."),
    ("q041", 1.0, "PRED 'IWSLT German-English' matches gold."),
    ("q042", 0.0, "PRED title fragment."),
    ("q043", 0.0, "PRED 'CoNLL-2010' WRONG."),
    ("q044", 0.0, "PRED 'three-step data collection TweetQA' WRONG."),
    ("q045", 0.0, "PRED 'Annotation tasks section' fragment."),
    ("q046", 0.0, "PRED 'ESIM and DecAtt' fragment."),
    ("q047", 0.5, "PRED 'CORD-19 clinical' partial."),
    ("q048", 0.0, "PRED title fragment."),
    ("q049", 0.25, "PRED 'word importance translation' partial."),
    ("q050", 0.0, "PRED 'DS with DDP' fragment."),
    ("q051", 0.0, "PRED 'NeuronBlocks CoNLL-2003' hallucinated."),
    ("q052", 0.0, "PRED 'lexical properties + PoS' fragment."),
    ("q053", 0.0, "PRED 'modify dataset BIBREF56' fragment."),
    ("q054", 0.0, "PRED 'Abstract Meaning Representations' hallucinated."),
    ("q055", 0.5, "PRED 'Table TABREF11 WER on WSJ' partial WSJ context."),
    ("q056", 0.0, "PRED 'Conclusion and future work' section header."),
    ("q057", 0.0, "PRED '1,949 pathology reports' hallucinated."),
    ("q058", 0.0, "PRED 'T5 WinoGrande Schemas' WRONG."),
    ("q059", 0.25, "PRED 'Table TABREF8 baseline HUMAN' partial."),
    ("q060", 0.0, "PRED 'Data Statistics' section header."),
    ("q061", 0.0, "PRED 'automization summarization' WRONG."),
    ("q062", 0.0, "PRED 'Reducing Gender Bias' title fragment."),
    ("q063", 0.0, "PRED 'GRUs comparable to LSTM' WRONG."),
    ("q064", 0.0, "PRED title fragment."),
    ("q065", 0.0, "PRED 'polarization legislatures votes' fragment."),
    ("q066", 0.0, "PRED 'multilingual models important' fragment."),
    ("q067", 0.5, "PRED 'Human Evaluation Results' section header partial unanswerable."),
    ("q068", 0.5, "PRED 'image captioning multimodal' partial."),
    ("q069", 0.5, "PRED 'fine-grained 5-class SST classification' partial."),
    ("q070", 0.0, "PRED 'rights about document' WRONG."),
    ("q071", 0.0, "PRED 'cosine similarity' fragment."),
    ("q072", 0.0, "PRED 'R1: kinds of topics' fragment."),
    ("q073", 0.0, "PRED '12 MOOC discussion forums' hallucinated."),
    ("q074", 0.0, "PRED 'dimensionality reduction RBM SMM' WRONG."),
    ("q075", 0.0, "PRED 'offensive target identification' fragment."),
    ("q076", 0.0, "PRED 'binary classification dataset combinations' fragment."),
    ("q077", 0.0, "PRED 'subword info Word embeddings' fragment."),
    ("q078", 0.5, "PRED 'Results General' section header partial unanswerable."),
    ("q079", 0.25, "PRED 'language identifier annotated' partial."),
    ("q080", 0.25, "PRED 'four DNN cyberbullying CNN/LSTM/BLSTM' partial."),
    ("q081", 0.0, "PRED title fragment."),
    ("q082", 0.25, "PRED 'DATEXIS-NER precision/recall' partial."),
    ("q083", 0.0, "PRED title fragment."),
    ("q084", 0.0, "PRED 'second experiment consumer health' fragment."),
    ("q085", 0.0, "PRED 'What is sentiment of text?' fragment."),
    ("q086", 0.0, "PRED title fragment."),
    ("q087", 0.0, "PRED 'Dependency parsing Freeling' fragment."),
    ("q088", 0.0, "PRED 'ALOHA + HLAs' WRONG."),
    ("q089", 0.0, "PRED '17,000 features' fragment."),
    ("q090", 0.0, "PRED 'edit distance mention' WRONG."),
    ("q091", 0.0, "PRED 'Additional Experiments' section header."),
    ("q092", 0.0, "PRED title fragment."),
    ("q093", 0.25, "PRED 'low-rank matrix factorization PMI' partial."),
    ("q094", 0.25, "PRED 'Multi-SimLex eng design' partial."),
    ("q095", 0.0, "PRED 'CoVoST transcripts speakers' WRONG."),
    ("q096", 0.25, "PRED 'frame MR as QA' partial."),
    ("q097", 0.5, "PRED 'examples model misclassified verify correct' partial."),
    ("q098", 0.25, "PRED '15 celebrities + 0.05 significance' partial count match."),
    ("q099", 0.0, "PRED 'SBERT not for transfer' WRONG."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__bm25__seed7__"
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
    print(f"qasper p4_main bm25 seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
