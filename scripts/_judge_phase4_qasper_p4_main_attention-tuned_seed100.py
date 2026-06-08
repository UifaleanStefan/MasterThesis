"""Phase 4 cross-vendor finishing — QASPER p4_main attention-tuned seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__attention-tuned__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Pattern: attention-tuned predictions are largely section headers or text
# fragments (not answers), or "I do not know" refusals — reflects how this
# memory architecture retrieves on QASPER's paper-structure data.
JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'active learning batch size' wrong context."),
    ("q001", 0.0, "PRED 'DeepMine database' hallucinated; gold unanswerable."),
    ("q002", 0.0, "PRED 'four languages English pivot' WRONG; gold=BIBREF19."),
    ("q003", 0.0, "PRED 'five policies' vague; gold=2.6 pp."),
    ("q004", 1.0, "PRED is verbatim copy of gold's first sentence."),
    ("q005", 0.0, "PRED 'I do not know'; gold=4 MT tasks."),
    ("q006", 0.0, "PRED 'seven experts legal training' wrong context."),
    ("q007", 0.0, "PRED 'social media Twitter' WRONG; gold=Mechanical Turk."),
    ("q008", 0.0, "PRED 'create labels for tweets' hallucinated."),
    ("q009", 0.0, "PRED 'INLINEFORM symbols' garbled."),
    ("q010", 0.0, "PRED 'I do not know'; gold='No'."),
    ("q011", 0.0, "PRED 'Mapudungu' WRONG; gold=Transformer."),
    ("q012", 0.0, "PRED 'Table TABREF20' fragment."),
    ("q013", 0.0, "PRED 'Testing Generalizability' section header."),
    ("q014", 0.0, "PRED 'Dataset' section header."),
    ("q015", 0.0, "PRED 'I do not know'."),
    ("q016", 0.0, "PRED 'Evaluating Keyphrase Generation' section header."),
    ("q017", 0.0, "PRED 'modify dataset BIBREF56' fragment."),
    ("q018", 0.5, "PRED 'PHI Healthcare' implies No commercial use."),
    ("q019", 1.0, "PRED 'German-English language pair' matches gold='De-En'."),
    ("q020", 0.0, "PRED 'ALCrowd compared' fragment."),
    ("q021", 0.0, "PRED 'RNN improvements 22.76%' WRONG; gold=Russian."),
    ("q022", 0.25, "PRED 'seven SentEval transfer tasks' vague."),
    ("q023", 0.0, "PRED 'Training ELMo' section header."),
    ("q024", 0.0, "PRED 'VLSP 2018+2019 testing sets' fragment."),
    ("q025", 0.0, "PRED 'CEI task DA AVG LSTM' WRONG."),
    ("q026", 0.75, "PRED 'sections 02-21 of WSJ' matches gold WSJ."),
    ("q027", 0.0, "PRED 'NLP within IPA' fragment."),
    ("q028", 0.0, "PRED 'Models' section header."),
    ("q029", 0.0, "PRED 'production of articulatory speech' fragment."),
    ("q030", 0.0, "PRED 'Results' section header fragment."),
    ("q031", 0.0, "PRED 'expectation inference' fragment."),
    ("q032", 0.0, "PRED 'Reading Comprehension' fragment."),
    ("q033", 0.0, "PRED 'future plan' fragment."),
    ("q034", 0.0, "PRED 'Introduction Detecting Propaganda' section header."),
    ("q035", 0.75, "PRED 'RoBERTa with language modeling' matches gold."),
    ("q036", 0.0, "PRED 'Models' fragment."),
    ("q037", 0.0, "PRED 'tag recommendation accuracy section' fragment."),
    ("q038", 0.0, "PRED 'Task and Evaluation' section header."),
    ("q039", 0.0, "PRED 'Attention Vectors' section header."),
    ("q040", 0.0, "PRED 'WebQSP' WRONG; gold=SimpleQuestions."),
    ("q041", 0.0, "PRED 'Correlation results' fragment."),
    ("q042", 0.0, "PRED 'Results and Analysis' section header."),
    ("q043", 0.5, "PRED 'RNN-based NMT and Transformer' includes gold."),
    ("q044", 0.0, "PRED 'I do not know'; gold=F-score."),
    ("q045", 0.0, "PRED 'Table TABREF8' fragment."),
    ("q046", 0.25, "PRED 'data weighting greatly improves' partial."),
    ("q047", 0.0, "PRED 'NeuronBlocks CoNLL-2003' hallucinated."),
    ("q048", 0.0, "PRED 'CoNLL-2010' WRONG; gold=MaxEnt + SVMs."),
    ("q049", 0.0, "PRED 'Experiment' section header."),
    ("q050", 0.0, "PRED 'layer-wise training' WRONG."),
    ("q051", 0.25, "PRED '3 different datasets' vague."),
    ("q052", 0.0, "PRED 'Rouge content quality' fragment."),
    ("q053", 0.0, "PRED 'I do not know'."),
    ("q054", 0.0, "PRED 'LangID NoLangID' WRONG."),
    ("q055", 0.0, "PRED 'Memory Generation Layer' fragment."),
    ("q056", 0.75, "PRED 'student vocab not complete subset' matches gold."),
    ("q057", 0.25, "PRED 'probing not straightforward + artifacts' partial."),
    ("q058", 0.0, "PRED 'predicted label true label' WRONG."),
    ("q059", 0.25, "PRED 'instance informative if changes params' partial."),
    ("q060", 0.0, "PRED 'Insight-driven analysis' fragment."),
    ("q061", 0.5, "PRED 'I do not know' partial match to gold unanswerable."),
    ("q062", 0.25, "PRED 'full text processing pipeline' partial."),
    ("q063", 0.0, "PRED 'Comparison with SOTA' section header."),
    ("q064", 0.25, "PRED 'review SOTA Vietnamese' general topic match."),
    ("q065", 0.25, "PRED 'encoder-decoder neural network' vague."),
    ("q066", 0.5, "PRED 'Context-Specificity higher layers' partial."),
    ("q067", 0.75, "PRED 'country-independent two news domains' matches gold concept."),
    ("q068", 0.5, "PRED 'GreenBioBERT eight NER' partial reference."),
    ("q069", 0.0, "PRED 'PhantomJS screenshot' WRONG; gold unanswerable."),
    ("q070", 0.5, "PRED 'necessity word importance' partial."),
    ("q071", 0.0, "PRED 'machine translated premises' WRONG."),
    ("q072", 0.0, "PRED 'extracted representation' question fragment."),
    ("q073", 0.0, "PRED 'Hashtag Segmentation Data' section header."),
    ("q074", 0.0, "PRED 'variability in generalization' WRONG."),
    ("q075", 1.0, "PRED '353 conversations 40 speakers + 11 nurses + 16 patients + 13 caregivers' matches gold exactly."),
    ("q076", 0.0, "PRED 'learning rate hyperparams' WRONG."),
    ("q077", 0.0, "PRED 'low dim word reps' fragment."),
    ("q078", 0.0, "PRED 'evaluating in style transfer' fragment."),
    ("q079", 0.0, "PRED 'I do not know'; gold=EI-Reg etc."),
    ("q080", 0.0, "PRED 'CFILT-preorder' hallucinated."),
    ("q081", 0.0, "PRED 'I do not know'; gold=Penn Treebank."),
    ("q082", 0.0, "PRED 'I do not know'."),
    ("q083", 0.5, "PRED 'Plank bi-LSTM MarMoT Czech French Italian' partial (3 of 16)."),
    ("q084", 0.0, "PRED 'twitter public-opinion' fragment."),
    ("q085", 0.0, "PRED 'UN roll call' WRONG."),
    ("q086", 0.0, "PRED 'four scores indicative of polarity' WRONG."),
    ("q087", 0.0, "PRED 'Machine Reading Comprehension' fragment."),
    ("q088", 0.0, "PRED 'sefe on ArXiv Senate' fragment."),
    ("q089", 0.5, "PRED 'dev data BIBREF19 + train+dev merged' partial WASSA reference."),
    ("q090", 0.0, "PRED 'Human Evaluation' section header."),
    ("q091", 0.0, "PRED 'Framework Description' section header."),
    ("q092", 0.0, "PRED 'AUC 30 terms' WRONG."),
    ("q093", 0.0, "PRED 'second experiment' fragment."),
    ("q094", 0.0, "PRED 'data noise patterns' fragment."),
    ("q095", 0.0, "PRED 'Existing Approaches' section header."),
    ("q096", 0.0, "PRED 'Impact of Attention Mechanism' section header."),
    ("q097", 0.0, "PRED 'Results' section header."),
    ("q098", 0.0, "PRED 'Baseline System' section header."),
    ("q099", 0.0, "PRED 'I do not know'."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__attention-tuned__seed100__"
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
    print(f"qasper p4_main attention-tuned seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
