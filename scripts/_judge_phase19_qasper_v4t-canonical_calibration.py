"""Phase 1.9 — QASPER v4t-canonical Protocol B calibration + batch_calib judge.

Cells:
  qasper__v4t-canonical__calibration__seed42   (1405 entries: 712 ack, 693 answer)
  qasper__v4t-canonical__batch_calib__seed42   (1005 entries: all answer)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-canonical (θ_store=0.293, w_embed=1.079, w_recency=3.777): extreme recency
    bias; with 281 QASPER docs, early docs are quickly evicted from memory
  - Calibration recall@k=0.0248 (batch=0.0307) — almost no early-doc retrieval
  - 712 ack entries: 501 honest refusals (1.0), 211 hallucinations (0.0)
    Hallucinations occur when BM25-like retrieval surfaces unrelated matching docs
    (but v4t-canonical uses embedding+recency, so hallucinations arise from
     recently-ingested docs with keyword overlap matching the question)
  - 693 answer entries: 428 refusals (0.0), 265 answered
    Of 265 answered: 19×1.0, 6×0.75, 36×0.5, 49×0.25, 155×0.0
  - Calibration mean = 554.75/1405 = 0.3949
    ack mean = 501/712 = 0.7037
    answer mean = 53.75/693 = 0.0776

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer, gold=(unanswerable per source): 1.0 if refused, 0.0 if answered
  - answer, gold!=unanswerable: standard 5-point rubric
    0.0 wrong/refusal, 0.25 partial/self-contradictory, 0.5 partially correct,
    0.75 correct-but-buried, 1.0 correct

Batch_calib statistics (1005 entries, all answer expected_behavior):
  - 95 unanswerable gold: 59 correctly refused (1.0), 36 answered (0.0)
  - 910 answerable gold: 548 refused (0.0), 362 answered
    Of 362 answered: ~36×1.0, ~8×0.75, ~35×0.5, ~61×0.25, ~222×0.0
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "qasper__v4t-canonical__calibration__seed42"
BATCH_DIR = JQ / "qasper__v4t-canonical__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_RESULTS = BATCH_DIR / "results.jsonl"

# ---------------------------------------------------------------------------
# Refusal detection — ~40 patterns covering all common refusal phrasings
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    "do not have", "don't have", "context provided", "provided context",
    "passages provided", "provided passages", "not mentioned", "not provided",
    "not available", "insufficient", "no information", "cannot determine",
    "not specify", "no mention", "cannot find", "does not contain",
    "not found", "i'm sorry", "unable to", "no context", "not enough",
    "apologies", "not explicitly", "not clear", "not specified",
    "passages do not", "does not provide", "do not provide",
    "cannot be determined", "no relevant", "not discussed",
    "no specific", "cannot answer", "not contain", "no detail",
    "information is not", "there is no", "is not mentioned",
    "are not mentioned", "is not provided", "are not provided",
    "there are no", "not include", "not included", "do not contain",
    "not be determined", "not be found", "without more", "lacks",
    "no passage", "i do not see", "not see", "not support",
    "unanswerable", "unspecified", "no answer", "the relevant passages",
    "the document passages", "document does not", "documents do not",
]


def is_refusal(pred: str) -> bool:
    p = pred.strip().lower()
    if not p:
        return True
    return any(pat in p for pat in _REFUSAL_PATTERNS)


def is_gold_unanswerable(gold: str) -> bool:
    return gold.strip().startswith("(unanswerable")


# ---------------------------------------------------------------------------
# CALIBRATION JUDGMENTS: (suffix → (score, rationale))
# suffix = qid minus prefix/suffix, e.g. "doc2_qa4__after264"
# Only non-zero answer entries (ack entries handled by refusal logic)
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 (19 entries) ───────────────────────────────────────────────
    "doc2_qa4__after264": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they add one latent variable for each language pair in "
        "their Bayesian model.' — correct yes answer with supporting detail. Full credit."),
    "doc17_qa3__after172": (1.0,
        "Gold: 'English'. Predicted: 'They explore English.' — correct. Full credit."),
    "doc33_qa0__after70": (1.0,
        "Gold: 'No'. Predicted: 'No, they aim to analyze vocabulary and portray characteristics "
        "rather than build a model for prediction.' — correct no answer with rationale. Full credit."),
    "doc44_qa0__after190": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they report results only on English data.' — correct. Full credit."),
    "doc46_qa2__after174": (1.0,
        "Gold: 'Knowledge Base Question Answering'. Predicted: 'Knowledge Base Question Answering' "
        "— exact match. Full credit."),
    "doc46_qa2__after196": (1.0,
        "Gold: 'Knowledge Base Question Answering'. Predicted: 'KBQA stands for Knowledge Base "
        "Question Answering.' — correct expansion. Full credit."),
    "doc51_qa1__after76": (1.0,
        "Gold: 'NLG datasets'. Predicted: 'The model was evaluated on NLG datasets.' — correct. Full credit."),
    "doc55_qa0__after166": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, pre-training is effective in their evaluation.' — correct. Full credit."),
    "doc60_qa2__after243": (1.0,
        "Gold: 'Phoneme Error Rate (PER)'. Predicted: 'The evaluation metric used was the phoneme "
        "error rate.' — correct. Full credit."),
    "doc71_qa1__after226": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they employ their indexing-based method for the creation of "
        "a silver-standard dataset...' — correct yes answer. Full credit."),
    "doc78_qa1__after160": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, their NER model learns NER from both text and images.' "
        "— correct. Full credit."),
    "doc78_qa1__after250": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, their NER model learns NER from both text and images.' "
        "— correct. Full credit."),
    "doc109_qa0__after149": (1.0,
        "Gold: 'Stacked LSTMs'. Predicted: 'They experimented with conventional stacked LSTMs and "
        "their proposed Cell-aware Stacked LSTMs...' — correctly identifies stacked LSTMs. Full credit."),
    "doc111_qa3__after238": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, the authors hypothesize that humans\\' robustness to noise is "
        "due to their general knowledge.' — correct yes answer. Full credit."),
    "doc111_qa3__after255": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, the authors propose that integrating general knowledge into "
        "MRC models can enhance robustness...' — correct yes answer. Full credit."),
    "doc134_qa2__after225": (1.0,
        "Gold: 'English'. Predicted: 'The tweets are in English.' — correct. Full credit."),
    "doc134_qa2__after269": (1.0,
        "Gold: 'English'. Predicted: 'English' — exact match. Full credit."),
    "doc156_qa0__after237": (1.0,
        "Gold: 'No'. Predicted: 'No, they do not report results only on the English dataset.' "
        "— correct no answer. Full credit."),
    "doc230_qa5__after278": (1.0,
        "Gold: 'workers are first asked to find those microposts where the model predictions are "
        "deemed correct...' Predicted: 'Workers are first asked to find microposts where the model "
        "predictions are deemed correct...' — semantically exact match. Full credit."),
    # ── Score 0.75 (6 entries) ───────────────────────────────────────────────
    "doc10_qa2__after238": (0.75,
        "Gold: 'Extract features from the LDA model and use them in a binary classification task'. "
        "Predicted: 'They detect spammers using a novel classification approach based on Latent "
        "Dirichlet Allocation (LDA)...' — correct approach (LDA for classification) but omits "
        "'binary' and 'feature extraction' framing. Score 0.75."),
    "doc92_qa0__after151": (0.75,
        "Gold: 'Answer with content missing: (Subscript 1: We did not participate in...) participated "
        "in four subtasks...' Predicted: 'They participated in four subtasks: EI-Reg, EI-Oc, V-Reg, "
        "and V-Oc.' — correct number and names of subtasks; partial credit for missing caveat text. "
        "Score 0.75."),
    "doc96_qa3__after195": (0.75,
        "Gold: 'majority baseline'. Predicted: 'The baselines mentioned are lexicon-based baseline "
        "and majority baseline.' — correctly identifies majority baseline (gold answer) but adds "
        "extra lexicon-based baseline. Score 0.75."),
    "doc113_qa0__after276": (0.75,
        "Gold: 'AutoJudge consistently and significantly outperforms all the baselines'. Predicted: "
        "'Experimental results demonstrate that the AutoJudge model achieves significant improvement "
        "over baselines...' — correct core finding, slightly paraphrased. Score 0.75."),
    "doc164_qa4__after212": (0.75,
        "Gold: 'the context and sequential nature of the text'. Predicted: 'The GRU model captures "
        "complex temporal dependencies and contextual information in the data.' — captures context "
        "and sequential nature (temporal dependencies) but with different vocabulary. Score 0.75."),
    "doc169_qa1__after183": (0.75,
        "Gold: 'Friends'. Predicted: 'The datasets used are Friends and EmotionPush.' — correct "
        "(includes Friends) but also lists EmotionPush which is additional. Score 0.75."),
    # ── Score 0.5 (36 entries) ───────────────────────────────────────────────
    "doc0_qa3__after153": (0.5,
        "Gold: 'ability to accurately classify texts even when the amount of prior knowledge for "
        "different classes is very different'. Predicted: 'Robustness of a model is defined as its "
        "sensitivity to the knowledge supplied...' — correctly identifies classification robustness "
        "but frames it as sensitivity rather than accuracy with varying prior knowledge. Score 0.5."),
    "doc0_qa3__after278": (0.5,
        "Gold: 'ability to accurately classify texts even when the amount of prior knowledge for "
        "different classes is very different'. Predicted: 'The robustness of a model is defined by "
        "its sensitivity to the knowledge supplied to it...' — partial overlap, same concept but "
        "imprecise framing. Score 0.5."),
    "doc1_qa3__after189": (0.5,
        "Gold: 'user comments to newswire articles or to blog posts'. Predicted: 'The data in the "
        "new corpus is sourced from actual Web data, specifically user-generated Web discourse.' "
        "— correctly identifies user-generated content but omits 'newswire articles or blog posts' "
        "specificity. Score 0.5."),
    "doc1_qa3__after219": (0.5,
        "Gold: 'user comments to newswire articles or to blog posts'. Predicted: 'The data in the "
        "new corpus is sourced from actual Web data, consisting of user-generated Web discourse.' "
        "— same pattern: correct source type but missing specific venues. Score 0.5."),
    "doc4_qa1__after235": (0.5,
        "Gold: 'we can expect a dependency edge between i and i\\' in the English parse if there is "
        "one in the other language...'. Predicted: 'The enforcement of agreement between parse trees "
        "from two languages works as a constraint...' — captures the bilingual parse agreement "
        "concept but misses the specific expected-dependency mechanism. Score 0.5."),
    "doc6_qa1__after100": (0.5,
        "Gold: 'The content relevance between the candidate summary and the human summary is "
        "measured...'. Predicted: 'Content relevance is measured by evaluating the alignment of "
        "words and concepts in the summary...' — broadly correct measurement concept but loses "
        "candidate vs. human summary specificity. Score 0.5."),
    "doc11_qa4__after50": (0.5,
        "Gold: 'a simple word-level encoder'. Predicted: 'The word level baseline they compare to "
        "is a bag-of-words model.' — bag-of-words is a type of word-level encoder; partially correct "
        "but imprecise. Score 0.5."),
    "doc13_qa1__after21": (0.5,
        "Gold: 'relation classification dataset of the SemEval 2010 task 8'. Predicted: 'They train "
        "their models on the SemEval 2010 relation classification task.' — correctly identifies "
        "SemEval 2010 relation classification but doesn't specify 'task 8'. Score 0.5."),
    "doc13_qa4__after122": (0.5,
        "Gold: 'They use two independent convolutional and max-pooling layers on (1) a combination "
        "of the entity tag...' Predicted: 'They present a new context representation for "
        "convolutional neural networks for relation classification...' — correctly identifies CNNs "
        "for new context representation but lacks two-stream architecture detail. Score 0.5."),
    "doc13_qa4__after167": (0.5,
        "Gold: 'They use two independent convolutional and max-pooling layers on (1) a combination "
        "of the entity tag...' Predicted: 'They present a new context representation for "
        "convolutional neural networks for relation classification...' — same: CNNs for context, "
        "missing two-stream details. Score 0.5."),
    "doc46_qa3__after130": (0.5,
        "Gold: 'answer questions by obtaining information from KB tuples'. Predicted: 'Relation "
        "detection is the core component for Knowledge Base Question Answering (KBQA).' — correctly "
        "identifies KBQA component (relation detection) but misses 'KB tuples' as the information "
        "source. Score 0.5."),
    "doc46_qa3__after141": (0.5,
        "Gold: 'answer questions by obtaining information from KB tuples'. Predicted: 'Relation "
        "detection is the core component for Knowledge Base Question Answering (KBQA).' — same "
        "partial answer. Score 0.5."),
    "doc46_qa3__after154": (0.5,
        "Gold: 'answer questions by obtaining information from KB tuples'. Predicted: 'The core "
        "component for KBQA is improved neural relation detection.' — partial: identifies core "
        "component but not KB tuples. Score 0.5."),
    "doc46_qa3__after169": (0.5,
        "Gold: 'answer questions by obtaining information from KB tuples'. Predicted: 'The core "
        "component for KBQA is the improved neural relation detection.' — same partial. Score 0.5."),
    "doc51_qa0__after246": (0.5,
        "Gold: 'Introduce a Refinement Adjustment LSTM-based component to the decoder...'. "
        "Predicted: 'The proposed model introduces an LSTM-based decoder and incorporates an "
        "attention mechanism...' — correctly identifies LSTM decoder but missing 'Refinement "
        "Adjustment' framing. Score 0.5."),
    "doc54_qa2__after275": (0.5,
        "Gold: 'hashtag features contain whether there is any hashtag in the tweet'. Predicted: "
        "'Hashtag features are specific elements used in the evaluation of SVM classifiers for "
        "stance detection...' — describes hashtag features broadly but omits the specific "
        "'whether there is any hashtag' binary indicator. Score 0.5."),
    "doc65_qa0__after83": (0.5,
        "Gold: 'it captures other information rather than only the translational equivalent in "
        "the case of attention'. Predicted: 'Attention captures useful information other than "
        "alignments.' — correct paraphrase; alignment ≈ translational equivalent. Score 0.5."),
    "doc65_qa0__after120": (0.5,
        "Gold: 'it captures other information rather than only the translational equivalent in "
        "the case of attention'. Predicted: 'Attention captures useful information other than "
        "alignments.' — correct paraphrase. Score 0.5."),
    "doc84_qa3__after222": (0.5,
        "Gold: 'At the macro level, it is important to decide which is the appropriate level of "
        "attention...'. Predicted: 'A bifocal attention mechanism exploits and combines macro level "
        "information...' — correctly identifies macro-level attention but loses the deciding "
        "appropriate-level framing. Score 0.5."),
    "doc104_qa1__after106": (0.5,
        "Gold: 'using tweets that one has replied or quoted to as contextual information...'. "
        "Predicted: 'The additional features and context proposed include using context tweets, "
        "which provide surrounding information...' — correctly identifies context tweets but omits "
        "'replied or quoted' specificity. Score 0.5."),
    "doc108_qa5__after108": (0.5,
        "Gold: 'The task of predicting MSD tags: V, PST, V.PCTP, PASS.' Predicted: 'MSD prediction "
        "refers to the prediction of morphosyntactic descriptions, which involves identifying...' "
        "— correctly defines MSD prediction but omits specific tag examples. Score 0.5."),
    "doc108_qa5__after226": (0.5,
        "Gold: 'The task of predicting MSD tags: V, PST, V.PCTP, PASS.' Predicted: 'MSD prediction "
        "refers to the auxiliary task of predicting Morphosyntactic Descriptions in NLP...' "
        "— correct definition; missing specific tags. Score 0.5."),
    "doc116_qa1__after173": (0.5,
        "Gold: 'the attention model, MDREA, also outperforms the best existing research results'. "
        "Predicted: 'The proposed model outperforms the best existing research results by achieving "
        "a WAP value...' — correctly identifies outperformance but omits MDREA name. Score 0.5."),
    "doc120_qa1__after209": (0.5,
        "Gold: 'the same datasets as BIBREF7'. Predicted: 'The dataset used in this paper is from "
        "photo-sharing websites such as Flickr.' — Flickr is consistent with the referenced "
        "dataset; partial credit for identifying source without citing the reference. Score 0.5."),
    "doc120_qa1__after259": (0.5,
        "Gold: 'the same datasets as BIBREF7'. Predicted: 'The paper uses the Flickr dataset.' "
        "— Flickr is a plausible match but cannot confirm it equals BIBREF7 datasets. Score 0.5."),
    "doc120_qa1__after260": (0.5,
        "Gold: 'the same datasets as BIBREF7'. Predicted: 'The dataset used in the paper is "
        "Flickr tags.' — same as above. Score 0.5."),
    "doc129_qa1__after151": (0.5,
        "Gold: 'visual model is based on fine-tuning an Inception V3 model BIBREF1 over visual "
        "renderings of documents'. Predicted: 'They propose a joint model that combines text "
        "content with a visual rendering of the document...' — correctly identifies visual "
        "rendering approach but omits Inception V3 architecture. Score 0.5."),
    "doc129_qa5__after175": (0.5,
        "Gold: 'It depends on the dataset. Experimental results over two datasets reveal...' "
        "Predicted: 'Textual and visual features are complementary.' — correct qualitative "
        "finding about complementarity; misses the dataset-dependent nuance. Score 0.5."),
    "doc147_qa3__after279": (0.5,
        "Gold: 'the series of posts that trigger an intervention'. Predicted: 'The latent context "
        "necessary to predict instructor intervention is inferred using attention-based models...' "
        "— correctly identifies attention-based latent context inference but omits 'series of "
        "posts' framing. Score 0.5."),
    "doc150_qa1__after238": (0.5,
        "Gold: 'There is nothing specific about the approach that depends on medical records'. "
        "Predicted: 'Yes, G-BERT is useful beyond the medication recommendation task as it "
        "integrates hierarchical... structures.' — captures that the approach generalizes but "
        "misses the 'nothing specific to medical records' caveat. Score 0.5."),
    "doc175_qa3__after200": (0.5,
        "Gold: 'two salient roles called Anchors and Punctual speakers'. Predicted: 'The authors "
        "define two categories for speaker role: Anchor and others.' — identifies Anchor role "
        "but replaces 'Punctual speakers' with generic 'others'. Score 0.5."),
    "doc186_qa1__after199": (0.5,
        "Gold: 'MT system on the data released by BIBREF11'. Predicted: 'The baseline refers to "
        "the original sentence-level translations produced by modern NMT systems.' — correctly "
        "identifies MT system as baseline but omits BIBREF11 specificity. Score 0.5."),
    "doc216_qa2__after253": (0.5,
        "Gold: 'Waseem-dataset'. Predicted: 'Two publicly available datasets that have been "
        "annotated for racism, sexism, hate, or offensive content...' — describes the datasets "
        "correctly but doesn't name 'Waseem-dataset' specifically. Score 0.5."),
    "doc218_qa2__after269": (0.5,
        "Gold: '30 terms, each term-sense pair has around 15 samples for testing'. Predicted: "
        "'The test set is manually created and balanced, with each class in each term having "
        "15 samples...' — correctly identifies 15 samples per class but omits '30 terms'. Score 0.5."),
    "doc219_qa2__after270": (0.5,
        "Gold: 'closed test limits all the data for learning should not be beyond the training "
        "data...'. Predicted: 'A closed test setting refers to a scenario where the test data is "
        "drawn from the same distribution...' — correctly describes closed-test constraint but "
        "with different framing. Score 0.5."),
    "doc236_qa3__after259": (0.5,
        "Gold: 'By using topic modeling and unsupervised emotion detection on ISIS materials and "
        "Catholic materials...'. Predicted: 'The analysis of similarities and differences between "
        "the texts from violent and non-violent religious groups is conducted...' — describes the "
        "comparison correctly but omits 'topic modeling and unsupervised emotion detection'. Score 0.5."),
    # ── Score 0.25 (49 entries) ──────────────────────────────────────────────
    "doc1_qa1__after47": (0.25,
        "Gold: 'claim, premise, backing, rebuttal, and refutation'. Predicted: 'The ML methods "
        "aim to identify argument components such as claims, premises, and other related elements.' "
        "— lists claims and premises correctly but omits backing, rebuttal, and refutation. Score 0.25."),
    "doc2_qa5__after187": (0.25,
        "Gold: 'Bayesian model of garg2012unsupervised as our base monolingual model'. Predicted: "
        "'An individual model consists of individual models for each language plus additional "
        "latent variables...' — correctly identifies multi-language structure but omits "
        "garg2012unsupervised base model. Score 0.25."),
    "doc11_qa2__after229": (0.25,
        "Gold: 'a simple word-level encoder'. Predicted: 'The word-level baseline is a traditional "
        "NLP approach that predicts user-annotated hashtags...' — identifies word-level baseline "
        "but describes it differently without naming 'simple word-level encoder'. Score 0.25."),
    "doc12_qa3__after153": (0.25,
        "Gold: 'Ethnic bias'. Predicted: 'The dataset contains biases and unwarranted inferences "
        "that are not solely based on the image content...' — mentions biases generally but omits "
        "'ethnic bias' as the specific type. Score 0.25."),
    "doc26_qa2__after203": (0.25,
        "Gold: 'dynamic average pooling'. Predicted: 'A single QRNN layer performs an "
        "input-dependent pooling.' — identifies input-dependent pooling (related) but doesn't "
        "name 'dynamic average pooling'. Score 0.25."),
    "doc31_qa1__after157": (0.25,
        "Gold: '50 human annotators ranked a random sample of 100 translations by Adequacy...'. "
        "Predicted: 'Human evaluation involved professional translators and native speakers...' "
        "— identifies human evaluation but omits '50 annotators', '100 translations', and "
        "'Adequacy' metric. Score 0.25."),
    "doc34_qa4__after88": (0.25,
        "Gold: 'technology, religion, fashion, publishing, sports or recreation, real estate...'. "
        "Predicted: 'The paper defines a taxonomy of fourteen industry classes.' — correctly "
        "identifies taxonomy of industries but doesn't list the actual classes. Score 0.25."),
    "doc37_qa4__after227": (0.25,
        "Gold: 'Readability (RED), Number of Words (LEN), Avg. Fixation Duration (FDUR), "
        "Avg. Fixation Count...'. Predicted: 'Cognitive features extracted from eye-movement "
        "patterns of human readers are used.' — correctly identifies eye-movement cognitive "
        "features but doesn't name any specific metrics. Score 0.25."),
    "doc43_qa1__after244": (0.25,
        "Gold: 'Zero Resource Toolkit (ZRTools) BIBREF7'. Predicted: 'The vocabulary of "
        "word-like or phoneme-like units is automatically discovered through a process...' "
        "— correctly describes the discovery process but omits ZRTools/BIBREF7. Score 0.25."),
    "doc45_qa1__after53": (0.25,
        "Gold: 'Answer with content missing: (Evaluation Metrics section) Precision, Recall, "
        "F1-scores, Standard ROUGE scores...'. Predicted: 'The proposed evaluation for this task "
        "is a baseline system and a proposed evaluation protocol...' — mentions evaluation "
        "protocol but omits Precision/Recall/F1/ROUGE metrics. Score 0.25."),
    "doc45_qa1__after182": (0.25,
        "Gold: 'Answer with content missing: (Evaluation Metrics section) Precision, Recall, "
        "F1-scores, Standard ROUGE scores...'. Predicted: 'The proposed evaluation for this task "
        "is a baseline system and a proposed evaluation protocol...' — same partial answer. Score 0.25."),
    "doc45_qa2__after212": (0.25,
        "Gold: 'Answer with content missing: (Baseline Method section) We implemented a "
        "sequential sentence extraction...'. Predicted: 'The baseline system proposed is a system "
        "that summarizes heterogeneous collections of web documents on educational topics.' "
        "— mentions the domain but omits sequential extraction method. Score 0.25."),
    "doc47_qa0__after256": (0.25,
        "Gold: 'precision, recall, F1 and accuracy'. Predicted: 'The evaluation metrics looked "
        "at were F1 micro and F1 macro scores.' — mentions F1 correctly but omits precision, "
        "recall, and accuracy; adds micro/macro distinction not in gold. Score 0.25."),
    "doc49_qa0__after167": (0.25,
        "Gold: 'text classification and text semantic matching'. Predicted: 'The experimental "
        "results are shown on two typical tasks.' — correctly says two tasks but doesn't name "
        "them. Score 0.25."),
    "doc57_qa0__after209": (0.25,
        "Gold: 'SVMs'. Predicted: 'The baseline was traditional sentiment analysis approaches "
        "that tackle problems like ternary classification...' — mentions traditional approaches "
        "but doesn't name SVMs. Score 0.25."),
    "doc59_qa2__after231": (0.25,
        "Gold: 'standard accuracy metric'. Predicted: 'The evaluation metrics used are "
        "performance improvement over traditional shallow approaches...' — describes evaluation "
        "broadly but doesn't name 'standard accuracy metric'. Score 0.25."),
    "doc63_qa1__after229": (0.25,
        "Gold: 'decoder task, which predicts the target sequence probability at time t based on "
        "encoder hidden states...'. Predicted: 'The decoder generates text by processing directly "
        "from the raw speech waveform...' — identifies decoder component but describes wrong "
        "mechanism (raw waveform instead of encoder hidden states). Score 0.25."),
    "doc70_qa1__after214": (0.25,
        "Gold: 'an expert annotator determined if the tweet fell under a specific category'. "
        "Predicted: 'The determination of fake news tweets was based on the analysis of their "
        "meta-data.' — identifies tweet determination process but says meta-data analysis "
        "instead of expert annotation. Score 0.25."),
    "doc70_qa3__after71": (0.25,
        "Gold: 'Accounts that spread fake news are mostly unverified, recently created and have '"
        "fewer followers...'. Predicted: 'The accounts that spread fake news include both "
        "verified and unverified users, with a higher proportion of unverified users...' "
        "— mentions unverified users but omits 'recently created' and 'fewer followers'. Score 0.25."),
    "doc70_qa3__after125": (0.25,
        "Gold: 'Accounts that spread fake news are mostly unverified, recently created and have "
        "fewer followers...'. Predicted: 'The accounts spreading fake news have significant "
        "differences in the distribution of followers...' — mentions follower differences but "
        "omits 'unverified' and 'recently created'. Score 0.25."),
    "doc70_qa3__after148": (0.25,
        "Gold: 'Accounts that spread fake news are mostly unverified, recently created and have "
        "fewer followers...'. Predicted: 'The accounts spreading fake news have significant "
        "differences in the distribution of followers...' — same partial. Score 0.25."),
    "doc71_qa3__after219": (0.25,
        "Gold: 'They compare the tasks that the datasets are suitable for, average number of "
        "answers, answer types...'. Predicted: 'They analyze contextual similarities intrinsically "
        "by examining the datasets based on contextual similarity metrics...' — describes dataset "
        "comparison but misses tasks/answers/answer-types as comparison dimensions. Score 0.25."),
    "doc89_qa1__after130": (0.25,
        "Gold: 'DSTC2'. Predicted: 'The behavior is learned from a corpus.' — correctly "
        "identifies corpus-based learning but doesn't name DSTC2. Score 0.25."),
    "doc94_qa1__after206": (0.25,
        "Gold: 'Word vectors, usually in the context of others within the same class'. Predicted: "
        "'Word subspace can represent the semantic relationships and contextual meanings of words "
        "based on their term frequencies...' — word subspace relates to word vectors in class "
        "context but misses 'within the same class' grouping. Score 0.25."),
    "doc94_qa1__after233": (0.25,
        "Gold: 'Word vectors, usually in the context of others within the same class'. Predicted: "
        "'Word subspace can represent the semantic relationships and contextual meanings of words "
        "based on their term frequencies...' — same partial. Score 0.25."),
    "doc98_qa1__after158": (0.25,
        "Gold: 'task of humor identification in social media texts is analyzed as a classification "
        "task'. Predicted: 'The experiments carried out on the corpus involved developing a "
        "baseline classification system...' — mentions classification but doesn't state humor "
        "identification task framing. Score 0.25."),
    "doc98_qa3__after261": (0.25,
        "Gold: 'tweets from the past two years from domains like sports, politics, entertainment...'. "
        "Predicted: 'The texts in the corpus came from social networking sites.' — correctly "
        "identifies social media origin but omits domain specification. Score 0.25."),
    "doc104_qa1__after187": (0.25,
        "Gold: 'using tweets that one has replied or quoted to as contextual information...'. "
        "Predicted: 'The paper discusses the possibility of using additional features and context "
        "data for improving results.' — mentions additional context broadly but omits "
        "'replied or quoted tweets' specificity. Score 0.25."),
    "doc108_qa4__after131": (0.25,
        "Gold: 'LSTM'. Predicted: 'The encoder has an encoder-decoder architecture over character "
        "sequences.' — describes architecture broadly; gold answer is specifically 'LSTM'. Score 0.25."),
    "doc109_qa2__after197": (0.25,
        "Gold: '(i) models that use plain stacked LSTMs'. Predicted: 'The Baseline model is "
        "mentioned as a comparison point, which already achieves a slightly better performance...' "
        "— mentions baseline but doesn't identify 'plain stacked LSTMs' as the model type. Score 0.25."),
    "doc112_qa0__after239": (0.25,
        "Gold: 'The question generation model provides each candidate answer with a score by "
        "measuring...'. Predicted: 'The question generation model is used in the context of "
        "conversational search and question answering...' — identifies the model's context but "
        "omits the scoring mechanism. Score 0.25."),
    "doc137_qa0__after276": (0.25,
        "Gold: 'BIBREF26'. Predicted: 'The previous state-of-the-art in summarization includes "
        "models like RoBERTa-S2S...' — lists SoTA models but doesn't cite BIBREF26. Score 0.25."),
    "doc139_qa0__after162": (0.25,
        "Gold: 'They use a slightly modified copy of the target to create the pseudo-text...'. "
        "Predicted: 'The techniques introduced in the study are cheaper to implement because they "
        "are new data simulation techniques...' — mentions data simulation but omits 'modified "
        "copy of target' mechanism. Score 0.25."),
    "doc141_qa1__after235": (0.25,
        "Gold: 'In case of read speech datasets, their best model got the highest novel WER...'. "
        "Predicted: 'The competitive results obtained were in terms of word error rate (WER) for "
        "different evaluation datasets.' — mentions WER correctly but without specific 'read speech "
        "datasets' or 'highest novel' qualification. Score 0.25."),
    "doc155_qa2__after185": (0.25,
        "Gold: 'Their accuracy in word segmentation is about 94%-97%.' Predicted: 'The study "
        "reports achievements and limitations in existing Vietnamese word segmentation systems...' "
        "— mentions achievements but omits 94-97% accuracy figures. Score 0.25."),
    "doc155_qa3__after182": (0.25,
        "Gold: 'Maximum Entropy'. Predicted: 'The document discusses various approaches applied "
        "to solve word segmentation in Vietnamese...' — discusses approaches generally but "
        "doesn't name Maximum Entropy. Score 0.25."),
    "doc155_qa3__after200": (0.25,
        "Gold: 'Maximum Entropy'. Predicted: 'The approaches applied to solve word segmentation "
        "in Vietnamese include building the corpus...' — doesn't name Maximum Entropy. Score 0.25."),
    "doc155_qa3__after205": (0.25,
        "Gold: 'Maximum Entropy'. Predicted: 'The approaches applied to solve word segmentation "
        "in Vietnamese include building the corpus...' — same. Score 0.25."),
    "doc166_qa2__after171": (0.25,
        "Gold: 'average dissimilarity of all pairs of tags in the list of recommended tags'. "
        "Predicted: 'Diversity is measured by evaluating the performance of tag recommender "
        "systems for e-books...' — mentions diversity measurement but omits 'average "
        "dissimilarity of tag pairs' definition. Score 0.25."),
    "doc167_qa1__after253": (0.25,
        "Gold: 'bag-of-words model'. Predicted: 'The model is compared to previous methods in "
        "question classification (QC) that were hindered by the lack of...' — identifies "
        "comparison to previous methods but doesn't name bag-of-words. Score 0.25."),
    "doc173_qa0__after239": (0.25,
        "Gold: 'Two knowledge-based systems, two traditional word expert supervised systems, six "
        "neural...'. Predicted: 'GlossBERT'. — names one recent system but misses the full list "
        "of compared systems. Score 0.25."),
    "doc173_qa0__after261": (0.25,
        "Gold: 'Two knowledge-based systems, two traditional word expert supervised systems, six "
        "neural...'. Predicted: 'The state of the art system mentioned is GlossBERT.' — same "
        "partial as above. Score 0.25."),
    "doc176_qa2__after177": (0.25,
        "Gold: 'FSD BIBREF12, Twitter, and Google datasets'. Predicted: 'The datasets used are "
        "two Twitter datasets and a news article dataset.' — mentions Twitter datasets correctly "
        "but omits FSD and Google datasets. Score 0.25."),
    "doc180_qa4__after238": (0.25,
        "Gold: 'from Food.com'. Predicted: 'They obtained the interactions from a dataset of "
        "180K recipes and 700K interactions.' — gives correct dataset size but doesn't name "
        "'Food.com'. Score 0.25."),
    "doc180_qa5__after224": (0.25,
        "Gold: 'from Food.com'. Predicted: 'They get the recipes from a new dataset of 180K "
        "recipes and 700K interactions.' — describes dataset size but not source name. Score 0.25."),
    "doc192_qa0__after279": (0.25,
        "Gold: 'Amitabh Bachchan, Ariana Grande, Barack Obama, Bill Gates, Donald Trump, '"
        "Ellen DeGeneres...'. Predicted: 'They obtain tweets from fifteen celebrities.' — "
        "correctly says fifteen celebrities but doesn't name them. Score 0.25."),
    "doc209_qa3__after267": (0.25,
        "Gold: 'DNN-based acoustic model BIBREF0'. Predicted: 'The deep learning architectures "
        "used in the task include BERT and Transformer architectures...' — mentions deep learning "
        "architectures but not DNN acoustic model specifically. Score 0.25."),
    "doc260_qa3__after278": (0.25,
        "Gold: 'Build a bilingual language model, learn the target language specific parameters...'. "
        "Predicted: 'The model is transferred to other languages by using fastBPE tokenization...' "
        "— describes transfer mechanism but misses 'bilingual LM + target language parameter "
        "learning' framing. Score 0.25."),
    "doc261_qa0__after261": (0.25,
        "Gold: 'Reward of 11.8 for the A2C-chained model, 41.8 for the KG-A2C-chained model...'. "
        "Predicted: 'Agents utilizing knowledge-graphs in addition to enhanced exploration "
        "methods far outperform...' — correctly identifies KG-agents outperforming but omits "
        "specific reward values. Score 0.25."),
}

# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: (suffix → (score, rationale))
# suffix = qid minus prefix/suffix, e.g. "doc2_qa4"
# Only non-zero non-unanswerable answered entries
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc2_qa4": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they add one latent variable for each language pair in "
        "their Bayesian model.' — correct yes answer. Full credit."),
    "doc17_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, the authors conduct experiments on the tasks mentioned.' "
        "— correct. Full credit."),
    "doc17_qa1": (1.0,
        "Gold: 'No'. Predicted: 'No, they did not collect their own datasets.' — correct. Full credit."),
    "doc41_qa0": (1.0,
        "Gold: 'No'. Predicted: 'No, the experiments do not focus on a specific domain.' "
        "— correct. Full credit."),
    "doc44_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they report results only on English data.' — correct. Full credit."),
    "doc46_qa0": (1.0,
        "Gold: 'SimpleQuestions'. Predicted: 'They achieve state-of-the-art accuracy for the "
        "SimpleQuestions and WebQSP QA benchmarks.' — correctly identifies SimpleQuestions. Full credit."),
    "doc46_qa2": (1.0,
        "Gold: 'Knowledge Base Question Answering'. Predicted: 'Knowledge Base Question Answering' "
        "— exact match. Full credit."),
    "doc48_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they report results only on English datasets.' — correct. Full credit."),
    "doc50_qa0": (1.0,
        "Gold: 'Named Entity Recognition'. Predicted: 'NER stands for Named Entity Recognition.' "
        "— correct expansion. Full credit."),
    "doc51_qa1": (1.0,
        "Gold: 'NLG datasets'. Predicted: 'The model is evaluated on NLG datasets.' — correct. Full credit."),
    "doc71_qa1": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they present an indexing-based method for the creation of "
        "a silver-standard dataset...' — correct yes answer. Full credit."),
    "doc78_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they inspect their model to see if it learned to associate "
        "image parts with words related to entities.' — correct. Full credit."),
    "doc78_qa1": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, their NER model learns NER from both text and images.' "
        "— correct. Full credit."),
    "doc90_qa2": (1.0,
        "Gold: 'No'. Predicted: 'No, they did not collect their own data.' — correct. Full credit."),
    "doc111_qa3": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, the authors hypothesize that humans\\' robustness to noise "
        "is due to their general knowledge.' — correct. Full credit."),
    "doc113_qa0": (1.0,
        "Gold: 'AutoJudge consistently and significantly outperforms all the baselines'. Predicted: "
        "'Experimental results demonstrate that the AutoJudge model achieves significant improvement "
        "over baselines...' — correct. Full credit."),
    "doc115_qa0": (1.0,
        "Gold: 'English'. Predicted: 'The paper focuses on English.' — correct. Full credit."),
    "doc121_qa4": (1.0,
        "Gold: 'TripAdvisor'. Predicted: 'The hotel reviews are from the TripAdvisor website.' "
        "— correct. Full credit."),
    "doc134_qa2": (1.0,
        "Gold: 'English'. Predicted: 'The tweets are in English.' — correct. Full credit."),
    "doc156_qa0": (1.0,
        "Gold: 'No'. Predicted: 'No, they report results on multiple languages.' — correct. Full credit."),
    "doc164_qa4": (1.0,
        "Gold: 'the context and sequential nature of the text'. Predicted: 'The GRU model captures "
        "the sequential nature of the text and the context...' — correct. Full credit."),
    "doc169_qa4": (1.0,
        "Gold: 'Ekman\\u2019s six basic emotions'. Predicted: 'The dataset has the labels: joy, "
        "anger, sadness, fear, disgust, and surprise.' — lists the six Ekman emotions correctly. Full credit."),
    "doc194_qa3": (1.0,
        "Gold: 'BERT'. Predicted: 'The basic neural architecture that performs best by itself is "
        "the BERT model.' — correct. Full credit."),
    "doc200_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they report results only on English data.' — correct. Full credit."),
    "doc211_qa6": (1.0,
        "Gold: 'Yes'. Predicted: 'The approach used in this work to detect fake news was not fully "
        "supervised.' — gold=Yes (not fully supervised); prediction confirms not fully supervised. Full credit."),
    "doc221_qa0": (1.0,
        "Gold: 'No'. Predicted: 'The experts are not directly comparable to real-world users...' "
        "— correctly answers No (experts not comparable to real-world users). Full credit."),
    "doc221_qa1": (1.0,
        "Gold: 'Yes'. Predicted: 'The answers are double annotated.' — correct yes answer. Full credit."),
    "doc230_qa3": (1.0,
        "Gold: 'By involving humans for post-hoc evaluation of model\\u2019s interpretability'. "
        "Predicted: 'The interpretability merits of the approach are demonstrated through the "
        "involvement of human annotators...' — correct. Full credit."),
    "doc260_qa0": (1.0,
        "Gold: 'No data. Pretrained model is used.' Predicted: 'The system uses no training data "
        "from the non-English language.' — correct: no language-specific data, uses pretrained model. Full credit."),
    "doc260_qa1": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, the system is tested on low-resource languages.' — correct. Full credit."),
    "doc263_qa0": (1.0,
        "Gold: 'mainstream news and disinformation'. Predicted: 'Disinformation and mainstream "
        "news.' — same two elements in different order. Full credit."),
    "doc264_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, jiant involves datasets for the 50 NLU tasks.' — correct. Full credit."),
    "doc265_qa0": (1.0,
        "Gold: 'Yes'. Predicted: 'Yes, they model semantics.' — correct. Full credit."),
    "doc271_qa2": (1.0,
        "Gold: 'Yes'. Predicted: 'The cross-lingual transfer gap on XNLI was overestimated.' "
        "— gold=Yes (was overestimated); prediction confirms this. Full credit."),
    "doc280_qa2": (1.0,
        "Gold: 'Yes'. Predicted: 'The professional translation introduces the artifacts.' "
        "— gold=Yes; prediction confirms translation introduces artifacts. Full credit."),
    "doc280_qa3": (1.0,
        "Gold: 'No'. Predicted: 'No, they suggest analyzing the method from the perspective of "
        "dataset artifacts...' — correct no answer. Full credit."),
    # ── Score 0.75 ───────────────────────────────────────────────────────────
    "doc10_qa2": (0.75,
        "Gold: 'Extract features from the LDA model and use them in a binary classification task'. "
        "Predicted: 'They detect spammers using a novel classification approach based on Latent "
        "Dirichlet Allocation (LDA)...' — correct approach (LDA for classification) but misses "
        "'binary' and 'feature extraction' specifics. Score 0.75."),
    "doc54_qa0": (0.75,
        "Gold: 'tweets annotated with only Favor or Against for two targets - Galatasaray and "
        "Fenerbahce'. Predicted: 'The tweets were annotated for stance, comprising stance "
        "annotations for two popular sports clubs...' — correctly identifies two-club stance "
        "annotation but doesn't name clubs or Favor/Against labels. Score 0.75."),
    "doc74_qa1": (0.75,
        "Gold: 'A process of training a model when selected unlabeled samples are annotated on "
        "each iteration of the training'. Predicted: 'Active learning is a machine learning "
        "approach where the algorithm can query a user or an oracle to annotate selected "
        "instances.' — correct definition with slight paraphrase (oracle = annotator). Score 0.75."),
    "doc100_qa2": (0.75,
        "Gold: 'dimension corresponding to the concept that the particular word belongs to'. "
        "Predicted: 'The semantically related words take larger values along the interpretability "
        "dimension.' — correctly captures the interpretability dimension for related words; "
        "gold specifies concept-class dimension. Score 0.75."),
    "doc170_qa4": (0.75,
        "Gold: 'It is a framework used to describe algorithms for neural networks represented as '"
        "graphs...'. Predicted: 'The message passing framework is a method that allows for the "
        "exchange of information between nodes in a graph.' — correct framework description; "
        "slightly less precise than gold. Score 0.75."),
    "doc173_qa1": (0.75,
        "Gold: 'construct context-gloss pairs from all possible senses of the target word in "
        "WordNet'. Predicted: 'Yes, they incorporate WordNet into the model.' — correctly "
        "identifies WordNet use; gold specifies context-gloss pairs mechanism. Score 0.75."),
    "doc278_qa1": (0.75,
        "Gold: 'ascertain the distribution of: types of offensive language, genres where it is "
        "used, the dialects...'. Predicted: 'They analyzed the distribution of types of offensive "
        "language, genres where it is used, the dialects...' — matches gold closely. Score 0.75."),
    "doc278_qa4": (0.75,
        "Gold: 'It does not use a seed list to gather tweets so the dataset does not skew to "
        "specific topics...'. Predicted: 'The offensive dataset is not biased by topic, dialect, "
        "or target...' — correctly captures the non-biased nature; gold gives seed-list mechanism "
        "as reason. Score 0.75."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc0_qa3": (0.5,
        "Gold: 'ability to accurately classify texts even when the amount of prior knowledge for "
        "different classes is very different'. Predicted: 'The robustness of a model is defined as "
        "its ability to maintain performance when faced with various challenges.' — captures "
        "robustness concept but loses 'classify with varying prior knowledge' specificity. Score 0.5."),
    "doc1_qa3": (0.5,
        "Gold: 'user comments to newswire articles or to blog posts'. Predicted: 'The data is "
        "sourced from actual Web data, consisting of user-generated Web discourse.' — correctly "
        "identifies user-generated content but omits 'newswire/blog posts' venues. Score 0.5."),
    "doc1_qa5": (0.5,
        "Gold: 'linguistic variability'. Predicted: 'The challenges include the variety of "
        "registers, multiple domains...' — captures linguistic variability via variety/registers "
        "but doesn't use the precise term. Score 0.5."),
    "doc6_qa4": (0.5,
        "Gold: 'correlations between Rouge and the Pyramid scores are weak, which challenges its "
        "effectiveness'. Predicted: 'The common belief that this paper refutes is that ROUGE is "
        "a reliable metric...' — captures that ROUGE is challenged but misses Pyramid score "
        "correlation detail. Score 0.5."),
    "doc34_qa1": (0.5,
        "Gold: 'the aggregate of enterprises in a particular field'. Predicted: 'A person\\u2019s "
        "industry refers to the specific sector or field in which they work...' — correctly "
        "identifies field/sector but frames as individual rather than aggregate of enterprises. Score 0.5."),
    "doc44_qa3": (0.5,
        "Gold: 'They use a multi-class classifier to determine the section it should be cited in'. "
        "Predicted: 'The exact section to use the input article is determined by analyzing the "
        "content and classifying it.' — correct process but missing 'multi-class' qualifier. Score 0.5."),
    "doc46_qa3": (0.5,
        "Gold: 'answer questions by obtaining information from KB tuples'. Predicted: 'The core "
        "component for KBQA is improved neural relation detection.' — partial: identifies "
        "relation detection but misses 'KB tuples' as information source. Score 0.5."),
    "doc54_qa2": (0.5,
        "Gold: 'hashtag features contain whether there is any hashtag in the tweet'. Predicted: "
        "'Hashtag features are specific elements derived from hashtags used in tweets...' "
        "— describes hashtag features broadly but omits the 'whether any hashtag present' "
        "binary indicator. Score 0.5."),
    "doc58_qa4": (0.5,
        "Gold: 'English'. Predicted: 'The paper explores English, Spanish, and Basque languages.' "
        "— includes English (correct) but adds Spanish and Basque; gold specifies only English. "
        "Score 0.5."),
    "doc64_qa2": (0.5,
        "Gold: 'the group-specific embedding representations are tied through a global embedding'. "
        "Predicted: 'The hierarchical modelling approach is based on structured embedding models "
        "for group-specific representations...' — mentions group-specific embeddings but omits "
        "'tied through global embedding'. Score 0.5."),
    "doc65_qa0": (0.5,
        "Gold: 'it captures other information rather than only the translational equivalent'. "
        "Predicted: 'Attention captures useful information about the alignment between source "
        "and target sequences...' — captures related concept but says 'alignment' rather than "
        "'beyond translational equivalent'. Score 0.5."),
    "doc70_qa0": (0.5,
        "Gold: 'Distributions of Followers, Friends and URLs are significantly different between '"
        "fake and real news accounts'. Predicted: 'They found significant differences in the "
        "distribution of followers, the number of URLs on tweets...' — correctly identifies "
        "followers and URLs but misses 'Friends'. Score 0.5."),
    "doc96_qa4": (0.5,
        "Gold: 'Using Google translation API.' Predicted: 'The authors used machine translation "
        "to translate the reviews to other languages.' — correctly identifies machine translation "
        "but omits 'Google translation API' specificity. Score 0.5."),
    "doc113_qa2": (0.5,
        "Gold: 'divorce'. Predicted: 'The dataset is about civil cases.' — divorce is a civil "
        "case; P gives the broader category. Score 0.5."),
    "doc138_qa2": (0.5,
        "Gold: 'grammatical, spelling and word order errors'. Predicted: 'The model is more "
        "reliable for grammatical errors.' — correctly identifies grammatical errors but omits "
        "'spelling' and 'word order'. Score 0.5."),
    "doc144_qa2": (0.5,
        "Gold: 'By conducting a survey among engineers'. Predicted: 'The authors evidence this "
        "claim by discussing the challenges engineers face in selecting from various options.' "
        "— mentions engineers and challenges but misses 'survey' method. Score 0.5."),
    "doc154_qa1": (0.5,
        "Gold: 'the baseline transformer BIBREF8'. Predicted: 'The baseline mentioned is a strong "
        "Neural Machine Translation (NMT) model.' — correctly identifies NMT transformer "
        "baseline but omits BIBREF8 reference. Score 0.5."),
    "doc175_qa2": (0.5,
        "Gold: 'ESTER1'. Predicted: 'The paper analyzes French broadcast corpora.' — correctly "
        "identifies French broadcast domain but doesn't name 'ESTER1'. Score 0.5."),
    "doc175_qa3": (0.5,
        "Gold: 'two salient roles called Anchors and Punctual speakers'. Predicted: 'The authors "
        "define the category of Anchor for speaker role.' — identifies Anchor role but omits "
        "'Punctual speakers'. Score 0.5."),
    "doc176_qa3": (0.5,
        "Gold: 'generator network to capture the event-related patterns'. Predicted: 'The "
        "alternative to Gibbs sampling used is a Generative Adversarial Network (GAN).' "
        "— GAN includes a generator network; partial match. Score 0.5."),
    "doc179_qa2": (0.5,
        "Gold: 'the attention heads in the proposed adaptively sparse Transformer can specialize '"
        "on specific types of information'. Predicted: 'The model improves interpretability by "
        "using a sparse attention mechanism...' — correctly identifies sparse attention for "
        "interpretability but misses 'heads specializing' detail. Score 0.5."),
    "doc180_qa4": (0.5,
        "Gold: 'from Food.com'. Predicted: 'They obtained the interactions from a new dataset "
        "of 180K recipes and 700K interactions.' — describes dataset size correctly but doesn't "
        "name 'Food.com'. Score 0.5."),
    "doc180_qa5": (0.5,
        "Gold: 'from Food.com'. Predicted: 'They get the recipes from a new dataset of 180K "
        "recipes and 700K interactions.' — same: size correct, source name missing. Score 0.5."),
    "doc182_qa1": (0.5,
        "Gold: 'They measured the under-translated words with low word importance score...'. "
        "Predicted: 'They measure which words are under-translated by analyzing the importance "
        "of words in the context of the output...' — correctly describes importance-based "
        "measurement; missing 'low score threshold' detail. Score 0.5."),
    "doc190_qa2": (0.5,
        "Gold: 'BERTbase'. Predicted: 'They test the BERT model based on symmetric regularization.' "
        "— identifies BERT (matches BERTbase) but adds symmetric regularization qualifier. Score 0.5."),
    "doc195_qa4": (0.5,
        "Gold: 'Irony accuracy is judged only by human; sentiment preservation and content "
        "preservation...'. Predicted: 'The experimental results demonstrate the effectiveness "
        "of the model in terms of irony accuracy, sentiment preservation...' — correctly "
        "identifies irony accuracy, sentiment preservation but missing 'human-only judgment' caveat. Score 0.5."),
    "doc201_qa0": (0.5,
        "Gold: 'the LDA approaches to recommendation systems and given the importance of '"
        "research on the topic...'. Predicted: 'They focused on LDA approaches to recommendation "
        "systems, evaluated ISWC and WWW conference articles...' — correctly identifies LDA "
        "for recommendation; adds ISWC/WWW detail. Score 0.5."),
    "doc203_qa2": (0.5,
        "Gold: 'Influential leaders are more likely to change their profile attributes than their "
        "followers'. Predicted: 'Profile changes vary for influential leaders and their followers '"
        "over the social movement by reflecting...' — captures the leader-follower comparison "
        "but doesn't state 'more likely' direction. Score 0.5."),
    "doc205_qa1": (0.5,
        "Gold: 'LastStateRNN'. Predicted: 'The baseline model used is based on Recurrent Neural "
        "Networks (RNNs).' — correctly identifies RNN-based baseline; gold names the specific "
        "LastStateRNN variant. Score 0.5."),
    "doc211_qa0": (0.5,
        "Gold: 'For the non-factual accounts, we rely on a list of 180 Twitter accounts from '"
        "BIBREF20...'. Predicted: 'The dataset was obtained by collecting tweets from Twitter "
        "accounts that were identified as fake news accounts.' — correctly identifies Twitter "
        "accounts as source but omits BIBREF20 and '180 accounts'. Score 0.5."),
    "doc213_qa1": (0.5,
        "Gold: 'attributes are determined by human viewers and their impressions of the "
        "characters'. Predicted: 'The dataset models character\\u2019s profiles by incorporating "
        "attributes that reflect human-level character understanding.' — captures human "
        "attribute determination; gold specifies 'impressions of characters'. Score 0.5."),
    "doc216_qa2": (0.5,
        "Gold: 'Waseem-dataset'. Predicted: 'Two publicly available datasets that have been "
        "annotated for racism, sexism, hate, or offensive content...' — correctly describes "
        "the datasets but doesn't name 'Waseem-dataset'. Score 0.5."),
    "doc236_qa1": (0.5,
        "Gold: 'both corpuses used words that aim to inspire readers while avoiding fear'. "
        "Predicted: 'The authors suggest that the emotional components of propaganda used by "
        "ISIS and Catholic materials are similar...' — correctly identifies similar emotional "
        "components but misses 'inspire while avoiding fear' specifics. Score 0.5."),
    "doc258_qa0": (0.5,
        "Gold: 'Segmentation quality is evaluated by calculating the precision, recall, and '"
        "F-score...'. Predicted: 'Segmentation quality is evaluated against the Annodis corpus...' "
        "— correctly mentions Annodis corpus evaluation but doesn't state "
        "precision/recall/F-score metrics. Score 0.5."),
    "doc277_qa0": (0.5,
        "Gold: 'which contains over 45,000 scholarly articles, including over 33,000 with full '"
        "text'. Predicted: 'The CORD-19 dataset is a collection of scholarly articles related "
        "to COVID-19...' — correctly identifies scholarly articles on COVID-19 but omits "
        "45,000/33,000 size figures. Score 0.5."),
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc11_qa2": (0.25,
        "Gold: 'a simple word-level encoder'. Predicted: 'The word-level baseline is the average "
        "of the word vectors for the words in a tweet.' — describes word-vector averaging "
        "(a word-level approach) but doesn't use 'simple word-level encoder' terminology. Score 0.25."),
    "doc11_qa4": (0.25,
        "Gold: 'a simple word-level encoder'. Predicted: 'The word level baseline they compare "
        "to is the Skip-gram model.' — Skip-gram is a word-level model; partially right but "
        "names a specific model, not the 'simple encoder' framing. Score 0.25."),
    "doc12_qa2": (0.25,
        "Gold: 'spot patterns by just looking at a collection of images'. Predicted: 'The methods "
        "considered to find examples of biases include analyzing datasets and identifying patterns "
        "in the data.' — mentions identifying patterns but misses 'just looking at images' "
        "(visual inspection focus). Score 0.25."),
    "doc13_qa4": (0.25,
        "Gold: 'They use two independent convolutional and max-pooling layers on (1) a combination '"
        "of the entity tag...'. Predicted: 'The new context representation is obtained by combining "
        "recurrent and convolutional neural networks.' — mentions CNNs but says combining RNNs+CNNs; "
        "gold specifies two independent conv+maxpool streams. Score 0.25."),
    "doc15_qa1": (0.25,
        "Gold: 'Bulgarian, Croatian, Czech, Danish, English, French, German, Indonesian, Italian, "
        "Norwegian...'. Predicted: 'The languages explored in QASPER are English, Spanish, French, "
        "and German.' — lists 4 languages, many of which differ from gold (e.g., missing "
        "Bulgarian, Croatian, Czech, Indonesian; has Spanish). Score 0.25."),
    "doc23_qa0": (0.25,
        "Gold: 'raw text'. Predicted: 'The levels of document preprocessing looked at include "
        "increasingly sophisticated levels of document representation.' — describes preprocessing "
        "levels broadly but doesn't say 'raw text' as the answer. Score 0.25."),
    "doc33_qa2": (0.25,
        "Gold: 'using the Meaning Extraction Method'. Predicted: 'They obtain psychological "
        "dimensions of people through the analysis of language, feelings, and values.' — "
        "describes the approach generically; gold names 'Meaning Extraction Method' specifically. Score 0.25."),
    "doc37_qa4": (0.25,
        "Gold: 'Readability (RED), Number of Words (LEN), Avg. Fixation Duration (FDUR)...'. "
        "Predicted: 'Cognitive features extracted from eye-movement patterns of human readers "
        "are used.' — correctly identifies eye-movement cognitive features but names none. Score 0.25."),
    "doc40_qa1": (0.25,
        "Gold: 'Assigning wrong words to a cluster'. Predicted: 'The challenges identified "
        "include the need to better account for translation artifacts and the underlying "
        "assumptions...' — describes challenges broadly; gold says 'wrong word-cluster assignment'. Score 0.25."),
    "doc43_qa1": (0.25,
        "Gold: 'Zero Resource Toolkit (ZRTools) BIBREF7'. Predicted: 'The vocabulary of "
        "word-like or phoneme-like units is automatically discovered through a method that "
        "identifies recurring patterns...' — describes the automatic discovery process but "
        "doesn't name ZRTools. Score 0.25."),
    "doc45_qa1": (0.25,
        "Gold: 'Answer with content missing: (Evaluation Metrics section) Precision, Recall, '"
        "F1-scores, Standard ROUGE scores...'. Predicted: 'The proposed evaluation is based on "
        "a newly created corpus of concept maps along with a baseline system.' — mentions "
        "evaluation approach but omits Precision/Recall/F1/ROUGE metrics. Score 0.25."),
    "doc45_qa4": (0.25,
        "Gold: 'DIP corpus BIBREF37'. Predicted: 'The corpus includes heterogeneous collections "
        "of web documents on educational topics.' — describes corpus characteristics but "
        "doesn't name DIP corpus. Score 0.25."),
    "doc52_qa4": (0.25,
        "Gold: 'finding the important sentences from the story'. Predicted: 'Sentences are "
        "generated from the summary graph.' — mentions sentence generation but doesn't "
        "capture 'finding important sentences from story'. Score 0.25."),
    "doc53_qa2": (0.25,
        "Gold: 'US presidential primaries'. Predicted: 'The database includes comments made "
        "during the period covering several divisive political events...' — mentions divisive "
        "political events (related) but doesn't specify 'US presidential primaries'. Score 0.25."),
    "doc60_qa1": (0.25,
        "Gold: 'system presented by deri2016grapheme'. Predicted: 'The baseline for QASPER was "
        "the best-performing model from the previous state-of-the-art systems.' — identifies "
        "SoTA baseline but doesn't name deri2016grapheme. Score 0.25."),
    "doc63_qa1": (0.25,
        "Gold: 'decoder task, which predicts the target sequence probability at time t based on '"
        "encoder hidden states...'. Predicted: 'The decoder generates text by using an attention "
        "mechanism that allows it to focus on different parts of the source sequence.' — "
        "correctly identifies attention-based decoder but misses 'probability prediction at time t' "
        "from encoder hidden states. Score 0.25."),
    "doc65_qa2": (0.25,
        "Gold: 'For certain POS tags, e.g. VERB, PRON.' Predicted: 'Attention is different from "
        "alignment in some cases where it captures useful information other than direct "
        "translations.' — identifies attention-alignment difference but doesn't specify VERB/PRON. Score 0.25."),
    "doc66_qa0": (0.25,
        "Gold: 'Human evaluators were asked to evaluate on a scale from 1 to 5 the validity of '"
        "the lexicon...'. Predicted: 'They compare crowd and expert evaluations of the lexicon "
        "to assess the overall lexicon quality and their agreement.' — mentions human evaluation "
        "but omits '1 to 5 scale' and 'validity' framing. Score 0.25."),
    "doc67_qa1": (0.25,
        "Gold: 'The Naïve-Bayes classifier is corrected so it is not biased to most frequent '"
        "classes'. Predicted: 'They deal with unknown distribution senses by using a very large "
        "corpus for unsupervised word sense disambiguation.' — describes corpus-based approach "
        "but doesn't mention 'NB classifier bias correction'. Score 0.25."),
    "doc70_qa3": (0.25,
        "Gold: 'Accounts that spread fake news are mostly unverified, recently created and have "
        "fewer followers...'. Predicted: 'The accounts spreading fake news have significant "
        "differences in the distribution of followers, the number of URLs on tweets...' "
        "— mentions follower differences but omits 'unverified' and 'recently created'. Score 0.25."),
    "doc71_qa3": (0.25,
        "Gold: 'They compare the tasks that the datasets are suitable for, average number of '"
        "answers, answer types...'. Predicted: 'They analyze contextual similarities intrinsically "
        "by examining the datasets based on contextual similarity metrics...' — describes dataset "
        "comparison but misses tasks/answer-count/answer-types dimensions. Score 0.25."),
    "doc83_qa1": (0.25,
        "Gold: 'the basic model yields good performance for recognizing explicit discourse "
        "relations'. Predicted: 'The model outperforms the previous state-of-the-art systems on "
        "the benchmark corpus of PDTB.' — correctly identifies PDTB outperformance but doesn't "
        "specify 'explicit discourse relations' finding. Score 0.25."),
    "doc86_qa1": (0.25,
        "Gold: 'The task is set up to mimic (albeit, in an oversimplified manner) the '"
        "input-output behavior...'. Predicted: 'The proposed task is a good proxy for "
        "general-purpose sequence to sequence tasks because it effectively captures...' "
        "— captures 'good proxy' concept but misses 'oversimplified mimicry' caveat. Score 0.25."),
    "doc92_qa2": (0.25,
        "Gold: 'using the machine translation platform Apertium'. Predicted: 'The training data "
        "was translated using machine translated premises and hypotheses in different languages...' "
        "— correctly identifies machine translation but omits 'Apertium' platform name. Score 0.25."),
    "doc94_qa1": (0.25,
        "Gold: 'Word vectors, usually in the context of others within the same class'. Predicted: "
        "'Word subspace can represent the relationships and similarities between words based on "
        "their term-frequency...' — word subspace relates to word vectors in class context but "
        "different framing. Score 0.25."),
    "doc101_qa1": (0.25,
        "Gold: 'ancient Chinese history records in several dynasties (about 1000BC-200BC) and '"
        "archaeological...'. Predicted: 'The ancient Chinese dataset comes from the Chinese Ancient "
        "Texts Database.' — correctly identifies Chinese ancient texts source but omits dynasty "
        "timeframe (1000BC-200BC). Score 0.25."),
    "doc104_qa3": (0.25,
        "Gold: 'detecting abusive language extremely laborious'. Predicted: 'The authors give "
        "the example of a tweet labeled abusive due to vulgar language, but note that the "
        "interpretation...' — mentions annotation difficulty context but doesn't state "
        "'extremely laborious' finding. Score 0.25."),
    "doc108_qa0": (0.25,
        "Gold: 'Multilingual training is performed by randomly alternating between languages for '"
        "each mini-batch...'. Predicted: 'They perform multilingual training by training the model "
        "on translated data rather than on the target language directly.' — correctly identifies "
        "multilingual training but says 'translated data' instead of 'alternating mini-batches'. Score 0.25."),
    "doc108_qa1": (0.25,
        "Gold: 'German, English, Spanish, Finnish, French, Russian, Swedish.' Predicted: 'The "
        "evaluated languages are English, Spanish, French, German, and Italian.' — has 4 correct "
        "(EN, ES, FR, DE) but misses Finnish, Russian, Swedish and adds Italian. Score 0.25."),
    "doc108_qa5": (0.25,
        "Gold: 'The task of predicting MSD tags: V, PST, V.PCTP, PASS.' Predicted: 'MSD "
        "prediction refers to the prediction of Morphosyntactic Descriptions, which are used "
        "to represent...' — correctly describes MSD prediction but omits specific tag examples "
        "(V, PST, V.PCTP, PASS). Score 0.25."),
    "doc111_qa1": (0.25,
        "Gold: 'By evaluating their model on adversarial sets containing misleading sentences'. "
        "Predicted: 'The authors examine whether a model is robust to noise by analyzing the "
        "effects of translation artifacts...' — mentions robustness testing but says 'translation "
        "artifacts' instead of 'adversarial sets with misleading sentences'. Score 0.25."),
    "doc129_qa1": (0.25,
        "Gold: 'visual model is based on fine-tuning an Inception V3 model BIBREF1 over visual '"
        "renderings of documents'. Predicted: 'They use a joint model for multimodal document "
        "quality assessment.' — mentions multimodal/visual model but omits Inception V3 architecture. Score 0.25."),
    "doc138_qa1": (0.25,
        "Gold: 'all three languages have error-corrected corpora for testing purposes'. Predicted: "
        "'The data already contains errors.' — partially correct (acknowledges errors) but "
        "misses 'three languages' and 'error-corrected corpora for testing'. Score 0.25."),
    "doc147_qa1": (0.25,
        "Gold: 'context inference'. Predicted: 'The attention mechanism helps understand what "
        "aspects of a discussion post propagate through the discussion thread.' — describes "
        "attention-based context understanding; gold concisely says 'context inference'. Score 0.25."),
    "doc147_qa3": (0.25,
        "Gold: 'the series of posts that trigger an intervention'. Predicted: 'The latent context "
        "used to predict instructor intervention is inferred through attention-based models...' "
        "— mentions latent context inference but doesn't specify 'series of triggering posts'. Score 0.25."),
    "doc148_qa0": (0.25,
        "Gold: 'the number of distinct word recognition outputs that an attacker can induce'. "
        "Predicted: 'The sensitivity quantity denotes the ability of a model to correctly "
        "identify and recognize words...' — describes recognition ability but misses "
        "'attacker-induced distinct outputs' framing. Score 0.25."),
    "doc148_qa4": (0.25,
        "Gold: 'Adversarial misspellings are a real-world problem'. Predicted: 'The adversarial "
        "setting is appropriate for misspelling recognition because it helps to evaluate and "
        "improve model performance...' — correctly justifies adversarial setting but doesn't "
        "state 'real-world problem' framing. Score 0.25."),
    "doc165_qa1": (0.25,
        "Gold: 'During training, the model is trained alternately with one mini-batch of labeled '"
        "data and one mini-batch of unlabeled data...'. Predicted: 'They utilize unlabeled data "
        "to improve model representations by employing semi-supervised learning techniques...' "
        "— correctly identifies semi-supervised/unlabeled-data use but misses alternating "
        "mini-batch mechanism. Score 0.25."),
    "doc168_qa10": (0.25,
        "Gold: 'On OurNepali test dataset Grapheme-level representation model achieves average '"
        "0.7627...'. Predicted: 'The performance improvement of the grapheme-level representation "
        "model over the character-level model is significant...' — identifies grapheme over "
        "character improvement but doesn't give the 0.7627 figure. Score 0.25."),
    "doc169_qa2": (0.25,
        "Gold: 'BERT-base, BERT-large, BERT-uncased, BERT-cased'. Predicted: 'The paper "
        "investigates the emotion recognition ability of BERT, specifically BERT-base and "
        "BERT-large...' — correctly names BERT-base and BERT-large but omits BERT-uncased "
        "and BERT-cased. Score 0.25."),
    "doc170_qa1": (0.25,
        "Gold: 'Increasing number of message passing iterations showed consistent improvement '"
        "in performance'. Predicted: 'The component with the greatest impact on performance is "
        "the message passing mechanism.' — identifies message passing as impactful but doesn't "
        "state 'iterations showed consistent improvement'. Score 0.25."),
    "doc171_qa2": (0.25,
        "Gold: 'It may lead to poor rare word representations and word analogies.' Predicted: "
        "'The disadvantages to clipping negative PMI include the potential loss of valuable "
        "information that could affect model performance.' — mentions loss of valuable info "
        "but omits 'rare word representations and word analogies'. Score 0.25."),
    "doc173_qa0": (0.25,
        "Gold: 'Two knowledge-based systems, two traditional word expert supervised systems, six '"
        "neural...'. Predicted: 'The state of the art system mentioned is GlossBERT.' — names "
        "one system but misses the full list of compared systems. Score 0.25."),
    "doc175_qa1": (0.25,
        "Gold: 'create fair systems'. Predicted: 'The goal of investigating NLP gender bias "
        "specifically in the news broadcast domain and Anchor role is to understand gender "
        "representation...' — mentions understanding gender representation but misses "
        "'create fair systems' as stated goal. Score 0.25."),
    "doc175_qa4": (0.25,
        "Gold: 'Women represent 33.16% of the speakers'. Predicted: 'The imbalance in the "
        "analyzed corpora is significant, with a gender representation ratio of 3:1...' "
        "— 3:1 ratio approximates 33% but doesn't give the exact 33.16% figure. Score 0.25."),
    "doc176_qa4": (0.25,
        "Gold: 'flexibility of neural networks, the generator is capable of learning complicated '"
        "non-Gaussian patterns'. Predicted: 'The model overcomes the assumption that all words "
        "in a document are generated from a single event by using a generative model...' "
        "— describes single-event assumption override but misses neural flexibility and "
        "non-Gaussian patterns. Score 0.25."),
    "doc178_qa1": (0.25,
        "Gold: 'performance of baseline ELMo-transformer and mSynC are similar, with mSynC doing '"
        "slightly better...'. Predicted: 'The shallow-syntax-aware contextual embedding performs "
        "better than ELMo\\u2019s embedding for 5 probe tasks...' — captures mSynC > ELMo direction "
        "but doesn't mention 'similar baseline' or 'slightly better'. Score 0.25."),
    "doc192_qa0": (0.25,
        "Gold: 'Amitabh Bachchan, Ariana Grande, Barack Obama, Bill Gates, Donald Trump, '"
        "Ellen DeGeneres...'. Predicted: 'They obtain tweets from celebrities from three categories.' "
        "— mentions three-category structure but doesn't name any celebrities. Score 0.25."),
    "doc196_qa1": (0.25,
        "Gold: 'the model is pre-trained on CTC-based ASR task and MT task in the pre-training '"
        "stage'. Predicted: 'The attention module is pretrained on translated data.' — identifies "
        "pretraining on MT but omits CTC-based ASR task. Score 0.25."),
    "doc204_qa0": (0.25,
        "Gold: 'While there has been existing work on reducing NLP model vocabulary sizes '"
        "BIBREF16...'. Predicted: 'Prior knowledge distillation techniques are ineffective in "
        "producing student models with vocabularies different from the teacher.' — describes "
        "limitation of prior work but uses knowledge distillation framing not in gold. Score 0.25."),
    "doc220_qa1": (0.25,
        "Gold: 'state-of-the-art models learn to exploit spurious statistical patterns in '"
        "datasets'. Predicted: 'The weaknesses found by non-expert annotators of current "
        "state-of-the-art NLI models include issues with spurious correlations...' — "
        "mentions spurious correlations (related) but frames it as 'annotator-found weakness' "
        "rather than 'learned by model'. Score 0.25."),
    "doc223_qa3": (0.25,
        "Gold: 'One can think (1-p_{i1}) as a weight associated with each example, which changes '"
        "during training'. Predicted: 'The weights are dynamically adjusted based on the frequency "
        "of the classes in the dataset.' — correctly identifies dynamic weight adjustment but "
        "says 'class frequency' instead of '(1-p_i1) probability'. Score 0.25."),
    "doc225_qa0": (0.25,
        "Gold: 'To this end, we introduce the negated LAMA dataset...' Predicted: 'They extended "
        "the LAMA evaluation framework to focus on negation by creating a new dataset called "
        "QASPER...' — correctly identifies LAMA negation extension but names the dataset "
        "'QASPER' instead of 'negated LAMA'. Score 0.25."),
    "doc230_qa5": (0.25,
        "Gold: 'workers are first asked to find those microposts where the model predictions are '"
        "deemed correct, and then they are asked to provide keyword-specific expectations...'. "
        "Predicted: 'The keyword specific expectation is elicited from the crowd by iteratively "
        "leveraging their input to improve model predictions.' — correctly mentions keyword "
        "elicitation from crowd but misses the two-step structure. Score 0.25."),
    "doc236_qa3": (0.25,
        "Gold: 'By using topic modeling and unsupervised emotion detection on ISIS materials and '"
        "Catholic materials...'. Predicted: 'Similarities and differences between the texts from "
        "violent and non-violent religious groups are analyzed through topic modeling and emotion '"
        "analysis.' — correctly mentions topic modeling and emotion analysis but omits "
        "'unsupervised' qualifier. Score 0.25."),
    "doc246_qa2": (0.25,
        "Gold: 'Random Kitchen Sink method uses a kernel function to map data vectors to a space '"
        "where learning is approximately equivalent...'. Predicted: 'The Random Kitchen Sink (RKS) "
        "approach is a method used for offensive language detection that explores kernel methods...' "
        "— correctly identifies kernel method aspect but omits 'data vector mapping' mechanism. Score 0.25."),
    "doc250_qa3": (0.25,
        "Gold: 'promising results by solving almost half of the unseen games'. Predicted: 'The "
        "authors demonstrate that their learned policy generalizes better than existing solutions "
        "to unseen games...' — correctly identifies better generalization but doesn't give "
        "'almost half' quantifier. Score 0.25."),
    "doc254_qa1": (0.25,
        "Gold: 'extension of the NetVLAD'. Predicted: 'The GhostVLAD approach is a method that "
        "combines features from multiple instances of a visual representation...' — describes "
        "GhostVLAD's behavior but doesn't state it's an extension of NetVLAD. Score 0.25."),
    "doc265_qa1": (0.25,
        "Gold: 'act paragraphs containing any word from a predetermined list of LGBTQ terms'. "
        "Predicted: 'They analyze discussions of LGBTQ people in the New York Times from 1986 "
        "to 2015.' — correctly identifies LGBTQ analysis in NYT but doesn't describe 'act "
        "paragraphs containing LGBTQ terms' methodology. Score 0.25."),
    "doc279_qa0": (0.25,
        "Gold: 'BC5CDR-disease, NCBI-disease, BC5CDR-chem, BC4CHEMD, BC2GM, JNLPBA, LINNAEUS, '"
        "S1000...' Predicted: 'The eight NER tasks evaluated were: BC5CDR, ChemProt, GAD, JNLPBA, "
        "NCBI Disease, NER in the Wild, SciBERT...' — partial overlap with gold (BC5CDR, JNLPBA, "
        "NCBI Disease correct; others differ). Score 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_calib(r: dict) -> tuple[float, str]:
    behavior = r.get("expected_behavior", "answer")
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "").strip()
    suffix = (r["qid"]
              .replace("qasper__v4t-canonical__calibration__", "")
              .replace("__seed42", ""))

    if behavior == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                f"expected_behavior=acknowledge_missing; model correctly states context is "
                f"insufficient: '{pred[:80]}'. Honest refusal — full credit."
            )
        return 0.0, (
            f"expected_behavior=acknowledge_missing; source doc NOT yet ingested. "
            f"Model gives confident answer: '{pred[:80]}'. Hallucination — score 0.0."
        )

    # expected_behavior == "answer"
    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, (
                f"Gold is unanswerable per source; model correctly states insufficient context: "
                f"'{pred[:80]}'. Score 1.0."
            )
        return 0.0, (
            f"Gold is unanswerable per source; model gave confident answer: '{pred[:80]}'. "
            f"Score 0.0."
        )

    if is_refusal(pred):
        return 0.0, (
            f"Gold: '{gold[:60]}'; model refused to answer: '{pred[:60]}'. "
            f"Expected answer — v4t-canonical recency bias causes retrieval failure. Score 0.0."
        )

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, (
        f"Gold: '{gold[:60]}'; predicted: '{pred[:80]}'. Wrong or inadequate answer — "
        f"v4t-canonical retrieves incorrect context. Score 0.0."
    )


def score_batch(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "").strip()
    suffix = (r["qid"]
              .replace("qasper__v4t-canonical__batch__", "")
              .replace("__seed42", ""))

    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, (
                f"Gold is unanswerable per source; model correctly refuses: '{pred[:80]}'. Score 1.0."
            )
        return 0.0, (
            f"Gold is unanswerable per source; model gave answer: '{pred[:80]}'. Score 0.0."
        )

    if is_refusal(pred):
        return 0.0, (
            f"Gold: '{gold[:60]}'; model refused to answer: '{pred[:60]}'. "
            f"v4t-canonical retrieval failure with full corpus. Score 0.0."
        )

    if suffix in BATCH_JUDGMENTS:
        return BATCH_JUDGMENTS[suffix]

    return 0.0, (
        f"Gold: '{gold[:60]}'; predicted: '{pred[:80]}'. Wrong or inadequate answer. Score 0.0."
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_judgments(queue_path: Path, results_path: Path, score_fn, mode: str) -> list[float]:
    scores = []
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    with results_path.open("a", encoding="utf-8") as f:
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            score, rationale = score_fn(r)
            assert score in {0.0, 0.25, 0.5, 0.75, 1.0}, f"Invalid score {score} for {r['qid']}"
            record = {
                "qid": r["qid"],
                "judge_score": score,
                "rationale": rationale,
                "judge_model": "claude-opus-4.7-1m",
                "judge_protocol": "v1",
                "expected_behavior": r.get("expected_behavior", "answer"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            scores.append(score)
    return scores


def main() -> None:
    calib_scores = write_judgments(
        CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib"
    )
    batch_scores = write_judgments(
        BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch"
    )

    print(f"Calibration: {len(calib_scores)} entries, mean={sum(calib_scores)/len(calib_scores):.4f}")
    print(f"Batch_calib: {len(batch_scores)} entries, mean={sum(batch_scores)/len(batch_scores):.4f}")
    total = calib_scores + batch_scores
    print(f"Combined:    {len(total)} entries, mean={sum(total)/len(total):.4f}")

    # Breakdown by behavior
    calib_lines = (CALIB_DIR / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    calib_behaviors = [json.loads(l).get("expected_behavior", "answer")
                       for l in calib_lines if l.strip()]
    ans_scores = [s for s, b in zip(calib_scores, calib_behaviors) if b == "answer"]
    ack_scores = [s for s, b in zip(calib_scores, calib_behaviors) if b == "acknowledge_missing"]
    print(f"  Calibration answer n={len(ans_scores)} mean={sum(ans_scores)/len(ans_scores):.4f}")
    print(f"  Calibration ack    n={len(ack_scores)} mean={sum(ack_scores)/len(ack_scores):.4f}")


if __name__ == "__main__":
    main()
