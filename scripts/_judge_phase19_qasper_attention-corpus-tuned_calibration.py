"""Phase 1.9 - QASPER attention-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  qasper__attention-corpus-tuned__calibration__seed42   (1405 entries)
  qasper__attention-corpus-tuned__batch_calib__seed42   (1005 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - attention-corpus-tuned: pure sparse retrieval via AttentionMemory, no tunable params
  - No recency bias (no theta). Retrieval driven entirely by attention-weighted sparse match.
  - Calibration: ack entries ~54% correct refusal, ~46% hallucination (no temporal signal)
    answer entries: low recall (consistent with Protocol A batch mean=0.157)
  - Batch_calib: end-of-corpus, similar to Protocol A batch performance

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer, gold=(unanswerable per source): 1.0 if refused, 0.0 if answered
  - answer, gold!=unanswerable: standard 5-point rubric
    0.0 wrong/refusal, 0.25 partial/one correct aspect, 0.5 partially correct,
    0.75 correct-but-imprecise, 1.0 correct
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "qasper__attention-corpus-tuned__calibration__seed42"
BATCH_DIR = JQ / "qasper__attention-corpus-tuned__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_RESULTS = BATCH_DIR / "results.jsonl"

# ---------------------------------------------------------------------------
# Refusal detection
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
# CALIBRATION JUDGMENTS: suffix -> (score, rationale)
# suffix = qid minus "qasper__attention-corpus-tuned__calibration__" and "__seed42"
# Only regular-answer entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # Batch 1 (entries 1-50)
    "doc13_qa1__after21": (1.0, "Gold: SemEval 2010 task 8 relation classification; prediction matches exactly. Full credit."),
    "doc24_qa2__after32": (1.0, "Gold: features extracted from CNN; prediction identifies CNN features exactly. Full credit."),
    "doc9_qa0__after32": (1.0, "Gold: Accuracy; prediction correctly names clustering accuracy. Full credit."),
    "doc34_qa0__after34": (0.25, "Gold: 22,880 users; prediction says over 20,000 (approximation, not exact). Score 0.25."),
    "doc13_qa2__after37": (0.75, "Gold: CNN+RNN models with voting; prediction captures CNN+RNN voting scheme. Score 0.75."),
    "doc44_qa5__after44": (0.25, "Gold: salience+positional+frequency+POS features; prediction gets frequency+salience only. Score 0.25."),
    "doc34_qa1__after49": (1.0, "Gold: aggregate of enterprises in a particular field; prediction matches exactly. Full credit."),
    "doc11_qa4__after50": (1.0, "Gold: simple word-level encoder; prediction: simple word-level encoder for tweets. Full credit."),
    "doc34_qa4__after57": (0.25, "Gold: 14 industry classes listed; prediction says 14 classes but doesn't name them. Score 0.25."),
    "doc27_qa1__after61": (0.75, "Gold: Affective Text; prediction: Affective Text + Fairy Tales + ISEAR. Correct answer present. Score 0.75."),
    "doc57_qa2__after62": (1.0, "Gold: SemEval-2016 Sentiment Analysis in Twitter; prediction matches exactly. Full credit."),
    "doc56_qa3__after71": (1.0, "Gold: grouped by objective function; prediction: organized by objective function. Full credit."),
    "doc70_qa3__after71": (0.25, "Gold: unverified+recently created+high ratio; prediction says both verified+unverified (partial overlap). Score 0.25."),
    "doc15_qa1__after73": (1.0, "Gold: 16 languages list; prediction matches same 16 languages. Full credit."),
    "doc0_qa2__after75": (0.25, "Gold: text classification themes (sentiment/web-page/science/medical etc); prediction gets text classification + semantic matching. Score 0.25."),
    "doc51_qa1__after76": (1.0, "Gold: NLG datasets; prediction: evaluated on NLG datasets. Full credit."),
    "doc70_qa5__after77": (1.0, "Gold: ground truth not established; prediction matches exactly. Full credit."),
    "doc65_qa0__after83": (0.75, "Gold: captures info beyond translational equivalent; prediction: beyond alignments + syntactic/morphological. Score 0.75."),
    "doc24_qa1__after85": (0.75, "Gold: SemEval 2014 Twitter Sentiment; prediction: SemEval 2014 Twitter + extra. Score 0.75."),
    "doc28_qa5__after88": (0.25, "Gold: SVM with uni/bi/trigram features; prediction: SVM classifier (misses n-gram detail). Score 0.25."),
    "doc34_qa4__after88": (0.25, "Gold: 14 industry categories named; prediction: 14 classes unnamed. Score 0.25."),
    "doc78_qa4__after90": (1.0, "Gold: 10K user-generated image+caption pairs; prediction matches exactly. Full credit."),
    "doc42_qa0__after91": (1.0, "Gold: 1000 hours data; prediction: 1000 hours exactly. Full credit."),
    "doc54_qa4__after92": (0.75, "Gold: Galatasaray; prediction: Galatasaray + Fenerbahce (extra). Score 0.75."),
    # Batch 2 (entries 51-100)
    "doc24_qa2__after94": (1.0, "Gold: CNN features; prediction: CNN baseline features exactly. Full credit."),
    "doc63_qa0__after94": (0.5, "Gold: encoder with convolutional layers; prediction: convolutional+NIN+Bi-LSTM layers. Partial match. Score 0.5."),
    "doc24_qa2__after95": (1.0, "Gold: CNN features; prediction: CNN features exactly. Full credit."),
    "doc6_qa1__after100": (0.75, "Gold: SERA measures content relevance; prediction captures SERA relevance concept. Score 0.75."),
    "doc98_qa2__after101": (1.0, "Gold: three; prediction: Three annotators. Full credit."),
    "doc44_qa3__after101": (0.25, "Gold: multi-class classifier for section; prediction: class-based section templates. Score 0.25."),
    "doc39_qa1__after102": (0.5, "Gold: CRC model with concept measures; prediction: CRC and 3C models. Captures CRC name. Score 0.5."),
    "doc48_qa2__after102": (1.0, "Gold: selection of word vectors; prediction matches exactly. Full credit."),
    "doc56_qa3__after104": (1.0, "Gold: grouped by objective function; prediction matches. Full credit."),
    "doc11_qa2__after107": (1.0, "Gold: simple word-level encoder; prediction: simple word-level encoder for tweets. Full credit."),
    "doc92_qa0__after108": (0.25, "Gold: content missing (subtask participation); prediction: four Spanish subtasks. Score 0.25."),
    "doc108_qa5__after108": (0.5, "Gold: predicting MSD tags V/PST/V.PCTP/PASS; prediction: MSD prediction auxiliary task (gets task, not specific tags). Score 0.5."),
    "doc34_qa0__after109": (0.25, "Gold: 22,880 users; prediction: over 20,000. Score 0.25."),
    "doc9_qa0__after109": (0.75, "Gold: Accuracy; prediction: accuracy of quasi-translation metric. Score 0.75."),
    "doc63_qa0__after112": (0.5, "Gold: convolutional layers encoder; prediction: convolutional+NIN+Bi-LSTM. Score 0.5."),
    "doc30_qa1__after112": (0.25, "Gold: NBOW, LSTM, attentive LSTM; prediction: NBOW, RNN, CNN. Gets NBOW correct. Score 0.25."),
    "doc30_qa2__after114": (1.0, "Gold: TransE; prediction: TransE. Full credit."),
    "doc96_qa4__after115": (0.25, "Gold: Google translation API; prediction: machine translation (no specific name). Score 0.25."),
    "doc78_qa2__after116": (0.75, "Gold: PER, LOC, ORG, MISC; prediction: persons, organizations, locations + other categories. Score 0.75."),
    "doc7_qa0__after117": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc80_qa2__after119": (1.0, "Gold: Amazon Mechanical Turk; prediction matches exactly. Full credit."),
    "doc65_qa0__after120": (0.75, "Gold: beyond translational equivalent; prediction: beyond alignments + syntactic/morphological. Score 0.75."),
    "doc79_qa0__after122": (1.0, "Gold: Yes; prediction: Yes + SERA metric. Full credit."),
    "doc100_qa3__after123": (0.75, "Gold: cost function modified by additive term; prediction: additive modification to objective function. Score 0.75."),
    "doc0_qa3__after123": (0.5, "Gold: classify despite unbalanced prior AND class distribution; prediction: handle bias in prior knowledge (partial). Score 0.5."),
    "doc116_qa0__after123": (0.75, "Gold: use text transcription; prediction: determine text from audio using speech transcription. Score 0.75."),
    "doc1_qa2__after125": (0.25, "Gold: Structural Support Vector Machine; prediction: supervised+semi-supervised ML methods. Score 0.25."),
    "doc70_qa3__after125": (0.25, "Gold: unverified+recently created+high ratio; prediction partial. Score 0.25."),
    "doc116_qa2__after125": (0.25, "Gold: combines using feed-forward neural model; prediction: combines using dual RNNs. Score 0.25."),
    "doc51_qa0__after126": (0.5, "Gold: Refinement Adjustment LSTM-based component; prediction: LSTM-based decoder with attention. Score 0.5."),
    # Batch 3 (entries 101-150)
    "doc52_qa4__after129": (0.75, "Gold: finding important sentences from story; prediction captures this concept. Score 0.75."),
    "doc23_qa0__after130": (0.75, "Gold: raw text; prediction includes raw text in three preprocessing levels. Score 0.75."),
    "doc46_qa3__after130": (0.25, "Gold: answer questions from KB tuples; prediction: Relation detection core for KBQA. Score 0.25."),
    "doc89_qa1__after130": (1.0, "Gold: DSTC2; prediction: DSTC2 exactly. Full credit."),
    "doc34_qa0__after131": (0.25, "Gold: 22,880 users; prediction: over 20,000. Score 0.25."),
    "doc108_qa4__after131": (0.25, "Gold: LSTM; prediction: character-based encoder-decoder architecture (not LSTM). Score 0.25."),
    "doc40_qa1__after132": (0.25, "Gold: Assigning wrong words to cluster; prediction: cross-speaker UTD difficulty (different framing). Score 0.25."),
    "doc26_qa1__after133": (1.0, "Gold: IMDb movie review dataset; prediction: IMDb movie review dataset. Full credit."),
    "doc96_qa5__after134": (0.25, "Gold: Amazon reviews; prediction: product reviews in English (captures concept but no name). Score 0.25."),
    "doc66_qa0__after135": (0.25, "Gold: human evaluators 1-5 scale validity; prediction: crowd and expert evaluations (captures evaluation but not scale). Score 0.25."),
    "doc48_qa1__after137": (0.25, "Gold: number of clusters, seed, word vectors, window...; prediction: number of clusters only. Score 0.25."),
    "doc70_qa5__after138": (1.0, "Gold: ground truth not established; prediction matches exactly. Full credit."),
    "doc73_qa1__after141": (1.0, "Gold: personal attack/racism/sexism; prediction matches exactly. Full credit."),
    "doc46_qa3__after141": (0.25, "Gold: KB tuples; prediction: KBQA. Score 0.25."),
    "doc122_qa0__after142": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc75_qa2__after143": (0.5, "Gold: hierarchical phrase-based system BIBREF29; prediction: phrase-based MT system (misses hierarchical). Score 0.5."),
    "doc55_qa0__after143": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc4_qa1__after143": (0.25, "Gold: dependency edge between i and i' in English parse tree; prediction: aligning words across languages (misses dependency edges). Score 0.25."),
    "doc39_qa1__after144": (0.5, "Gold: CRC model with concept measures; prediction: CRC and 3C. Score 0.5."),
    "doc54_qa3__after144": (1.0, "Gold: 700; prediction: 700 tweets. Full credit."),
    "doc26_qa1__after145": (1.0, "Gold: IMDb dataset; prediction: IMDb dataset. Full credit."),
    "doc134_qa5__after146": (1.0, "Gold: CNN; prediction: CNN-based best results. Full credit."),
    "doc31_qa1__after147": (0.25, "Gold: 50 annotators ranked 100 translations; prediction: adequacy/fluency/ranking (metrics only, misses 50/100). Score 0.25."),
    "doc47_qa1__after148": (0.5, "Gold: custom dataset + documents + twitter + news; prediction: debate dataset + 1900 dialogs (partial). Score 0.5."),
    "doc70_qa3__after148": (0.25, "Gold: unverified+recently created; prediction: both verified+unverified. Score 0.25."),
    "doc109_qa0__after149": (0.75, "Gold: Stacked LSTMs; prediction: CAS-LSTM + stacked LSTMs (includes gold). Score 0.75."),
    "doc76_qa0__after149": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc55_qa1__after150": (0.75, "Gold: Asian Scientific Paper Excerpt Corpus (ASPEC); prediction: ASPEC + NTCIR PatentMT. Score 0.75."),
    "doc92_qa0__after151": (0.25, "Gold: content missing; prediction: four Spanish subtasks. Score 0.25."),
    "doc129_qa1__after151": (0.75, "Gold: Inception V3; prediction: joint model with Inception V3. Score 0.75."),
    "doc0_qa3__after153": (0.5, "Gold: classify despite unbalanced prior + class distribution; prediction: leverage prior while insensitive to quality (partial). Score 0.5."),
    "doc12_qa3__after153": (0.25, "Gold: Ethnic bias; prediction: linguistic bias + unwarranted inferences (different). Score 0.25."),
    "doc46_qa3__after154": (0.25, "Gold: KB tuples; prediction: KBQA. Score 0.25."),
    "doc2_qa1__after154": (0.25, "Gold: CLV as parent of role variables; prediction: CLVs used in model (misses structural detail). Score 0.25."),
    # Batch 4 (entries 151-200)
    "doc151_qa0__after155": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc41_qa0__after156": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc48_qa3__after156": (0.75, "Gold: k-means on word embeddings; prediction: clustering text embeddings (captures k-means word embedding concept). Score 0.75."),
    "doc31_qa1__after157": (0.5, "Gold: 50 annotators, 100 translations, Adequacy/Fluency/Ranking; prediction: adequacy, fluency, ranking metrics (misses 50/100). Score 0.5."),
    "doc89_qa1__after158": (0.25, "Gold: DSTC2; prediction: dialogues from SDS restaurant domain (no DSTC2 name). Score 0.25."),
    "doc98_qa1__after158": (0.75, "Gold: humor identification analyzed as classification; prediction: humor detection using classification approaches. Score 0.75."),
    "doc94_qa0__after159": (0.25, "Gold: Reuters-8 without stop words; prediction: Reuters database (partial). Score 0.25."),
    "doc78_qa1__after160": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc54_qa3__after161": (1.0, "Gold: 700; prediction: 700 tweets. Full credit."),
    "doc48_qa3__after161": (0.75, "Gold: k-means word embeddings; same concept. Score 0.75."),
    "doc99_qa0__after162": (0.75, "Gold: words user wants in generated output; prediction: user-defined keywords DMK forces model to generate. Score 0.75."),
    "doc111_qa3__after164": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc101_qa1__after165": (0.75, "Gold: ancient Chinese history records several dynasties 1000BC-200BC; prediction matches description. Score 0.75."),
    "doc111_qa1__after165": (0.25, "Gold: evaluating on adversarial sets with misleading sentences; prediction: testing with 20-80% training subsets. Score 0.25."),
    "doc0_qa0__after165": (0.25, "Gold: labeled features; prediction: leverage prior knowledge (paraphrase). Score 0.25."),
    "doc55_qa0__after166": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc13_qa4__after167": (0.25, "Gold: two independent conv+maxpool on adjacent contexts + longer text; prediction: extended middle context (partial). Score 0.25."),
    "doc42_qa0__after168": (1.0, "Gold: 1000 hours; prediction: 1000 hours. Full credit."),
    "doc45_qa1__after169": (0.25, "Gold: (content missing) Precision/Recall/F1 concept maps; prediction: evaluation protocol baseline (captures framework). Score 0.25."),
    "doc46_qa3__after169": (0.25, "Gold: answer questions from KB tuples; prediction: KBQA. Score 0.25."),
    "doc31_qa1__after171": (0.5, "Gold: 50 annotators/100 translations; prediction: adequacy/fluency/ranking metrics (same as above). Score 0.5."),
    "doc166_qa2__after171": (0.5, "Gold: average dissimilarity of all tag pairs; prediction: novel diversity metric semantic diversity. Score 0.5."),
    "doc137_qa1__after172": (0.25, "Gold: German newscrawl WMT'18; prediction: French and German biography datasets (partial). Score 0.25."),
    "doc116_qa1__after173": (0.5, "Gold: attention model MDREA outperforms; prediction: outperforms 68.8-71.8% (no MDREA name). Score 0.5."),
    "doc108_qa1__after174": (0.25, "Gold: German/English/Spanish/Finnish/French/Russian/Swedish; prediction: overlaps significantly (3 correct). Score 0.25."),
    "doc46_qa2__after174": (1.0, "Gold: Knowledge Base Question Answering; prediction: Knowledge Base Question Answering. Full credit."),
    "doc129_qa5__after175": (0.5, "Gold: depends on dataset; prediction: visual+textual complementary. Score 0.5."),
    "doc172_qa1__after176": (0.25, "Gold: ARAM improved all baselines via reverse perplexity; prediction: ARAML better than GAN baselines (name mismatch). Score 0.25."),
    "doc54_qa0__after176": (0.5, "Gold: Favor/Against for Galatasaray + Fenerbahce; prediction: favor/against stance (no target names). Score 0.5."),
    "doc15_qa0__after177": (0.25, "Gold: UD v1.2 treebanks 16 languages; prediction: 16 languages (misses UD name). Score 0.25."),
    "doc90_qa0__after177": (0.25, "Gold: Back Translation; prediction: data augmentation monolingual (no specific name). Score 0.25."),
    "doc176_qa2__after177": (0.25, "Gold: FSD/Twitter/Google datasets; prediction: two Twitter + news article (partial). Score 0.25."),
    "doc78_qa2__after177": (0.75, "Gold: PER/LOC/ORG/MISC; prediction: organizations/locations/persons (misses MISC). Score 0.75."),
    # Batch 5 (entries 201-250)
    "doc31_qa1__after178": (0.5, "Gold: 50 annotators/100 translations Adequacy/Fluency/Ranking; prediction: metrics correct (same pattern). Score 0.5."),
    "doc39_qa0__after180": (0.25, "Gold: benchmark ceccarelli2013/CoNLL 2003; prediction: benchmark for entity semantic relatedness (captures concept). Score 0.25."),
    "doc45_qa1__after182": (0.25, "Gold: (content missing) Precision/Recall/F1; prediction: evaluation protocol (captures framework). Score 0.25."),
    "doc114_qa2__after182": (0.75, "Gold: multi-turn answer module for span detector; prediction: multi-turn answer module with bilinear function. Score 0.75."),
    "doc169_qa1__after183": (0.75, "Gold: Friends; prediction: Friends + EmotionPush (extra). Score 0.75."),
    "doc168_qa6__after183": (1.0, "Gold: daily newspaper 2015-2016; prediction matches exactly. Full credit."),
    "doc177_qa2__after184": (1.0, "Gold: 20 minutes; prediction: less than 20 minutes. Full credit."),
    "doc155_qa2__after185": (1.0, "Gold: 94-97%; prediction: 94-97%. Full credit."),
    "doc182_qa3__after186": (1.0, "Gold: Transformer+RNN-Search; prediction: Transformer+RNN-Search. Full credit."),
    "doc45_qa3__after186": (0.5, "Gold: description+topic+proposed concept map; prediction: score proposition importance. Score 0.5."),
    "doc42_qa0__after186": (1.0, "Gold: 1000 hours; prediction: 1000 hours. Full credit."),
    "doc132_qa0__after187": (0.75, "Gold: sentences in document are disordered; prediction: sentences not in specific order (captures concept). Score 0.75."),
    "doc138_qa1__after187": (0.5, "Gold: error-corrected corpora for testing; prediction: data already contains errors. Score 0.5."),
    "doc2_qa5__after187": (0.5, "Gold: Bayesian model garg2012unsupervised; prediction: Bayesian model per language. Score 0.5."),
    "doc104_qa1__after187": (0.5, "Gold: tweets replied/quoted as contextual info; prediction: context data+ensemble. Score 0.5."),
    "doc183_qa1__after188": (0.5, "Gold: first principal component contextualized; prediction: static embedding from contextualized. Score 0.5."),
    "doc1_qa3__after189": (0.5, "Gold: user comments to newswire/blogs; prediction: user-generated Web discourse. Score 0.5."),
    "doc96_qa3__after189": (0.75, "Gold: majority baseline; prediction: majority+lexicon baselines (includes gold). Score 0.75."),
    "doc104_qa1__after189": (0.5, "Gold: replied/quoted tweets as context; prediction: context+Latent Topic Clustering. Score 0.5."),
    "doc180_qa0__after190": (1.0, "Gold: BPE PPL/BLEU-1/4/ROUGE-L/distinct n-grams; prediction matches all. Full credit."),
    "doc121_qa0__after190": (0.75, "Gold: 8 tasks requiring different competencies; prediction: 8 groups based on competency. Score 0.75."),
    "doc68_qa4__after191": (0.25, "Gold: LR/MNB/RF/AdaBoost/LinSVM; prediction: SVM/multi-label/LR (partial). Score 0.25."),
    "doc54_qa4__after192": (0.75, "Gold: Galatasaray; prediction: Galatasaray+Fenerbahce. Score 0.75."),
    "doc59_qa2__after192": (0.75, "Gold: standard accuracy; prediction: prediction accuracy. Score 0.75."),
    "doc178_qa4__after192": (0.5, "Gold: token-level chunk label embeddings; prediction: chunk boundary info. Score 0.5."),
    "doc192_qa0__after193": (0.75, "Gold: 15 specific celebrities; prediction: fifteen celebrities various domains. Score 0.75."),
    "doc26_qa1__after194": (1.0, "Gold: IMDb movie review; prediction: IMDb movie review. Full credit."),
    "doc0_qa0__after194": (0.5, "Gold: labeled features; prediction: regularization terms leverage prior knowledge. Score 0.5."),
    "doc39_qa1__after195": (0.5, "Gold: CRC with concept measures; prediction: CRC+3C. Score 0.5."),
    "doc96_qa3__after195": (0.75, "Gold: majority baseline; prediction: majority+lexicon. Score 0.75."),
    "doc46_qa2__after196": (1.0, "Gold: KBQA; prediction: KBQA. Full credit."),
    # Batch 6 (entries 251-300)
    "doc109_qa2__after197": (0.75, "Gold: plain stacked LSTMs; prediction: typical stacked LSTMs. Score 0.75."),
    "doc54_qa4__after197": (0.75, "Gold: Galatasaray; prediction: Galatasaray+Fenerbahce. Score 0.75."),
    "doc168_qa8__after197": (0.25, "Gold: OurNepali 3 types/ILPRL 4 types; prediction: PER/LOC/ORG (partial). Score 0.25."),
    "doc129_qa5__after198": (0.5, "Gold: depends on dataset; prediction: visual+textual complementary. Score 0.5."),
    "doc186_qa1__after199": (0.5, "Gold: MT system BIBREF11; prediction: context-agnostic MT sentence level. Score 0.5."),
    "doc68_qa4__after199": (0.25, "Gold: LR/MNB/RF/AdaBoost/LinSVM; prediction: SVM/multi-label/LR (partial). Score 0.25."),
    "doc52_qa1__after199": (0.5, "Gold: ROUGE/Recall/Precision/F1; prediction: Pyramid+ROUGE family. Score 0.5."),
    "doc175_qa3__after200": (1.0, "Gold: Anchors and Punctual speakers; prediction: Anchor and Punctual speakers. Full credit."),
    "doc41_qa2__after201": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc129_qa3__after202": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc38_qa3__after203": (0.25, "Gold: BIBREF12/13; prediction: annotated Twitter dataset 9,473 annotations (describes). Score 0.25."),
    "doc183_qa1__after203": (0.5, "Gold: first principal component; prediction: static embedding from contextualized. Score 0.5."),
    "doc171_qa0__after204": (1.0, "Gold: clipped PMI; NNEGPMI; prediction: NNEGPMI and clipped PMI. Full credit."),
    "doc12_qa1__after205": (0.75, "Gold: 30,000; prediction: over 30,000 images. Score 0.75."),
    "doc203_qa0__after205": (0.75, "Gold: username; prediction: username+display name+profile image (includes gold). Score 0.75."),
    "doc116_qa0__after205": (0.75, "Gold: use text transcription; prediction: speech transcription. Score 0.75."),
    "doc94_qa1__after206": (0.75, "Gold: word vectors in context of same class; prediction: word subspace retaining variability. Score 0.75."),
    "doc71_qa3__after206": (0.25, "Gold: compare tasks/average number; prediction: contextual similarities examination. Score 0.25."),
    "doc55_qa1__after206": (0.25, "Gold: ASPEC; prediction: English-Japanese+Japanese-English parallel (no ASPEC name). Score 0.25."),
    "doc85_qa2__after207": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc45_qa2__after207": (0.5, "Gold: (content missing) baseline simple implementation; prediction: implementation along with corpus. Score 0.5."),
    "doc122_qa0__after208": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc142_qa5__after208": (1.0, "Gold: 14; prediction: 14 participants. Full credit."),
    "doc118_qa1__after208": (1.0, "Gold: ilur.am; prediction: ilur.am. Full credit."),
    "doc199_qa1__after209": (0.75, "Gold: font type; prediction: font type and font style (includes gold). Score 0.75."),
    "doc120_qa1__after209": (0.25, "Gold: same as BIBREF7 datasets; prediction: 70 million Flickr photos (describes). Score 0.25."),
    "doc143_qa1__after210": (0.25, "Gold: EM 51.10/F1 63.11; prediction: 2% EM improvement (no absolute numbers). Score 0.25."),
    "doc59_qa1__after210": (1.0, "Gold: pre-trained GloVe vectors; prediction: GloVe Wikipedia+Gigaword5+Common Crawl. Full credit."),
    "doc165_qa1__after211": (0.25, "Gold: model trained alternately mini-batch labeled; prediction: CVT semi-supervised. Score 0.25."),
    "doc92_qa2__after211": (0.25, "Gold: Apertium platform; prediction: translated from other languages (no Apertium). Score 0.25."),
    # Batch 7 (entries 301-350)
    "doc0_qa0__after212": (0.5, "Gold: labeled features; prediction: regularization terms leverage prior knowledge. Score 0.5."),
    "doc45_qa2__after212": (0.5, "Gold: (content missing) baseline; prediction: implementation along corpus. Score 0.5."),
    "doc164_qa4__after212": (1.0, "Gold: context+sequential nature; prediction: GRU captures sequential+context. Full credit."),
    "doc129_qa3__after212": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc92_qa0__after212": (0.25, "Gold: (content missing) subtask participation; prediction: four subtasks. Score 0.25."),
    "doc212_qa1__after213": (1.0, "Gold: VAE; prediction: variational auto-encoder (VAE). Full credit."),
    "doc97_qa0__after213": (0.75, "Gold: ambiguous or too common; prediction: unresolved ambiguities. Score 0.75."),
    "doc103_qa0__after214": (0.5, "Gold: WSJ Penn Treebank; prediction: Penn Treebank (misses WSJ). Score 0.5."),
    "doc142_qa1__after214": (1.0, "Gold: presence/absence consonants/nasal/bilabial/high-front vowels; prediction: five binary tasks same list. Full credit."),
    "doc6_qa1__after215": (0.75, "Gold: content relevance candidate vs human summary; prediction: SERA measures content relevance. Score 0.75."),
    "doc96_qa2__after215": (0.25, "Gold: SemEval-2016 Task 5; prediction: large product reviews dataset (no SemEval name). Score 0.25."),
    "doc76_qa2__after216": (1.0, "Gold: corpus of state speeches UN General Debate; prediction: state speeches UN General Debate+voting. Full credit."),
    "doc96_qa5__after217": (0.25, "Gold: Amazon reviews; prediction: large product reviews English (no Amazon name). Score 0.25."),
    "doc189_qa0__after217": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc69_qa1__after218": (1.0, "Gold: MLP; prediction: MLP. Full credit."),
    "doc63_qa0__after218": (0.75, "Gold: convolutional layers+...; prediction: convolutional+NIN+Bi-LSTM. Score 0.75."),
    "doc1_qa3__after219": (0.5, "Gold: user comments newswire/blogs; prediction: user-generated Web discourse. Score 0.5."),
    "doc110_qa0__after219": (0.5, "Gold: Tree-LSTM variants; prediction: tree-structured+neural models. Score 0.5."),
    "doc90_qa0__after219": (0.25, "Gold: Back Translation; prediction: data augmentation monolingual. Score 0.25."),
    "doc71_qa3__after219": (0.25, "Gold: compare tasks/average number; prediction: contextual similarities. Score 0.25."),
    "doc54_qa0__after220": (0.5, "Gold: Favor/Against Galatasaray/Fenerbahce; prediction: favor/against (no targets). Score 0.5."),
    "doc173_qa3__after220": (1.0, "Gold: small BERT; prediction: small BERT. Full credit."),
    "doc54_qa3__after221": (1.0, "Gold: 700; prediction: 700 tweets. Full credit."),
    "doc134_qa2__after221": (1.0, "Gold: English; prediction: English. Full credit."),
    "doc84_qa3__after222": (0.5, "Gold: macro level decide which field; prediction: bifocal macro level decides field. Score 0.5."),
    "doc63_qa1__after222": (0.5, "Gold: decoder predicts target sequence probability; prediction: attention matrices. Score 0.5."),
    "doc180_qa5__after224": (1.0, "Gold: from Food.com; prediction: Food.com. Full credit."),
    "doc168_qa8__after224": (0.25, "Gold: 3/4 entity types; prediction: PER/LOC/ORG (partial). Score 0.25."),
    "doc70_qa4__after224": (1.0, "Gold: 1000; prediction: more than 1000 retweets. Full credit."),
    "doc10_qa1__after224": (0.25, "Gold: Social Honeypot+Weibo; prediction: English Honeypot only. Score 0.25."),
    "doc163_qa2__after225": (0.5, "Gold: archived snapshots CNN/NBC; prediction: crawling news articles with tweets. Score 0.5."),
    "doc106_qa2__after225": (0.25, "Gold: F1/precision/recall/accuracy; prediction: one/zero scoring BIBREF33. Score 0.25."),
    "doc134_qa2__after225": (1.0, "Gold: English; prediction: English. Full credit."),
    "doc108_qa5__after226": (0.5, "Gold: V/PST/V.PCTP/PASS tags; prediction: MSD auxiliary objective (no specific tags). Score 0.5."),
    "doc71_qa1__after226": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    # Batch 8 (entries 351-400)
    "doc37_qa4__after227": (0.25, "Gold: RED/LEN/FDUR list; prediction: cognitive eye movement features (general). Score 0.25."),
    "doc46_qa0__after227": (0.75, "Gold: SimpleQuestions; prediction: SimpleQuestions+WebQSP (includes gold). Score 0.75."),
    "doc107_qa0__after227": (0.25, "Gold: PDTB taggers; prediction: SVM+BiLSTMs (captures method, not name). Score 0.25."),
    "doc11_qa1__after228": (1.0, "Gold: established task; prediction: addressed earlier = established. Full credit."),
    "doc81_qa1__after228": (1.0, "Gold: Euclidean distance context vectors; prediction: Euclidean distance context vectors. Full credit."),
    "doc80_qa3__after228": (0.25, "Gold: 3 components pilot studies; prediction: via crowdsourcing. Score 0.25."),
    "doc11_qa2__after229": (1.0, "Gold: simple word-level encoder; prediction: simple word-level encoder for tweets. Full credit."),
    "doc63_qa1__after229": (0.5, "Gold: decoder predicts target sequence probability; prediction: attention matrices. Score 0.5."),
    "doc177_qa5__after231": (0.25, "Gold: GloVe/BERT/USE/TF-IDF/InferSent; prediction: InferSent+USE (2/5). Score 0.25."),
    "doc59_qa2__after231": (0.75, "Gold: standard accuracy; prediction: prediction accuracy. Score 0.75."),
    "doc107_qa0__after232": (0.25, "Gold: PDTB taggers; prediction: SVM+BiLSTMs. Score 0.25."),
    "doc179_qa0__after232": (0.25, "Gold: 4 MT language pairs; prediction: translation accuracy+interpretability (no specific pairs). Score 0.25."),
    "doc94_qa1__after233": (0.75, "Gold: word vectors in context of same class; prediction: word subspace variability. Score 0.75."),
    "doc162_qa0__after233": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc13_qa4__after233": (0.25, "Gold: two conv+maxpool on adj contexts + longer text; prediction: extended middle context (partial). Score 0.25."),
    "doc111_qa1__after234": (0.25, "Gold: adversarial sets misleading sentences; prediction: 20-80% training subset. Score 0.25."),
    "doc107_qa0__after235": (0.25, "Gold: PDTB taggers; prediction: SVM+BiLSTMs. Score 0.25."),
    "doc151_qa0__after235": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc141_qa1__after235": (0.5, "Gold: highest nov93 score; prediction: competitive WSJ+Hub5'00. Score 0.5."),
    "doc122_qa0__after235": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc85_qa0__after236": (1.0, "Gold: Automatic; prediction: automatic (freely available tools). Full credit."),
    "doc223_qa1__after236": (0.75, "Gold: +0.29 CoNLL2003/+0.96 OntoNotes5.0; prediction matches numbers. Score 0.75."),
    "doc156_qa0__after237": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc10_qa2__after238": (0.75, "Gold: LDA features for binary classification; prediction: LDA-based feature extraction. Score 0.75."),
    "doc111_qa3__after238": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc180_qa4__after238": (0.25, "Gold: from Food.com; prediction: historical preferences from recipes (no Food.com). Score 0.25."),
    "doc24_qa2__after238": (1.0, "Gold: CNN features; prediction: CNN baseline features. Full credit."),
    "doc100_qa3__after239": (1.0, "Gold: cost function modified by additive term; prediction: additive modification objective. Full credit."),
    "doc112_qa0__after239": (0.75, "Gold: question generation model scores candidates; prediction: QG model in framework. Score 0.75."),
    "doc208_qa3__after240": (0.75, "Gold: 3.7 content score; prediction: content 3.7, creativity 3.9, style. Score 0.75."),
    "doc214_qa1__after240": (0.75, "Gold: news articles free-text; prediction: news articles annotated 18 techniques. Score 0.75."),
    "doc177_qa2__after241": (1.0, "Gold: 20 minutes; prediction: less than 20 minutes. Full credit."),
    "doc59_qa2__after242": (0.75, "Gold: standard accuracy; prediction: prediction accuracy. Score 0.75."),
    "doc204_qa1__after242": (0.25, "Gold: NoKD baseline; prediction: knowledge distillation techniques (no NoKD name). Score 0.25."),
    "doc60_qa2__after243": (1.0, "Gold: Phoneme Error Rate (PER); prediction: phoneme error rate (PER). Full credit."),
    # Batch 9 (entries 401-450)
    "doc57_qa2__after243": (1.0, "Gold: SemEval-2016 Sentiment Analysis Twitter; prediction: SemEval2016 subtask 4 Sentiment Analysis. Full credit."),
    "doc70_qa5__after245": (1.0, "Gold: ground truth not established; prediction: ground truth not established. Full credit."),
    "doc51_qa0__after246": (0.5, "Gold: Refinement Adjustment LSTM component; prediction: LSTM-based decoder attention. Score 0.5."),
    "doc152_qa0__after247": (0.75, "Gold: perceptual illusion speech+sight; prediction: hearing influenced by seeing. Score 0.75."),
    "doc39_qa0__after247": (0.25, "Gold: ceccarelli2013/CoNLL 2003 benchmark; prediction: benchmark for entity relatedness. Score 0.25."),
    "doc19_qa1__after248": (0.5, "Gold: predicted number redundant answers; prediction: annotated by experts or crowd by difficulty. Score 0.5."),
    "doc156_qa1__after249": (0.25, "Gold: English-German dataset; prediction: multimodal MT datasets (partial). Score 0.25."),
    "doc216_qa1__after249": (0.25, "Gold: tweets neither + implicit hatred; prediction: model captures biases. Score 0.25."),
    "doc78_qa1__after250": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc202_qa2__after250": (1.0, "Gold: pre-trained multi-BERT; prediction: multi-BERT multilingual. Full credit."),
    "doc93_qa1__after251": (0.5, "Gold: decoder with whole sequence context; prediction: encoder LSTM context vector. Score 0.5."),
    "doc108_qa0__after253": (0.5, "Gold: multilingual training alternating languages; prediction: encoder-decoder multilingual training. Score 0.5."),
    "doc216_qa2__after253": (0.25, "Gold: Waseem-dataset; prediction: two publicly available datasets racism/sexism. Score 0.25."),
    "doc185_qa0__after253": (0.5, "Gold: Conversations Gone Awry personal attacks; prediction: labels for antisocial events. Score 0.5."),
    "doc178_qa4__after254": (0.5, "Gold: token-level chunk label embeddings; prediction: chunk boundary info. Score 0.5."),
    "doc24_qa1__after255": (0.75, "Gold: SemEval 2014 Twitter Sentiment; prediction: SemEval 2014 Twitter + 25K sarcastic. Score 0.75."),
    "doc111_qa3__after255": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc39_qa3__after255": (0.25, "Gold: 212 times reduction; prediction: much less dimensions (no 212). Score 0.25."),
    "doc1_qa1__after256": (0.25, "Gold: claim/premise/backing/rebuttal/refutation; prediction: argument components in discourse. Score 0.25."),
    "doc173_qa3__after256": (1.0, "Gold: small BERT; prediction: small BERT. Full credit."),
    "doc228_qa6__after257": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc214_qa1__after257": (0.75, "Gold: news articles free-text; prediction: news articles annotated 18 techniques. Score 0.75."),
    "doc211_qa3__after258": (0.75, "Gold: consecutive account tweets in time; prediction: groups of tweets read together. Score 0.75."),
    # Batch 10 (entries 451-526)
    "doc218_qa2__after259": (0.75, "Gold: 30 terms, 15 samples per term; prediction: 15 samples per class per term balanced. Score 0.75."),
    "doc236_qa3__after259": (0.5, "Gold: topic modeling+unsupervised emotion ISIS; prediction: analyzed using topic modeling. Score 0.5."),
    "doc120_qa1__after259": (0.25, "Gold: BIBREF7 datasets; prediction: 70M Flickr photos (describes). Score 0.25."),
    "doc107_qa0__after260": (0.25, "Gold: PDTB taggers; prediction: SVM+BiLSTMs. Score 0.25."),
    "doc120_qa1__after260": (0.25, "Gold: BIBREF7 datasets; prediction: 70M Flickr photos. Score 0.25."),
    "doc98_qa3__after261": (0.25, "Gold: tweets sports/politics/entertainment; prediction: code-mixed tweets social media. Score 0.25."),
    "doc180_qa4__after261": (0.25, "Gold: from Food.com; prediction: historical preferences from recipes (no Food.com). Score 0.25."),
    "doc177_qa1__after263": (0.5, "Gold: Spearman's cosine similarity sentence embeddings; prediction: cosine+logistic regression. Score 0.5."),
    "doc93_qa2__after263": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc173_qa2__after264": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc175_qa0__after265": (0.25, "Gold: ASR; prediction: WER scores by gender/role/speech (describes usage, not ASR name). Score 0.25."),
    "doc34_qa1__after265": (1.0, "Gold: aggregate of enterprises in particular field; prediction matches exactly. Full credit."),
    "doc138_qa2__after266": (0.75, "Gold: grammatical/spelling/word order; prediction: punctuation/word order/grammatical (overlaps). Score 0.75."),
    "doc143_qa1__after266": (0.25, "Gold: EM 51.10/F1 63.11; prediction: 2% EM improvement (no absolute). Score 0.25."),
    "doc119_qa2__after267": (0.25, "Gold: MIMIC-III; prediction: 1102 Discharge Summaries + 1000 Nursing Notes ICU (no MIMIC name). Score 0.25."),
    "doc100_qa3__after267": (1.0, "Gold: additive modification cost function; prediction: additive modification objective. Full credit."),
    "doc26_qa1__after268": (1.0, "Gold: IMDb movie review; prediction: IMDb movie review. Full credit."),
    "doc75_qa1__after268": (0.25, "Gold: Spearman's phrasal similarity; prediction: context-dependent scoring likelihood. Score 0.25."),
    "doc54_qa4__after269": (0.75, "Gold: Galatasaray; prediction: Galatasaray+Fenerbahce. Score 0.75."),
    "doc134_qa2__after269": (1.0, "Gold: English; prediction: English. Full credit."),
    "doc218_qa2__after269": (0.75, "Gold: 30 terms 15 samples; prediction: 15 samples balanced. Score 0.75."),
    "doc182_qa0__after270": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc218_qa2__after270": (0.75, "Gold: 30 terms 15 samples; prediction: 15 samples balanced. Score 0.75."),
    "doc243_qa5__after271": (0.5, "Gold: (+1 or -1) opposite polarity words; prediction: supervised scores indicative of polarities. Score 0.5."),
    "doc37_qa4__after271": (0.25, "Gold: RED/LEN/FDUR list; prediction: eye movement cognitive features. Score 0.25."),
    "doc207_qa0__after271": (0.75, "Gold: BLEU; prediction: BLEU-1/Meteor/Rouge-L (includes gold). Score 0.75."),
    "doc60_qa1__after272": (0.25, "Gold: deri2016grapheme system; prediction: adapting monolingual g2p models. Score 0.25."),
    "doc273_qa1__after273": (0.75, "Gold: precision; prediction: precision/recall/weighted F1 (includes gold). Score 0.75."),
    "doc68_qa0__after273": (0.5, "Gold: LSA/TextRank/LexRank/ILP; prediction: ILP summarization vs manual. Score 0.5."),
    "doc34_qa2__after273": (0.25, "Gold: AllWords model; prediction: content-based classifier+feature selection. Score 0.25."),
    "doc57_qa2__after273": (1.0, "Gold: SemEval-2016; prediction: SemEval2016 subtask 4. Full credit."),
    "doc183_qa1__after273": (1.0, "Gold: first principal component contextualized; prediction: first principal component. Full credit."),
    "doc205_qa3__after274": (0.5, "Gold: classic/avg/attention/multiattention RNN; prediction: multi-attention+single attention. Score 0.5."),
    "doc90_qa0__after275": (0.25, "Gold: Back Translation; prediction: data augmentation monolingual. Score 0.25."),
    "doc54_qa2__after275": (1.0, "Gold: whether any hashtag in tweet; prediction: existence of hashtags. Full credit."),
    "doc113_qa0__after276": (0.75, "Gold: AutoJudge outperforms all baselines; prediction: AutoJudge achieves improvement. Score 0.75."),
    "doc183_qa1__after276": (1.0, "Gold: first principal component; prediction: first principal component. Full credit."),
    "doc275_qa5__after277": (0.75, "Gold: CRF; prediction: BioBERT/MTL/BiLSTM-CRF/CRF (includes gold). Score 0.75."),
    "doc146_qa0__after277": (0.5, "Gold: improvement mixing difficult expert+crowd; prediction: expert higher quality augmenting helps. Score 0.5."),
    "doc230_qa5__after278": (0.5, "Gold: workers find microposts where model prediction; prediction: keyword expectation by finding microposts. Score 0.5."),
    "doc0_qa3__after278": (0.5, "Gold: classify despite unbalanced prior+class distribution; prediction: prior knowledge while insensitive quality. Score 0.5."),
    "doc118_qa1__after278": (1.0, "Gold: ilur.am; prediction: ilur.am. Full credit."),
    "doc260_qa3__after278": (0.75, "Gold: Build bilingual LM + target lang params; prediction: builds bilingual LM. Score 0.75."),
    "doc192_qa0__after279": (0.75, "Gold: 15 celebrities; prediction: fifteen celebrities. Score 0.75."),
    "doc147_qa3__after279": (1.0, "Gold: series of posts triggering intervention; prediction matches exactly. Full credit."),
    "doc63_qa0__after280": (0.25, "Gold: convolutional layers encoder; prediction: attention-based encoder-decoder. Score 0.25."),
    "doc129_qa5__after280": (0.5, "Gold: depends on dataset; prediction: visual+textual complementary. Score 0.5."),
}

# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix -> (score, rationale)
# suffix = qid minus "qasper__attention-corpus-tuned__batch_calib__" and "__seed42"
# Only entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # B1 (entries 0-50)
    "doc0_qa0": (0.25, "Gold: labeled features; prediction: leverage prior knowledge (paraphrase). Score 0.25."),
    "doc0_qa3": (0.5, "Gold: classify despite unbalanced prior+class distribution; prediction: insensitive to quality. Score 0.5."),
    "doc1_qa1": (0.25, "Gold: claim/premise/backing/rebuttal/refutation; prediction: argument components (no names). Score 0.25."),
    "doc1_qa3": (0.5, "Gold: user comments newswire/blogs; prediction: user-generated Web discourse. Score 0.5."),
    "doc1_qa5": (0.5, "Gold: linguistic variability; prediction: variety formats/styles/contexts. Score 0.5."),
    "doc2_qa1": (0.25, "Gold: CLV as parent of role variables; prediction: CLVs used in model. Score 0.25."),
    "doc2_qa5": (0.5, "Gold: Bayesian garg2012; prediction: Bayesian model per language. Score 0.5."),
    "doc4_qa1": (0.75, "Gold: dependency edge i-i' English parse tree; prediction: expecting dependency edge i-i'. Score 0.75."),
    "doc6_qa1": (0.75, "Gold: content relevance candidate vs human summary; prediction: relevance system vs human. Score 0.75."),
    "doc6_qa2": (0.75, "Gold: ROUGE variants no high correlation; prediction: different ROUGE variants different correlations. Score 0.75."),
    "doc6_qa3": (0.25, "Gold: higher tiers of pyramid; prediction: pyramid organizes content quality. Score 0.25."),
    "doc6_qa4": (0.75, "Gold: ROUGE-Pyramid correlations weak; prediction: ROUGE not reliable. Score 0.75."),
    "doc7_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc8_qa0": (0.5, "Gold: self/opponent-coverage/number; prediction: promoting own+attacking opponents. Score 0.5."),
    "doc8_qa1": (0.75, "Gold: Intelligence Squared Debates; prediction: IQ2 debates (IQ2=Intelligence Squared). Score 0.75."),
    "doc10_qa1": (0.25, "Gold: Social Honeypot+Weibo; prediction: English Honeypot only. Score 0.25."),
    "doc10_qa2": (0.75, "Gold: LDA features binary classification; prediction: LDA-based classification. Score 0.75."),
    "doc11_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc11_qa1": (1.0, "Gold: established task; prediction: addressed earlier=established. Full credit."),
    "doc11_qa2": (1.0, "Gold: simple word-level encoder; prediction: simple word-level encoder. Full credit."),
    "doc11_qa4": (1.0, "Gold: simple word-level encoder; prediction: simple word-level encoder. Full credit."),
    "doc12_qa1": (0.75, "Gold: 30,000; prediction: over 30,000. Score 0.75."),
    "doc13_qa1": (1.0, "Gold: SemEval 2010 relation classification; prediction: SemEval 2010 relation classification. Full credit."),
    "doc13_qa4": (0.25, "Gold: two conv+maxpool adj contexts+longer text; prediction: three disjoint regions. Score 0.25."),
    "doc14_qa0": (0.25, "Gold: attentional encoder-decoder BIBREF0; prediction: phrase-based+NMT (partial). Score 0.25."),
    "doc15_qa0": (0.5, "Gold: UD v1.2 treebanks 16 languages; prediction: Chinese POS+16 datasets. Score 0.5."),
    "doc15_qa1": (0.25, "Gold: 16 languages listed; prediction: 16 languages (no list). Score 0.25."),
    "doc17_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc17_qa2": (0.25, "Gold: WSC collection; prediction: Winograd schemas (no WSC name). Score 0.25."),
    "doc19_qa1": (0.5, "Gold: predicted number redundant answers; prediction: annotated by experts or crowd by difficulty. Score 0.5."),
    # B2 (entries 50-100)
    "doc20_qa0": (0.25, "Gold: politics/business/science/AskReddit; prediction: abortion/climate/cooking. Score 0.25."),
    "doc21_qa0": (0.75, "Gold: frequencies of words co-occurring with both; prediction: second-order co-occurrence matrix. Score 0.75."),
    "doc22_qa2": (0.25, "Gold: testing humans 50 NE+50 common; prediction: human study showing space. Score 0.25."),
    "doc23_qa0": (0.75, "Gold: raw text; prediction: three preprocessing levels (gets raw text). Score 0.75."),
    "doc24_qa2": (1.0, "Gold: CNN features; prediction: CNN baseline features. Full credit."),
    "doc26_qa0": (1.0, "Gold: German-English; prediction: German-English. Full credit."),
    "doc26_qa1": (1.0, "Gold: IMDb movie review; prediction: IMDb movie review. Full credit."),
    "doc27_qa1": (0.75, "Gold: Affective Text; prediction: Affective Text+Fairy Tales+ISEAR. Score 0.75."),
    "doc28_qa5": (0.5, "Gold: SVM unigram/bigram/trigram; prediction: SVM classifiers (gets SVM). Score 0.5."),
    "doc30_qa2": (1.0, "Gold: TransE; prediction: TransE. Full credit."),
    "doc31_qa1": (0.5, "Gold: 50 annotators/100 translations; prediction: adequacy/fluency/ranking metrics. Score 0.5."),
    "doc34_qa0": (0.25, "Gold: 22,880; prediction: over 20,000. Score 0.25."),
    "doc34_qa1": (1.0, "Gold: aggregate enterprises in particular field; prediction matches. Full credit."),
    "doc34_qa4": (0.25, "Gold: 14 industry classes named; prediction: 14 classes (no names). Score 0.25."),
    "doc37_qa4": (0.25, "Gold: RED/LEN/FDUR list; prediction: eye movement cognitive features. Score 0.25."),
    "doc38_qa2": (0.75, "Gold: no evidence of depression; prediction: annotated as no evidence. Score 0.75."),
    "doc38_qa3": (0.25, "Gold: BIBREF12/13; prediction: annotated Twitter dataset 9,473. Score 0.25."),
    "doc39_qa0": (0.25, "Gold: ceccarelli2013/CoNLL 2003; prediction: benchmark entity relatedness. Score 0.25."),
    "doc39_qa1": (0.5, "Gold: CRC concept measures; prediction: CRC+3C. Score 0.5."),
    "doc39_qa3": (0.25, "Gold: 212 times; prediction: much less dimensions. Score 0.25."),
    "doc40_qa1": (0.25, "Gold: Assigning wrong words to cluster; prediction: cross-speaker UTD. Score 0.25."),
    "doc40_qa2": (0.25, "Gold: 104 telephone calls; prediction: CALLHOME corpus (no number). Score 0.25."),
    "doc41_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc42_qa0": (1.0, "Gold: 1000 hours; prediction: 1000 hours. Full credit."),
    "doc42_qa2": (0.25, "Gold: Unidirectional LSTM 2/6/7/8/9 layers; prediction: deep LSTM models. Score 0.25."),
    # B3 (entries 100-150)
    "doc44_qa3": (0.5, "Gold: multi-class classifier section; prediction: class-based section templates. Score 0.5."),
    "doc44_qa5": (0.25, "Gold: salience+positional+frequency; prediction: frequency of entity features. Score 0.25."),
    "doc45_qa1": (0.25, "Gold: Precision/Recall/F1; prediction: evaluation protocol. Score 0.25."),
    "doc45_qa3": (0.5, "Gold: description+topic+concept map; prediction: identify important by description. Score 0.5."),
    "doc45_qa4": (0.25, "Gold: DIP corpus; prediction: heterogeneous web educational. Score 0.25."),
    "doc45_qa5": (1.0, "Gold: concept map labeled graph concepts+nodes relationships+edges; prediction matches. Full credit."),
    "doc46_qa0": (0.75, "Gold: SimpleQuestions; prediction: SimpleQuestions+WebQSP. Score 0.75."),
    "doc46_qa2": (1.0, "Gold: KBQA; prediction: KBQA. Full credit."),
    "doc46_qa3": (0.25, "Gold: answer questions KB tuples; prediction: Relation detection. Score 0.25."),
    "doc48_qa1": (0.25, "Gold: clusters/seed/word vectors/window; prediction: clusters only. Score 0.25."),
    "doc48_qa2": (0.75, "Gold: selection word vectors; prediction: word vectors+clusters. Score 0.75."),
    "doc50_qa0": (1.0, "Gold: Named Entity Recognition; prediction: NER. Full credit."),
    "doc50_qa1": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc51_qa0": (0.5, "Gold: Refinement Adjustment LSTM; prediction: LSTM-based decoder. Score 0.5."),
    "doc51_qa1": (1.0, "Gold: NLG datasets; prediction: NLG datasets. Full credit."),
    "doc52_qa1": (0.5, "Gold: ROUGE/Recall/Precision/F1; prediction: Pyramid+ROUGE. Score 0.5."),
    "doc52_qa4": (0.75, "Gold: finding important sentences from story; prediction: selected by important info. Score 0.75."),
    "doc53_qa2": (0.25, "Gold: US presidential primaries; prediction: divisive political events 2015-2017. Score 0.25."),
    "doc54_qa0": (0.5, "Gold: Favor/Against Galatasaray/Fenerbahce; prediction: favor/against (no targets). Score 0.5."),
    "doc54_qa2": (1.0, "Gold: whether any hashtag in tweet; prediction: existence of hashtags. Full credit."),
    "doc54_qa4": (0.75, "Gold: Galatasaray; prediction: Galatasaray+Fenerbahce. Score 0.75."),
    "doc55_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc55_qa1": (0.25, "Gold: ASPEC; prediction: English-Japanese+Japanese-English (no ASPEC). Score 0.25."),
    "doc56_qa3": (1.0, "Gold: grouped by objective function; prediction: organized by objective function. Full credit."),
    "doc57_qa2": (1.0, "Gold: SemEval-2016; prediction: SemEval2016 subtask 4. Full credit."),
    "doc59_qa1": (1.0, "Gold: GloVe pre-trained; prediction: GloVe Wikipedia+Gigaword+Common Crawl. Full credit."),
    "doc59_qa2": (0.75, "Gold: standard accuracy; prediction: prediction accuracy. Score 0.75."),
    "doc60_qa1": (0.25, "Gold: deri2016grapheme; prediction: adapting g2p models. Score 0.25."),
    "doc60_qa2": (1.0, "Gold: PER; prediction: PER. Full credit."),
    # B4 (entries 150-200)
    "doc63_qa0": (0.25, "Gold: convolutional layers encoder; prediction: attention-based encoder-decoder. Score 0.25."),
    "doc63_qa1": (0.5, "Gold: decoder predicts target probability; prediction: weighted average. Score 0.5."),
    "doc63_qa2": (0.75, "Gold: WSJ; prediction: WSJ-SI84+WSJ-SI284. Score 0.75."),
    "doc64_qa2": (0.5, "Gold: tied through global; prediction: hierarchical structure sharing. Score 0.5."),
    "doc65_qa0": (0.75, "Gold: beyond translational equivalent; prediction: beyond alignments syntactic/morphological. Score 0.75."),
    "doc65_qa2": (0.5, "Gold: VERB/PRON POS tags; prediction: verbs. Score 0.5."),
    "doc66_qa0": (0.5, "Gold: 1-5 scale human evaluators; prediction: crowd+expert evaluations. Score 0.5."),
    "doc67_qa1": (0.75, "Gold: Naive-Bayes corrected bias; prediction: modified for unknown distribution. Score 0.75."),
    "doc68_qa0": (0.5, "Gold: LSA/TextRank/LexRank/ILP; prediction: ILP+short-text. Score 0.5."),
    "doc68_qa1": (0.75, "Gold: ROUGE unigram; prediction: ROUGE+Sera+other. Score 0.75."),
    "doc68_qa4": (0.25, "Gold: LR/MNB/RF/AdaBoost/LinSVM; prediction: SVM+LR. Score 0.25."),
    "doc68_qa5": (1.0, "Gold: 15.5; prediction: 15.5 words. Full credit."),
    "doc69_qa1": (1.0, "Gold: MLP; prediction: modified MLP. Full credit."),
    "doc70_qa0": (0.75, "Gold: followers/friends/URLs significantly different; prediction: differences followers/URLs/verification. Score 0.75."),
    "doc70_qa2": (1.0, "Gold: retweeted >1000 times; prediction: retweeted >1000. Full credit."),
    "doc70_qa3": (0.25, "Gold: unverified/recently created/high ratio; prediction: both verified+unverified. Score 0.25."),
    "doc70_qa4": (1.0, "Gold: 1000; prediction: 1000. Full credit."),
    "doc70_qa5": (1.0, "Gold: ground truth not established; prediction matches. Full credit."),
    "doc71_qa1": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc71_qa3": (0.25, "Gold: compare tasks/average number; prediction: contextual similarities. Score 0.25."),
    "doc72_qa0": (0.25, "Gold: F1 85.99/75.15/71; prediction: +1.08 improvements. Score 0.25."),
    "doc73_qa1": (1.0, "Gold: personal attack/racism/sexism; prediction matches. Full credit."),
    "doc74_qa1": (0.25, "Gold: training when unlabeled annotated; prediction: active learning increases performance. Score 0.25."),
    "doc75_qa1": (0.25, "Gold: Spearman's phrasal similarity; prediction: context-dependent scoring. Score 0.25."),
    "doc75_qa2": (0.5, "Gold: hierarchical phrase-based BIBREF29; prediction: phrase-based MT. Score 0.5."),
    "doc76_qa2": (1.0, "Gold: state speeches UN General Debate; prediction matches. Full credit."),
    "doc77_qa3": (0.5, "Gold: newly acquired facts retained KB; prediction: LiLi imitates human knowledge. Score 0.5."),
    "doc77_qa5": (0.25, "Gold: LiLi capabilities; prediction: general knowledge LiLi. Score 0.25."),
    "doc78_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc78_qa1": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc78_qa4": (1.0, "Gold: 10K image+caption pairs; prediction matches. Full credit."),
    "doc79_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    # B5 (entries 200-250)
    "doc80_qa2": (1.0, "Gold: Amazon Mechanical Turk; prediction: Amazon Mechanical Turk. Full credit."),
    "doc80_qa3": (0.25, "Gold: 3 components pilot studies; prediction: via crowdsourcing. Score 0.25."),
    "doc81_qa0": (0.5, "Gold: b-emb/CBOW/...; prediction: CBOW/PV-DM/GloVe/EqEmb. Score 0.5."),
    "doc81_qa1": (1.0, "Gold: Euclidean distance context vectors; prediction matches. Full credit."),
    "doc83_qa1": (0.5, "Gold: basic model explicit discourse; prediction: outperforms SOA on PDTB. Score 0.5."),
    "doc84_qa3": (0.75, "Gold: macro level decide which field; prediction: bifocal macro level. Score 0.75."),
    "doc86_qa1": (0.75, "Gold: mimic sequence to sequence; prediction: proxy for general S2S. Score 0.75."),
    "doc87_qa2": (0.75, "Gold: n-gram subwords; prediction: n-grams+unsupervised morphemes. Score 0.75."),
    "doc87_qa3": (0.75, "Gold: weighted factorization word-context; prediction: explicit matrix factorization. Score 0.75."),
    "doc88_qa1": (0.5, "Gold: race/gender-associated words; prediction: tease out biases. Score 0.5."),
    "doc89_qa0": (0.25, "Gold: 2.6% higher success rate; prediction: NUS outperformed ABUS. Score 0.25."),
    "doc89_qa1": (1.0, "Gold: DSTC2; prediction: DSTC2. Full credit."),
    "doc90_qa0": (0.25, "Gold: Back Translation; prediction: data augmentation+BPE. Score 0.25."),
    "doc92_qa0": (0.25, "Gold: subtask; prediction: four Spanish subtasks. Score 0.25."),
    "doc92_qa2": (0.25, "Gold: Apertium; prediction: translated from other languages. Score 0.25."),
    "doc92_qa3": (0.75, "Gold: tweets with intensity labels; prediction: tweets predicting intensity. Score 0.75."),
    "doc92_qa5": (0.5, "Gold: first train then annotate; prediction: semi-supervised assigns tweets. Score 0.5."),
    "doc93_qa0": (1.0, "Gold: CNN; prediction: CNN. Full credit."),
    "doc93_qa1": (0.5, "Gold: decoder with whole sequence context; prediction: encoder LSTM context vector. Score 0.5."),
    "doc93_qa3": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc94_qa0": (0.25, "Gold: Reuters-8 without stop words; prediction: Reuters database. Score 0.25."),
    "doc94_qa1": (0.75, "Gold: word vectors in context same class; prediction: word subspace variability. Score 0.75."),
    "doc95_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc96_qa2": (0.25, "Gold: SemEval-2016 Task 5; prediction: sentiment reviews (no SemEval name). Score 0.25."),
    "doc96_qa3": (0.75, "Gold: majority baseline; prediction: majority+lexicon. Score 0.75."),
    "doc96_qa4": (0.25, "Gold: Google translation API; prediction: machine translation. Score 0.25."),
    "doc96_qa5": (0.25, "Gold: Amazon reviews; prediction: product reviews English. Score 0.25."),
    "doc97_qa0": (0.75, "Gold: ambiguous or too common; prediction: unresolved ambiguities. Score 0.75."),
    "doc98_qa1": (0.75, "Gold: humor identification as classification; prediction: humor detection classification. Score 0.75."),
    "doc98_qa2": (1.0, "Gold: three; prediction: Three annotators. Full credit."),
    "doc98_qa3": (0.25, "Gold: sports/politics/entertainment; prediction: code-mixed social media. Score 0.25."),
    "doc99_qa0": (0.75, "Gold: words user wants in output; prediction: user-defined keywords DMK. Score 0.75."),
    "doc99_qa3": (1.0, "Gold: roughly 40,000 Manhattan listings; prediction matches. Full credit."),
    "doc100_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc100_qa1": (0.5, "Gold: word intrusion test interpretability; prediction: interpretability scores. Score 0.5."),
    "doc100_qa2": (0.5, "Gold: dimension corresponding to concept; prediction: along specified dimension. Score 0.5."),
    # B6 (entries 250-300)
    "doc100_qa3": (1.0, "Gold: additive modification cost function; prediction: additive modification objective. Full credit."),
    "doc101_qa0": (0.75, "Gold: RNN-based NMT; prediction: RNN-based NMT+Transformer. Score 0.75."),
    "doc101_qa1": (1.0, "Gold: ancient Chinese history 1000BC-200BC; prediction: ancient Chinese 1000BC-200BC. Full credit."),
    "doc103_qa0": (0.5, "Gold: WSJ Penn Treebank; prediction: Penn Treebank (missing WSJ). Score 0.5."),
    "doc103_qa2": (0.75, "Gold: neural projector must be invertible; prediction: invertibility condition exact inference. Score 0.75."),
    "doc104_qa2": (0.75, "Gold: Naive Bayes; prediction: BiGRU/NB/SVM (includes NB). Score 0.75."),
    "doc107_qa0": (0.25, "Gold: PDTB taggers; prediction: SVM+BiLSTMs. Score 0.25."),
    "doc108_qa0": (0.5, "Gold: multilingual alternating languages; prediction: encoder-decoder multilingual. Score 0.5."),
    "doc108_qa5": (0.5, "Gold: V/PST/V.PCTP/PASS tags; prediction: MSD morpho-syntactic (no specific tags). Score 0.5."),
    "doc109_qa1": (0.5, "Gold: SNLI best model 87.4% accuracy; prediction: new SOA SNLI+Quora. Score 0.5."),
    "doc109_qa2": (0.75, "Gold: plain stacked LSTMs; prediction: typical stacked LSTMs. Score 0.75."),
    "doc109_qa3": (0.75, "Gold: SNLI+MultiNLI; prediction: SNLI/MultiNLI/Quora/SST (includes gold). Score 0.75."),
    "doc110_qa0": (0.5, "Gold: Tree-LSTM variants; prediction: tree-structured+neural models. Score 0.5."),
    "doc111_qa3": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc112_qa0": (0.75, "Gold: QG model scores candidates; prediction: QG model in framework. Score 0.75."),
    "doc113_qa0": (0.75, "Gold: AutoJudge outperforms all baselines; prediction: significant improvement SOA. Score 0.75."),
    "doc113_qa2": (1.0, "Gold: divorce; prediction: divorce proceedings. Full credit."),
    "doc114_qa2": (0.75, "Gold: multi-turn answer module span detector; prediction: multi-turn+bilinear. Score 0.75."),
    "doc115_qa0": (1.0, "Gold: English; prediction: English. Full credit."),
    "doc115_qa2": (0.25, "Gold: improvement 2.11 BLEU; prediction: substantial improvements. Score 0.25."),
    "doc115_qa4": (0.75, "Gold: 89,042 train/100 test WikiSmall; prediction: WikiSmall 89,042+100. Score 0.75."),
    "doc116_qa0": (0.75, "Gold: use text transcription; prediction: speech transcription. Score 0.75."),
    "doc116_qa1": (0.5, "Gold: MDREA outperforms; prediction: outperforms 68.8-71.8%. Score 0.5."),
    "doc116_qa2": (0.25, "Gold: feed-forward neural; prediction: dual recurrent neural. Score 0.25."),
    "doc117_qa0": (0.25, "Gold: average unique predictions; prediction: new metrics variable phrases. Score 0.25."),
    "doc117_qa5": (0.25, "Gold: average unique predictions; prediction: two new metrics. Score 0.25."),
    "doc118_qa1": (1.0, "Gold: ilur.am; prediction: ilur.am. Full credit."),
    "doc118_qa2": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc119_qa0": (0.25, "Gold: Demographics/Age/DiagnosisHistory list; prediction: labeled topics. Score 0.25."),
    "doc119_qa2": (0.25, "Gold: MIMIC-III; prediction: 1102 Discharge+1000 Nursing ICU. Score 0.25."),
    "doc120_qa0": (0.75, "Gold: BOW-Tags; prediction: bag-of-words nearby tags. Score 0.75."),
    # B7 (entries 300-350)
    "doc121_qa0": (0.5, "Gold: 8 tasks different competencies; prediction: relational questions linked to competencies. Score 0.5."),
    "doc121_qa4": (1.0, "Gold: TripAdvisor; prediction: TripAdvisor. Full credit."),
    "doc122_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc124_qa0": (0.75, "Gold: Document to Vector (Doc2Vec); prediction: paragraph vectors. Score 0.75."),
    "doc125_qa1": (1.0, "Gold: NCEL consistently outperforms baselines favorable generalization; prediction matches. Full credit."),
    "doc125_qa2": (0.25, "Gold: Macro F1 at document level; prediction: evaluate linking+generalization. Score 0.25."),
    "doc129_qa1": (1.0, "Gold: fine-tuning Inception V3; prediction: joint model with Inception V3. Full credit."),
    "doc129_qa5": (0.5, "Gold: depends on dataset; prediction: visual+textual complementary. Score 0.5."),
    "doc131_qa1": (0.75, "Gold: Logistic Regression; prediction: deep learning+logistic regression. Score 0.75."),
    "doc132_qa0": (0.75, "Gold: sentences disarranged; prediction: not arranged in logical order. Score 0.75."),
    "doc133_qa0": (0.5, "Gold: similar to cloze task in BERT pre-training; prediction: mask allows refine draft. Score 0.5."),
    "doc134_qa4": (0.25, "Gold: profanity/swearing/insults/cyberbullying/hate speech; prediction: hate speech/cyberbullying/cybertrolling. Score 0.25."),
    "doc134_qa8": (0.25, "Gold: Level A Offensive language Detection; prediction: detect/categorize/identify target. Score 0.25."),
    "doc135_qa1": (0.5, "Gold: nurse-initiated telephone congestive heart failure; prediction: minimal nurse-to-patient conversations. Score 0.5."),
    "doc135_qa2": (0.25, "Gold: time/activities list; prediction: query symptom+attribute. Score 0.25."),
    "doc137_qa3": (0.25, "Gold: uni-directional augment decoder; prediction: BERT+seq2seq. Score 0.25."),
    "doc138_qa2": (0.25, "Gold: grammatical/spelling/word order errors; prediction: error types+style transfer. Score 0.25."),
    "doc139_qa0": (0.25, "Gold: modified copy of target pseudo-training; prediction: cheaper data simulation. Score 0.25."),
    "doc139_qa5": (0.5, "Gold: English; prediction: monolingual target language. Score 0.5."),
    # B8 (entries 350-400)
    "doc141_qa1": (0.5, "Gold: read speech best model novel results; prediction: competitive on WSJ+Hub5'00. Score 0.5."),
    "doc142_qa0": (0.25, "Gold: T-SNE plot; prediction: EEG discriminative information. Score 0.25."),
    "doc142_qa1": (1.0, "Gold: consonants/nasal/bilabial/high-front list; prediction matches. Full credit."),
    "doc142_qa3": (0.25, "Gold: 7 phonemic/syllabic sounds; prediction: imagined speech sounds. Score 0.25."),
    "doc142_qa5": (1.0, "Gold: 14; prediction: 14 participants. Full credit."),
    "doc143_qa0": (1.0, "Gold: Exact Match (EM); prediction: EM (Exact Match). Full credit."),
    "doc143_qa1": (0.25, "Gold: EM 51.10 F1 63.11; prediction: 2% improvement (no numbers). Score 0.25."),
    "doc143_qa3": (1.0, "Gold: Spoken-SQuAD testing set; prediction: Spoken-SQuAD testing set. Full credit."),
    "doc144_qa2": (0.5, "Gold: survey among engineers; prediction: engineers significant overhead. Score 0.5."),
    "doc145_qa2": (0.75, "Gold: entities linked to Wikidata; prediction: ELMo+Wikidata augment entities. Score 0.75."),
    "doc146_qa0": (0.75, "Gold: improvement expert+difficult subset; prediction: expert higher quality, augmenting helps. Score 0.75."),
    "doc146_qa1": (0.5, "Gold: experts if already collected; prediction: routing difficult to experts. Score 0.5."),
    "doc146_qa4": (1.0, "Gold: sentence; prediction: instance is a sentence. Full credit."),
    "doc147_qa1": (0.25, "Gold: context inference; prediction: attention discussion propagation. Score 0.25."),
    "doc147_qa3": (1.0, "Gold: series of posts that trigger intervention; prediction matches. Full credit."),
    "doc148_qa0": (0.75, "Gold: distinct recognition outputs attacker can induce; prediction: unique outputs adversarial. Score 0.75."),
    "doc148_qa1": (0.5, "Gold: sentiment analysis+paraphrase detection; prediction: text classification+sentiment. Score 0.5."),
    "doc148_qa2": (0.5, "Gold: ScRNN first+last character; prediction: semicharacter RNN sub-word. Score 0.5."),
    "doc148_qa4": (1.0, "Gold: adversarial misspellings real-world problem; prediction matches. Full credit."),
    "doc148_qa6": (0.5, "Gold: pass-through passes misspelled word; prediction: backoff handles rare. Score 0.5."),
    "doc149_qa0": (0.5, "Gold: CDA; prediction: counterfactual data augmentation (CDA by description). Score 0.5."),
    "doc149_qa2": (1.0, "Gold: gendered pairs he/she; prediction: he/she. Full credit."),
    "doc152_qa0": (0.75, "Gold: perceptual illusion listening while watching; prediction: perception influenced by seeing. Score 0.75."),
    "doc154_qa1": (0.75, "Gold: baseline transformer BIBREF8; prediction: NMT baseline+Transformer. Score 0.75."),
    "doc155_qa2": (1.0, "Gold: 94%-97% accuracy; prediction: around 94-97%. Full credit."),
    "doc156_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc156_qa1": (0.5, "Gold: English-German dataset; prediction: multimodal MT tasks. Score 0.5."),
    "doc157_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc157_qa1": (0.5, "Gold: GRU encoder/interaction/classifier; prediction: attention+conflict 2-head. Score 0.5."),
    "doc157_qa2": (0.25, "Gold: Task 1 Quora Duplicate Question Pair; prediction: Task 1 (no name). Score 0.25."),
    # B9 (entries 400-450)
    "doc162_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc163_qa0": (0.75, "Gold: BLEU-1; prediction: BLEU-1/Meteor/Rouge-L. Score 0.75."),
    "doc163_qa1": (1.0, "Gold: 13,757; prediction: includes 13,757. Full credit."),
    "doc163_qa2": (0.5, "Gold: archived CNN/NYT snapshots; prediction: crawling news articles. Score 0.5."),
    "doc164_qa4": (1.0, "Gold: context and sequential nature; prediction: GRU captures sequential+context. Full credit."),
    "doc165_qa1": (0.25, "Gold: alternately one mini-batch; prediction: CVT semi-supervised. Score 0.25."),
    "doc166_qa2": (0.5, "Gold: avg dissimilarity all tag pairs; prediction: recommendation diversity. Score 0.5."),
    "doc166_qa4": (0.25, "Gold: 48,705 e-books 13 publishers; prediction: editor tags+Amazon (no numbers). Score 0.25."),
    "doc166_qa5": (0.75, "Gold: popularity-based; prediction: popularity-based+similarity+... Score 0.75."),
    "doc168_qa2": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc168_qa6": (1.0, "Gold: daily newspaper 2015-2016; prediction: daily newspapers 2015-2016. Full credit."),
    "doc168_qa8": (0.25, "Gold: OurNepali 3 types ILPRL 4 types; prediction: PER/LOC/ORG. Score 0.25."),
    "doc168_qa10": (0.5, "Gold: grapheme-level achieves; prediction: grapheme outperforms character. Score 0.5."),
    "doc168_qa11": (0.25, "Gold: BiLSTM/BiLSTM+CNN list; prediction: SVM/HMM/neural. Score 0.25."),
    "doc169_qa3": (0.75, "Gold: Friends TV sitcom; prediction: Friends+EmotionPush. Score 0.75."),
    "doc169_qa4": (0.25, "Gold: Ekman's six basic emotions; prediction: anger/joy/neutral/sadness. Score 0.25."),
    "doc170_qa4": (0.75, "Gold: framework algorithms for NNs; prediction: method graph NN. Score 0.75."),
    "doc171_qa0": (1.0, "Gold: clipped PMI; NNEGPMI; prediction: NNEGPMI and clipped PMI. Full credit."),
    "doc171_qa2": (0.5, "Gold: poor rare word+word analogies; prediction: loss of negative PMI info. Score 0.5."),
    "doc171_qa3": (0.5, "Gold: PMI→-inf when unobserved; prediction: unreliable finite corpora. Score 0.5."),
    "doc172_qa1": (0.5, "Gold: ARAML improved over all baselines; prediction: better than GAN baselines. Score 0.5."),
    "doc173_qa1": (0.25, "Gold: context-gloss pairs all senses; prediction: incorporate WordNet. Score 0.25."),
    "doc173_qa2": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc173_qa3": (1.0, "Gold: small BERT; prediction: BERT-Base. Full credit."),
    "doc173_qa4": (0.5, "Gold: converts WSD to sequence learning; prediction: accommodates unknown senses. Score 0.5."),
    "doc174_qa0": (0.25, "Gold: BERT 512 limit overcome by; prediction: document-level BERT encoder. Score 0.25."),
    "doc175_qa1": (0.5, "Gold: create fair systems; prediction: study gender imbalance impact on ASR. Score 0.5."),
    "doc175_qa3": (1.0, "Gold: Anchors and Punctual speakers; prediction: Anchor and Punctual speakers. Full credit."),
    # B10 (entries 450-500)
    "doc175_qa4": (0.25, "Gold: 33.16%; prediction: under-represented (no %). Score 0.25."),
    "doc175_qa5": (0.75, "Gold: ESTER1; prediction: ESTER1/ESTER2/ETAPE/REPERE. Score 0.75."),
    "doc176_qa2": (0.25, "Gold: FSD/Twitter/Google; prediction: two Twitter+news article. Score 0.25."),
    "doc176_qa3": (1.0, "Gold: generator network capture event patterns; prediction: AEM generator event patterns. Full credit."),
    "doc176_qa4": (0.75, "Gold: neural network flexibility; prediction: generator learns complex distributions. Score 0.75."),
    "doc177_qa1": (0.5, "Gold: Spearman cosine-similarity; prediction: cosine-similarity+logistic regression. Score 0.5."),
    "doc177_qa2": (1.0, "Gold: 20 minutes; prediction: less than 20 minutes. Full credit."),
    "doc177_qa5": (0.5, "Gold: GloVe/BERT/USE/TF-IDF/InferSent; prediction: InferSent+USE. Score 0.5."),
    "doc178_qa1": (0.5, "Gold: ELMo-transformer+mSynC similar; prediction: doesn't perform better than ELMo. Score 0.5."),
    "doc178_qa3": (0.5, "Gold: modest gains three/four tasks; prediction: no significant gain four. Score 0.5."),
    "doc178_qa4": (0.5, "Gold: token-level chunk label embeddings; prediction: chunk boundary shallow syntactic. Score 0.5."),
    "doc179_qa2": (0.5, "Gold: attention heads can choose; prediction: heads choose between focusing. Score 0.5."),
    "doc180_qa0": (1.0, "Gold: BPE PPL/BLEU-1/4/ROUGE-L/per; prediction: BPE perplexity/BLEU-1/4/ROUGE-L/distinct. Full credit."),
    "doc180_qa1": (0.5, "Gold: English; prediction: natural text. Score 0.5."),
    "doc180_qa2": (0.25, "Gold: coherence 1.78-1.82; prediction: plausible+personalized (no numbers). Score 0.25."),
    "doc180_qa4": (1.0, "Gold: from Food.com; prediction: Food.com historical preferences. Full credit."),
    "doc180_qa5": (1.0, "Gold: from Food.com; prediction: Food.com. Full credit."),
    "doc181_qa3": (0.75, "Gold: tags words before+after pun; prediction: three tags to capture property. Score 0.75."),
    "doc182_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc182_qa1": (0.75, "Gold: low word importance measured; prediction: gradient-based comparing. Score 0.75."),
    "doc182_qa2": (0.75, "Gold: contribution matrix word importance; prediction: contribution of input words. Score 0.75."),
    "doc182_qa3": (1.0, "Gold: Transformer+RNN-Search; prediction: Transformer+RNN-Search. Full credit."),
    "doc183_qa0": (0.5, "Gold: self-similarity/intra-sentence/max expressiveness; prediction: intra-sentence. Score 0.5."),
    "doc183_qa1": (1.0, "Gold: first principal component; prediction: first principal component. Full credit."),
    "doc184_qa3": (1.0, "Gold: 8 languages list; prediction: matches 8 languages. Full credit."),
    "doc185_qa0": (0.75, "Gold: antisocial event or not; prediction: labels for antisocial events. Score 0.75."),
    "doc185_qa1": (0.75, "Gold: Conversations Gone Awry; prediction: CGA+ChangeMyView. Score 0.75."),
    "doc186_qa1": (0.5, "Gold: BIBREF11 MT system; prediction: context-agnostic sentence-level. Score 0.5."),
    "doc186_qa2": (0.25, "Gold: deixis/lexical cohesion/VP ellipsis list; prediction: discourse phenomena. Score 0.25."),
    "doc187_qa0": (0.5, "Gold: intents annotated manually; prediction: 23,700 queries 150 intents. Score 0.5."),
    "doc187_qa1": (0.75, "Gold: SVM; prediction: 1NN+SVM. Score 0.75."),
    "doc187_qa2": (1.0, "Gold: 23,700; prediction: 23,700. Full credit."),
    "doc189_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc190_qa0": (0.25, "Gold: RTE 4% absolute; prediction: notable improvement BERT. Score 0.25."),
    # B11 (entries 500-550)
    "doc190_qa2": (0.75, "Gold: BERTbase; prediction: pre-trained BERT. Score 0.75."),
    "doc191_qa2": (1.0, "Gold: Pointer-Gen; prediction: Pointer-Gen baseline. Full credit."),
    "doc191_qa4": (0.75, "Gold: classifying sensational vs non-sensational; prediction: classifying clickbait. Score 0.75."),
    "doc193_qa1": (0.25, "Gold: proposed ontology populated with; prediction: evaluating relevance retrieved answers. Score 0.25."),
    "doc194_qa1": (0.75, "Gold: Linguistic; prediction: Linguistic+layout features. Score 0.75."),
    "doc194_qa5": (0.5, "Gold: output layer per task; prediction: multi-tasking jointly. Score 0.5."),
    "doc195_qa1": (0.75, "Gold: irony accuracy; prediction: harmonic mean irony+sentiment. Score 0.75."),
    "doc195_qa2": (0.25, "Gold: obscure and hard to understand; prediction: lack dataset+challenge. Score 0.25."),
    "doc195_qa3": (0.5, "Gold: classifier for ironic sentences; prediction: 2M tweets 262K ironic. Score 0.5."),
    "doc195_qa4": (1.0, "Gold: irony judged by human; sentiment+content; prediction matches. Full credit."),
    "doc197_qa0": (0.25, "Gold: ROUGE scores Arxiv/PubMed; prediction: outperforms on ROUGE (no numbers). Score 0.25."),
    "doc197_qa1": (0.75, "Gold: global=whole document; prediction: global=whole document. Score 0.75."),
    "doc199_qa0": (0.75, "Gold: paragraphs; prediction: paragraphs and lines. Score 0.75."),
    "doc199_qa1": (0.75, "Gold: font type; prediction: font type and font style. Score 0.75."),
    "doc200_qa1": (0.5, "Gold: language modeling objective; prediction: creative text generation. Score 0.5."),
    "doc200_qa2": (0.75, "Gold: 740 English poems; prediction: 740 English poems+14,950. Score 0.75."),
    "doc201_qa0": (0.5, "Gold: LDA approaches recommendation; prediction: LDA+Gibbs on ISWC+WWW. Score 0.5."),
    "doc202_qa2": (1.0, "Gold: pre-trained multi-BERT; prediction: multi-BERT. Full credit."),
    "doc202_qa3": (0.25, "Gold: official BERT training script; prediction: language-independent strategies. Score 0.25."),
    "doc203_qa0": (0.75, "Gold: username; prediction: username+display name+profile image. Score 0.75."),
    "doc203_qa1": (1.0, "Gold: political party names in profile; prediction: political party names. Full credit."),
    "doc203_qa2": (1.0, "Gold: influential leaders more change; prediction: political handles more changes. Full credit."),
    "doc204_qa0": (0.75, "Gold: prior KD ineffective; prediction: prior KD ineffective different vocab. Score 0.75."),
    "doc205_qa2": (1.0, "Gold: multi-attention+projected layer; prediction: multi-attention+projected layer. Full credit."),
    "doc205_qa3": (0.5, "Gold: classic/avgRNN/attentionRNN/multiattention; prediction: multi+single attention. Score 0.5."),
    "doc205_qa4": (0.5, "Gold: Twitter dataset by organizers; prediction: OLID. Score 0.5."),
    "doc205_qa5": (0.75, "Gold: indirect/sexual/physical harassment; prediction: indirect/information/sexual/physical. Score 0.75."),
    "doc207_qa0": (0.5, "Gold: BLEU; prediction: automatic evaluation metrics. Score 0.5."),
    "doc208_qa0": (0.25, "Gold: actor-critic architecture; prediction: CNN-RNN+language style. Score 0.25."),
    "doc208_qa3": (1.0, "Gold: content score 3.7; prediction: content 3.7 creativity 3.9. Full credit."),
    "doc208_qa5": (0.25, "Gold: seq2seq global attention best; prediction: BLEU 45.9 (no model). Score 0.25."),
    "doc209_qa2": (0.5, "Gold: decoding lattice level; prediction: language model combination. Score 0.5."),
    # B12 (entries 550-600)
    "doc210_qa0": (0.75, "Gold: OpenIE+heuristic rules; prediction: Open IE extract structured. Score 0.75."),
    "doc210_qa2": (0.5, "Gold: BLEU-1/2/3/4/METEOR; prediction: BLEU-1/Meteor/Rouge-L. Score 0.5."),
    "doc210_qa3": (0.75, "Gold: SQuAD; prediction: includes SQuAD in list. Score 0.75."),
    "doc211_qa3": (0.75, "Gold: group from single account consecutive; prediction: groups read together. Score 0.75."),
    "doc211_qa4": (0.75, "Gold: Sentiment; prediction: sentiment+morality+other. Score 0.75."),
    "doc211_qa7": (0.5, "Gold: word embeddings/style/morality; prediction: sentiment/morality/text. Score 0.5."),
    "doc211_qa9": (0.5, "Gold: sorted sequence labeled by label; prediction: group read together. Score 0.5."),
    "doc212_qa0": (1.0, "Gold: CJFA encoder; prediction: Contextual Joint Factor Analysis encoder. Full credit."),
    "doc213_qa1": (0.5, "Gold: human viewers impressions; prediction: HLAs with dialogue data. Score 0.5."),
    "doc213_qa3": (0.75, "Gold: Poly-encoder BIBREF7; prediction: includes Poly-encoder. Score 0.75."),
    "doc214_qa1": (0.75, "Gold: news articles free-text; prediction: PTC 350 news articles. Score 0.75."),
    "doc216_qa1": (0.25, "Gold: implicit hatred tweets; prediction: captures annotation biases. Score 0.25."),
    "doc216_qa4": (0.5, "Gold: BERT fine-tuning; prediction: BERT embedding layers. Score 0.5."),
    "doc216_qa5": (0.5, "Gold: systematic+substantial racial biases; prediction: biases annotation/collection. Score 0.5."),
    "doc216_qa6": (0.75, "Gold: annotation biases disrespectful words; prediction: captures biases. Score 0.75."),
    "doc218_qa1": (0.75, "Gold: SVM; prediction: NB+SVM. Score 0.75."),
    "doc218_qa2": (0.75, "Gold: 30 terms ~15 samples; prediction: 15 samples each class. Score 0.75."),
    "doc219_qa1": (0.75, "Gold: adjacent chars localness; prediction: Gaussian-masked directional localness. Score 0.75."),
    "doc220_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc220_qa1": (0.5, "Gold: SOA exploit spurious patterns; prediction: non-experts found weaknesses. Score 0.5."),
    "doc220_qa3": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc221_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc221_qa1": (1.0, "Gold: Yes; prediction: double annotated. Full credit."),
    "doc221_qa2": (0.75, "Gold: legal training individuals; prediction: seven experts legal training. Score 0.75."),
    "doc222_qa0": (0.75, "Gold: crawling+pre-processing OSG forum; prediction: OSG online asynchronous. Score 0.75."),
    "doc223_qa1": (1.0, "Gold: +0.29 CoNLL2003 +0.96 OntoNotes; prediction: +0.29/+0.96. Full credit."),
    "doc223_qa2": (1.0, "Gold: +1.86 CTB5; prediction: +1.86 CTB5. Full credit."),
    "doc223_qa3": (0.5, "Gold: weight per example; prediction: dynamically adjusted easy-negatives. Score 0.5."),
    "doc224_qa1": (1.0, "Gold: MLP/NBC/SVM; prediction: MLP/NBC/SVM. Full credit."),
    "doc225_qa0": (1.0, "Gold: negated LAMA constructed by; prediction: extended LAMA negated LAMA. Full credit."),
    "doc226_qa0": (0.25, "Gold: Spearman GM_KL benchmark; prediction: qualitative experiments benchmark. Score 0.25."),
    # B13 (entries 600-650)
    "doc226_qa1": (0.5, "Gold: GM_KL better correlation; prediction: semantic similarity improves. Score 0.5."),
    "doc227_qa2": (0.5, "Gold: completion times+accuracies; prediction: efficiency/accuracy/interpretability. Score 0.5."),
    "doc228_qa3": (0.25, "Gold: average classification accuracy; prediction: evaluated on short texts. Score 0.25."),
    "doc228_qa6": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc230_qa3": (0.25, "Gold: humans post-hoc evaluation; prediction: training more transparent. Score 0.25."),
    "doc230_qa4": (0.75, "Gold: significant improvements; prediction: empirically real-world datasets. Score 0.75."),
    "doc230_qa5": (1.0, "Gold: workers find microposts model predicts; prediction matches. Full credit."),
    "doc231_qa0": (0.75, "Gold: read normally no special instructions; prediction: own speed control pad. Score 0.75."),
    "doc231_qa2": (0.75, "Gold: Wikipedia corpus; prediction: 739 Wikipedia sentences. Score 0.75."),
    "doc232_qa2": (0.75, "Gold: Europarl; prediction: Europarl+MultiUN. Score 0.75."),
    "doc234_qa1": (1.0, "Gold: automatic+human evaluation; prediction: automatic+human. Full credit."),
    "doc235_qa3": (0.5, "Gold: three setups different speaker numbers; prediction: text-dependent/prompted/independent. Score 0.5."),
    "doc236_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc236_qa1": (0.75, "Gold: both inspire avoiding fear; prediction: both inspire readers. Score 0.75."),
    "doc236_qa2": (0.25, "Gold: crowd-annotated matrix×emotion-word; prediction: lexical-base emotion. Score 0.25."),
    "doc236_qa3": (0.25, "Gold: topic modeling+unsupervised emotion; prediction: similarities analyzed. Score 0.25."),
    "doc238_qa0": (0.5, "Gold: hybrid NER F1 0.995; prediction: better on mixture. Score 0.5."),
    "doc239_qa0": (0.75, "Gold: No; prediction: paper does not mention (effectively No). Score 0.75."),
    "doc239_qa2": (1.0, "Gold: seeker interacts real conversational interface; prediction matches. Full credit."),
    "doc239_qa4": (1.0, "Gold: text/speech/image/click; prediction: text/speech/image/click. Full credit."),
    "doc240_qa3": (1.0, "Gold: ratio observed vs optimal; prediction: ratio observed vs optimal. Full credit."),
    "doc242_qa1": (1.0, "Gold: MULTIPLE CHOICE QA; prediction: multiple choice QA. Full credit."),
    "doc242_qa2": (1.0, "Gold: 1-hop to 2-hops; prediction: accuracy decreases 1-hop to 2-hops. Full credit."),
    "doc243_qa0": (0.75, "Gold: word2vec for SVM; prediction: word2vec approach. Score 0.75."),
    "doc243_qa1": (0.25, "Gold: noisier+shorter tweets; prediction: domain-specific vectors. Score 0.25."),
    "doc243_qa2": (0.25, "Gold: 20,244 positive/negative; prediction: movie+Twitter Turkish. Score 0.25."),
    "doc243_qa4": (0.5, "Gold: domain-specific embeddings; prediction: word2vec+dictionary. Score 0.5."),
    "doc243_qa5": (0.5, "Gold: (+1/-1) polarities; prediction: supervised scores polarities. Score 0.5."),
    # B14 (entries 650-700)
    "doc244_qa0": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc245_qa0": (0.25, "Gold: Max Entropy to SVM; prediction: rule-based+deep learning. Score 0.25."),
    "doc245_qa2": (0.25, "Gold: BF/BA/SFU/Sherlock; prediction: BioScope+SFU. Score 0.25."),
    "doc246_qa2": (0.75, "Gold: kernel function map data; prediction: RKS feature maps. Score 0.75."),
    "doc248_qa2": (0.25, "Gold: metric fits gold rank; prediction: annotated data objective reasoning. Score 0.25."),
    "doc249_qa0": (0.75, "Gold: IMDB+Yelp three datasets; prediction: IMDB+shorter Yelp. Score 0.75."),
    "doc249_qa1": (0.75, "Gold: architecture of classifier; prediction: architecture+sentence length. Score 0.75."),
    "doc250_qa1": (0.5, "Gold: keeps track previously visited; prediction: high reward trajectories extraction. Score 0.5."),
    "doc250_qa3": (0.5, "Gold: almost half unseen games; prediction: generalizes better unseen. Score 0.5."),
    "doc252_qa1": (0.75, "Gold: Farasa; prediction: RDI+Farasa+others. Score 0.75."),
    "doc252_qa3": (0.75, "Gold: POS/gender/number/stem POS; prediction: POS/gender/number/morphological. Score 0.75."),
    "doc253_qa1": (0.25, "Gold: berard2018end 3 decoder layers; prediction: end-to-end multilingual. Score 0.25."),
    "doc253_qa2": (0.25, "Gold: voice clips reading from bank; prediction: sentence-level without alignments. Score 0.25."),
    "doc253_qa3": (0.75, "Gold: French/German/Dutch/Russian/Spanish/Italian list; prediction matches+more. Score 0.75."),
    "doc253_qa4": (0.5, "Gold: professional translators; prediction: measured translations. Score 0.5."),
    "doc253_qa6": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc254_qa0": (1.0, "Gold: All India Radio actors read; prediction: All India Radio actor read. Full credit."),
    "doc254_qa1": (0.5, "Gold: extension NetVLAD; prediction: GhostVLAD pooling strategy. Score 0.5."),
    "doc255_qa1": (1.0, "Gold: BioASQ; prediction: BioASQ 5b. Full credit."),
    "doc256_qa2": (0.75, "Gold: car; prediction: physical attributes of cars. Score 0.75."),
    "doc256_qa5": (1.0, "Gold: KNN/RF/SVM/MLP; prediction: KNN/RF/SVM/MLP. Full credit."),
    "doc256_qa6": (0.5, "Gold: we do not know exactly; prediction: abstract language. Score 0.5."),
    "doc257_qa1": (0.25, "Gold: 20 evaluators from institution; prediction: doctor/librarian/researcher. Score 0.25."),
    "doc257_qa2": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc257_qa3": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc258_qa0": (0.25, "Gold: precision/recall/F1; prediction: Annodis automatic evaluations. Score 0.25."),
    "doc260_qa1": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc260_qa3": (1.0, "Gold: bilingual LM target-specific; prediction: bilingual LM from BERT. Full credit."),
    # B15 (entries 700-752)
    "doc262_qa1": (0.25, "Gold: backward greedy search; prediction: boundary assembling DNN Chinese NER. Score 0.25."),
    "doc263_qa0": (1.0, "Gold: mainstream+disinformation; prediction: mainstream+disinformation. Full credit."),
    "doc263_qa1": (1.0, "Gold: political bias label+training; prediction: assigning political bias label. Full credit."),
    "doc263_qa2": (0.75, "Gold: US dataset; prediction: United States and another. Score 0.75."),
    "doc263_qa3": (0.75, "Gold: SCC count; prediction: traditional indicators density/strong/weak. Score 0.75."),
    "doc264_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc265_qa0": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc265_qa1": (0.25, "Gold: LGBTQ word list; prediction: LGBTQ NYT framework. Score 0.25."),
    "doc265_qa2": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc266_qa1": (1.0, "Gold: 15 clinical patient phenotypes; prediction: 15 clinical patient phenotypes. Full credit."),
    "doc267_qa1": (0.25, "Gold: NER model; prediction: logistic regression+deep learning. Score 0.25."),
    "doc267_qa2": (1.0, "Gold: No; prediction: No. Full credit."),
    "doc267_qa3": (0.75, "Gold: MEDDOCAN; prediction: NUBes-PHI+MEDDOCAN. Score 0.75."),
    "doc268_qa1": (0.75, "Gold: Restrictivity; prediction: absence of semantics-altering modifiers. Score 0.75."),
    "doc269_qa0": (0.25, "Gold: integer 0-6; prediction: human judgments semantic similarity. Score 0.25."),
    "doc269_qa1": (0.25, "Gold: 12 languages list; prediction: Mandarin/Spanish/Russian. Score 0.25."),
    "doc271_qa1": (0.25, "Gold: M-BERT 76.6 F1; prediction: zero-shot transfer. Score 0.25."),
    "doc271_qa2": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc271_qa3": (0.25, "Gold: BERT 76.6 x-stance; prediction: zero-shot transfer. Score 0.25."),
    "doc272_qa1": (0.5, "Gold: Pseudo-perplexity; prediction: BERT joint probability. Score 0.5."),
    "doc273_qa0": (0.25, "Gold: intentional+multicast; prediction: sentences with propaganda technique. Score 0.25."),
    "doc273_qa1": (0.75, "Gold: precision; prediction: precision/recall/F1. Score 0.75."),
    "doc274_qa0": (0.75, "Gold: dimension/window/architecture/algorithm/epochs/hidden; prediction: dimension/epochs/window/vocab. Score 0.75."),
    "doc274_qa2": (1.0, "Gold: IMDb movie reviews; prediction: IMDb movie reviews. Full credit."),
    "doc274_qa3": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc275_qa1": (0.5, "Gold: CRF; prediction: BioBERT/BiLSTM-CRF. Score 0.5."),
    "doc275_qa3": (0.75, "Gold: doctors linguistic annotation tool; prediction: doctors medical knowledge. Score 0.75."),
    "doc275_qa5": (0.75, "Gold: CRF; prediction: BioBERT/MTL/BiLSTM-CRF/CRF. Score 0.75."),
    "doc276_qa0": (1.0, "Gold: recovers masked document; prediction: MDG recovers masked document. Full credit."),
    "doc277_qa0": (0.75, "Gold: 45,000+/33,000+; prediction: 45,000 COVID-related. Score 0.75."),
    "doc277_qa1": (1.0, "Gold: 45,000 inc 33,000 full text; prediction: 45,000 inc 33,000 full text. Full credit."),
    "doc278_qa1": (0.5, "Gold: types/genres distribution; prediction: topics/dialects/gender. Score 0.5."),
    "doc278_qa2": (1.0, "Gold: One; prediction: One annotator. Full credit."),
    "doc278_qa3": (1.0, "Gold: 10,000 Arabic tweets; prediction: 10,000 tweets. Full credit."),
    "doc278_qa4": (0.5, "Gold: no seed list; prediction: not biased by topic/dialect. Score 0.5."),
    "doc280_qa0": (0.75, "Gold: lexical overlap degree; prediction: reduction of lexical overlap. Score 0.75."),
    "doc280_qa2": (1.0, "Gold: Yes; prediction: Yes. Full credit."),
    "doc280_qa3": (1.0, "Gold: No; prediction: No. Full credit."),
}


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def score_calib_entry(entry: dict) -> tuple[float, str]:
    qid = entry["qid"]
    expected = entry.get("expected_behavior", "answer")
    gold = entry.get("gold_answer", "")
    pred = entry.get("predicted", "")

    # Extract suffix
    prefix = "qasper__attention-corpus-tuned__calibration__"
    suffix_raw = qid.replace(prefix, "").replace("__seed42", "")

    # Auto-score acknowledge_missing
    if expected == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                f"expected_behavior=acknowledge_missing; source doc not yet ingested. "
                f"Model correctly refuses: '{pred[:80]}'. Honest refusal -- full credit."
            )
        else:
            return 0.0, (
                f"expected_behavior=acknowledge_missing; source doc not yet ingested. "
                f"Model hallucinated: '{pred[:80]}'. Confident wrong answer -- zero credit."
            )

    # Auto-score unanswerable gold
    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, (
                f"Gold: (unanswerable per source). Model correctly refuses. Full credit."
            )
        else:
            return 0.0, (
                f"Gold: (unanswerable per source). Model answered when it should refuse: "
                f"'{pred[:80]}'. Zero credit."
            )

    # Refusal on regular answer entry
    if is_refusal(pred):
        return 0.0, (
            f"Gold: {gold[:60]}. Model refused when answer expected. Zero credit."
        )

    # Lookup manual judgment
    if suffix_raw in CALIB_JUDGMENTS:
        score, rationale = CALIB_JUDGMENTS[suffix_raw]
        return score, rationale

    # Default: 0.0
    return 0.0, f"Gold: {gold[:60]}. Prediction incorrect/off-topic. Zero credit."


def score_batch_entry(entry: dict) -> tuple[float, str]:
    qid = entry["qid"]
    gold = entry.get("gold_answer", "")
    pred = entry.get("predicted", "")

    prefix = "qasper__attention-corpus-tuned__batch_calib__"
    suffix_raw = qid.replace(prefix, "").replace("__seed42", "")
    # Also handle "batch" mode QIDs (batch_calib uses mode=batch in qid)
    suffix_raw = suffix_raw.replace("qasper__attention-corpus-tuned__batch__", "")

    # Auto-score unanswerable
    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, "Gold: (unanswerable per source). Model correctly refuses. Full credit."
        else:
            return 0.0, f"Gold: (unanswerable per source). Model answered: '{pred[:80]}'. Zero credit."

    # Refusal on answer entry
    if is_refusal(pred):
        return 0.0, f"Gold: {gold[:60]}. Model refused when answer expected. Zero credit."

    # Lookup manual judgment
    if suffix_raw in BATCH_JUDGMENTS:
        score, rationale = BATCH_JUDGMENTS[suffix_raw]
        return score, rationale

    # Default: 0.0
    return 0.0, f"Gold: {gold[:60]}. Prediction incorrect. Zero credit."


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
def write_results(queue_path: Path, results_path: Path, score_fn, prefix: str, mode: str):
    if not queue_path.exists():
        print(f"  SKIP (no queue): {queue_path.name}")
        return 0

    entries = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    for entry in entries:
        score, rationale = score_fn(entry)
        result = {
            "qid": entry["qid"],
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
            "expected_behavior": entry.get("expected_behavior", "answer"),
        }
        results.append(result)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    scores = [r["judge_score"] for r in results]
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"  {mode}: {len(results)} entries, mean={mean:.4f}")
    return len(results)


if __name__ == "__main__":
    print("Writing QASPER attention-corpus-tuned Protocol B results...")
    n_calib = write_results(
        CALIB_DIR / "queue.jsonl", CALIB_RESULTS,
        score_calib_entry, "qasper__attention-corpus-tuned__calibration__", "Calibration"
    )
    n_batch = write_results(
        BATCH_DIR / "queue.jsonl", BATCH_RESULTS,
        score_batch_entry, "qasper__attention-corpus-tuned__batch_calib__", "Batch_calib"
    )
    print(f"  Combined: {n_calib + n_batch} entries, mean={(n_calib + n_batch) and 0:.4f}")
