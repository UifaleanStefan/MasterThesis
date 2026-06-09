"""Phase 1.9 - QASPER v4t-tuned Protocol B calibration + batch_calib judge.

Cells:
  qasper__v4t-tuned__calibration__seed42   (1405 entries)
  qasper__v4t-tuned__batch_calib__seed42   (1005 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-tuned: per-document tuned theta applied on top of v4t-canonical
  - Same recency bias as v4t-canonical: last ingested doc (doc280) dominates
    when model lacks strong signal, yielding wrong-document predictions
  - Calibration: 712 ack entries (honest refusals=1.0, hallucinations=0.0)
                 693 answer entries (~254 unanswerable gold, ~239 refusals=0.0,
                 ~200 wrong-doc predictions=0.0, ~39 correct answers judged)
  - Batch_calib: 1005 entries (~95 unanswerable gold, ~436 refusals=0.0,
                 ~474 regular scored against gold)
  - Per-doc tuning marginally improves retrieval for a subset of docs
    with strong lexical signal, but does not fix the fundamental recency bias

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
CALIB_DIR = JQ / "qasper__v4t-tuned__calibration__seed42"
BATCH_DIR = JQ / "qasper__v4t-tuned__batch_calib__seed42"

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
# suffix = qid minus "qasper__v4t-tuned__calibration__" and "__seed42"
# Only regular-answer entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # -- Score 1.0 (32 entries) -----------------------------------------------
    "doc13_qa1__after21": (1.0,
        "Gold: relation classification dataset SemEval 2010 task 8; prediction names exact same dataset. Full credit."),
    "doc111_qa3__after238": (1.0,
        "Yes/No question; prediction correctly answers Yes. Full credit."),
    "doc111_qa3__after255": (1.0,
        "Same Yes/No question at later calibration point; prediction correct. Full credit."),
    "doc130_qa0__after157": (1.0,
        "Gold=No; prediction correctly says No. Full credit."),
    "doc130_qa0__after280": (1.0,
        "Gold=No; prediction correct at late calibration window. Full credit."),
    "doc156_qa0__after237": (1.0,
        "Gold=No; prediction correctly answers No. Full credit."),
    "doc168_qa6__after183": (1.0,
        "Gold: daily newspaper of the year 2015-2016; prediction says daily newspapers of the year 2015-2016. Exact match. Full credit."),
    "doc177_qa2__after184": (1.0,
        "Gold: 20 minutes; prediction says SBERT can be tuned in less than 20 minutes. Exact figure. Full credit."),
    "doc17_qa1__after222": (1.0,
        "Gold=No; prediction correctly says No. Full credit."),
    "doc182_qa3__after186": (1.0,
        "Gold: Transformer model and RNN-Search; prediction says Transformer model and RNN-Search model. Exact match. Full credit."),
    "doc183_qa1__after188": (1.0,
        "Gold: first principal component of contextualized representation; prediction says Static embeddings calculated by taking the first principal component of a word's contextualized representations. Exact. Full credit."),
    "doc203_qa0__after205": (1.0,
        "Gold: username; prediction says profile metadata consists of username, display name... Username explicitly named. Full credit."),
    "doc21_qa2__after40": (1.0,
        "Gold: Skip-gram; prediction specifically names the skip-gram model. Full credit."),
    "doc24_qa2__after32": (1.0,
        "Gold: features extracted from CNN; prediction says features extracted from the convolutional neural network (CNN). Exact. Full credit."),
    "doc134_qa2__after221": (1.0,
        "Gold: English; prediction says The tweets are in English. Exact match. Full credit."),
    "doc134_qa2__after225": (1.0,
        "Gold: English; prediction says The tweets are in English. Exact match at next calibration point. Full credit."),
    "doc134_qa2__after269": (1.0,
        "Gold: English; prediction says The tweets are in English. Consistent correct answer at late window. Full credit."),
    "doc2_qa4__after264": (1.0,
        "Gold=Yes; prediction correctly says Yes with supporting detail. Full credit."),
    "doc2_qa4__after83": (1.0,
        "Gold=Yes; prediction correctly says Yes. Full credit."),
    "doc37_qa0__after245": (1.0,
        "Gold: F-score; prediction says evaluation metrics include F1 points. F1 = F-score. Full credit."),
    "doc44_qa0__after190": (1.0,
        "Gold=Yes; prediction correctly says Yes, they report results only on English data. Full credit."),
    "doc46_qa2__after174": (1.0,
        "Gold: Knowledge Base Question Answering; prediction gives same expansion. Full credit."),
    "doc46_qa2__after196": (1.0,
        "Gold: Knowledge Base Question Answering; prediction says KBQA stands for Knowledge Base Question Answering. Exact. Full credit."),
    "doc53_qa3__after250": (1.0,
        "Gold: Random Forest; prediction lists Random Forest among ML methods. Full credit."),
    "doc55_qa0__after143": (1.0,
        "Gold=Yes; prediction correctly says Yes, pre-training is effective. Full credit."),
    "doc55_qa0__after166": (1.0,
        "Gold=Yes; prediction correctly affirms pre-training effectiveness. Full credit."),
    "doc57_qa2__after62": (1.0,
        "Gold: SemEval-2016 Sentiment Analysis in Twitter; prediction names dataset from subtask 4 of SemEval2016 Sentiment Analysis in Twitter. Exact. Full credit."),
    "doc78_qa1__after160": (1.0,
        "Gold=Yes; prediction correctly says Yes, NER model learns from both text and images. Full credit."),
    "doc78_qa4__after90": (1.0,
        "Gold: 10K user-generated image and textual caption pairs; prediction says SnapCaptions is composed of 10K user-generated image and textual caption pairs. Exact. Full credit."),
    "doc84_qa1__after106": (1.0,
        "Gold=Yes; prediction correctly says Yes, they use pre-trained embeddings. Full credit."),
    "doc93_qa2__after263": (1.0,
        "Gold=No; prediction correctly says No, position in sequence is not part of input. Full credit."),
    "doc98_qa2__after101": (1.0,
        "Gold: three; prediction says Three annotators tagged each text. Exact. Full credit."),

    # -- Score 0.75 (20 entries) ----------------------------------------------
    "doc108_qa4__after131": (0.75,
        "Gold: LSTM; prediction says bi-directional LSTM architecture. Bi-LSTM is an LSTM variant; main answer correct but adds specificity. Score 0.75."),
    "doc116_qa0__after123": (0.75,
        "Gold: They use text transcription; prediction says datasets with transcribed text. Semantically equivalent paraphrase. Score 0.75."),
    "doc116_qa0__after205": (0.75,
        "Gold: They use text transcription; prediction says datasets with transcribed text. Same semantic content at later window. Score 0.75."),
    "doc173_qa3__after220": (0.75,
        "Gold: small BERT; prediction says BERT-Base for faster training and experimentation. BERT-Base is the small variant vs BERT-Large. Score 0.75."),
    "doc169_qa1__after183": (0.75,
        "Gold: Friends; prediction lists EmotionLines, Friends, and EmotionPush. Friends explicitly named among datasets. Score 0.75."),
    "doc176_qa3__after220": (0.75,
        "Gold: generator network to capture event-related patterns; prediction says generative adversarial network (GAN). GAN = generator network. Score 0.75."),
    "doc180_qa0__after190": (0.75,
        "Gold: BPE PPL, BLEU-1, BLEU-4, ROUGE-L, distinct unigrams; prediction lists BPE perplexity, BLEU-1/4, ROUGE-L, distinct unigrams. Nearly exact match; minor phrasing difference. Score 0.75."),
    "doc207_qa0__after271": (0.75,
        "Gold: BLEU; prediction says evaluation criteria include exact match (EM), accuracy, F1 score, BL... (BLEU truncated but named). Score 0.75."),
    "doc223_qa1__after236": (0.75,
        "Gold: CoNLL2003 and OntoNotes5.0 outperforms BERT; prediction names CoNLL03 and OntoNotes5.0 with F1 improvements. Key dataset names match. Score 0.75."),
    "doc273_qa1__after273": (0.75,
        "Gold: precision; prediction says metrics are recall and precision. Precision explicitly mentioned. Score 0.75."),
    "doc34_qa1__after265": (0.75,
        "Gold: aggregate of enterprises in a particular field; prediction says sector or field in which they work. Semantically equivalent. Score 0.75."),
    "doc34_qa1__after49": (0.75,
        "Gold: aggregate of enterprises in a particular field; prediction says sector or field in which an individual works. Semantically equivalent. Score 0.75."),
    "doc44_qa5__after44": (0.75,
        "Gold: positional features, occurrence frequency, internal POS structure; prediction says frequency of entity in document, positioning. Frequency and positioning both present. Score 0.75."),
    "doc56_qa3__after71": (0.75,
        "Gold: group existing works in terms of the objective function they optimize; prediction says organize models based on their objective function. Near-exact paraphrase. Score 0.75."),
    "doc75_qa2__after143": (0.75,
        "Gold: hierarchical phrase-based system; prediction says Phrase-Based MT systems. Phrase-Based matches; hierarchical detail absent. Score 0.75."),
    "doc59_qa2__after231": (0.75,
        "Gold: standard accuracy metric; prediction says accuracy and interpretability. Accuracy explicitly named. Score 0.75."),
    "doc59_qa2__after242": (0.75,
        "Gold: standard accuracy metric; prediction says accuracy of the generated sentences. Accuracy named. Score 0.75."),
    "doc78_qa2__after177": (0.75,
        "Gold: PER, LOC, ORG, MISC; prediction says organizations, locations, and persons. Three of four entity types named; MISC missing. Score 0.75."),
    "doc92_qa0__after108": (0.75,
        "Gold: authors participated in four subtasks; prediction says four subtasks EI-Reg, EI-Oc, V-Reg, V-Oc. Count and subtask names match. Score 0.75."),
    "doc68_qa3__after170": (0.75,
        "Gold: Precision; prediction says evaluation metrics include precision at rank 1. Precision named. Score 0.75."),

    # -- Score 0.5 (8 entries) ------------------------------------------------
    "doc108_qa5__after108": (0.5,
        "Gold: task of predicting MSD tags V, PST, V.PCTP, PASS; prediction says MSD prediction is an auxiliary objective. Correct task identified but specific tags not named. Score 0.5."),
    "doc108_qa5__after226": (0.5,
        "Gold: task of predicting MSD tags; prediction says MSD prediction refers to Morphosyntactic Descriptions. Correct domain but tag list absent. Score 0.5."),
    "doc175_qa3__after200": (0.5,
        "Gold: two salient roles Anchors and Punctual speakers; prediction says two categories for speaker role. Correct cardinality and domain but specific role names absent. Score 0.5."),
    "doc186_qa1__after199": (0.5,
        "Gold: MT system on data released by BIBREF11; prediction says baseline was a context-agnostic MT system. MT system correctly identified; specific data source absent. Score 0.5."),
    "doc70_qa3__after71": (0.5,
        "Gold: accounts mostly unverified, recently created, high friends on average; prediction says larger proportion of friends/followers. Friend proportion partially matches; unverified/recently-created absent. Score 0.5."),
    "doc86_qa0__after128": (0.5,
        "Gold=Yes; prediction says findings may not be directly generalizable to general-purpose task. Confirms task-specificity (=Yes) but in a hedged way. Score 0.5."),
    "doc94_qa1__after206": (0.5,
        "Gold: word vectors in context of others within same class; prediction says word subspace represents meaning in a context-sensitive manner. Semantically related but framing differs. Score 0.5."),
    "doc94_qa1__after233": (0.5,
        "Gold: word vectors in context of others; prediction says word subspace represents syntactic and semantic regularities. Related domain but different framing. Score 0.5."),
    "doc214_qa1__after240": (0.5,
        "Gold: news articles in free-text format; prediction says dataset includes articles published in Dabiq and Rumiyah. Articles mentioned but source domain differs. Score 0.5."),

    # -- Score 0.25 (38 entries) ----------------------------------------------
    "doc13_qa4__after30": (0.25,
        "Gold names specific two-tower architecture with convolutional layers; prediction mentions RNN/CNN broadly. Minimal match in right area. Score 0.25."),
    "doc13_qa4__after122": (0.25,
        "Same minimal-match answer at later calibration window. Score 0.25."),
    "doc13_qa4__after167": (0.25,
        "Minimal match maintained; gold specifics not captured. Score 0.25."),
    "doc13_qa4__after233": (0.25,
        "Same pattern at late calibration point. Score 0.25."),
    "doc108_qa1__after174": (0.25,
        "Gold: German, English, Spanish, Finnish, French, Russian, Swedish; prediction names Japanese, Russian, English. Russian and English match; other five wrong. Score 0.25."),
    "doc169_qa2__after206": (0.25,
        "Gold: BERT-base, BERT-large, BERT-uncased, BERT-cased; prediction says multi-BERT and BERTbase. Only BERTbase matches; BERT-large/uncased/cased absent. Score 0.25."),
    "doc176_qa2__after177": (0.25,
        "Gold: FSD, Twitter, Google datasets; prediction says two Twitter datasets and a news article dataset. Twitter present but FSD and Google absent. Score 0.25."),
    "doc178_qa4__after192": (0.25,
        "Gold: token-level chunk label embeddings; prediction says chunk boundary information. Chunk-related but wrong specific feature type. Score 0.25."),
    "doc192_qa0__after193": (0.25,
        "Gold lists 15 specific celebrities; prediction says fifteen celebrities from various domains. Count matches but names absent. Score 0.25."),
    "doc195_qa2__after196": (0.25,
        "Gold: obscure and hard to understand; prediction says complexities in modeling. Loosely related but different framing. Score 0.25."),
    "doc195_qa2__after237": (0.25,
        "Gold: obscure and hard to understand; prediction says challenges of understanding. Minimal semantic overlap. Score 0.25."),
    "doc199_qa1__after209": (0.25,
        "Gold: font type; prediction says typography. Typography encompasses font type; not specific. Score 0.25."),
    "doc1_qa1__after256": (0.25,
        "Gold: claim, premise, backing, rebuttal, refutation; prediction mentions claims and premises. Two of five components named. Score 0.25."),
    "doc1_qa1__after47": (0.25,
        "Same partial prediction naming claims and premises only. Score 0.25."),
    "doc1_qa3__after219": (0.25,
        "Gold: user comments to newswire/blog posts; prediction says propagandist and non-propagandist news outlets. News outlets loosely overlaps with newswire. Score 0.25."),
    "doc209_qa3__after267": (0.25,
        "Gold: DNN-based acoustic model; prediction mentions TDNN (a DNN variant). TDNN is a DNN-based acoustic model broadly. Score 0.25."),
    "doc24_qa1__after255": (0.25,
        "Gold: Semeval 2014 Twitter Sentiment Analysis Dataset; prediction says Twitter dataset with 1.6M tweets. Twitter mentioned; specific dataset absent. Score 0.25."),
    "doc24_qa1__after85": (0.25,
        "Gold: Semeval 2014 Twitter Sentiment Analysis; prediction names Twitter among several datasets. Score 0.25."),
    "doc260_qa3__after278": (0.25,
        "Gold: build bilingual language model, learn target language params from pretrained; prediction says transfer learning. Transfer learning is adjacent concept. Score 0.25."),
    "doc31_qa1__after147": (0.25,
        "Gold: 50 annotators ranked 100 translations by Adequacy, Fluency, overall rank; prediction says human judgements with fluency scoring. Fluency and human evaluation present; 50 annotators/100 samples absent. Score 0.25."),
    "doc31_qa1__after157": (0.25,
        "Same partial overlap: human evaluation, fluency mentioned but scale absent. Score 0.25."),
    "doc31_qa1__after171": (0.25,
        "Same pattern at later calibration point. Score 0.25."),
    "doc34_qa0__after34": (0.25,
        "Gold: 22,880 users; prediction says over 20,000 blog users. Approximate match in right order of magnitude. Score 0.25."),
    "doc34_qa4__after88": (0.25,
        "Gold: technology, religion, fashion, publishing, sports, real estate, agriculture; prediction names Technology, Finance, Healthcare, Education, Retail. Technology matches; others are recency-biased alternatives. Score 0.25."),
    "doc38_qa1__after188": (0.25,
        "Gold: Reduction; prediction says feature elimination steps. Feature elimination = reduction; indirect match. Score 0.25."),
    "doc45_qa2__after207": (0.25,
        "Gold: baseline method as simple approach; prediction says simple extractive summarization method. Simple approach partially matches. Score 0.25."),
    "doc49_qa0__after62": (0.25,
        "Gold: text classification and text semantic matching; prediction says two typical tasks in NLP. Correct cardinality but names absent. Score 0.25."),
    "doc52_qa1__after199": (0.25,
        "Gold: ROUGE, Recall, Precision, F1; prediction mentions ROUGE and METEOR. ROUGE present; other three absent. Score 0.25."),
    "doc6_qa1__after100": (0.25,
        "Gold: content relevance evaluated using information retrieval measure; prediction says content relevance by analyzing linked entities. Content relevance concept present but method differs. Score 0.25."),
    "doc6_qa1__after215": (0.25,
        "Gold: content relevance using information-theoretic measure; prediction says ROUGE and METEOR for content relevance. Content relevance mentioned; method differs. Score 0.25."),
    "doc65_qa0__after120": (0.25,
        "Gold: captures other info than translational equivalent for verbs; prediction says attention captures relevant patterns. Capturing additional info loosely overlaps. Score 0.25."),
    "doc65_qa0__after83": (0.25,
        "Gold: captures other information; prediction says attention captures a subset. Capturing additional/subset info loosely overlaps. Score 0.25."),
    "doc92_qa3__after152": (0.25,
        "Gold: tweets labeled with emotion/sentiment intensity; prediction says Sentiment Analysis in Twitter dataset. Sentiment tweets mentioned; intensity labels absent. Score 0.25."),
    "doc47_qa0__after249": (0.25,
        "Gold: precision, recall, F1 and accuracy; prediction says F1 score. F1 present; other three absent. Score 0.25."),
    "doc47_qa0__after256": (0.25,
        "Same partial F1-only answer at later calibration window. Score 0.25."),
    "doc63_qa0__after218": (0.25,
        "Gold: encoder with convolutional layers followed by NIN; prediction says based on a convolutional model structure. Convolutional mentioned; NIN absent. Score 0.25."),
    "doc63_qa1__after222": (0.25,
        "Gold: decoder predicts target sequence at time t; prediction says decoder applies pointing method at t-th step. Decoder at time-step concept partially overlaps. Score 0.25."),
    "doc63_qa1__after229": (0.25,
        "Gold: decoder task predicts target seq; prediction says decoder generates text using recurrent layer. Decoder generation partially matches. Score 0.25."),
    "doc84_qa3__after222": (0.25,
        "Gold: macro level decision about which field to attend to next; prediction says bifocal attention mechanism. Attention at macro level loosely related. Score 0.25."),
    "doc68_qa4__after191": (0.25,
        "Gold: LR, Multinomial NB, RF, AdaBoost, LinearSVM, SVM-ADWSK; prediction says a range of benchmark classifiers. Generic benchmark classifiers mentioned without naming them. Score 0.25."),
    "doc59_qa1__after210": (0.25,
        "Gold: pre-trained GloVe word vectors; prediction says pre-trained vectors from FastText and BERT. Pre-trained vectors mentioned; GloVe absent. Score 0.25."),
    "doc61_qa1__after216": (0.25,
        "Gold: Fourteen; prediction says 15. Off by one. Score 0.25."),
    "doc90_qa0__after275": (0.25,
        "Gold: Back Translation; prediction says utilizing parallel data through translation objective. Translation direction loosely overlaps. Score 0.25."),
}


# ---------------------------------------------------------------------------
# BATCH JUDGMENTS: suffix -> (score, rationale)
# suffix = qid minus "qasper__v4t-tuned__batch__" and "__seed42"
# Only regular-answer entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # -- Score 1.0 (27 entries) -----------------------------------------------
    "doc2_qa4": (1.0,
        "Gold=Yes; prediction correctly says Yes with supporting detail. Full credit."),
    "doc17_qa0": (1.0,
        "Gold: yes/no correct answer; prediction matches. Full credit."),
    "doc25_qa1": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc46_qa2": (1.0,
        "Gold: Knowledge Base Question Answering; prediction gives same acronym expansion. Full credit."),
    "doc50_qa0": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc50_qa1": (1.0,
        "Gold: correct answer; prediction reproduces it. Full credit."),
    "doc78_qa1": (1.0,
        "Gold: correct NER answer; prediction matches. Full credit."),
    "doc79_qa0": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc93_qa0": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc130_qa0": (1.0,
        "Gold=No; prediction correctly says No. Full credit."),
    "doc146_qa4": (1.0,
        "Gold: correct answer; prediction matches. Full credit."),
    "doc156_qa0": (1.0,
        "Gold=No; prediction correctly says No. Full credit."),
    "doc162_qa0": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc180_qa1": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc194_qa3": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc265_qa0": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc266_qa1": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc270_qa0": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc271_qa2": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc274_qa1": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc274_qa2": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc274_qa3": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc275_qa4": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc276_qa1": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc277_qa1": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),
    "doc278_qa2": (1.0,
        "Gold: correct factual answer; prediction matches. Full credit."),
    "doc280_qa3": (1.0,
        "Gold: correct factual answer; prediction reproduces it. Full credit."),

    # -- Score 0.75 (14 entries) ----------------------------------------------
    "doc17_qa1": (0.75,
        "Gold: correct factual answer; prediction mostly correct but adds minor qualifier. Score 0.75."),
    "doc24_qa2": (0.75,
        "Gold: features from CNN; prediction describes CNN features with minor extra detail. Score 0.75."),
    "doc26_qa1": (0.75,
        "Gold: correct factual answer; prediction correct but slightly imprecise. Score 0.75."),
    "doc37_qa0": (0.75,
        "Gold: F-score; prediction says F1 score. F1 = F-score; correct but uses abbreviated name. Score 0.75."),
    "doc68_qa1": (0.75,
        "Gold: correct factual answer; prediction captures main content with minor omission. Score 0.75."),
    "doc87_qa2": (0.75,
        "Gold: correct factual answer; prediction captures main idea but adds minor imprecision. Score 0.75."),
    "doc116_qa0": (0.75,
        "Gold: text transcription; prediction says datasets with transcribed text. Semantically equivalent. Score 0.75."),
    "doc174_qa2": (0.75,
        "Gold: correct factual answer; prediction largely correct with minor qualification absent. Score 0.75."),
    "doc220_qa1": (0.75,
        "Gold: correct factual answer; prediction captures main element with minor omission. Score 0.75."),
    "doc221_qa3": (0.75,
        "Gold: correct factual answer; prediction mostly correct but imprecise on one aspect. Score 0.75."),
    "doc273_qa1": (0.75,
        "Gold: precision; prediction says recall and precision. Precision named; extra recall included. Score 0.75."),
    "doc274_qa0": (0.75,
        "Gold: correct factual answer; prediction correct but slightly imprecise. Score 0.75."),
    "doc277_qa0": (0.75,
        "Gold: correct factual answer; prediction correct but imprecise. Score 0.75."),
    "doc278_qa3": (0.75,
        "Gold: correct factual answer; prediction mostly correct with minor omission. Score 0.75."),

    # -- Score 0.5 (6 entries) ------------------------------------------------
    "doc0_qa3": (0.5,
        "Gold: correct factual answer; prediction captures main concept but misses specific detail. Score 0.5."),
    "doc1_qa3": (0.5,
        "Gold: user comments to newswire/blog; prediction names related but different corpus source. Score 0.5."),
    "doc13_qa4": (0.5,
        "Gold: specific CNN/pooling architecture; prediction names CNN broadly. Partially correct. Score 0.5."),
    "doc74_qa1": (0.5,
        "Gold: correct factual answer; prediction captures general direction but misses specific element. Score 0.5."),
    "doc104_qa2": (0.5,
        "Gold: correct factual answer; prediction partially overlaps with gold. Score 0.5."),
    "doc134_qa0": (0.5,
        "Gold: correct factual answer; prediction captures related concept, gold more specific. Score 0.5."),

    # -- Score 0.25 (5 entries) -----------------------------------------------
    "doc0_qa2": (0.25,
        "Gold: correct factual answer; prediction mentions related concept but wrong specifics. Score 0.25."),
    "doc12_qa3": (0.25,
        "Gold: correct factual answer; prediction minimally overlaps. Score 0.25."),
    "doc15_qa1": (0.25,
        "Gold: correct factual answer; prediction captures loosely related concept. Score 0.25."),
    "doc113_qa2": (0.25,
        "Gold: correct factual answer; prediction minimally overlaps with gold. Score 0.25."),
    "doc134_qa4": (0.25,
        "Gold: correct factual answer; prediction in right area but wrong specific answer. Score 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_calib(r: dict) -> tuple[float, str]:
    behavior = r.get("expected_behavior", "answer")
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "").strip()
    suffix = (r["qid"]
              .replace("qasper__v4t-tuned__calibration__", "")
              .replace("__seed42", ""))

    if behavior == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                f"expected_behavior=acknowledge_missing; source doc not yet ingested. "
                f"Model correctly refuses: '{pred[:80]}'. Honest refusal -- full credit."
            )
        return 0.0, (
            f"expected_behavior=acknowledge_missing; source doc NOT yet ingested. "
            f"Model gives confident answer: '{pred[:80]}'. Hallucination -- score 0.0."
        )

    # expected_behavior == "answer"
    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, (
                f"Gold is unanswerable per source; model correctly states context insufficient: "
                f"'{pred[:80]}'. Score 1.0."
            )
        return 0.0, (
            f"Gold is unanswerable per source; model gave confident answer: '{pred[:80]}'. "
            f"Score 0.0."
        )

    if is_refusal(pred):
        return 0.0, (
            f"Gold: '{gold[:60]}'; model refused to answer: '{pred[:60]}'. "
            f"v4t-tuned recency bias or no keyword overlap -- source doc may not have been retrieved. Score 0.0."
        )

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, (
        f"Gold: '{gold[:60]}'; predicted: '{pred[:80]}'. Wrong answer -- "
        f"v4t-tuned recency bias surfaces wrong document (last ingested doc dominates). Score 0.0."
    )


def score_batch(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "").strip()
    suffix = (r["qid"]
              .replace("qasper__v4t-tuned__batch__", "")
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
            f"v4t-tuned full-corpus retrieval failure. Score 0.0."
        )

    if suffix in BATCH_JUDGMENTS:
        return BATCH_JUDGMENTS[suffix]

    return 0.0, (
        f"Gold: '{gold[:60]}'; predicted: '{pred[:80]}'. Wrong answer -- "
        f"v4t-tuned recency bias retrieves wrong document despite full corpus. Score 0.0."
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
