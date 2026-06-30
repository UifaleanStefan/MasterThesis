"""Apply Claude 1-by-1 judgments for qasper__bm25-corpus-tuned__batch__seed42.

Each (sfx -> (score, rationale)) below is a hand judgment by Claude against the
QASPER 5-point rubric (evaluation/claude_judge_protocol.md): 1.0 fully correct;
0.75 substantially correct, minor omission; 0.5 partial; 0.25 weakly related /
right category wrong specific; 0.0 wrong, or a refusal/abstention when the gold
answer exists. NO heuristics. Writes results.jsonl with judge_model +
judge_protocol + rationale, matching the established cell schema.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "results" / "stage3" / "judge_queue" / "qasper__bm25-corpus-tuned__batch__seed42"

# sfx (doc{D}_qa{Q}) -> (judge_score, rationale)
J = {
 "doc0_qa0": (0.25, "Gold 'labeled features'; pred says generic prior NLP knowledge/indicators -- weakly related, misses the term."),
 "doc0_qa1": (0.75, "Names the gold neutral-features regularization term plus two others; substantially correct."),
 "doc0_qa2": (0.5,  "Captures text categorization + sentiment but misses the breadth (web-page/science/medical/healthcare)."),
 "doc0_qa3": (0.75, "Captures robustness to unbalanced prior knowledge; omits the class-distribution imbalance clause."),
 "doc1_qa0": (0.0,  "Gold Yes (English only); pred says No -- opposite."),
 "doc1_qa1": (0.25, "Restates 'argument components' without naming claim/premise/backing/rebuttal/refutation."),
 "doc1_qa2": (0.0,  "Gold is Structural SVM; pred abstains though the answer exists."),
 "doc1_qa3": (0.75, "Captures user comments to articles/blog posts/newswire -- overlaps gold well."),
 "doc1_qa4": (0.0,  "Gold unanswerable; pred fabricates an answer."),
 "doc1_qa5": (0.75, "Captures linguistic variability via 'variety of formats/styles, less formal/ambiguous'."),
 "doc2_qa0": (0.5,  "Describes the monolingual SRI baseline type but misses the lang2011 reference."),
 "doc2_qa1": (0.75, "Correctly names crosslingual latent variables (CLV); omits 'parent of role variables' detail."),
 "doc2_qa2": (0.0,  "Gold EN/DE CoNLL-2009; pred abstains though the answer exists."),
 "doc2_qa3": (0.5,  "Gold No; pred says small improvements but biggest impact monolingual -- partial directional match."),
 "doc2_qa4": (0.0,  "Gold Yes; pred says No -- opposite on the yes/no."),
 "doc2_qa5": (0.5,  "Captures 'Bayesian model per language' but misses garg2012 base-monolingual specifics."),
 "doc2_qa6": (1.0,  "Gold unanswerable; pred correctly says no direct answer."),
 "doc3_qa0": (0.0,  "Gold unanswerable; pred fabricates a 'Yes'."),
 "doc4_qa0": (0.0,  "Gold 1500 sentences; pred says 100 -- wrong number."),
 "doc4_qa1": (0.75, "Captures cross-language edge correspondence via alignment + dual decomposition."),
 "doc5_qa0": (0.0,  "Gold CrowdFlower; pred describes data nature, not the collection method."),
 "doc5_qa1": (0.0,  "Gold 4.49 turns; pred abstains though the answer exists."),
 "doc6_qa0": (0.0,  "Gold Yes (English only); pred abstains though the answer exists."),
 "doc6_qa1": (1.0,  "Matches gold: IR with summaries as queries, comparing retrieved-result overlaps."),
 "doc6_qa2": (0.5,  "Partly aligns (variants differ) but claims ROUGE-2/3 'better correlated', softening gold's low-correlation point."),
 "doc6_qa3": (0.75, "Captures higher-tier pyramid scoring; substantially correct."),
 "doc6_qa4": (0.75, "Captures the refuted belief that ROUGE is reliable; omits the weak-correlation framing."),
 "doc7_qa0": (0.75, "Gold No; pred conveys 'no other datasets mentioned' -- consistent with No."),
 "doc7_qa1": (0.0,  "Gold None; pred fabricates parameter counts/accuracies for baselines."),
 "doc8_qa0": (0.75, "Captures self- and opponent-coverage; misses 'adopted discussion points'."),
 "doc8_qa1": (1.0,  "IQ2 US debates = Intelligence Squared Debates -- correct."),
 "doc9_qa0": (0.75, "Names accuracy (gold) with extra edit-distance qualifier."),
 "doc10_qa0": (0.0, "Gold No; pred says Yes (unsupervised) -- opposite."),
 "doc10_qa1": (0.25,"Says 'a public dataset' but misses Social Honeypot + Weibo naming."),
 "doc10_qa2": (0.5, "Captures LDA-based feature extraction; doesn't state binary classification explicitly."),
 "doc11_qa0": (1.0, "Gold Yes; pred correctly says Yes with supporting detail."),
 "doc11_qa1": (1.0, "Gold 'established task'; pred says addressed earlier = established."),
 "doc11_qa2": (1.0, "Matches gold simple word-level encoder, with detail."),
 "doc11_qa3": (0.0, "Gold None; pred fabricates LM/sentiment/NMT tasks."),
 "doc11_qa4": (1.0, "Matches gold simple word-level encoder."),
 "doc12_qa0": (0.0, "Gold unanswerable; pred fabricates evaluation methods."),
 "doc12_qa1": (1.0, "Gold 30,000; pred says over 30,000 -- correct."),
 "doc12_qa2": (0.5, "Manual inspection ~ 'spot patterns by looking'; partially captures it."),
 "doc12_qa3": (0.25,"Gold ethnic bias; pred says linguistic bias -- wrong specific bias."),
 "doc13_qa0": (0.0, "Gold 0.8% F1; pred abstains though the answer exists."),
 "doc13_qa1": (1.0, "Matches gold SemEval-2010 task 8 dataset."),
 "doc13_qa2": (0.0, "Gold majority-vote over CNN/RNN models; pred abstains though the answer exists."),
 "doc13_qa3": (0.0, "Gold uni-directional RNN; pred says bi-directional -- opposite."),
 "doc13_qa4": (0.5, "Captures the three-region context split; misses conv/max-pool/concatenate specifics."),
 "doc14_qa0": (0.25,"Generic 'translation systems'; doesn't name attentional encoder-decoder baseline."),
 "doc15_qa0": (0.25,"Says '16 languages' but misses UD v1.2 treebanks and the language list."),
 "doc15_qa1": (0.25,"Says '16 languages' without naming them."),
 "doc16_qa0": (0.0, "Gold Yes (one pair); pred says No -- wrong yes/no."),
 "doc17_qa0": (0.0, "Gold Yes; pred abstains (and shows a prompt artifact) though the answer exists."),
 "doc17_qa1": (0.75,"Gold No; pred says No (hedged)."),
 "doc17_qa2": (0.75,"Gold WSC collection; pred describes Winograd schemas = WSC."),
 "doc17_qa3": (0.0, "Gold English; pred says Spanish and Italian -- wrong."),
 "doc18_qa0": (0.25,"Names Stanford NER (one of the gold systems) but mostly generic."),
 "doc18_qa1": (0.0, "Gold GENIA Corpus; pred says Reuters/Medline -- wrong dataset."),
 "doc19_qa0": (0.0, "Gold LSTM+VGG16 multimodal; pred says random forest + DL -- wrong architecture."),
 "doc19_qa1": (0.5, "Crowd-agreement framing partially overlaps gold's redundant-answer diversity idea."),
 "doc20_qa0": (0.25,"Generic 'diverse topics'; names none of politics/business/science/AskReddit."),
 "doc20_qa1": (0.25,"Generic predictive-model description; doesn't name logistic regression."),
 "doc21_qa0": (0.75,"Captures second-order = co-occurrence of term pairs; substantially correct."),
 "doc21_qa1": (1.0, "Gold unanswerable; pred correctly says not specified."),
 "doc21_qa2": (0.25,"Doesn't name skip-gram; generic word-embedding mention."),
 "doc22_qa0": (0.0, "Gold averaging constituent predictions; pred abstains though the answer exists."),
 "doc22_qa1": (0.25,"Gold a specific magnitude; pred only qualitative 'much larger margin'."),
 "doc22_qa2": (0.75,"Captures human study on unanswered questions; misses the 50+50 subset specifics."),
 "doc23_qa0": (1.0, "Correctly enumerates the preprocessing levels incl. gold 'raw text'."),
 "doc23_qa1": (0.25,"Garbled gold; pred gives a plausible 5-model description not matching listed items."),
 "doc23_qa2": (0.0, "Gold 244; pred abstains though the answer exists."),
 "doc24_qa0": (0.25,"Gold is a citation; pred gives generic CNN-feature description."),
 "doc24_qa1": (1.0, "Matches gold SemEval-2014 Twitter Sentiment dataset."),
 "doc24_qa2": (0.75,"Captures 'features from baseline CNN'; substantially correct."),
 "doc25_qa0": (1.0, "Gold unanswerable; pred correctly says not mentioned."),
 "doc25_qa1": (0.0, "Gold 300; pred abstains though the answer exists."),
 "doc25_qa2": (0.25,"Says 'tweets' but misses the 250k count."),
 "doc26_qa0": (0.5, "Gold German-English; pred lists it among 4 pairs (over-inclusive)."),
 "doc26_qa1": (0.0, "Gold IMDb; pred abstains though the answer exists."),
 "doc26_qa2": (0.25,"Gold dynamic average pooling; pred says 'minimalist recurrent pooling' -- wrong specific."),
 "doc27_qa0": (0.25,"On-topic but gives no f-scores; gold has specific Table-3 numbers."),
 "doc27_qa1": (1.0, "Includes gold Affective Text (plus Fairy Tales, ISEAR)."),
 "doc27_qa2": (1.0, "Facebook-pages list matches gold exactly."),
 "doc28_qa0": (1.0, "Gold anti-nuclear-power; pred says anti-nuclear power -- correct."),
 "doc28_qa1": (0.0, "Gold eight layers; pred abstains though the answer exists."),
 "doc28_qa2": (0.0, "Gold abortion; pred abstains though the answer exists."),
 "doc28_qa3": (0.0, "Gold 32,595 posts; pred abstains though the answer exists."),
 "doc28_qa4": (0.0, "Gold No; pred says Yes -- opposite."),
 "doc28_qa5": (1.0, "Includes gold SVM n-gram baseline as item 1 of the full baseline list."),
 "doc29_qa0": (1.0, "Gold unanswerable; pred correctly says not mentioned."),
 "doc29_qa1": (0.75,"Gold Yes (purely attention); pred conveys yes via 'attention-based NMT'."),
 "doc29_qa2": (0.0, "Gold Yes; pred says No -- opposite."),
 "doc29_qa3": (0.0, "Gold English; pred abstains though the answer exists."),
}


def main() -> int:
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    missing = []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed42", "")
        if sfx not in J:
            missing.append(sfx); continue
        score, rat = J[sfx]
        out.append({"qid": q["qid"], "judge_score": score, "rationale": rat,
                    "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
    if missing:
        raise SystemExit(f"missing judgments for {len(missing)}: {missing[:10]}")
    (CELL / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    mean = sum(o["judge_score"] for o in out) / len(out)
    print(f"wrote {len(out)} judgments; mean judge = {mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
