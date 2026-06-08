"""Phase 4 cross-vendor finishing — QASPER p4_main bm25 seed=42 (76 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__bm25__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q001", 0.0, "PRED hallucinated stereotype-detection methods; gold unanswerable."),
    ("q003", 0.25, "PRED 'TAC2010, ACE2004, AQUAINT, CoNLL' partial; gold=CoNLL-YAGO specifically."),
    ("q005", 0.75, "PRED 'does not provide info' implicit match to gold unanswerable."),
    ("q006", 0.0, "PRED 'competitive with Lead-3-AMR' hallucinated; gold unanswerable."),
    ("q008", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q010", 0.5, "PRED '16 languages' correct count but doesn't name them; gold=full UD1.2 list."),
    ("q011", 0.0, "PRED 'time, memory, CPU, bandwidth' WRONG; gold=precision, recall, F1, accuracy."),
    ("q012", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q013", 0.25, "PRED 'demographics, diagnosis, symptoms' partial; gold has 10 categories."),
    ("q014", 0.75, "PRED 'comparing Annodis segmentation, word pairs for borders' partial match to gold."),
    ("q015", 0.0, "PRED 'new SOTA on SemEval 2010' vague; gold=0.8% F1 specifically."),
    ("q016", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold RNN-based NMT."),
    ("q017", 0.0, "PRED 'Straw man, two options, scapegoating...' WRONG; gold=Loaded language."),
    ("q018", 0.25, "PRED 'in the QA+QG framework' partial; gold specifies scoring via semantic relevance."),
    ("q019", 0.25, "PRED '280M vs 20M Latvian' implicitly 14×; gold=14 times explicitly."),
    ("q020", 0.0, "PRED '77.9% accuracy, 22.5% improvement' WRONG; gold=tSNE."),
    ("q021", 0.0, "PRED hallucinated jargon-quality link; gold unanswerable."),
    ("q022", 1.0, "PRED 'b-emb, CBOW, PV-DM, GloVe' exact match to gold."),
    ("q024", 0.25, "PRED 'built on BERT, stacking layers' partial; gold=extra BERT position embeddings specifically."),
    ("q025", 0.0, "PRED 'monolingual data from non-English' WRONG; gold=No data, pretrained model."),
    ("q027", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q029", 0.5, "PRED 'ironic to non-ironic and vice versa' partial; gold=Irony Classifier specifically."),
    ("q030", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q032", 0.75, "PRED 'raw text, text cleaning, keyphrase sparse' includes gold=raw text."),
    ("q033", 0.75, "PRED 'unresolved ambiguities, irrelevant entities' close to gold=ambiguous/too common."),
    ("q034", 0.75, "PRED 'BIBREF8 and BIBREF9' includes gold BIBREF9."),
    ("q035", 0.75, "PRED 'De-En, Ja-En, Ro-En' includes gold De-En."),
    ("q037", 0.0, "PRED 'systems tend to mark one gender/race higher' vague; gold has specific gender+race direction."),
    ("q038", 0.0, "PRED 'Italian-English 16% improvement' WRONG; gold=MAP only."),
    ("q039", 1.0, "PRED 'No' matches gold='No'."),
    ("q040", 0.25, "PRED 'F1 0.92 intent, 0.97 slot' different from gold=2.3-6.8 F1 intent, 0.8-3.5 slot."),
    ("q041", 1.0, "PRED 'Yes, rare words, noisy social media' matches gold='Yes'."),
    ("q042", 0.75, "PRED 'not specified' implicit match to gold unanswerable."),
    ("q043", 0.0, "PRED 'two metrics quantitative/qualitative' vague; gold=average unique predictions."),
    ("q045", 0.0, "PRED '7.05 BLEU improvement' hallucinated; gold unanswerable."),
    ("q047", 0.0, "PRED 'climate change, abortion, world news' WRONG; gold=politics, business, science, AskReddit."),
    ("q048", 0.0, "PRED hallucinated Jasper baselines; gold unanswerable."),
    ("q050", 1.0, "PRED 'CJFA encoder' matches gold."),
    ("q051", 0.0, "PRED 'weak model b3' WRONG; gold=pivot-based translation."),
    ("q053", 0.25, "PRED 'nearest-neighbour qualitative results' partial; gold=Spearman correlation + entailment datasets."),
    ("q054", 1.0, "PRED 'No' matches gold='No'."),
    ("q055", 1.0, "PRED 'EI-Reg, EI-Oc, V-Reg, V-Oc' matches gold."),
    ("q056", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q057", 1.0, "PRED 'five text classification + semantic matching' matches gold."),
    ("q058", 0.75, "PRED 'UMA, MRR, BPE PPL, BLEU-1/4, ROUGE-L, Distinct-1/2' matches 8/9 gold metrics; missing PP."),
    ("q059", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q060", 1.0, "PRED 'No' matches gold='No'."),
    ("q061", 0.75, "PRED 'BOW, TFIDF, LR, RF, TextCNN' matches gold components BOW-LR, BOW-RF, TFIDF-RF, TextCNN."),
    ("q063", 0.25, "PRED 'narrow margin' vague; gold has specific per-dataset Rouge/Meteor numbers."),
    ("q065", 0.0, "PRED 'speech recognition, synthesis, MT' (tasks not models); gold=Transformer architecture."),
    ("q067", 0.75, "PRED 'does not mention seasonality' implicit match to gold unanswerable."),
    ("q068", 0.75, "PRED 'does not provide info' implicit match to gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes, RE task' but gold='No' — Y/N flip."),
    ("q070", 1.0, "PRED 'transcribed text' matches gold."),
    ("q071", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q073", 1.0, "PRED 'No, diverse range' matches gold='No'."),
    ("q075", 0.75, "PRED 'Back translation, mix-source, BPE variant' includes gold=Back Translation."),
    ("q076", 0.0, "PRED 'Reddit comment analysis' WRONG; gold=SGD, naive bayes, decision tree."),
    ("q077", 0.25, "PRED '-PMI spectrum variants' vague; gold=clipped PMI, NNEGPMI specifically."),
    ("q079", 1.0, "PRED 'ro→en, en→de, de→en, ja→en' matches gold four MT tasks."),
    ("q080", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q081", 0.0, "PRED 'local n-gram + self-attention' WRONG; gold=Bi-LSTM-CRF."),
    ("q082", 0.25, "PRED 'GENIA chunker, DATEXIS-NER, Stanford NER' partial; gold=8 named systems."),
    ("q083", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q085", 0.0, "PRED 'Uzbek, Mandarin, Turkish' WRONG; gold=Switchboard Telephone Speech Corpus."),
    ("q088", 0.0, "PRED 'maxpool with kernel for sentence-level attention' hallucinated; gold unanswerable."),
    ("q089", 0.75, "PRED 'WER, BLEU, NIST, METEOR, ROUGE-L, CIDEr' includes gold=BLEU."),
    ("q091", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q092", 1.0, "PRED 'MLP' matches gold=MLP."),
    ("q093", 0.75, "PRED 'BLEU-1, Meteor, Rouge-L' includes gold BLEU-1."),
    ("q094", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q095", 1.0, "PRED 'always YES baseline' matches gold exactly."),
    ("q096", 0.0, "PRED 'PPDB dataset' hallucinated; gold unanswerable."),
    ("q097", 0.25, "PRED 'ESTER1/2, ETAPE, REPERE French broadcast' partial; gold=ASR."),
    ("q098", 0.5, "PRED 'best=implicit contingency, comparable for explicit' partial; gold=explicit discourse relations."),
    ("q099", 0.0, "PRED 'WMT16 trained models' WRONG; gold=attentional encoder-decoder networks."),
]


def main() -> None:
    assert len(JUDGMENTS) == 76
    qid_prefix = "p4_main__qasper__bm25__seed42__"
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
                skipped += 1; continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_main bm25 seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
