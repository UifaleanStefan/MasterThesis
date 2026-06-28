"""
Corpus descriptors for the θ-from-task study (Phase 4).

Goal: represent each benchmark corpus by a small vector of cheap, deterministic
statistics ("descriptors") that a meta-learner could map to a good θ. This
module only EXTRACTS descriptors; modelling (predict θ from descriptors) lives
in optimization/theta_predict.py and is gated on the power analysis below.

Critical caveat (why this file also ships an analysis, not just features):
there are only ~5 benchmarks with a tuned θ. Fitting a 10-D regression on 5
points is hopeless, so before any predictor we first ask the falsifiable
question this whole idea rests on: *do corpora with similar descriptors want
similar θ?* If pairwise descriptor-distance does not correlate with
θ-distance, the meta-learner premise is already in trouble and we report that
honestly rather than fitting a meaningless model.

No LLM, no API — pure corpus statistics + the committed tuned-θ vectors.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import get_adapter

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAPWORD = re.compile(r"\b[A-Z][a-z]+\b")  # cheap proper-noun proxy

# Descriptor feature names (order matters; used as a vector downstream).
FEATURES = [
    "n_docs",
    "mean_paras_per_doc",
    "mean_para_words",
    "type_token_ratio",
    "mean_q_words",
    "mean_gold_per_q",
    "mean_gold_rel_pos",
    "gold_pos_dispersion",
    "capword_ratio",
]


def extract_descriptors(benchmark: str, limit_docs: int = 50) -> dict:
    """Compute the descriptor vector for one benchmark corpus.

    Each feature is a cheap statistic over the first `limit_docs` docs:
      n_docs              — corpus size (capped)
      mean_paras_per_doc  — document length in passages
      mean_para_words     — passage length
      type_token_ratio    — lexical diversity (vocab / tokens)
      mean_q_words        — question length
      mean_gold_per_q     — evidence concentration (#relevant paragraphs)
      mean_gold_rel_pos   — where gold sits in the doc (0=start, 1=end);
                            a recency proxy — high values mean recency could help
      gold_pos_dispersion — std of gold relative position
      capword_ratio       — capitalized-word fraction (entity density proxy)
    """
    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=limit_docs))
    d = descriptors_from_docs(docs)
    d["benchmark"] = benchmark
    return d


def descriptors_from_docs(docs: list) -> dict:
    """Core descriptor computation over an arbitrary list of docs (so it works
    on random sub-samples, not just a benchmark's first-N). Same features as
    extract_descriptors, minus the 'benchmark' label."""
    n_docs = len(docs)
    paras_per_doc, para_words = [], []
    vocab, total_tokens = set(), 0
    cap_tokens = 0
    q_words, gold_per_q, gold_rel_pos = [], [], []

    for doc in docs:
        paras = doc.get("paragraphs", []) or []
        paras_per_doc.append(len(paras))
        for p in paras:
            toks = _WORD.findall(p.lower())
            para_words.append(len(toks))
            total_tokens += len(toks)
            vocab.update(toks)
            cap_tokens += len(_CAPWORD.findall(p))
        npar = max(1, len(paras))
        for qa in (doc.get("qa_pairs", []) or []):
            q_words.append(len(_WORD.findall(str(qa.get("question", "")))))
            rel = [i for i in (qa.get("relevant_paragraphs", []) or []) if 0 <= i < len(paras)]
            if rel:
                gold_per_q.append(len(rel))
                for i in rel:
                    gold_rel_pos.append(i / npar)

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))

    return {
        "n_docs": float(n_docs),
        "mean_paras_per_doc": _mean(paras_per_doc),
        "mean_para_words": _mean(para_words),
        "type_token_ratio": float(len(vocab) / total_tokens) if total_tokens else 0.0,
        "mean_q_words": _mean(q_words),
        "mean_gold_per_q": _mean(gold_per_q),
        "mean_gold_rel_pos": _mean(gold_rel_pos),
        "gold_pos_dispersion": _std(gold_rel_pos),
        "capword_ratio": float(cap_tokens / total_tokens) if total_tokens else 0.0,
    }


def descriptor_vector(desc: dict) -> list[float]:
    return [float(desc[f]) for f in FEATURES]


def _load_tuned_thetas() -> dict[str, list[float]]:
    out = {}
    for f in sorted((ROOT / "results" / "stage3").glob("tuned_theta_v4t_corpus_*.json")):
        bench = f.name.replace("tuned_theta_v4t_corpus_", "").replace(".json", "")
        vec = json.loads(f.read_text()).get("tuned_theta_vec")
        if isinstance(vec, list) and len(vec) == 10:
            out[bench] = [float(v) for v in vec]
    return out


def _zscore(rows: list[list[float]]) -> list[list[float]]:
    cols = list(zip(*rows))
    means = [sum(c) / len(c) for c in cols]
    stds = [(sum((x - m) ** 2 for x in c) / len(c)) ** 0.5 or 1.0 for c, m in zip(cols, means)]
    return [[(x - m) / s for x, m, s in zip(r, means, stds)] for r in rows]


def _euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return float(num / den) if den else 0.0


def analyze_descriptor_theta_link(limit_docs: int = 50) -> dict:
    """The falsifiable pre-check: across benchmarks with a tuned θ, does
    pairwise descriptor-distance correlate with pairwise θ-distance?

    Positive ρ → similar corpora want similar θ (meta-learning is plausible).
    ρ ≈ 0 / negative → θ is idiosyncratic to the task (meta-learning premise
    weak; reinforces the thesis's task-dependence finding). Either is honest.
    """
    thetas = _load_tuned_thetas()
    benches = sorted(thetas)
    descs = {b: extract_descriptors(b, limit_docs) for b in benches}
    dvecs = _zscore([descriptor_vector(descs[b]) for b in benches])
    tvecs = [thetas[b] for b in benches]

    pair_labels, d_dist, t_dist = [], [], []
    for i in range(len(benches)):
        for j in range(i + 1, len(benches)):
            pair_labels.append(f"{benches[i]}~{benches[j]}")
            d_dist.append(_euclid(dvecs[i], dvecs[j]))
            t_dist.append(_euclid(tvecs[i], tvecs[j]))

    rho = _spearman(d_dist, t_dist)
    return {
        "benchmarks": benches,
        "n_benchmarks": len(benches),
        "n_pairs": len(pair_labels),
        "descriptors": descs,
        "descriptor_distance_vs_theta_distance_spearman": round(rho, 3),
        "pairs": [
            {"pair": p, "descriptor_dist": round(dd, 3), "theta_dist": round(td, 3)}
            for p, dd, td in zip(pair_labels, d_dist, t_dist)
        ],
        "interpretation": (
            "positive rho => similar corpora want similar theta (meta-learning plausible); "
            "rho near 0 or negative => theta is task-idiosyncratic, reinforcing task-dependence."
        ),
        "power_caveat": (
            f"only {len(benches)} benchmarks => {len(pair_labels)} pairs; this is a "
            "directional pre-check, NOT a fitted model. A credible predictor needs many "
            "(descriptor, theta) pairs from corpus sub-samples (see plan Phase 4)."
        ),
    }


def main() -> int:
    out = analyze_descriptor_theta_link()
    path = ROOT / "results" / "stage3" / "theta_descriptors.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "benchmarks", "n_pairs",
        "descriptor_distance_vs_theta_distance_spearman", "power_caveat")}, indent=2))
    print(f"-> saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
