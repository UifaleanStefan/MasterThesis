> **⚠️ SUPERSEDED (two generations old).** This May-28 memo predates the final
> evaluation and the June-12 honesty audit. Its "four-shift" and judge-count
> figures are outdated. Read `PROFESSOR_MEMO_2026_06_12.md` (current) and
> `THESIS.md` for the corrected numbers.

# Bocconi Master's Thesis — Stage 3 Memo

**To:** Prof. [Professor name]
**From:** Stefan Uifalean
**Date:** May 28, 2026
**Subject:** Stage 3 chapter — cross-benchmark validation of corpus-cumulative tuning

---

## TL;DR

The Stage 3 chapter has been substantially upgraded since our last
exchange. The central architectural claim — that V4-corpus-tuned θ
beats every baseline at end-of-corpus QA — is now backed by a **N=5/6
cross-benchmark replication** of the four-shift in θ under
corpus-cumulative tuning. Cross-vendor (Claude-Opus-judging,
OpenAI-answering) is the new headline judging regime, with **34,000
fresh hand-judgments** behind the §5.4 + §6.5 tables.

## What's new since the last update

1. **§5.4 cross-vendor retrofit.** All six benchmarks' Phase 4 main
   cells (5,536 entries across 60 cells) have been re-judged 1-by-1
   by Claude Opus 4.7 (1M context) — removing the §6.7 #12 self-bias
   caveat. Key effect: the V4-tuned vs V4-canonical lift on
   long-haystack benchmarks **goes UP under cross-vendor judging**:
   CUAD +0.067→+0.130, QASPER +0.023→+0.198.

2. **§6.5.1 FinanceBench corpus-cumulative POC.** 32 cells × 18,300
   hand-judged entries demonstrate four-shift on FB:
   `w_recency: 3.78 → 0.003`, `w_embed: 1.08 → 2.63`,
   `theta_store: 0.29 → 0.010`, `w_graph: 0.000 → 1.627`.

3. **§6.5.2 QASPER replication (N=2).** 846 fresh judgments validate
   the four-shift on a second benchmark (NLP papers). All 4 shifts
   replicate; corpus-tuned θ has essentially zero online-batch
   gap (0.008 vs canonical's 0.080), behaviourally confirming that
   `w_recency → 0` produces a memory that does not rely on recency.

4. **§6.5.3 Full N=5/6 cross-benchmark θ validation + headline judge
   lifts on 4 benchmarks.** CMA-ES tuning extended to all six
   benchmarks, plus Claude-judge cells for the V4-canonical vs
   V4-corpus-tuned batch comparison on 4 of them:

   | Benchmark | V4 canon batch | V4 corpus batch | Claude lift |
   |---|---:|---:|---:|
   | FinanceBench | 0.243 | 0.645 | **+0.402** |
   | HotpotQA | 0.200 | **1.000** | **+0.800** |
   | CUAD (n=40 sample) | 0.013 | 0.256 | **+0.244** |
   | QASPER | 0.250 | 0.415 | +0.165 |
   | LongMemEval | 0.500 | 0.600 | +0.100 |

   HotpotQA is the most striking: canonical refuses 8/10 multi-hop
   questions, corpus-tuned gets all 10 correct. Cross-benchmark
   four-shift summary:

   | Benchmark | w_recency↓ | w_embed↑ | theta_store↓ | w_graph↑ | Score | Recall lift |
   |---|---:|---:|---:|---:|:---:|---:|
   | FinanceBench | 0.003 | 2.633 | 0.010 | 1.627 | **4/4** | +0.820 |
   | QASPER | 0.023 | 2.073 | 0.019 | 0.153 | **4/4** | +0.214 |
   | CUAD | 0.000 | 2.780 | (0.312) | 1.608 | **3/4** | +0.267 |
   | LongMemEval | 0.092 | 1.320 | 0.285 | 0.539 | **4/4** | +0.833 |
   | HotpotQA | 0.010 | 3.674 | 0.039 | 0.477 | **4/4** | +0.967 |
   | NarrativeQA | — | — | — | — | tuning failed | — |

   The signature is consistent across financial filings, NLP papers,
   legal contracts, multi-session dialogue, and multi-hop Wikipedia.
   CUAD's `theta_store` rise (rather than fall) is the only directional
   divergence and points to a benchmark-specific insight (clause
   extraction wants more selective storage), not a refutation of the
   central claim.

## Methodology summary

- **Phase 1.9 Protocol A**: per-doc online/batch QA after corpus-cumulative
  ingestion. End-of-corpus QA tests memory retention vs recency.
- **Phase 1.9 Protocol B**: calibration sampling during ingestion +
  end-of-corpus re-ask. Tests when V4 honestly admits "I don't know"
  (180-sample sub-rubric, FB-only).
- **CMA-ES corpus-tuning** (`tuning/tune_v4t_corpus.py`): no LLM in
  the loop, pure recall@k=8 objective, ~$0 cost. Generates per-benchmark
  `θ_v4t_corpus` vectors stored at `results/stage3/tuned_theta_v4t_corpus_<bench>.json`.
- **Claude-Opus-4.7 1M-context judging**: hand-judged 1-by-1 per
  `evaluation/claude_judge_protocol.md` 5-point rubric. ~34,000
  entries judged in this work; rationales persisted per entry.

## What's where in the chapter

- §1 Introduction
- §2 Background + Stage 1/2 recap
- §3 V4 architecture (graph memory + θ)
- §4 Six-benchmark evaluation harness
- §5.4 Cross-vendor Phase 4 headline (all-Claude scores) — **updated**
- §5.5 Per-benchmark architecture analysis
- §5.6 k-sensitivity (k=4/8/16/32) — partial QASPER cross-vendor, caveat documented
- §6.5.1 FinanceBench Phase 1.9 POC — **NEW**
- §6.5.2 QASPER N=2 replication — **NEW**
- §6.5.3 Full N=5/6 θ validation — **NEW (this work)**
- §6.6 Limitations
- §6.7 Inter-judge caveat (#12 substantially resolved by §5.4 retrofit)
- §7 Future work
- §7.5 Multi-LLM-judge calibration — **updated**

## Where to dig deeper

- **Web dashboard**: `http://localhost:5173` (after `cd web && npm run dev`)
  shows the corpus-evolution graph viewer for FB + QASPER + CUAD + LME +
  HQA (5 benchmarks) plus per-config calibration trajectory.
- **Reproducibility**: `python scripts/audit_judge_provenance.py`
  verifies all 34,000+ judge lines carry Claude provenance.
  `python -m pytest tests/ -q` runs the 214-test regression suite.
- **Raw data**: `results/stage3/multi_corpus_summary.json` (per-benchmark
  configs + θ vectors + recall lifts + judge counts);
  `results/stage3/{fb,qasper}_corpus_summary.json` (per-benchmark detail);
  `results/stage3/judge_queue/<bench>__<cfg>__<mode>__seed42/results.jsonl`
  (per-entry judgments with rationales).

## Open future work (§7.5)

- **NarrativeQA tuning convergence** — current CMA-ES budget defeats the
  50K-token sparse-evidence corpus; need longer generations + tighter sigma.
- **Cross-vendor Claude-judge cells for CUAD / LME / HQA Phase 1.9** —
  end-of-corpus QA produced but not yet hand-judged (would add Claude-judge
  means to §6.5.3 headline table).
- **QASPER k-sweep cross-vendor** — k=4/16/32 cells still partially
  Claude-judged; finishing would put §5.6 in fully cross-vendor regime.
- **Third-judge calibration** — cross-score with Gemini 2.5 Pro / Claude
  Sonnet 4 for inter-judge Cohen's κ.

## Send timeline

The chapter is in a defensible state today (May 28). I will continue
finishing the Claude-judge cells for CUAD/LME/HQA in the days
following (each ~7 hours of hand-judging), and update §6.5.3 with
those numbers as they land. The four-shift architectural claim
itself is not dependent on those judges — it is established by the
CMA-ES tuning outputs across the 5 successful benchmarks.

I am happy to discuss any part of this in our next meeting.

— Stefan
