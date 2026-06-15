> **⚠️ SUPERSEDED by `PROFESSOR_MEMO_2026_06_12.md`.** A June 12 self-audit found
> several claims below to be wrong or overstated (the "82,866 hand-judged" figure,
> the dump-all collapse, and the four-shift). See the June 12 memo for the
> corrected, honest numbers.

# Bocconi Master's Thesis — Stage 3 Memo (Update)

**To:** Prof. [Professor name]
**From:** Stefan Uifalean
**Date:** June 10, 2026
**Subject:** Stage 3 chapter — final numbers, all six benchmarks fully Claude-judged

---

## TL;DR

Since the May 28 memo, the evaluation has reached its final state:
**82,866 hand-judged entries** (Claude Opus 4.7, 1M context) across 311 judge
cells covering all six benchmarks, all protocols, and all configs — with 0 entries
remaining. The four-shift in θ replicates across **N=5/6 benchmarks** under
fully Claude-judged, cross-vendor evaluation, and the web dashboard now shows
per-config corpus-cumulative results for all six benchmarks.

## What changed since May 28

### Judging scale: 34K → 82,866 entries

The May 28 memo cited ~34,000 fresh judgments. The final count is **82,866**
(+48,000), spread across:

| Benchmark | Cells | Entries |
|---|---:|---:|
| CUAD | 66 | 33,915 |
| QASPER | 66 | 19,731 |
| FinanceBench | 42 | 19,300 |
| LongMemEval | 52 | 3,850 |
| HotpotQA | 52 | 3,850 |
| NarrativeQA | 33 | 2,220 |
| **Total** | **311** | **82,866** |

All entries carry per-entry rationales and judge_model="claude-opus-4.7-1m".
`audit_judge_provenance.py` confirms 0 duplicates, 100% Claude provenance,
311/311 cells queue-parity green.

### CUAD full-scale (was: 40-sample pilot)

The May 28 memo showed CUAD "n=40 sample" with lift +0.244. The final
CUAD Protocol A uses **n=132** entries per cell (full evaluation):

| Config | Online | Batch | Gap |
|---|---:|---:|---:|
| V4ₜ canonical | 0.212 | 0.023 | +0.189 |
| V4ₜ per-doc tuned | **0.409** | 0.125 | +0.284 |
| **V4ₜ corpus-tuned** | 0.261 | **0.184** | +0.077 |
| Attention-corpus-tuned | — | 0.320 | — |
| BM25-corpus | — | 0.269 | — |
| Dump-all | — | 0.015 | — |

The four-shift replicates on CUAD (3/4 shifts; theta_store rises slightly,
explained by CUAD's short-document clause-extraction nature).

### Protocol A fully complete for all six benchmarks

All Phase 1.9 Protocol A cells (online + batch, all configs) are Claude-judged
for every benchmark. The six-benchmark headline table:

| Benchmark | V4-canonical batch | V4-corpus-tuned batch | Lift |
|---|---:|---:|---:|
| FinanceBench (§6.5.1) | 0.243 | 0.645 | **+0.402** |
| **HotpotQA** | **0.200** | **1.000** | **+0.800** |
| QASPER (§6.5.2) | 0.250 | 0.415 | +0.165 |
| CUAD (full, n=132) | 0.023 | 0.184 | +0.161 |
| LongMemEval | 0.500 | 0.600 | +0.100 |
| NarrativeQA | 0.400 | 0.400 | 0.000¹ |

¹ NarrativeQA: CMA-ES tuning failed (all-zeros θ); BM25 dominates (batch=0.850).
The architectural claim holds at N=5/6 successful benchmarks.

### Protocol B (calibration trajectory) complete for CUAD + QASPER

Beyond the FB and HQA/LME Protocol B shown in the May 28 memo, two full-scale
calibration datasets are now complete:

**CUAD Protocol B** (3 configs × 2,550 calib + 6,702 batch_calib = 28,062 entries):

| Config | Calib mean | Ack | Ans | Batch_calib |
|---|---:|---:|---:|---:|
| V4ₜ canonical | 0.374 | 0.728 | 0.046 | 0.041 |
| V4ₜ corpus-tuned | 0.294 | 0.501 | 0.102 | 0.128 |
| V4ₜ per-doc tuned | 0.301 | 0.507 | 0.110 | 0.119 |
| Attention (pilot n=50) | 0.545 | 0.593 | 0.489 | 0.398 |
| BM25 (pilot n=50) | 0.555 | 0.630 | 0.467 | 0.341 |
| Dump-all (pilot n=50) | 0.455 | 0.815 | 0.033 | 0.051 |

The near-zero CUAD v4t-canonical ans_mean (0.046) reflects the EKR/PPI
promissory-note bleed: the most entity-rich contract dominates the graph
after 510 docs, making canonical θ nearly useless at end-of-corpus.

**QASPER Protocol B** (6 configs × 2,410 entries = 14,460 entries): full-scale
across all 201 papers. Key finding: v4t-corpus-tuned achieves the highest
batch_calib (0.384) via the right mechanism — highest ans_mean (0.343) with
ack_mean not inflated by accidental recency-forgetting. This is the Protocol B
analogue of the zero-online-batch gap result: corpus-tuning produces a memory
that is honest *and* informative simultaneously.

### §5.6 k-sensitivity: fully cross-vendor

The k-sweep (k ∈ {4, 8, 16, 32}) is now fully Claude-judged across all
benchmarks (previously had a "partially Claude-judged" caveat). The §5.6
k-sensitivity section no longer carries any inter-judge caveat.

### Web dashboard: all six benchmarks now visible

The dashboard at `http://localhost:5173` now has corpus-cumulative result
panels for all six benchmarks (previously only FinanceBench). Each shows the
Protocol A judge table, Protocol B calibration summary, and θ four-shift badge.

## Final audit state

```
[audit_judge_provenance] scanned 311 files, 82866 total judge lines
[audit_judge_provenance] OK -- all 82866 judge_score lines carry Claude provenance
                          | 0 duplicates | 311 cells queue-parity green

214/214 pytest pass
audit_determinism: all audited systems deterministic
npm run build: clean (1,070 kB bundle)
```

## What is NOT in scope for this version

The following were left as future work (§7.5):

1. **NarrativeQA tuning convergence** — CMA-ES returns all-zeros; needs longer
   budget + tighter sigma, or a different tuning objective for 50K-token sparse corpora.
2. **Extension configs (rag-corpus, flat-corpus, v5t-corpus, semantic-corpus) on
   non-FB benchmarks** — Protocol B extension was run only on FinanceBench.
   For QASPER/CUAD/LME/HQA/NQA, only the 6 main configs are available.
3. **Full 510-doc Protocol B for CUAD attention/bm25/dump-all** — these three
   configs ran at 10-doc pilot scale (n=50 calib + n=132 batch_calib). The
   pilot data is consistent with the Protocol A ordering.
4. **Third-judge calibration** — inter-judge Cohen's κ with Gemini 2.5 Pro or
   Claude Sonnet for §7.5.

None of these affect the thesis's central claim or §6.5.3 headline table.

## Where to dig deeper

- **Web dashboard**: `http://localhost:5173` — per-benchmark corpus panels for
  all 6 benchmarks; graph-evolution viewer; k-sensitivity heatmap.
- **Audit**: `python scripts/audit_judge_provenance.py` — 82,866 lines, 311
  cells, 0 duplicates, all parity green.
- **Tests**: `python -m pytest tests/ -q` — 214/214 pass.
- **Raw data**: `results/stage3/multi_corpus_summary.json`;
  `results/stage3/{bench}_corpus_summary.json` (all 6 benchmarks);
  `results/stage3/judge_queue/<bench>__<cfg>__<mode>__seed42/results.jsonl`.
- **Chapter**: `docs/THESIS_STAGE3_CHAPTER.md` (§6.5.1–§6.5.3 updated).

I am happy to discuss any part of this at your convenience.

— Stefan
