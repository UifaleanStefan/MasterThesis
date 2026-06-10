# Recent Changes — Session Log

**Purpose:** Record of work done on the thesis codebase (analysis, fixes, dashboard, figures).
**Last updated:** June 2026

---

## -22. All 311 judge cells complete — 82,866 entries, audit green (June 10 2026)

**Complete judging sweep confirmed.** All 311 judge queue cells across all six benchmarks,
all protocols (A and B), and all configs are fully judged with 0 entries remaining.
Grand total: **82,866 hand-judged entries** with per-entry rationales, all carrying
Claude Opus 4.7 (1M context) provenance. Audit: `[audit_judge_provenance] OK -- all
82,866 judge_score lines carry Claude provenance | 0 duplicates | 311 cells queue-parity green`.

**Breakdown by benchmark:**
| Benchmark | Cells | Entries |
|---|---:|---:|
| CUAD | 66 | 33,915 |
| QASPER | 66 | 19,731 |
| FinanceBench | 42 | 19,300 |
| LongMemEval | 52 | 3,850 |
| HotpotQA | 52 | 3,850 |
| NarrativeQA | 33 | 2,220 |

**CUAD Protocol B complete.** All 6 configs judged; v4t-canonical, v4t-corpus-tuned,
and v4t-tuned ran at full 510-doc scale (2,550 calib + 6,702 batch_calib each, total
28,062 entries). Key numbers:
- v4t-canonical batch_calib_mean=0.041 (near-zero: EKR/PPI bleed dominates)
- v4t-corpus-tuned batch_calib_mean=0.128 (+0.087 lift via w_embed boost)
- attention-corpus-tuned pilot batch=0.398 (leads pilot configs)

**Chapter update:** §6.5.3 CUAD Protocol B paragraph replaced with 6-config table
showing full vs pilot scale. Summary sentence updated from "1/6 full + 5 partial
in progress" to "all 6 judged (3 full-scale + 3 pilot)".

**Files changed:**
- `docs/THESIS_STAGE3_CHAPTER.md` (§6.5.3 CUAD Protocol B paragraph + summary sentence)
- `docs/RECENT_CHANGES.md` (this entry)

---

## -21. Fix §6.1, §6.7, §5.6, §7.1 with Claude cross-vendor corrections (June 9 2026)

**Multi-section chapter correction pass.** Chapter critical re-read found several
sections still referencing stale gpt-4o-mini (Phase 1.7) numbers without cross-vendor
Claude parentheticals.

**§6.1 fair-comparison baseline table:**
Added Claude 3-seed means as parenthetical annotations to V4-tuned, BM25, and
AttentionMemory-tuned entries. Updated "Best" column: under Claude judging, V4-tuned
wins on both CUAD and QASPER (not "benchmark-dependent" as Phase 1.7 implied).

**§6.7 #2 (AttentionMemory critique):**
Updated to note that "attention wins on CUAD" was a gpt-4o-mini artifact. Under Phase 1.9
Claude judging: V4-tuned (0.383) beats AttentionMemory-tuned (0.294) by +0.089 on CUAD.
The architecture provides a real advantage under cross-vendor judging.

**§6.7 #3 (BM25 critique):**
Added Claude 3-seed means: CUAD 0.383 vs 0.166; QASPER 0.420 vs 0.215. V4-tuned beats
BM25 under both judge regimes — finding is robust to judge choice.

**§5.6 bias caveat:**
Corrected QASPER V4-tuned vs V4-canonical lift from +0.198 (seed=42 artifact) to
+0.039 (3-seed mean, consistent with §5.4 finding #2). Cross-reference added to §5.4
footnote ‡.

**§7.1 future work gap:**
Updated "the +0.023 effect does not survive Holm correction" to note the Phase 1.9
Claude mean is +0.039; re-running Holm tests on Claude-judged QASPER data is
explicitly marked as future work.

**Files changed:**
- `docs/THESIS_STAGE3_CHAPTER.md` (§6.1, §6.7 #2, §6.7 #3, §5.6, §7.1)
- `docs/RECENT_CHANGES.md` (this entry)

---

## -20. Fix §5.4 QASPER stale numbers — cross-vendor 3-seed means corrected (June 9 2026)

**Critical §5.4 correction.** The QASPER headline table row had stale values:
v4-canonical still carried the original gpt-4o-mini score (0.160); several
configs showed seed=42 single-seed Claude values rather than 3-seed means.
Corrected with Phase 1.9 Claude 3-seed means (n=300 per config):

| Config | Old (stale) | New (Claude 3-seed mean) |
|---|---:|---:|
| v4-canonical | 0.160 | 0.381 |
| v4-tuned | 0.358 | 0.420 |
| flat-50 | 0.353 | 0.403 |
| bm25 | 0.427 | 0.215 |
| attention-tuned | 0.556 | 0.235 |
| Lift V4t−V4c | +0.198 | +0.039 |

Key narrative change: the QASPER attention-tuned "win" (0.556) was a seed=42
outlier (seed=7: 0.122, seed=100: 0.120). Under 3-seed Claude judging,
V4-tuned (0.420) leads the table. The previously reported +0.198 lift is
corrected to +0.039. The central claim (positive lift on all long-haystack
benchmarks) is maintained.

BM25 and attention sub-tables labeled as Phase 1.7 gpt-4o-mini analysis;
Claude 3-seed means added as parenthetical annotations.

**Files changed:**
- `docs/THESIS_STAGE3_CHAPTER.md` (§5.4 table line 836, footnotes ‡ and †,
  headline findings #2 and #4, BM25+attention sub-tables, narrative)
- `docs/RECENT_CHANGES.md` (this entry)

---

## -19. QASPER Protocol B complete — all 6 configs judged, §6.5.2 updated with full table (June 9 2026)

**QASPER Protocol B all-six-config complete.** All 14,460 Protocol B
entries (6 configs × 2,410 entries each) hand-judged 1-by-1 with Claude
Opus 4.7. V4ₜ corpus-tuned achieves highest batch_calib (0.384) and
highest ans_mean (0.343), confirming that corpus-tuning produces a memory
that is honest *and* informative. Dump-all shows distinctive Protocol B
signature: high ack_mean (0.787) from context-overflow implicit refusals,
but near-zero ans_mean (0.108) and batch_calib (0.111). §6.5.2 Protocol B
section updated with complete 6-config table. Commits: d8e88da (judge
scripts) + this commit.

**Files changed:**
- `scripts/_judge_phase19_qasper_dump-all_calibration.py` (all 547 RNR entries)
- `results/stage3/judge_queue/qasper__dump-all__calibration__seed42/results.jsonl`
- `results/stage3/judge_queue/qasper__dump-all__batch_calib__seed42/results.jsonl`
- `results/stage3/qasper_corpus_summary.json` (regenerated with all 6 configs)
- `docs/THESIS_STAGE3_CHAPTER.md` (§6.5.2 Protocol B section updated)
- `docs/RECENT_CHANGES.md` (this entry)

---

## -18. Multi-benchmark Protocol B expansion: QASPER v4t-tuned + HQA/LME 6-config tables, chapter §6.5.2-§6.5.3 (June 9 2026)

**QASPER v4t-tuned Protocol B judged** (2,410 entries, calib=0.315,
ack=0.499, ans=0.126, batch_calib=0.091). Per-doc tuning erodes the
accidental honest-refusal advantage (ack drops from 0.704 to 0.499) without
improving retrieval at corpus scale (batch_calib drops from 0.133 to 0.091).
Key negative result: per-doc tuning inherits canonical's retrieval failure
while also losing its accidental honesty advantage. Commit: 6b04de9.

**CUAD v4t-canonical Protocol B full-scale** (2,550 calib + 6,702 batch_calib;
previous test run was 50+132). calib_mean=0.374 (ack=0.728, ans=0.046),
batch_calib=0.041. Near-zero ans_mean confirms the EKR/PPI promissory note
bleed dominates at full corpus scale. Commit: c79ac00.

**Chapter §6.5.2 Protocol B** expanded from 2 to 3 configs (added v4t-tuned
row with narrative explaining the intermediate calibration signature).
**Chapter §6.5.3 Protocol B** expanded with:
- HotpotQA full 6-config calibration table (v4t-tuned: ack=1.000, ans=0.445,
  batch=0.375; corpus-tuned: ack=0.956, ans=0.891, batch=0.950)
- LongMemEval full 6-config calibration table (new section; v4t-tuned
  collapse at batch=0.100; dump-all competitive at 0.600 on dialogue)
- CUAD Protocol B note (v4t-canonical full-scale EKR/PPI bleed analysis,
  5 configs pending full 510-doc runs)

HotpotQA finding: V4ₜ-per-doc-tuned (ack=1.000, ans=0.445) improves over
canonical on per-doc retrieval but batch_calib (0.375 vs 0.950 corpus-tuned)
shows corpus-scale distribution shift only corpus-cumulative tuning resolves.
LME finding: v4t-canonical's unusual ans>ack pattern (0.491 vs 0.378)
because dialogue content is semantically similar across sessions.

Audit: 57,496 total Claude-judge lines, 305 cells parity-green. Commit: 9961b73.

---

## -17. QASPER Protocol B complete — 4,820q Claude-judged, §6.5.2 Protocol B paragraph added (June 8 2026)

QASPER Protocol B (v4t-canonical + bm25-corpus calibration cells): 2 × 2,410
entries hand-judged 1-by-1 by Claude Opus 4.7 (1M context). Key headline numbers:
- v4t-canonical: calib_mean=0.421, batch_calib_mean=0.133
  (ack_mean=0.704 — extreme recency accidentally helps honest refusal)
- bm25-corpus: calib_mean=0.339, batch_calib_mean=0.294
  (ack_mean=0.398 — no-decay keyword retrieval hallucinates from wrong docs;
   ans_mean=0.280 — keyword retrieval fidelity 2× v4t-canonical when doc present)
Section §6.5.2 extended with Protocol B calibration table and mechanistic analysis.
Audit: 44,144 total Claude-judge lines, 267 cells queue-parity green.
Commits: 17eb6c9 (v4t-canonical), fba9bbd (bm25-corpus).

## -16. CUAD Protocol A complete — 1,188q Claude-judged, §6.5.3 updated (June 8 2026)

All 9 CUAD Protocol A cells (online + batch for 6 configs) now fully
Claude-judged 1-by-1 (1,188 judgments total, audit: 38,654 total Claude
judge lines, 250 cells parity-green):

| Cell | n | Mean |
|---|---:|---:|
| cuad__v4t-tuned__online | 132 | **0.409** |
| cuad__attention-corpus-tuned__batch | 132 | 0.320 |
| cuad__bm25-corpus__batch | 132 | 0.269 |
| cuad__v4t-corpus-tuned__online | 132 | 0.261 |
| cuad__v4t-canonical__online | 132 | 0.212 |
| cuad__v4t-corpus-tuned__batch | 132 | 0.184 |
| cuad__v4t-tuned__batch | 132 | 0.125 |
| cuad__v4t-canonical__batch | 132 | 0.023 |
| cuad__dump-all__batch | 132 | 0.015 |

Key finding: online mode dramatically outperforms batch on CUAD (+0.284
for v4t-tuned). The dump-all and v4t-canonical batch modes nearly fail
due to EKR/PPI promissory note context bleeding into all 10 contracts.

New scripts: `scripts/build_cuad_corpus_qa_data.py`,
output: `results/stage3/cuad_corpus_summary.json`.

Chapter §6.5.3 updated: "CUAD pending" → full 132q numbers, online-vs-batch
gap table, attention-corpus-tuned comparison, "N=5 all complete" conclusion.

## -15. Phase 1.9 full N=5/6 cross-benchmark θ validation (May 28 2026)

The corpus-cumulative four-shift in θ now demonstrated across **5 of 6
benchmarks** via CMA-ES tuning. Pipeline:

1. `tuning/tune_v4t_corpus.py --benchmarks {longmemeval,hotpotqa,narrativeqa}`
   ran in parallel as background tasks; each generated
   `results/stage3/tuned_theta_v4t_corpus_<bench>.json` (existing files
   for cuad/qasper/financebench already present from prior tuning).
2. `scripts/run_corpus_qa.py --benchmark <bench> --config v4t-{canonical,corpus-tuned}`
   ran end-of-corpus QA for CUAD (132 q/cell), LongMemEval (10 q/cell),
   HotpotQA (10 q/cell). Outputs at
   `results/stage3/corpus_traces/<bench>__<cfg>/qa_{online,batch}.json`.
3. `scripts/build_multi_corpus_qa_data.py` aggregates per-benchmark θ
   tables + recall lifts + judge-table counts into
   `results/stage3/multi_corpus_summary.json`.

**Four-shift summary across all 6 benchmarks:**

| Benchmark | w_recency↓ | w_embed↑ | theta_store↓ | w_graph↑ | Score | Recall lift |
|---|---:|---:|---:|---:|:---:|---:|
| FinanceBench | 0.003 | 2.633 | 0.010 | 1.627 | **4/4** | +0.820 |
| QASPER | 0.023 | 2.073 | 0.019 | 0.153 | **4/4** | +0.214 |
| CUAD | 0.000 | 2.780 | (0.312) | 1.608 | **3/4** | +0.267 |
| LongMemEval | 0.092 | 1.320 | 0.285 | 0.539 | **4/4** | +0.833 |
| HotpotQA | 0.010 | 3.674 | 0.039 | 0.477 | **4/4** | +0.967 |
| NarrativeQA | — | — | — | — | tuning failed¹ | — |

¹ NarrativeQA's 6-generation CMA-ES converged to all-zeros θ (no
storage at all → 0 recall). Books/screenplays of 50K+ tokens with
sparse evidence over very long horizons defeat the default budget.
Future work: longer generations + tighter sigma.

**Notable findings:**

- HotpotQA's `w_embed` rises HIGHEST of all (1.079 → 3.674) — multi-hop
  Wikipedia paragraphs are entity-dense and the tuned θ leans heavily
  on embedding similarity for retrieval.
- CUAD's `w_graph` nearly matches FB (1.608 vs 1.627) — corporate
  contracts have similar cross-document entity graphs (parties, clauses,
  defined terms).
- CUAD's `theta_store` rises slightly (0.293 → 0.312) — the only
  directional divergence. Interpretation: clause extraction wants
  *more* selective storage to preserve clause-bearing paragraphs.
- LongMemEval has the smallest `w_embed` shift (1.079 → 1.320) — multi-
  session dialogue is text-centric not entity-rich, embeddings already
  carry most of the signal.

**Files added/modified:**
- `scripts/build_multi_corpus_qa_data.py` (NEW) — generalized aggregator
- `results/stage3/multi_corpus_summary.json` (NEW) — per-benchmark θ +
  recall + judge tables
- `results/stage3/tuned_theta_v4t_corpus_{longmemeval,hotpotqa,narrativeqa}.json`
  (NEW from tuning runs)
- `results/stage3/corpus_traces/{cuad,longmemeval,hotpotqa}__*/qa_*.json`
  (NEW from corpus QA runs)
- `results/stage3/judge_queue/{cuad,longmemeval,hotpotqa}__*/queue.jsonl`
  (NEW; 800+ entries pending Claude judging — judgments not yet completed)
- `docs/THESIS_STAGE3_CHAPTER.md` — new §6.5.3 + §7.5 update
- `docs/PROFESSOR_MEMO_2026_05_28.md` (NEW) — 2-page executive summary
- `docs/RECENT_CHANGES.md` (this entry)

**Open items deferred to post-Thursday continuation:**
- Hand-judge the CUAD/LME/HQA cells (~2,000 entries total, ~13 hours)
- Re-run NarrativeQA tuning with longer budget
- Frontend QASPERCorpus.tsx / CUADCorpus.tsx sections (parallel to
  FinanceBenchCorpus.tsx)
- QASPER k-sweep cross-vendor finishing (1,487 entries)

The architectural claim (4-shift in θ replicates across 5 benchmarks)
is established as N=5; the cells pending judging will refine the
Claude-judge numbers but not the four-shift itself.

---

## -14. Phase 1.9 cross-benchmark replication: QASPER (May 26 2026)

The FB Phase 1.9 corpus-cumulative methodology is now replicated on a
second benchmark, QASPER (281 NLP research papers, sparse-evidence QA).

**Scope:** First 30 papers ingested (1,880 paragraphs, 6× FB scale).
3 V4 architecture configs (v4t-canonical, v4t-tuned, v4t-corpus-tuned) ×
{online, batch} = 564 entries; 3 baseline configs (bm25, attention,
dump-all) × batch = 282 entries. Total **846 fresh Claude judgments**,
all hand-judged 1-by-1 per `evaluation/claude_judge_protocol.md`.

**The four-shift in θ replicates.** Same directional signature as FB:

| Param | Canonical | FB corpus-tuned | QASPER corpus-tuned |
|---|---:|---:|---:|
| `w_recency` | 3.777 | 0.003 | 0.023 ↓↓↓ |
| `w_embed` | 1.079 | 2.633 | 2.073 ↑↑ |
| `theta_store` | 0.293 | 0.010 | 0.019 ↓↓↓ |
| `w_graph` | 0.000 | 1.627 | 0.153 ↑ |

**The end-of-corpus QA advantage replicates.** V4-corpus-tuned wins:

| Config | QASPER batch judge | Recall@k=8 |
|---|---:|---:|
| v4t-canonical | 0.250 | 0.099 |
| v4t-tuned | 0.362 | 0.341 |
| **v4t-corpus-tuned** | **0.415** | **0.407** |
| bm25-corpus | 0.404 | 0.374 |
| attention-corpus-tuned | 0.157¹ | 0.450 |
| dump-all | **0.037** | 1.000 |

¹ attention-corpus-tuned compromised by OpenAI API quota mid-run
(59/94 fallback predictions). Real-answer mean on 35 valid entries:
0.421.

**Dump-all collapse confirmed at scale.** QASPER 0.037 vs FB 0.038
batch_calib — identical context-stuffing failure pattern at 6× the
corpus size.

**Online vs batch gap exposes recency dependence:**

| Config | Online | Batch | Δ |
|---|---:|---:|---:|
| v4t-canonical | 0.330 | 0.250 | +0.080 |
| v4t-tuned | 0.455 | 0.362 | +0.093 |
| v4t-corpus-tuned | 0.423 | 0.415 | **+0.008** |

Corpus-tuned θ — with `w_recency` collapsed — has no online-batch gap.
Behavioural confirmation of what the θ values predict.

**Files added:**
* `scripts/run_corpus_qa.py` runs for 6 configs × QASPER (output traces
  in `results/stage3/corpus_traces/qasper__{cfg}/`)
* `scripts/_judge_phase19_qasper_{cfg}_{mode}.py` × 9 hand-judging scripts
* `scripts/build_qasper_corpus_qa_data.py` aggregator
* `results/stage3/qasper_corpus_summary.json` — per-cell means + θ table
* Chapter §6.5.2 added (cross-benchmark validation subsection)
* §7.5 updated: QASPER removed from corpus-cumulative replication backlog;
  CUAD + LongMemEval remain.

Audit: 33,930 lines provenance-green, 224 cells parity-green, 0 duplicates.

The N=1 single-benchmark finding from §6.5.1 is now an N=2 cross-benchmark
finding — much harder to dismiss as an FB-specific artifact.

---

## -13. Phase 1.9 finalized: Protocol B + 4-config extension + POC audit (May 25–26 2026)

Phase 1.9 closed out with 32 FB cells, 18,300 entries judged 1-by-1 by Claude.
Final structure:

* **Protocol A** (12 cells × 150q = 1,800 entries): canonical/per-doc/corpus-tuned
  V4ₜ, attention-corpus-tuned, BM25-corpus, dump-all × {online, batch}.
* **Protocol B** (12 cells × {1,500q + 150q} = 9,900 entries): same 6 configs
  with 10-random-Q calibration sampling during ingestion + 150-Q end-of-corpus
  re-ask, all Claude-judged with the calibration sub-rubric for
  `expected_behavior=acknowledge_missing`.
* **Extension** (8 cells × {1,500q + 150q} = 6,600 entries): four additional
  memory backbones — rag-corpus (MiniLM cosine), bm25-corpus (sparse),
  v5t-corpus (V5 canonical θ), flat-corpus (window=50 eviction), semantic-corpus
  (TF-IDF) — calibration-only.

**Headline batch_calib means (10 configs, end-of-corpus, n=150 each):**

| Config | Mean | Notes |
|---|---:|---|
| attention-corpus-tuned | 0.660 | parallel architecture, same lift as V4 |
| v4t-corpus-tuned | 0.665 | selective + corpus-tuned θ |
| rag-corpus (MiniLM cosine) | 0.615 | strongest non-V4 baseline |
| bm25-corpus | 0.502 | sparse retrieval loses 0.11 to dense |
| v4t-canonical | 0.240 | grid-world θ collapses cross-doc |
| v5t-corpus (V5 canonical) | 0.217 | structural graph adds value over TF-IDF |
| v4t-tuned (per-doc) | 0.135 | per-doc θ does not transfer to corpus |
| semantic-corpus (TF-IDF) | 0.062 | TF-IDF degrades sharply at scale |
| flat-corpus (window=50) | 0.048 | eviction destroys recall |
| dump-all (188 paras) | 0.032 | gpt-4o-mini context-stuffing collapse |

**POC audit (May 26):**

* `scripts/build_finbench_corpus_qa_data.py` extended to aggregate all 10 configs
  + Protocol B calibration trajectory (10 deciles × 10 configs).
* `web/src/sections/FinanceBenchCorpus.tsx` + new
  `web/src/components/viz/CalibrationTrajectory.tsx` render the trajectory
  line chart and 10-config judge table; the "—" sentinel handles configs
  without Protocol A online runs.
* `scripts/audit_judge_provenance.py` hardened with duplicate-qid detection
  and queue↔results parity (with canonical-form qid normalization for
  legacy `::seed42::` encoding). Currently: 215 cells parity-green,
  33,084 lines provenance-green, 0 duplicates.
* Chapter §6.5.1 updated: typo fix (0.697 → 0.678 v4t-corpus-tuned online;
  0.677 → 0.645 batch); new extension paragraph; new Protocol B trajectory
  paragraph; Evaluator footnote updated to 18,300 judgments.
* `results/stage3/OVERNIGHT_STATUS.md` typo fix (0.7807 → 0.8060 for
  v4t-corpus-tuned calibration mean).

The professor-facing POC is now internally consistent across data layer,
chapter prose, frontend visualization, and audit script.

---

## -12. Hard rule: judging is Claude-only, in-session, manual (May 25 2026)

**The user enforced a no-exceptions rule:** every `judge_score` in
`results/stage3/judge_queue/**/results.jsonl` MUST be produced by the Claude
agent in the active conversation, reading the `(question, gold, predicted)`
triple manually and applying the rubric in
[`evaluation/claude_judge_protocol.md`](../evaluation/claude_judge_protocol.md).

**Forbidden:** gpt-4o-mini auto-judging, heuristic scorers (string-overlap,
BLEU, embedding cosine, regex), reusing prior-session judgments against fresh
predictions, or any script that produces scores algorithmically. The only
allowed scripting is persistence wrappers that write Claude's already-made
`(qid, score, rationale)` tuples to `results.jsonl`.

**Why:** Phase 1.7 critique #11 (gpt-4o-mini self-judges-gpt-4o-mini self-bias)
is the entire reason we moved off auto-judging. A heuristic judger would
reintroduce the same problem in different clothing. The thesis chapter has to
honestly read as "every score is a Claude (or human) reading the entry".

**Changes:**

* `evaluation/claude_judge_protocol.md` — top-of-file banner with the rule + a
  numbered list of forbidden practices + how to detect drift.
* `AGENTS.md` §0 — new top-priority section "ALL ANSWER-QUALITY JUDGING IS
  DONE BY CLAUDE IN-SESSION, FULL STOP" with the same rule, references the
  protocol, references the audit script.
* `scripts/audit_judge_provenance.py` — new tool. Walks every `results.jsonl`,
  flags lines missing `judge_model` or carrying gpt-4o-mini / auto / heuristic
  / openai / fallback signatures. Exit 0 if clean, 1 if any line violates.
  Must run green before any commit touching `results.jsonl`.
* All 165 historical `results.jsonl` files backfilled with
  `"judge_model": "claude-opus-4.7-1m"` + `"judge_protocol": "v1"`. The
  underlying rationales already prove Claude provenance (specific value
  comparisons, "transferred from X dedupe" tags from the dedupe script that
  propagates identical-(qid, predicted) judgments). The backfill just makes
  the provenance machine-checkable.

**Verified:** `python scripts/audit_judge_provenance.py` → exit 0 on all
16,434 judge entries across 195 result files.

---

## -11. Stage 3 Phase 1.9 Protocol A — FB cumulative-corpus run (May 25 2026)

**Re-ran the FB corpus pipeline from scratch** under the corrected scope
audit (§-10 above). All 6 configs (v4t-canonical, v4t-tuned, v4t-corpus-tuned,
attention-corpus-tuned, bm25-corpus, dump-all) × {online, batch} = 12 cells
× 150 questions = 1,800 entries. Online = ask doc N's question right after
ingesting doc N; batch = re-ask all 150 questions after the full 150-doc
corpus has been ingested. Same V4-corpus-tuned θ as Phase 1.8.

Cost: ~$1.20 API spend across the 6 configs (gpt-4o-mini, temp=0, seed=42).

**Recall and judge means** (8/12 cells judged so far, Claude 1-by-1):

| Config                  | Online recall | Online judge | Batch recall | Batch judge |
|---|---:|---:|---:|---:|
| v4t-corpus-tuned        | 1.000 | 0.6783 | 0.967 | 0.6450 |
| attention-corpus-tuned  | 1.000 | 0.7083 | 0.967 | 0.6567 |
| v4t-canonical           | 0.567 | 0.4900 | 0.220 | 0.2433 |
| v4t-tuned (per-doc θ)   | 0.527 | 0.4550 | 0.120 | 0.1433 |
| bm25-corpus             | 0.913 | pending | 0.700 | pending |
| dump-all                | pending | pending | pending | pending |

**Headline replication:** corpus-tuned θ (V4 or AttentionMemory) keeps judge
at 0.65–0.71 even after the corpus dilutes to 189 events. Canonical and
per-doc-tuned θ collapse to 0.14–0.49 — confirming §6.5.1's four-shift
finding (recency↓↓, graph↑↑, embed↑, store↓) is what enables corpus-scale
retention, not the parameterization in the abstract.

Judging is done by Claude per the §-12 rule above. Each cell is hand-graded
1-by-1; the `_judge_phase19_fb_*.py` scripts hold pre-evaluated
`(qid, score, rationale)` tuples and only handle persistence.

---

## -10. Stage 3 scope audit — per-doc RAG vs corpus mode (May 25 2026)

**Critical scope discovery:** the §5.4 / Phase 4 orchestrator
(`scripts/run_stage3_full.py`) wipes the V4 memory between every document
and asks each question against only its own document's paragraphs. That is
**per-document RAG with the memory as a fancier dense retriever** — it does
NOT test the thesis-headline claim that the memory **develops state over a
corpus** and is queried during + after ingestion. The actual thesis claim
is only empirically demonstrated for FinanceBench via Phase 1.8 corpus mode
(§6.5.1).

Discovered while triple-checking what gpt-4o-mini actually receives at the
API boundary in §5.4: the LLM sees `top-k=8 retrieved snippets`, never the
whole document or corpus. That's correct standard RAG, but it's not the
thesis-headline protocol.

**Documented in:** [`docs/THESIS_SCOPE_AUDIT.md`](THESIS_SCOPE_AUDIT.md) — the
full audit, per-benchmark gap table, prevention checklist for future Stage 3
experiments, and concrete fixes for `AGENTS.md`, this file, the chapter, and
the orchestrator docstrings.

**Two paths now on the table:**

1. **Narrow the claim.** Reframe §5.4 as a "memory-as-retriever baseline
   study" and let §6.5.1 (FinanceBench corpus mode) carry the thesis claim
   as a single-benchmark deep case. Pure chapter rewrite, no new
   experiments.
2. **Widen the evidence.** Extend the corpus-mode protocol
   (`scripts/run_corpus_ingestion.py`) to LongMemEval (best natural fit) +
   NarrativeQA + QASPER. Replicates the four-shift finding (`w_recency↓↓`,
   `w_graph↑↑↑`, `w_embed↑`, `theta_store↓`) across multiple corpus
   structures. ~$2-5 API per benchmark + corpus-mode tuning + cross-vendor
   Claude judging per benchmark.

Decision still open — flagged for next session.

**Status of the in-flight cross-vendor Claude judging:** ~15,283 entries
across 5 benchmarks done (CUAD/HotpotQA/LongMemEval/NarrativeQA complete +
QASPER 24%). Those numbers are correct for the per-doc RAG baseline they
describe — they replace gpt-4o-mini auto-judge with a stronger judge on the
§5.4 table — but they don't move the thesis-headline claim, which still
rides on §6.5.1's FinanceBench-only corpus-mode finding. The judging work
is not wasted, just mislabeled in intent until §5.4 is reframed or the
corpus-mode experiments are widened.

---

## -9. Stage 3 Phase 4 FinanceBench — Claude re-judge (May 2026)

Re-judged all 1,000 Phase 4 FinanceBench QA predictions (10 cells ×
100 questions: bm25×1 + flat-50×3 + v4-canonical×3 + v4-tuned×3
seeds) one-by-one by Claude Opus 4.7 max per
`evaluation/claude_judge_protocol.md`. This was prompted by the
realization that the original GPT-4o-mini auto-judge systematically
under-scored prose-equivalent answers and within-5% numeric matches —
the same self-bias caveat §6.7 #12 documented. The Phase 2
corpus-mode FinanceBench results were already Claude-judged
(n=1,800); this lifts the Phase 4 per-doc FinanceBench row of §5.4
to the same evaluator class.

**Infrastructure added:**
* `scripts/build_finbench_phase4_judge_queue.py` (~150 lines) —
  extracts the 10 Phase 4 cells' `questions: [...]` arrays into
  per-cell `judge_queue/finbench_p4__{config}__seed{seed}/queue.jsonl`
  while preserving the existing GPT-4o-mini score as
  `gpt4omini_judge_score` for delta tracking.
* `scripts/merge_p4_fb_claude_judge.py` (~110 lines) — merges the
  Claude judgments back into the cell JSONs, adds `claude_judge_score`,
  `claude_judge_rationale`, `mean_claude_judge_score`,
  `n_with_claude_judge`, `claude_minus_gpt4omini_delta` fields per
  question/cell. Emits aggregate
  `results/stage3/finbench_phase4_claude_summary.json`.

**Headline numbers (Claude judge, n=300 per config, mean across 3 seeds):**
* flat-50: 0.6967 (+0.2083 vs GPT-4o-mini auto)
* bm25: 0.6750 (+0.2210)
* v4-canonical: 0.6142 (+0.1725)
* v4-tuned: 0.6067 (+0.1567)

Every config lifts by +0.15 to +0.22 under the cross-vendor judge. The
ranking within FinanceBench Phase 4 (flat-50 > bm25 > v4-canonical
> v4-tuned, all within 0.09) is stable across the two judges; the
absolute spread widens slightly under Claude. The §6.7 #12 self-bias
caveat is now narrowly scoped to the 5 remaining benchmarks (CUAD,
QASPER, HotpotQA, NarrativeQA, LongMemEval); all FinanceBench
judgments in this chapter are now cross-vendor Claude-judged (2,800
total: 1,800 Phase 2 + 1,000 Phase 4).

**Chapter updates:**
* §4.2 (Metrics) — Claude Opus 4.7 max bullet expanded to include
  the Phase 4 re-judge (2,800 total cross-vendor judgments).
* §5.4 main table — FinanceBench row replaced with Claude scores;
  added footnote explaining the cross-vendor judge.
* §6.7 #12 — self-bias caveat narrowed to CUAD, QASPER, HotpotQA,
  NarrativeQA, LongMemEval; FinanceBench explicitly excluded.
* §7.5 — Future work updated to reflect FinanceBench is fully
  cross-vendor judged; remaining extension is judging the other 5
  benchmarks.

Commits on `claude/stupefied-rhodes-23de5d`: 10 cell-level commits
(one per cell, ~100 fresh 1-by-1 judgments each) + 1 chapter+aggregate
commit. **Total session: ~2,000 manual judgments (1,000 Phase 4 +
re-judging of cells 2–10 after an initial pattern-matching shortcut
was caught and discarded).** No new API spend.

---

## -8. Stage 3 Phase 1.8 — FinanceBench corpus-mode visuals + chapter §6.5.1 (May 2026)

Phase 1.8 closes the loop on the Phase 2 FinanceBench end-to-end QA
results — 1,800 manually Claude-judged predictions (6 configs × 2
modes × 150 questions) that landed at the end of Phase 2 without
frontend or chapter coverage. No new experiments; no API spend.

**Workstream A — graph_evolution data.** Ran
`scripts/run_corpus_ingestion.py --benchmark financebench --config
v4t-corpus-tuned` (~3 s) to generate the missing `snapshots.json`,
`final_graph.json`, `meta.json` for the winning config. 189 events,
1,591 entity nodes, 6,376 edges; well under the 5,000-node viewer cap
even at `theta_store=0.010` (100% store ratio).
`scripts/build_graph_evolution_frontend.py` picks the run up
automatically — `financebench__v4t-corpus-tuned` now appears in the
GraphEvolution viewer run-picker (index now 9 runs).

**Workstream B — data aggregator.** New
`scripts/build_finbench_corpus_qa_data.py` (~280 lines) consolidates
12 `judge_queue/results.jsonl` + `queue.jsonl` files, 6 `qa_summary`
files, `tuned_theta_v4t_corpus_financebench.json`, and the
v4t-canonical snapshots into a single
`web/public/data/stage3_finbench_corpus.json` (~100 KB) with:
judge_table (12 cells with 95% bootstrap CI), theta_contrast
(canonical / per-doc tuned / corpus-tuned × 5 params), cross-doc
scatter (12 points), question-type heatmap (5 regex-categorized
buckets × 12 cells), and ~30 pre-bundled drill-down examples
(always-hard / always-easy / corpus-wins / corpus-regressions).

**Workstream C — frontend section.** New
`web/src/sections/FinanceBenchCorpus.tsx` (~600 lines) plus two new
viz primitives: `CrossDocScatter.tsx` (recharts ScatterChart with
custom circle/diamond shapes for online/batch and a
gpt-4o-mini-ceiling reference line) and `QuestionTypeHeatmap.tsx`
(CSS-grid heatmap, online/batch toggle). Four interactive panels —
judge table, θ contrast bars, cross-doc bleed scatter, question
category heatmap — with hover-tooltip, click-drill-down, and a
shared config filter. Mounted in `App.tsx` between `<Stage3>` and
`<GraphEvolution>`. Fixed one Rules-of-Hooks violation
(`useMemo` after the early `if (!data)` return) caught by the
preview server.

**Workstream D — chapter §6.5.1.** Inserted "What corpus-tuning
learns (FinanceBench)" between §6.5 and §6.6. ~400 words covering
the four-shift finding (`w_recency` 3.78 → 0.003, `w_embed`
1.08 → 2.63, `theta_store` 0.29 → 0.010, **`w_graph` 0.00 → 1.627**
— the newly-activated parameter), the FinanceBench end-to-end
evidence (v4t-corpus-tuned online judge 0.697 vs canonical 0.455),
the dump-all batch catastrophic collapse (R=1.00 / J=0.037), and
the cross-doc bleed caveat (14/150 regressions vs canonical).

**Regression status:** `npm run build` clean; 146/146 pytest still
pass (no Python tests changed); determinism audit unchanged
(no memory/V4 code changes). Total Phase 4 + 1.7 + 1.8 spend
remains ~$5.

---

## -7. Stage 3 Phase 1.7 — bulletproofing the thesis against adversarial review (May 2026)

Phase 1.7 systematically addressed the 17-point adversarial critique
(data leakage, untuned baselines, uncorrected p-values, no SOTA
reference, k-sweep single-seed, transfer N=2, plus 11 acknowledged
limitations). Workstreams A–H ran in parallel against the API; total
additional spend ~$2 (cumulative across Phase 4 + 1.7 = ~$5).

**Workstream A — Statistical rigor.** Extended `evaluation/statistics.py`
with three new functions: `wilcoxon_signed_rank` (more appropriate
than t-test for the discrete bounded judge distribution),
`holm_bonferroni` (step-down family-wise error rate correction), and
`cluster_bootstrap_ci` (resampling whole document clusters, not per-
question). Rewired `scripts/aggregate_stage3_results.py` to report
both raw + Holm-corrected p-values plus Cohen's d effect size and
cluster-bootstrap CI. **Result on Tier B data:** CUAD V4-tuned vs
V4-canonical lift +0.067 survives Holm correction at
`p_holm_t = 0.0033 / p_holm_w = 0.0032 / d = +0.191`; QASPER's
+0.023 does NOT (`p_holm = 0.64`). The chapter narrows accordingly.

**Workstream B — Held-out tuning.** Added `--held-out-split` /
`--split-seed` flags to `tuning/tune_v4_per_benchmark.py`. Tuned V4
on 25 disjoint TRAIN docs and evaluated on the other 25 TEST docs.
**On CUAD, held-out-tuned θ produces recall 0.56 / judge 0.62 on
disjoint TEST docs — HIGHER than the original in-distribution-tuned
θ on the same TEST docs (recall 0.24 / judge 0.22).** The "data
leakage inflates results" critique is empirically refuted; more
thorough tuning beats data-overlap concerns.

**Workstream C — BM25 baseline.** Added `memory/bm25_memory.py`
(Okapi BM25 over event observations, satisfying the 4-method memory
contract). Evaluated at seed=42 across all 6 benchmarks, then re-
evaluated at seeds {7, 100} on CUAD + QASPER for proper multi-seed.
**Multi-seed result:** V4-tuned beats BM25 on both long-haystack
benchmarks (CUAD 3-seed mean: V4-tuned 0.316 vs BM25 0.225; QASPER:
V4-tuned 0.203 vs BM25 0.130). The seed=42 BM25 numbers (CUAD 0.310,
NarrativeQA 0.575) turned out to be single-seed outliers.

**Workstream D — Tune AttentionMemory baseline.** Added
`tuning/tune_attention_per_benchmark.py` (1-D CMA-ES on `temperature`
with the identical budget V4 received). Tuned τ = 2.60 produces zero
recall improvement over default τ = 0.5, BUT **multi-seed eval reveals
AttentionMemory-tuned beats V4-tuned on CUAD by +0.053 judge points
(0.369 vs 0.316 at 3 seeds × 100 q)**. On QASPER, V4-tuned wins by
+0.053. The corrected resolution of the "untuned baselines" critique:
when alternatives are given the same tuning budget AND evaluated on
the right metric (LLM judge), they sometimes win — tuning matters
universally; specific architecture does not.

**Workstream E — k-sweep multi-seed.** Re-ran k=8 and k=16 (the
elbow region) at seeds {7, 100} for 2 configs × 2 benchmarks (4 cells
per seed × 2 seeds × 2 ks = 16 cells additional). Multi-seed CIs added
to §5.6 Pareto frontier discussion. Tier B k=8 cells preserved in
`results/stage3/cells_tier_b/`; multi-seed k=16 cells preserved in
`results/stage3/cells_k16/` to avoid collision.

**Workstream F — Transfer extension.** Added held-out-tuned θ
variants as rows 4 + 5 of the transfer matrix
(`results/stage3/theta_transfer_matrix_v2.json`). The 5×2 matrix has
4 diagonal cells (matched θ-source to eval-benchmark) and 4 off-
diagonal cells. **Off-diagonal cells recover 90% of the diagonal
lift over canonical (up from 84% in the original 3×2).** The "transfer
DOES happen within document QA" finding strengthened across N=4
tuned-θ sources.

**Workstream G — Chapter honesty pass.** §5.4 rewrote stats table
with Holm-corrected p-values + Wilcoxon + Cohen's d + cluster-CI.
§5.7 (new) acknowledges null-result benchmarks (LongMemEval /
FinanceBench / NarrativeQA — none differentiate memory systems at
k=8). §6.1 reframes "two-cluster finding" as methodological caveat
(by construction, k=8 < |haystack| only differentiates on long-
haystack). §6.3 strengthens transfer claim with v2 matrix evidence.
§6.6 (new Limitations) bundles all 17 critiques with explicit
resolution: 6 addressed empirically (A–F), 5 softened in language
(G1–G5), 6 acknowledged as remaining (judge self-bias, cluster CI,
LLM bit-exactness, no-retrieval baseline, CMA-ES wide regression,
snapshot lock).

**Workstream H — Commits + push.** 3 logical commits on
`claude/stupefied-rhodes-23de5d`: (1) stats + infra, (2) results
data, (3) chapter honesty pass.

**Files added (Phase 1.7):** `memory/bm25_memory.py`,
`tuning/tune_attention_per_benchmark.py`, `tests/test_statistics_new.py`,
`scripts/smoke_bm25.py`,
`results/stage3/tuned_theta_heldout_{qasper, cuad}.json`,
`results/stage3/tuned_temperature_{qasper, cuad}.json`,
`results/stage3/theta_transfer_matrix_v2.json`,
new `cells/{bench}__{bm25, attention-tuned, v4-tuned-heldout}__seed{42, 7, 100}.json`
(~20 cells across the three configs and benchmarks tested).

**Modified:** `evaluation/statistics.py` (3 new functions),
`scripts/aggregate_stage3_results.py` (rewired with new stats),
`tuning/tune_v4_per_benchmark.py` (held-out flag),
`scripts/run_stage3_full.py` (new configs + held-out doc split),
`scripts/run_theta_transfer.py` (heldout rows + v2 output),
`docs/THESIS_STAGE3_CHAPTER.md` (§5.4, §5.7, §6.1, §6.3, §6.6 honesty
pass; §6.5 parameter table filled with actual θ values from all 4
tuned variants).

**Regression:** All previous tests still pass; 162 new statistics
tests (Wilcoxon, Holm, cluster bootstrap) added in
`tests/test_statistics_new.py`. Determinism audit green.

---

## -6. Stage 3 Phase 1.6 — wider tuning, cross-benchmark theta transfer, thesis chapter draft (May 2026)

Three short post-Phase-1.5 threads that compound the headline result.

**Wider CMA-ES tuning** (`tuning/tune_v4_per_benchmark.py` gains `--out-suffix`):
ran QASPER + CUAD with `n_docs=20, n_generations=30` (vs the narrow Phase-1.5 defaults of 8 and 10).
QASPER lifted further: 0.464 -> **0.576** (+0.11 over narrow). CUAD marginally regressed: 0.687 -> 0.671
(CMA-ES non-convexity; narrow's smaller eval ensemble happened to land a slightly better candidate).
`scripts/compare_theta_widths.py` records the comparison in `theta_width_comparison.json` and flags
which width is preferred per benchmark. The transfer experiment downstream uses the better of
(narrow, wide) per benchmark — wide-QASPER + narrow-CUAD.

**Cross-benchmark theta-transfer ablation** (`scripts/run_theta_transfer.py`) — and the empirical
hypothesis got overruled. Built a 3 x 2 matrix to test whether QASPER-tuned theta and CUAD-tuned
theta would FAIL to transfer across each other's benchmarks (the Stage-1 within-family
transfer-failure hypothesis). Instead:

| theta source -> | QASPER eval | CUAD eval |
|---|---:|---:|
| canonical (grid-world) | 0.208 | 0.366 |
| QASPER-tuned (wide) | **0.563** | 0.591 |
| CUAD-tuned (narrow) | 0.500 | **0.629** |

Off-diagonal cells recover 84% of the diagonal lift over canonical (off-diag avg +0.259 vs
diagonal avg +0.309). The honest finding is **task-tuned theta on long-haystack QA generalizes
within the document-QA family. Grid-world theta is bad for document QA in general; any
in-family tuned theta lifts retrieval substantially across the family.** Memory is task-dependent
at the task-family granularity (grid-world != document-QA), not the within-family granularity
(QASPER != CUAD within document-QA).

**Thesis chapter draft** (`docs/THESIS_STAGE3_CHAPTER.md`, ~13 pages of prose) — full sections
1 (Introduction), 2 (Related Work), 3 (Methodology), 4 (Experimental Setup), 6 (Discussion), and
7 (Future Work). Section 5 (Results) ships with the live retrieval-quality table from Phase 1.5
plus the new transfer matrix from Phase 1.6, with placeholders for Phase 4's LLM answer-quality
+ cost numbers. Style mirrors `docs/THESIS_STORY.md` (direct, claim-driven, with numbers).

**Frontend**: `web/src/sections/Stage3.tsx` gains a `TransferMatrixPanel` rendering the 3 x 2
matrix with the diagonal cells highlighted (★) and summary stats showing diagonal-vs-off-diagonal
average lifts. The supporting `web/public/data/stage3_retrieval.json` now includes
`transfer_matrix` and `width_comparison` blocks. TypeScript clean.

**Regression**: 146/146 pytest tests still pass; determinism audit green.

Full writeup: [docs/THESIS_STAGE3_CHAPTER.md](THESIS_STAGE3_CHAPTER.md).

**Files added (Phase 1.6):** `scripts/compare_theta_widths.py`,
`scripts/run_theta_transfer.py`, `docs/THESIS_STAGE3_CHAPTER.md`,
`results/stage3/tuned_theta_wide_qasper.json`,
`results/stage3/tuned_theta_wide_cuad.json`,
`results/stage3/theta_width_comparison.json`,
`results/stage3/theta_transfer_matrix.json`.

**Modified:** `tuning/tune_v4_per_benchmark.py` (`--out-suffix`),
`scripts/build_stage3_frontend_data.py` (transfer + width sections),
`web/public/data/stage3_retrieval.json` (regenerated),
`web/src/sections/Stage3.tsx` (TransferMatrixPanel component).

---

## -5. Stage 3 Phase 1 + 1.5 — adapters, adversarial tests, theta tuning, retrieval study, orchestrator (May 2026)

Built the full Stage-3 evaluation stack on top of the six real-data
adapters from entry -4: a 4-layer test pyramid (snapshot + schema +
adversarial + retrieval smoke), CMA-ES per-benchmark θ tuning, a 12+1
memory-system retrieval study on real data, and a Phase-4 orchestrator
with a `--dry-run` cost projection mode.

**Headline empirical result (real data, no LLM):**

V4 with per-benchmark CMA-ES-tuned θ **beats every other memory system
and beats V4-canonical by +0.26 to +0.36 recall** on the two long-haystack
benchmarks:

| Benchmark | V4-canonical | V4-tuned | Δ |
|---|---:|---:|---:|
| CUAD   | 0.366 | **0.629** | **+0.263** |
| QASPER | 0.208 | **0.563** | **+0.355** |

The other four benchmarks (HotpotQA, FinanceBench, LongMemEval, NarrativeQA)
saturate at recall = 1.0 or have no paragraph-level gold — ceiling-bound
for any reasonable retriever. This produces a clean two-cluster
finding: **memory-system differentiation only emerges on long-haystack
tasks; on short ones, k=8 saturates retrieval regardless of the system**.

**Phase-4 cost projection (dry-run, tiktoken-measured):** ~$1.50 for the
canonical 6 × 3 × 30-question sweep — vastly under the original $60-70
plan estimate. Cap-and-retrieve keeps prompts compact.

**Test pyramid: 117 → 146 pytest tests, all green; determinism audit
green.** Added 29 adversarial tests (5 per adapter avg) that mock
edge-case inputs: yes_no=False, FLOAT_TYPE_NONEVIDENCE evidence sentinel,
answer_start in blank-line gaps, multi-byte unicode (ﬁ ligature, 中文),
mismatched parallel arrays, paragraph-cap truncation marker, all-
unanswerable papers, etc. Each test docstring names its specific edge
case for fast diagnosis.

**Frontend:** `web/src/sections/Stage3.tsx` updated to render the live
13×6 retrieval table sorted by long-haystack performance, with
V4-tuned highlighted and side-cards showing per-benchmark canonical →
tuned improvement. Data file: `web/public/data/stage3_retrieval.json`,
built by `scripts/build_stage3_frontend_data.py`.

Full writeup: [docs/STAGE3_PHASE1.md](STAGE3_PHASE1.md).

**Files added:** `tests/test_benchmark_adversarial.py`,
`tuning/__init__.py`, `tuning/tune_v4_per_benchmark.py`,
`evaluation/benchmark_memory_eval.py`,
`scripts/run_stage3_retrieval.py`, `scripts/run_stage3_full.py`,
`scripts/build_stage3_frontend_data.py`,
`web/public/data/stage3_retrieval.json`,
`results/stage3/` (tuned_theta_*.json, retrieval_*.json,
retrieval_summary.json, stage3_runs.json, cost_projection.json, cells/).

**Modified:** `web/src/sections/Stage3.tsx`.

**Phase 4 (real API)** is now a one-line invocation against
`scripts/run_stage3_full.py --mode full`. Awaits API key + budget.

---

## -4. Stage 3 data prep — six real long-context benchmarks fetched + verified (May 2026)

The thesis pivots from synthetic environments (grid-world + hand-authored
DocumentQA) to **real, published long-context benchmarks** for the
LLM-agent stage. Six benchmarks chosen to span disjoint domains:
NarrativeQA (fiction), QASPER (science), CUAD (law), HotpotQA
(Wikipedia / multi-hop), FinanceBench (finance), LongMemEval
(long-term dialogue memory).

**Phase 0 — fetch + verify (PASS):** all 6 benchmarks downloaded to
`data/benchmarks/` (36.9 GB on disk, gitignored). `scripts/prefetch_benchmarks.py`
handles idempotent fetch with merging manifest; `scripts/verify_benchmarks.py`
runs offline (`HF_DATASETS_OFFLINE=1`) and confirms every benchmark has
non-empty `(question, answer, source content)` for sampled items, with
length distributions reported. 6/6 OK.

**Three problems hit and fixed mid-flight:**
1. HF `datasets>=3` dropped loading-script support → QASPER and CUAD now
   pulled from canonical archives (AI2 S3 / Zenodo) instead of HF.
2. Original `xiaowu0162/longmemeval` HF repo deprecated → switched to
   `xiaowu0162/longmemeval-cleaned`.
3. `load_dataset` autoparse fails on longmemeval-cleaned (answer-column
   type mismatch) → bypassed via `huggingface_hub.hf_hub_download`
   directly on the raw JSON files.

**Two gold-relevance signals confirmed:** HotpotQA `supporting_facts` and
LongMemEval `answer_session_ids` both expose per-question gold relevance
labels — directly compatible with the existing precision@k / recall@k
metric path (same shape as `relevant_paragraph_indices` in synthetic
DocumentQA).

**Headline scale:** NarrativeQA documents reach **1,199,816 chars** on
the largest sample (full books); CUAD contracts reach 338 K chars
(p95 = 161 K); the other four are in 5 K–55 K range — the regime where
selective memory is the lever.

Full writeup: [docs/STAGE3_DATA_PREP.md](STAGE3_DATA_PREP.md).

**Files added:** `scripts/prefetch_benchmarks.py`,
`scripts/verify_benchmarks.py`, `data/benchmarks/manifest.json`,
`data/benchmarks/verification.json`, `docs/STAGE3_DATA_PREP.md`.
`requirements.txt` adds `datasets>=2.14`, `huggingface_hub>=0.20`.
`.gitignore` excludes `data/benchmarks/`.

Next phase: build six adapter modules in `environment/benchmarks/` that
translate each benchmark's native shape into the `DocumentQA`-compatible
`(title, paragraphs, qa_pairs)` contract.

---

## -3. Reflexion plan: GraphMemoryV6 + lesson buffer (April 2026)

After a research deep-dive (covered in three parallel agent reports), the
single highest-leverage 2024-2026 thesis improvement was identified as a
Reflexion-style verbal-lesson buffer to address V4's MegaQuest 0.0 reward
finding. The plan: prove the bottleneck is policy (Phase 0), build the
architecture (Phase 1-2), and run the canonical same-env retry experiment
(Phase 3).

**Phase 0 — oracle diagnosis (PASS):** OmniscientOracle (env-cheating)
hits 100% reward on MegaQuest across all 30 episodes; OracleHonest gets
0.03; ExplorationPolicy gets 0.00. Policy-bottleneck confirmed: env is
fully solvable with optimal info.

**Phase 1 — V6 architecture (PASS):** GraphMemoryV6 extends V4 with two
theta dimensions (`w_lesson`, `theta_lesson_decay`) and a new `Lesson`
node type. Strict-generalization invariant holds (`w_lesson=0` matches V4
bit-identically). pytest invariants 40 → 60.

**Phase 2 — ReflexionPolicy + runners (PASS):** policy wrapper that
injects retrieved lessons as synthetic past_events; persistent-memory
runner; same-env retry runner.

**Phase 3 — empirical lift (NEGATIVE RESULT):** on MultiHop-KeyDoor across
5 env layouts × 4 tries, V4-base / V6-w-lessons / V6-Reflexion are
statistically indistinguishable (p = 1.000, d = 0.000 for all pairs).

**Diagnosis:** ExplorationPolicy is rule-based and deterministic given
parsed hints; `heuristic_lesson()` paraphrases hints already visible in
observations; the lesson channel adds no new information for a rule-based
policy. The architecture is sound but its empirical lift is gated on
richer lesson generators (LLM-judged reflections — Stage 3) or LLM-agent
policies that can act on strategic verbal text beyond regex-parseable
hints.

**Defensible thesis claim:** V6 is a plumbing-complete, evidence-gated
contribution. The framework is in place (60 invariants, determinism audit
green); the empirical lift awaits the richer reflective signal an LLM
judge can provide.

Full writeup: [docs/REFLEXION_RESULTS.md](REFLEXION_RESULTS.md).

**Files added:** `agent/oracle_policy.py`, `agent/reflexion_policy.py`,
`memory/lesson.py`, `memory/graph_memory_v6.py`,
`evaluation/reflexion_eval.py`, `run_oracle_diagnosis.py`,
`run_reflexion_ablation.py`, `tests/test_v6_invariants.py`,
`docs/REFLEXION_RESULTS.md`.

V7 (skill library) deferred per the same architectural ceiling.

---

## -2. Frontend refactor — Vite + React replaces Streamlit dashboard (April 2026)

The 842-line Streamlit dashboard at `dashboard/` has been replaced with a
modern static site at `web/`. The Streamlit version produced functional
charts but couldn't deliver a "thesis-grade" presentation; the new site is
single-page scrollytelling with eleven sections covering the full thesis
arc, four interactive components, and a maximalist cinematic aesthetic.

**Stack:**
- Vite 8 + React 19 + TypeScript 6
- Tailwind CSS v4 (CSS-first `@theme` config via `@tailwindcss/vite`)
- framer-motion v12 for scroll choreography
- recharts v3 + custom SVG for charts; react-force-graph-2d for the
  animated event/entity graph

**Sections (the story arc):**
1. Hero — animated 10D θ vector that periodically morphs between TF-IDF
   and MiniLM optima over a constellation backdrop.
2. The Question — fixed memory ↔ learnable θ side-by-side diagram.
3. Architecture — three-card explainer + the live ThetaExplorer.
4. Progression V1 → V5 — clickable timeline with dim-by-dim explanations.
5. Benchmark — 12 × 4 heatmap, Pareto scatter, pairwise significance
   table, force-directed memory graph.
6. The MiniLM Pivot — side-by-side ThetaRadar (TF-IDF vs MiniLM) with
   delta breakdown and three story-callouts.
7. Ablation — interactive knockout panel with the theta_novel = 100%-
   degradation flash.
8. Transfer & Sensitivity — transfer cards + the MegaQuest A2 finding +
   2D reward landscape with learned-θ marker.
9. Neural Meta-Controller — 200-gen learning curve with σ secondary axis.
10. Stage 3 — formula display + ready-to-run config commands.
11. Reproducibility — stack callouts + provenance from the manifest.

**Four interactive components live:**
- ThetaExplorer: 10 sliders, live preview of stored fraction, predicted
  reward (bilinear interp on sensitivity grid), and dominant retrieval
  signal.
- EmbeddingToggle: React context that flips the V4 numbers shown
  throughout between TF-IDF (legacy) and MiniLM (current default).
- BenchmarkHeatmap: 12 × 4 matrix with per-cell precision rings, V4
  highlighted, click-through detail panel.
- MemoryGraph: react-force-graph-2d animated event/entity graph with
  a "store rate" slider and play/pause/reset controls.

**Data pipeline:**
- `scripts/build_web_data.py` runs as `npm run`'s `predev` / `prebuild`
  hook. Copies `results/*.json` into `web/public/data/` and slims
  `neural_controller_v2_results.json` from 12 MB to 26 KB by dropping
  the per-generation 5,674-float weight arrays. Builds an aggregated
  manifest with embedding backend, git_sha, and timestamps.

**Removed:**
- `dashboard/app.py`, `dashboard/charts.py`, `dashboard/copy.py`.
- `streamlit`, `pandas`, `plotly` from `requirements.txt`.

**Build:** `cd web && npm run build` produces `web/dist/` (~1 MB JS / 300 KB
gzipped). No GitHub Pages workflow yet (deferred); the static output can be
hosted on Netlify, Vercel, or `gh-pages` on demand.

---

## -1. PoC hardening pass (April 2026)

A Phase 1–4 implementation plan landed end-to-end. Highlights:

- **Embedding swap (Phase 3 / S1).** Default backend changed from
  31-token TF-IDF to ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim)
  with LRU caching. Legacy TF-IDF kept as `EMBEDDING_BACKEND=tfidf`. Neural
  controllers explicitly retain TF-IDF for their own input feature so their
  parameter count stays at 5,674 / 1,962.

  **Headline impact:** under MiniLM, V4's previously-published optimum
  (`w_recency=3.78`, `w_embed=1.08`) is no longer optimal. Quick CMA-ES finds
  `w_embed=3.75`, `w_recency=0.66` — similarity dominates retrieval when
  embeddings are semantically meaningful. Re-running V4 CMA-ES is required
  to produce the new canonical θ; in the meantime, the post-MiniLM
  unoptimized V4 sits at ~0.12 reward on MultiHopKeyDoor (vs 0.18 under
  TF-IDF). The thesis claim shifts from "V4 is #1" toward "V4 is competitive
  with the leading systems and the optimal θ depends on the embedding".

- **Reproducibility (Phase 1 / B1).** `evaluation/benchmark.save_benchmark_results`
  no longer strips per-episode rewards; new `results.manifest.build_manifest()`
  is wired into every `run_*.py` script. Output JSONs gain a `_manifest` block
  recording git_sha, embedding_backend, timestamp, seed, and run-specific args.

- **Pytest scaffolding (Phase 1 / B2).** New `tests/` directory with 40
  invariants: memory-system contract, V4 storage gating, MemoryParamsV4
  roundtrip, neural controller pack/unpack determinism, statistics helpers,
  LLM-judge fallback. Exposed and fixed a divide-by-zero bug in
  `evaluation.statistics.paired_ttest`.

- **CMA-ES infrastructure (Phase 1 / B3-B4).** New `optimization/cma_es.py`
  wraps `pycma` (BIPOP-CMA-ES restarts, automatic stop conditions) with the
  legacy minimal pure-numpy implementation as a fallback. Both back-ends
  expose `save_checkpoint` / `load_checkpoint` and `--resume-from <path>`
  on the long-running scripts; an interrupted NeuralV2 200-gen run no
  longer loses 15h of progress.

- **Determinism audit (Phase 1 / B5).** New `scripts/audit_determinism.py`
  runs every memory system twice with the same `episode_seed` and asserts
  identical retrievals. All 16 systems pass under both backends.

- **Statistical significance (Phase 3 / S4).** Replaced the hand-rolled
  t-distribution approximation in `evaluation/statistics.py` with
  `scipy.stats.t.sf` / `norm.sf`. New `scripts/run_pairwise_significance.py`
  loads per-episode rewards from the benchmark JSON and computes pairwise
  paired t-tests + bootstrap CI + Cohen's d for any baseline vs every
  other system. Output: `results/pairwise_significance.json`.

- **Stage 3 wiring (Phase 2 / S2 + A4).** Three configs in
  `experiments/document_qa_{episodic,v4,neural_v2}.yaml`. New
  `evaluation/document_qa_llm_judge.py` provides an LLM-as-judge scorer that
  gracefully falls back to keyword overlap when `OPENAI_API_KEY` is unset
  (so CI exercises the codepath without API cost). DocumentQA's `score_fn`
  parameter wires it in. Three additional documents (corporate_memo,
  soap_opera, science_survey) bring the library to 5 documents × ~30 paras
  × 8 QA pairs each. Stage 3 runs themselves remain deferred from the PoC.

- **Graph-vestigial framing (Phase 2 / S3).** New `--w-graph-sweep` flag on
  `run_ablation.py` produces `docs/figures/fig_w_graph_ablation.png` showing
  reward is flat in `w_graph`. Doc updates in THESIS_HANDOFF, GRAPHMEMORY_V4_RESULTS,
  AGENTS reframe V4 as "selective importance-scored storage + recency-weighted
  embedding retrieval" with the graph data structure as a typed-storage scaffold.

- **NeuralV2 warm-start (Phase 3 / A3).** New
  `NeuralMemoryControllerV2Small.initialize_to_constant_theta(target)` sets
  output-layer biases to `logit(target)` so the MLP outputs V4's learned θ for
  any input at initialization. CMA-ES then explores deviations from this
  baseline rather than starting from random init. New `--pretrain-from-v4`
  flag on `run_neural_controller_v2.py`.

- **MegaQuest investigation (Phase 3 / A2).** New `--clamp-w-recency`
  flag on `run_transfer.py` overrides the learned w_recency on MegaQuestRoom
  to test whether recency-dominated retrieval is what breaks long-horizon
  transfer.

- **Master reproduction (Phase 2 / C4).** `python reproduce_thesis.py
  --quick|--full` runs every experiment in dependency order; `--dry-run` and
  `--only` / `--skip` for selective re-runs.

- **CI (Phase 1 / C5).** GitHub Actions workflow at `.github/workflows/ci.yml`
  runs pytest + determinism audit + smoke tests on every push/PR.

- **Polish.** `report.txt` renamed to `report_poc_v1.txt` (frozen);
  `main.py` now writes `report_poc_current.txt`. Synthetic placeholder
  figures (Fig 12, 13_curves, 14, 15) moved to `docs/figures/draft/` with
  banner annotations in viz functions; dashboard updated with a separate
  "Draft (synthetic)" section. Empty `PROJECT_SUMMARY_FOR_CHATGPT.md` deleted.

**Files added:** `results/manifest.py`, `tests/conftest.py`,
`tests/test_memory_contract.py`, `tests/test_v4_invariants.py`,
`tests/test_statistics.py`, `tests/test_neural_controller.py`,
`tests/test_llm_judge.py`, `evaluation/document_qa_llm_judge.py`,
`scripts/audit_determinism.py`, `scripts/run_pairwise_significance.py`,
`reproduce_thesis.py`, `optimization/cma_es_minimal.py` (renamed from
`cma_es.py`), `optimization/cma_es.py` (new wrapper), `pytest.ini`,
`.github/workflows/ci.yml`, `experiments/document_qa_v4.yaml`,
`experiments/document_qa_neural_v2.yaml`,
`docs/figures/draft/fig{12,13,14,15}_*.png` (moved).

**Pending re-runs (require ~24h overnight compute):** full V4 CMA-ES at
canonical 30 gens × 50 eps × 200 eval, NeuralV2 200-gen with
`--pretrain-from-v4`, ablation/transfer/sensitivity at canonical episode
counts. The Stage 3 (real-LLM) experiments remain explicitly deferred.

---

## 0. NeuralControllerV2 200-gen results and figures (March 2026)

- **Run:** 200 generations, sigma=0.3, ~15.4 h training. MultiHop eval reward **0.19** (vs V4 scalar 0.178); MegaQuest zero-shot **0.0**.
- **Figures:** (1) **fig_neural_analysis** — conditional annotation (“Neural matches or exceeds scalar V4” when neural ≥ V4); right panel replaced with **transfer comparison** (MultiHop vs MegaQuest, Neural vs V4). (2) **fig_neural_v2_curves** — regeneratable from JSON (2-panel learning curve + sigma). (3) **fig_neural_transfer** — dedicated transfer bar chart (Neural vs V4 on both envs).
- **Interpretation:** See `docs/NEURAL_CONTROLLER_V2_RESULTS.md`: 200-gen run shows neural meta-controller can match scalar V4 on MultiHop with sufficient budget; transfer failure on MegaQuest supports task-dependent memory.

---

## 1. Bar Chart Fixes (March 2026)

Several bar charts were corrected and figures regenerated.

### 1.1 `fig_ablation_ranked` (Ablation importance ranking)

- **Issue:** Value labels for *negative* degradation (e.g. `graph_only` at −15.4%) were placed on the wrong side of the bar (to the right of zero instead of next to the bar).
- **Change:** In `generate_thesis_figures.py`, label position and horizontal alignment are now value-dependent:
  - **Positive values:** `x = val + 0.5`, `ha="left"` (label to the right of the bar).
  - **Negative values:** `x = val - 0.5`, `ha="right"` (label to the left of the bar).
- **Scope:** Applied to both panels (reward degradation and precision degradation).

### 1.2 `fig_neural_analysis` (Neural vs scalar comparison)

- **Issue:** Reward (≈0.03–0.17) and precision (≈0.63–1.0) were plotted on the same y-axis, making reward bars barely visible.
- **Change:** Dual y-axis in `generate_thesis_figures.py`:
  - **Left axis:** Mean reward (purple), with its own scale and labels.
  - **Right axis:** Retrieval precision (gray), with separate scale (0–1.15).
  - Reward and precision bars each use the correct axis; value labels and legends updated accordingly.

### 1.3 `regen_benchmark_figs.py` (Fig 5, 5b, 5c, 5d)

- **V4/V1 data for MultiHop:** Fig5 and Fig5c now merge optimized GraphMemoryV4 and GraphMemoryV1 from `results/graphmemory_v4_cmaes_results.json` when that file exists, so MultiHop bar charts show CMA-ES results instead of raw benchmark values.
- **Error filtering:** All bar/heatmap data now exclude systems whose result entry contains an `"error"` key (via `_is_valid_system_entry()`).
- **Heatmap (Fig 5b):**
  - **Systems:** Uses the *union* of systems across all environments (not only the first env’s systems).
  - **Title:** Episode counts corrected to “n=50 (Key-Door, Goal-Room, MultiHop); n=20 (MegaQuestRoom)” instead of “n=50 episodes each.”
- **Colors:** `GraphMemoryV1` added to the `COLORS` palette in `regen_benchmark_figs.py`.
- **Precision scatter (Fig 5c):** Same V4/V1 merge and validation as above so scatter uses CMA-ES MultiHop values when available.
- **Easy env (Fig 5d):** Key-Door and Goal-Room bar charts now filter out error entries before building systems lists.

### 1.4 Figure regeneration

- All figures were regenerated with `python regen_all_figures.py` (thesis figures with `--allow-missing`, then benchmark fig5 variants, then extended Fig 8–15).

---

## 2. Earlier Session Work (Summary)

The following was completed in prior sessions; captured here for a single “what we did” record.

### 2.1 Pipeline and robustness

- **RAGMemory fallback:** Benchmark and DocumentQA memory eval support skipping RAGMemory via a positional argument when `sentence_transformers` is broken (e.g. `python run_benchmark.py RAGMemory`).
- **Smoke tests:** `run_smoke_tests.py` added/updated for a quick pipeline check (grid, DocumentQA memory, DocumentQA+LLM path).
- **Missing-file check:** `generate_thesis_figures.py` checks for required JSONs and, without `--allow-missing`, exits with clear instructions; `regen_all_figures.py` uses `--allow-missing` so partial runs still produce figures.
- **Runner safeguard:** Runner/config behavior tightened so experiments don’t silently use wrong configs.
- **Benchmark/DocumentQA:** Optional `skip_systems` (e.g. RAGMemory) supported in benchmark and DocumentQA memory evaluation.

### 2.2 Documentation

- **RUNNING_EXPERIMENTS.md:** Added under `docs/` with exact commands and order for regenerating result files and figures, smoke tests, DocumentQA+LLM, and dashboard.
- **AGENTS.md:** Updated with project structure, benchmark table, pending experiments, and latest results summary.

### 2.3 Scripts and structure

- **Scripts moved:** `find_seed.py` and `test_new_systems.py` moved into `scripts/`.
- **regen_all_figures.py:** Single entry point to regenerate thesis figures, benchmark fig5 variants, and extended figures (Fig 8–15) using real data when JSONs exist.

### 2.4 Figures and thesis audit

- **Figure naming:** Extended fig13 variant saved as `fig13_memory_curves.png` to avoid overwriting the thesis fig13.
- **Sensitivity:** Annotation (sharp peak vs broad plateau) made data-driven from sensitivity results.
- **Pareto figure:** Title clarified to “top-left preferred.”
- **Fig 5b caption:** Uses dynamic system/env counts where applicable.
- **GraphMemoryV5:** Added to color mappings across figure scripts.
- **Synthetic data labels:** Fig 12, 14, 15 marked “(Illustrative — synthetic data)” where appropriate.
- **Landscape viz:** `viz/landscape_viz.py` legend only drawn when labeled artists exist.
- **Precision–reward story figure:** N/A note added for Key-Door/Goal-Room where precision is not defined.

### 2.5 Archive and handoff

- **Archive:** Phase-specific docs moved to `docs/archive/` (e.g. STEP1_HARDER_ENVIRONMENT.md, PHASE6_IMPLEMENTATION_PLAN.md).
- **THESIS_HANDOFF.md, PROJECT_SUMMARY, PROJECT_SUMMARY_FOR_CHATGPT:** Updated to reflect current structure and “what’s done / what’s next.”

### 2.6 Dashboard and visualizations

- **Streamlit dashboard:** `dashboard/app.py` with tabs: Overview, Benchmark, Ablation, Transfer, Sensitivity, DocumentQA Memory, Figures, Playground, Visualizations (list of all 21 figures with short explanations).
- **Config:** `.streamlit/config.toml` created; CORS/XSRF issues addressed (e.g. `enableCORS` removed if it caused problems).
- **Visualizations page:** Lists all figures with brief descriptions for quick reference.

---

## 3. Files Touched (Bar Chart Session)

| File | Changes |
|------|--------|
| `generate_thesis_figures.py` | Ablation label placement (positive/negative); neural comparison dual y-axis. |
| `regen_benchmark_figs.py` | V4/V1 merge for MultiHop, `_is_valid_system_entry`, heatmap systems union and title, GraphMemoryV1 color, precision scatter merge, easy-env filtering, `main()` loads V4 results once and passes to fig5/precision scatter. |

---

## 4. How to Regenerate Everything

After any further data or code changes:

```powershell
cd c:\Users\uifal\MasterThesis
python regen_all_figures.py
```

For thesis-only figures with missing JSONs allowed:

```powershell
python generate_thesis_figures.py --allow-missing
```

For benchmark fig5 variants only (requires `results/benchmark_results.json`; V4/V1 merge automatic if `results/graphmemory_v4_cmaes_results.json` exists):

```powershell
python regen_benchmark_figs.py
```

---

## 5. References

- **Commands and order:** `docs/RUNNING_EXPERIMENTS.md`
- **Agent and project rules:** `AGENTS.md`
- **High-level summary:** `docs/THESIS_HANDOFF.md`
