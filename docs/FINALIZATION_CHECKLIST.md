# Thesis Finalization Checklist

**Purpose:** make finalization mechanical once the last results land. Created
2026-06-16 after a whole-thesis coherence audit. Authoritative numbers live in
`THESIS.md` + `PROFESSOR_MEMO_2026_06_12.md`.

---

## 1. What is already coherent (done)

- **Six-benchmark batch-lift headline table** (FB +0.402, HQA +0.540, QASPER
  +0.165, CUAD +0.161, LME +0.165, NQA 0.000) — consistent across `THESIS.md`,
  the memo, and `THESIS_STAGE3_CHAPTER.md`.
- **Stage-2 phrasing** ("V4 reaches the top cluster, 0.178 vs 0.173 — a
  statistical tie, not a clean #1") — consistent across spine, memo,
  `ALGORITHMS_AND_FINDINGS.md`, `RECENT_CHANGES.md`.
- **Dump-all framing** ("accuracy-tied, ~18× cost"; old "structurally
  necessary" framing appears only as explicitly-retracted text) — consistent.
- **Held-out verdicts** (FB/CUAD/HQA/LME survive, QASPER n.s., NQA undefined) —
  consistent.
- **Transfer + sensitivity docs/figures** — refreshed to the reproducible
  MiniLM-era θ (commit b9d78c8): transfer 0.140/0.690/0.173/0.000; sensitivity
  `is_sharp_peak=True`.
- **All 24 thesis figures** reviewed for readability; 3 fixed + regenerated.
- **Stage-1/2 appendix detail docs** (`GRAPHMEMORY_V4_RESULTS`, `ABLATION_RESULTS`,
  `BENCHMARK_RESULTS`, `NEURAL_CONTROLLER_V2_RESULTS`) carry currency caveats
  noting their tables are the original **TF-IDF-era** Stage-2 run while the
  committed JSONs + figures reflect the later **MiniLM** re-tune. *(Optional
  alternative: fully renumber these appendix docs to the MiniLM JSONs — not
  done, because the spine deliberately reports the 0.178 TF-IDF number with a
  "doesn't survive MiniLM" caveat. Decide if a full renumber is wanted.)*

---

## 2. Pending "last results" (the moving targets)

### 2a. CUAD Protocol B — remaining 3 configs at full 510-doc scale
- **Owner:** scheduled task `cuad-protocol-b-pipeline-watch` (runs pipeline →
  judges 1-by-1 → commits + pushes, one config at a time).
- **Status:** 3 main V4 configs done at 510-scale; attention-corpus-tuned,
  bm25-corpus, dump-all still at **10-doc pilot scale**. Chapter §6.5.3 Protocol B
  table currently labels these "pilot n=50".
- **Do NOT run a competing CUAD pipeline** while these scheduled tasks are
  enabled — collision + push races. Coordinate or disable them first.

### 2b. CUAD Protocol A — 10 → 50 contract scaling (currently ORPHANED)
- `tuned_theta_v4t_corpus_cuad.json` exists at **limit_docs=50** (w_graph=1.985,
  w_recency=0.003, lift +0.12 in-sample).
- `cuad__v4t-canonical__{batch,online}` queues regenerated to **644 entries**
  but only **132 judged** (512 new unjudged per mode).
- `cuad__v4t-corpus-tuned` **not yet re-run at 50** (still 132/10-doc).
- **To finish:** run corpus-tuned-50 (`run_corpus_qa.py --benchmark cuad
  --config v4t-corpus-tuned --limit-docs 50 --protocol online_batch`), then
  hand-judge the ~2k new entries 1-by-1 (scores in {0,.25,.5,.75,1},
  `judge_model="claude-opus-4.7-1m"`, `rationale` field). **This is the judging
  bottleneck.** Decide: finish it, or keep CUAD at 10 contracts and leave
  `THESIS.md` §6 "scaling in progress" as honest future work.

---

## 3. On results landing — mechanical ingestion

1. **Rebuild aggregator:** `python scripts/build_cuad_corpus_qa_data.py`
   (writes `results/stage3/cuad_corpus_summary.json` + frontend JSON).
2. **Re-run provenance audit for ground-truth counts:**
   `python scripts/audit_judge_provenance.py` → record `<N> judge lines`,
   `<M> cells`, `<P>% rule-assisted`. **As of 2026-06-16: 83,883 lines / 313
   cells / 39.8% rule-assisted, full parity green.**
3. **Update the count figures everywhere** (these are the moving targets, do
   them LAST):
   - `THESIS_STAGE3_CHAPTER.md` §"All judge cells complete" para (judge-line
     count; cell count — make consistent, two places say 311 vs 313) and the
     `audit_judge_provenance.py` description line.
   - `PROFESSOR_MEMO_2026_06_12.md` line ~43 (judge-line count).
   - `RECENT_CHANGES.md` top entry.
4. **Update CUAD numbers** (only if 2a/2b change the CUAD means). Chapter
   anchors to edit — locate by content, not raw line (lines drift):
   - §6.5.3 batch table CUAD row (`CUAD (n=132, full)` → new n + means).
   - §6.5.3 CUAD online-vs-batch table + surrounding prose.
   - §6.5.3 Protocol B CUAD table (pilot n=50 rows) + prose.
   - §6.5.4 held-out CUAD lift (+0.135, tuned-on 0.575 / held-out 0.152).
   - §6.5.3 four-shift table CUAD recall row.
   - `THESIS.md` §4 batch-lift table **CUAD row** (0.023 → 0.184) + §6
     "subset of its 510 contracts; scaling in progress" limitation.
   - `cuad_corpus_summary.json` header `n_docs_ingested` / `n_questions_per_cell`.

---

## 4. Final verification gate (run before declaring done)

```
python -m pytest tests/ -q              # expect 214 passed
python scripts/audit_determinism.py     # expect OK
python scripts/audit_judge_provenance.py# expect parity green, 0 dup qids
cd web && npm run build                 # expect clean
python scripts/build_web_data.py        # refresh frontend JSONs from results/
```

---

## 5. Optional polish (non-blocking, disclosed as future work in the memo)

- Full renumber of the four Stage-1/2 appendix docs to MiniLM JSONs (§1 note).
- Delete 8 unreferenced stale figures (`fig08_ablation`, `fig09_landscape`,
  `fig10_transfer` plain; `fig_story_precision_reward`; draft fig12/14/15 +
  fig15) so `docs/figures/` holds only thesis content.
- Multi-seed corpus replication; dump-all Protocol-B calibration re-run.
