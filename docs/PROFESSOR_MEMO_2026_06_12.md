# Bocconi Master's Thesis — Stage 3 Memo (Honesty Pass)

**To:** Prof. [Professor name]
**From:** Stefan Uifalean
**Date:** June 12, 2026
**Subject:** Stage 3 — adversarial self-critique and the corrections it forced

> This memo **supersedes** `PROFESSOR_MEMO_2026_06_10.md`. That memo repeated
> several claims that a subsequent self-audit found to be wrong or overstated
> (notably "82,866 hand-judged", the dump-all collapse, and the four-shift).
> The numbers below are the corrected, honest state.

---

## TL;DR

I ran an adversarial critique of the entire Stage 3 evaluation and found five
material problems. All five are now addressed in the data and the chapter:

1. **Judging was not all hand-done.** ~53% of the 82,866 judgments were
   rule-assisted (a refusal/acknowledgment classifier + content templates),
   not one-by-one Claude judgments as previously implied. I re-judged the
   **~9,500 content-templated entries one-by-one** with Claude (Opus-class,
   1M context), each independently re-checked by a second adversarial Claude
   pass. The remaining ~22k entries are refusal/acknowledgment cases that keep
   rule-assisted scores; this is now **disclosed** (sample-validation is
   future work, see below).
2. **"Dump-all collapses" was a bug, not a finding.** A 12-event truncation
   cap meant the dump-all (context-stuffing) condition never received the full
   corpus. Fixed and re-run: dump-all is **competitive on accuracy** and only
   loses on **cost**.
3. **The headline tuning was tuned-on-test.** CMA-ES corpus tuning optimized
   recall on questions that the evaluation then re-used. I split every cell
   into tuned-on vs held-out and recomputed.
4. **The "four-shift in θ" is honestly a three-shift.** A `w_graph=0` ablation
   shows the graph term carries no retrieval load, and re-tuning on a MiniLM
   backend reproduces most shifts from the encoder change alone.
5. **n=10 single-seed headline cells** (HotpotQA, LongMemEval) are now flagged
   as underpowered rather than presented beside n=150 cells.

**Verification:** `pytest` 214/214, determinism audit green, provenance audit
green (83,163 judge lines, 0 duplicate rationales, 313/313 cells queue-parity),
frontend build clean.

---

## 1 · Judging honesty

The re-judging was done with a deterministic two-stage workflow: one Claude
subagent scores a 150-question shard one-by-one against the 5-point rubric
({0, .25, .5, .75, 1}), then a second independent Claude adversarially
re-judges the same shard and writes the final scores. Provenance is stamped
`judge_pass=2_manual`. Cells re-judged: all CUAD batch_calib + calibration for
v4t-canonical / v4t-tuned / v4t-corpus-tuned plus the attention/BM25/dump-all
baselines; all QASPER cells; and the FinanceBench / HotpotQA / LongMemEval /
NarrativeQA residuals — **~9,500 entries**.

Honest framing of the judge protocol is now two-tier: (a) one-by-one Claude
judging for content answers, (b) a rule-assisted classifier for
refusal/acknowledgment answers, disclosed as such. The chapter no longer
claims all 82,866 were hand-judged.

## 2 · Dump-all (the biggest single correction)

With the cap removed and the condition re-run + re-judged one-by-one
(n=148 online / 149 batch):

| FinanceBench (Protocol A) | online | batch | cost / 150q |
|---|---:|---:|---:|
| Dump-all (fixed, 188 paras) | **0.689** | 0.607 | **$2.68** |
| V4ₜ corpus-tuned (k=8) | 0.678 | 0.645 | $0.15 |
| Attention corpus-tuned (k=8) | 0.708 | 0.657 | $0.15 |
| BM25-corpus (k=8) | 0.650 | 0.503 | $0.17 |

The 95% bootstrap CIs for dump-all and corpus-tuned overlap, so they are
**statistically indistinguishable on accuracy** at n≈150. The honest argument
for selective retrieval at this corpus scale is **efficiency (~18× cheaper),
not an accuracy cliff.** The old "selective retrieval is structurally
necessary" sentence has been removed from §6.5.1.

*(Caveat, disclosed in the chapter: the Protocol-B calibration trajectory
still reflects the pre-fix capped dump-all; re-running that 1,500-question
cell uncapped is deferred as it does not change the headline.)*

## 3 · Contamination / held-out splits

Splitting each cell into tuned-on vs held-out questions and recomputing with
cluster-bootstrap CIs + Holm-corrected exact tests:

- **FinanceBench corpus-tuned held-out batch lift +0.335 (p_holm < 0.0001) —
  survives.** The effect is real on data the optimizer never saw.
- **CUAD held-out lift +0.135 — significant**, though pooled means were
  inflated by tuned-on questions (0.575 tuned-on vs 0.152 held-out).
- **QASPER batch +0.125 — not significant after Holm correction.**

Held-out recall for the corpus-tuned θ also holds (FB 0.98, HotpotQA 1.00 on
70 unseen questions, LongMemEval 0.83).

## 4 · Four-shift → three-shift

A `w_graph=0` ablation removes essentially zero retrieval load everywhere
(QASPER even improves, 0.508 vs 0.460), so the graph-term "shift" is not load-
bearing. Re-tuning on the MiniLM backend reproduces 3 of 4 shift directions
from the encoder change alone; shift scores drop {4,4,3,4,3} → {4,3,3,3,2}.
The chapter reframes the four-shift as an exploratory observation and reports
it against the MiniLM-era baseline with a noise floor.

## 5 · Other corrections in the chapter

- Corpus tuning spend stated honestly at **~$18.5** (prior "~$5" understated
  it ~4×); per-cell cost columns added.
- NarrativeQA: the adapter never emits paragraph gold, so the recall@k tuning
  objective is undefined by construction — stated plainly (the prior "returned
  all-zeros θ" footnote was wrong).
- n=10 HotpotQA / LongMemEval cells carry per-row n and are described as
  directionally positive but underpowered.

---

## Remaining / future work (disclosed, not hidden)

- **Refusal-classifier sample validation** — hand-judge a stratified 300-entry
  sample of the kept rule-assisted scores to bound their error.
- **HotpotQA / LongMemEval held-out** — re-tune θ at 100 docs and re-run on
  disjoint documents for a clean (non-incoherent-control) held-out number.
- **Dump-all Protocol-B calibration re-run** (deferred, disclosed).
- **Multi-seed corpus replication** (compute is cheap; judging volume is the
  constraint).

All changes are committed on branch `claude/stupefied-rhodes-23de5d`.
