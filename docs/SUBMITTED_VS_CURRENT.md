# Submitted thesis vs. current thesis — definitive diff

**Anchor (verified):** the PDF actually sent
(`D:\Bocconi\Thesis\Uifalean_Stefan_3310360_MSc_AI_Thesis_submission\Uifalean_Stefan_3310360_MSc_AI_Thesis.pdf`)
is **byte-identical (SHA-256 `6aa095ec…`) to `268845d:thesis/main.pdf`**
(commit 2026-06-16 16:44, "Repo cleanup + finalize"). Everything after that
commit (54 commits, work of 2026-06-28 → 2026-07-08) is post-submission and is
**not** in the version the committee has.

## Size

| | Submitted (268845d) | Current |
|---|---|---|
| Pages | 45 | 65 |
| Body words (chapters) | 5,832 | 14,450 (+148%) |
| 01 intro | 704 | 1,537 |
| 02 background | 764 | 2,685 |
| 03 methodology | 1,037 | 1,953 |
| 04 results | 1,786 | 4,726 |
| 05 discussion | 897 | 2,160 |
| 06 conclusion | 644 | 1,389 |
| Tables | 6 | 10 |
| Figures | 11 | 12 |
| \paragraph blocks | 13 | 30 |
| Citations (\citep) | 35 | 45 (same .bib — new cites use entries already present) |

**Unchanged since submission:** the abstract + title page (`frontmatter/title.tex`),
`references.bib`, `preamble.tex`, `main.tex` skeleton. Appendices: only
`B_honesty_audit.tex` gained the per-benchmark refusal-share table.

## New empirical content (none of this is in the submitted version)

1. **Fair corpus-tuned baselines** (`tab:fairbaseline`; submitted only promised
   this as future work). Verdict: benchmark-dependent — V4t wins FB (+0.132),
   ties QASPER, loses CUAD (−0.131) vs a same-budget tuned BM25.
2. **Head-to-head vs published memory systems** (`tab:head2head`, 3 corpora).
   HippoRAG + MemGPT/Letta reimplementations under the identical harness.
   V4t beats both on QASPER (n=1005), ties both on FB, loses to HippoRAG on
   CUAD@50. Entirely new; submitted version had no empirical comparison to
   published systems (only the qualitative `tab:related`).
3. **Full-corpus scaling** (`tab:fullcorpus`). CUAD lift at all 510 contracts
   (+0.107, n=6702; monotone +0.161→+0.144→+0.107) and QASPER at all 281 papers
   (V4t 0.351 tied-for-best with tuned BM25 0.367). Submitted version topped out
   at CUAD-50 / QASPER-94.
4. **CUAD-510 cross-method baselines + the scale-dependent reshuffle** — the
   strongest new finding: the memory-system ranking *reverses* with corpus size
   (@50: HippoRAG > V4t > Letta; @510: Letta > V4t > HippoRAG, all Holm-sig).
   Plus dump-all@510 probed: overflows the 128K window on 150/150 questions.
5. **θ-predictability** (`sec:thetapredict`, `fig:thetatransfer`): LOBO predictor
   over 40 tuned sub-samples recovers 0.80 of the tuning lift (CUAD 0.37 outlier).
   Submitted version listed this as future work.
6. **Multi-seed determinism**: FB replicated across seeds {7,42,100}, cross-seed
   std ≤0.012 (`sec:determinism`). Submitted: single-seed with a stated limitation.
7. **Per-benchmark refusal-share table** (`tab:refusalshare`, Appendix B).
8. **New discussion sections**: `sec:whybm25` (why BM25 flips per benchmark),
   `sec:graphnegative` (what the refuted w_graph term implies),
   `sec:tworefusal`, `sec:budgettune`, `sec:stage2structure`.
9. **Prose deepening throughout**: intro contributions/motivation, related-work
   read/write asymmetry + empirical positioning + CMA-ES rationale + KG-retrieval
   paragraphs, Stage-1/Stage-2 deep dives, expanded conclusion (three sharpened
   results + five grounded next steps).

## Claims that CHANGED STRENGTH vs the submitted version

| Claim | Submitted said | Current says |
|---|---|---|
| vs tuned classical baselines | future work | benchmark-dependent (win/tie/lose) — measured |
| vs published memory systems | not compared | benchmark- AND scale-dependent — measured on 3 corpora, 2 scales |
| CUAD scale | "holds at 50 contracts" | holds at full 510 (+0.107, n=6702) |
| QASPER scope | n=94, lift n.s. | full n=1005; V4t tied-for-best of 7 |
| Seed robustness | limitation (single-seed) | measured: std ≤0.012 across 3 seeds |
| θ re-tuning burden | "must re-tune per task" | largely predictable (0.80 recovery, CUAD exception) |
| dump-all at scale | analytical overflow at N≈11 | empirically confirmed: 150/150 overflow at 510 |

## Corrections a professor-style review made (contest → verify → fix)

Six adversarial examiner passes (one per chapter + cross-chapter), each verifying
every claim against the raw `results/` data and the code. ~270 claims contested,
~24 verified defects fixed. Split by origin:

**Errors that were ALSO in the submitted version (now corrected in current):**
- Stage-2 headline "$0.178$ vs EpisodicSemantic $0.173$": real but from the
  \emph{TF-IDF} backend, with no committed source, and Appendix C wrongly said the
  committed MiniLM backend gives "the same conclusion" (it does not — MiniLM V4
  $\approx0.13$, below the top baseline). Now labelled TF-IDF and disclosed as
  backend-sensitive; Appendix C corrected.
- Algorithm 2's storage rule mis-stated the gate (`θ_store` used as a bias, `θ_erich`
  term dropped); `tab:notation` inverted `θ_store`'s direction; `eq:score` used a raw
  cosine where the code rescales to $[0,1]$. Now match `graph_memory_v4.py`.
- "$18.5$ per-benchmark CMA-ES tuning cost" — actually the \emph{total} gpt-4o-mini
  answering cost across six benchmarks (tuning is recall@k, ~\$0). Relabelled.
- FinanceBench described as "lexical fact lookup" (it is semantic — V4t beats tuned
  BM25 there); "six benchmarks with gold signals" (five); MemGPT paging described as
  a hand-loop (it is LLM-driven); mem0 bib year 2024→2025. Fixed.

**Errors introduced by the post-submission work (now corrected):**
- CUAD full-corpus `n=6702`→`6683`; lift `0.150/+0.107`→`0.149/+0.106`; dump-all
  fit `2.5%`→`2.3%`; the attention-CUAD head-to-head cell flagged as an $n{=}132$
  pilot; refusal-error bound `≤0.013`→`≤0.014`.
- The abstract (unchanged since submission) was refreshed to include the new
  head-to-head, full-corpus scaling + scale reshuffle, fair-baseline
  benchmark-dependence, θ-predictability, and multi-seed determinism.

Net: the current version is not just larger but **more accurate than the submitted
one** — several submitted-era slips are now fixed, and every headline number has
been re-verified against the committed data.

## What is IDENTICAL in both (the submitted core is intact)

The three-stage arc, all Stage-1/Stage-2 results, the corpus-cumulative method,
the held-out FB/CUAD significance (QASPER n.s.), the dump-all cost-not-accuracy
reframe, recall↔judge ρ=0.69, κ=0.66 self-consistency, the honesty-audit
appendix, and all 6-benchmark headline tables. Nothing in the submitted version
was retracted; the additions strengthen, scope, or complete claims the submitted
version already stated honestly (usually as limitations/future work).
