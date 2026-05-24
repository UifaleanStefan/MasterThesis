# Thesis Scope Audit — The Per-Doc-RAG vs Corpus-Mode Confusion

**Status:** CRITICAL — read before planning any Stage 3 experiment
**Date:** May 25 2026
**Author:** discovered during cross-vendor Claude judging session
**Severity:** spent ~1 day of judging effort on the wrong experiment before catching it

---

## The one-sentence summary

**The thesis claim is about a memory that ingests a corpus cumulatively and is
queried during + after ingestion. The §5.4 / Phase 4 experimental setup wipes
the memory between every document, so it never tests that claim — it only
benchmarks V4 as a fancier per-document dense retriever. The actual thesis
claim is only empirically demonstrated on 1 of the 6 benchmarks (FinanceBench
via Phase 1.8 corpus mode).**

---

## What the thesis claim actually is

From `AGENTS.md`:

> Can an agent learn how to construct its own memory representation, and
> should that structure be task-dependent?

The contribution the chapter advertises:

1. A parameterized graph memory whose θ controls **what gets stored** (θ_store),
   **how it ages** (w_recency, θ_decay), **how it's weighted at retrieval**
   (w_embed, w_graph), etc.
2. Evidence that this memory **develops a useful state over a stream of
   inputs** — not that it retrieves well from one document.
3. Evidence that the **optimal θ depends on the task / corpus structure**
   (the §6.5.1 four-shift finding).

What this **requires** to be demonstrated empirically:

- A **single memory instance** ingests **multiple documents cumulatively**.
- The memory **state evolves** as new content arrives (entities accrete,
  edges form, older nodes decay).
- Queries are asked **online** (during ingestion, at specific checkpoints)
  and **batch** (at the end, with the full corpus in memory).
- The retrieval cost / quality trade-off is measured as the corpus grows.

---

## What Phase 4 / §5.4 actually does (the experimental setup we have)

Reference: `scripts/run_stage3_full.py:300-360`,
`evaluation/document_qa_memory.py:_run_reading_phase`.

For each (benchmark, config, seed, k) cell:

```
for each document independently:
    memory = MemoryFactory()        # ← FRESH MEMORY, no state from prior docs
    for paragraph in document.paragraphs:
        memory.add_event(paragraph)
    for question, gold in document.qa_pairs:
        retrieved = memory.get_relevant_events(question, k=8)  # top-k
        predicted = gpt4o_mini.answer(question, context=retrieved)
        judge = gpt4o_mini.judge(predicted, gold)
```

Notice:

- The memory is **wiped between documents**.
- Each question is asked against only its own document's paragraphs.
- The LLM sees the **top-k=8 retrieved snippets**, not the document, not the
  corpus.

This is **standard per-document RAG benchmarking with V4 as the retriever**.
It is a valid baseline ("does our memory at least work as a retriever?") but
it is NOT a test of the thesis claim. The memory never develops a state.
The "evolution" story is invisible. θ_decay, w_recency, w_graph all collapse
to weak signals because the corpus a question sees is N=10-200 paragraphs from
one document — not the cumulative product of hundreds of documents.

---

## What Phase 1.8 / §6.5.1 actually does (the experimental setup we want everywhere)

Reference: `scripts/run_corpus_ingestion.py`,
`results/stage3/corpus_traces/financebench__v4t-corpus-tuned/`.

For FinanceBench (and only FinanceBench so far):

```
memory = V4Memory(theta_corpus_tuned)
for doc in all_150_docs:                              # ← cumulative
    for paragraph in doc.paragraphs:
        memory.add_event(paragraph)                   # state accretes
        snapshot if checkpoint                        # ← saved to snapshots.json
    if doc in online_query_docs:
        for q in doc.online_questions:
            ask(q, memory_state_at_this_point)        # ← online
# end of ingestion
for q in batch_questions:
    ask(q, full_memory_state)                         # ← batch
```

Notice:

- **One memory instance** spans all 150 docs.
- State evolves; `final_graph` has 935 entities and 2,520 edges after the
  whole corpus, not 10-15 per document.
- Questions ask both **online** (during ingestion, retrieving from the memory
  as it was at that point) and **batch** (against the full final memory).
- The CMA-ES tuning that produced `theta_corpus_tuned` was done under THIS
  protocol, which is why it found `w_recency 3.78 → 0.003`, `w_graph 0.000
  → 1.627`, `w_embed 1.08 → 2.63`, `theta_store 0.29 → 0.010` — those shifts
  ONLY make sense when the memory is asked to remember 188 paragraphs of
  multi-doc state, not when it's wiped every 10-20 paragraphs.

This is the actual thesis claim. We have it for **1 of 6 benchmarks**.

---

## What gpt-4o-mini actually receives (the truth I was sloppy about earlier)

Across **every** Phase 4 / §5.4 cell — all 6 benchmarks:

```
system: "You are a question-answering assistant. You receive a question and relevant document passages. Answer the question concisely using only the provided context."
user:   "Relevant passages:
        step 12: <paragraph text>
        step 47: <paragraph text>
        ... (k=8 of these)

        Question: <q>

        Answer:"
```

The LLM **never sees** the whole document and **never sees** the whole corpus.
It always sees the **top-k retrieved snippets** (k=8 by default for §5.4,
varied in §5.6 k-sweep).

The **only** place a "whole corpus" ever hit the OpenAI API was the
**dump-all baseline in Phase 1.8 FinanceBench corpus mode** — that one passes
all ~188 paragraphs of the cumulative FB memory in one prompt, and it
collapses to judge=0.037 (the §6.5.1 "context-stuffing breaks at scale"
finding). That dump-all cell exists ONLY for FinanceBench.

---

## Per-benchmark gap table

| Benchmark | Has corpus-mode? | Natural corpus structure? | Effort to add |
|---|---|---|---|
| **FinanceBench** | YES (Phase 1.8) | medium (150 SEC filings) | done |
| **LongMemEval** | **NO** | **excellent** — multi-session dialogue IS a cumulative corpus | low |
| **NarrativeQA** | NO | strong — each book = one corpus, questions span chapters | medium |
| **QASPER** | NO | strong — each paper = a corpus of sections + abstract | medium |
| **HotpotQA** | NO | weak — each q has 10 distractors, no cross-q accumulation | high (semi-synthetic) |
| **CUAD** | NO | weak — each q is "name of THIS contract", no cross-doc thesis fit | high (semi-synthetic) |

**LongMemEval is the obvious next benchmark to add corpus-mode to** — its
structure (haystack_sessions accumulating over time, queries that depend on
prior session content) is born for this. It's also the one §5.4 currently
shows as "no spread between configs," which is unsurprising under per-doc
RAG and is the wrong question to be asking.

---

## What this means for the chapter

The chapter currently has:

- §5.4 (per-doc RAG table for all 6 benchmarks) ← **measures wrong thing**
- §5.6 (Pareto k-sweep) ← also per-doc RAG
- §5.7 ("no spread" benchmarks) ← actually just "no spread under per-doc RAG"
- §6.5.1 (FinanceBench four-shift finding) ← **measures the right thing**

If we ship the chapter as-is, the headline is FB-only and the rest is
baseline / methodology work that doesn't directly support the thesis claim.
The §6.5.1 finding is real but is a single-benchmark anecdote until corpus
mode is replicated.

**Two paths forward:**

1. **Narrow the claim.** Reframe §5.4 explicitly as "memory-as-retriever
   baseline study" (which it is) and let §6.5.1 carry the thesis claim as a
   single deep case study. Requires chapter rewrite but no new experiments.
2. **Widen the evidence.** Extend corpus mode to LongMemEval (+
   NarrativeQA + QASPER if budget permits) and replicate the four-shift
   finding. Requires ~$2-5 API per benchmark + corpus-mode tuning runs +
   cross-vendor Claude judging.

Both are defensible. Path 1 is faster and honest. Path 2 is stronger but
takes 1-2 weeks of focused work.

---

## How this happened (so we don't repeat it)

- The codebase was built bottom-up: per-doc DocumentQA (Stage 1) → adapters
  (Phase 1) → orchestrator (Phase 1.5) → tuning + retrieval study
  (Phase 1.5) → corpus mode (Phase 1.8 — added LATE).
- Phase 4 inherited the per-doc DocumentQA shape because that's what the
  orchestrator was built against. Nobody re-examined whether that shape was
  testing the actual claim.
- The chapter was drafted in Phase 1.6 against the existing Phase 1.5 data,
  which baked in the per-doc framing.
- §6.5.1 came in Phase 1.8 but stayed scoped to FinanceBench because the
  corpus-mode pipeline was new.
- The cross-vendor Claude judging session (this session) was about lifting
  §5.4 to a stronger judge — but §5.4 is the per-doc RAG table. So the
  judging is a quality improvement on a table that doesn't directly support
  the thesis claim. Useful but secondary.

---

## Prevention checklist (READ BEFORE PLANNING ANY NEW STAGE 3 EXPERIMENT)

Before running anything that costs API spend or sets up a new experimental
cell, answer these:

1. **Does the experiment use one memory instance per corpus, or one per document?**
   - One per corpus → testing the thesis claim. Proceed.
   - One per document → benchmarking V4-as-retriever. Useful as baseline,
     but call it that in the chapter; don't let it carry the headline.

2. **Does the memory state evolve across the experiment, or is it reset?**
   - Evolves → corpus mode. Save snapshots (`snapshots.json`).
   - Reset → per-doc RAG. Don't claim "memory evolution" anywhere
     downstream of this cell.

3. **Are questions asked both during and after ingestion (online + batch)?**
   - Both → matches the thesis claim and lets you separate temporal effects.
   - Only batch / only at end → the experiment can't say anything about
     "memory developing during ingestion."

4. **What does gpt-4o-mini actually see in the user prompt?**
   - Top-k retrieved snippets → standard RAG.
   - All-paragraphs (dump-all baseline) → context-stuffing study.
   - Anything else → trace `scripts/run_stage3_full.py:300-360` and
     `agent/llm_agent.py:205-253` to be sure. **Do not assume.**

5. **For tuning: which protocol is the tuner optimizing for?**
   - `tuning/tune_v4_per_benchmark.py` → optimizes per-doc retrieval recall.
     The θ it produces is for the per-doc RAG setup. It does NOT
     necessarily transfer to the corpus-mode setup (see §6.5.1's massively
     different `theta_corpus_tuned`).
   - `tuning/tune_v4t_corpus.py` (Phase 1.8) → optimizes corpus-mode
     end-to-end. This is the tuner that produces `theta_corpus_tuned`.
   - Mixing the two leaks information and produces misleading numbers.

6. **For the chapter: which table is this cell going into, and does that
   section claim "memory evolution" or "retrieval quality"?**
   - If "memory evolution," this cell must be corpus-mode.
   - If "retrieval quality," per-doc is fine, but say so explicitly.

If any answer to 1-3 is "per-doc" or "reset" or "only at end," and the cell
is intended to support the thesis-headline claim, stop and re-scope before
running the experiment.

---

## Concrete things to fix in existing artifacts

1. **`docs/THESIS_STAGE3_CHAPTER.md`** — add a "Scope caveat" subsection
   right after §5.0 / before §5.4 that says explicitly: "The numbers in
   §§5.4-5.7 are per-document RAG; the thesis-headline claim about memory
   evolution over a corpus is empirically demonstrated only in §6.5.1
   (FinanceBench)." Then either narrow the claim or commit to the
   replication work.
2. **`docs/RECENT_CHANGES.md`** — add an entry pointing to this audit doc.
3. **`AGENTS.md`** — add a "READ THIS FIRST" warning at the top with a
   pointer to this doc, so future agents (including future-me) check before
   spending API or judging effort.
4. **`scripts/run_stage3_full.py`** — add a module-level docstring noting
   that this orchestrator runs per-doc RAG, not corpus mode, and pointing
   to `run_corpus_ingestion.py` for the corpus protocol.
5. **`evaluation/document_qa_memory.py`** — add a comment on
   `_run_reading_phase` noting the per-doc memory reset and what that means
   for the thesis claim.

---

## What's done in this session that's still valuable

The cross-vendor Claude judging covers ~15,283 entries across 5 benchmarks
(CUAD, HotpotQA, LongMemEval, NarrativeQA + 24% of QASPER). Those numbers
are correct for the per-doc RAG setup — they replace gpt-4o-mini auto-judge
with a stronger judge for the §5.4 baseline table. They are NOT wasted
effort, just mislabeled in intent. After the scope reframe, §5.4 reads as:

> "We benchmarked our V4 memory as a per-document retriever on six standard
> long-context QA benchmarks against four reference systems (BM25, flat-50,
> AttentionMemory-tuned, plus a held-out leak-free V4-tuned). The direction
> of effect from §1.5's headline (V4-tuned > V4-canonical on CUAD and
> QASPER) persists under cross-vendor Claude judging. This is a useful
> retrieval-quality baseline but does not test the cumulative-memory thesis
> claim, which is addressed in §6.5.1."

That framing is honest and the Claude-judged numbers are exactly the right
evidence for it.
