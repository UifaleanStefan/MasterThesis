# Key Concepts & Clarifications
*Companion glossary to the 10-lesson study guide — plain-English definitions of the jargon.*


## Is there one θ, or one per entity? (how each item is considered)

**There is exactly ONE θ.** It is a single 10-number vector — the *policy* — shared by the whole memory. Entries and entities do **not** each carry their own θ.

Think of θ as the **settings on a machine** and each event/entity as a **workpiece** passing through it. The 10 dials are identical for every item; what differs is each item's *own* data that the dials act on.

Each **event** carries its own 384-dim **embedding** and a **step** (timestamp). Each **entity** carries its own **mention count** and last/first-seen step. θ is the shared set of weights applied to those private attributes — it is not stored per item.

*Two different spaces:* CMA-ES searches the **10-D θ space** (one point = one whole policy); each passage lives in the separate **384-D embedding space** from the frozen MiniLM encoder.


## Corpus tuning

**Re-running CMA-ES to fit the single θ to one benchmark's document collection.** A *corpus* = the documents of one benchmark (all CUAD contracts, all QASPER papers, …).

The loop: ingest that corpus cumulatively with a candidate θ, measure **recall@8** of its gold evidence on a tuning-question split, and let CMA-ES push θ toward whatever maximizes it. Each benchmark gets its **own** tuned θ — which is how task-dependence (RQ2) shows up.

It is **cheap and LLM-free** (recall@8 uses no answerer/judge) and measured on a **held-out** split, so 'corpus-tuned' can't just mean 'memorized the tuning questions.'

*Canonical θ* = a fixed, not-tuned-to-this-corpus default. *Corpus-tuned θ* = the re-optimized result. The gap between them is the **lift**.


## Lift

**Plain English for 'how much better the tuned version does than the untuned baseline' — the gain.** It is a difference of two scores:

```
lift = (score with corpus-tuned θ)  −  (score with canonical θ)
```

FinanceBench: 0.645 − 0.243 = **+0.402**. A *positive* lift = tuning helped; *zero* = it did nothing; *negative* = it hurt (e.g. vs tuned BM25 on CUAD the lift is −0.131 → V4t loses).

'The lift survives held-out testing' = the gain is still there, and statistically real, on questions the tuner never saw. 'QASPER's lift didn't survive' = the gain shrank to non-significant, so it is not claimed. '2 of 3 survive' = real on FinanceBench and CUAD, not QASPER.


## Dump-all

**A baseline: no retrieval — paste the *entire* corpus into the LLM's prompt and let the model find the answer.** It's the brute-force 'why bother with memory when context windows are huge?' alternative to selective retrieval.

It is the **most important** baseline for two reasons: (1) it's the **accuracy ceiling** — if the answer is anywhere in the corpus, dump-all has it in context, so tying it proves selective retrieval lost no accuracy by being selective; (2) it *is* the field's 'long-context kills retrieval' argument, operationalized.

Outcome: on accuracy it **ties** selective memory (after a truncation bug was fixed) — so the win is **efficiency**: ≈18× cheaper, and it **structurally overflows** the 128K context window at **N≈11 CUAD contracts** (~43× the window at 510), where the selective prompt stays flat at ~704 tokens.


## What the answerer LLM actually receives

For each question, gpt-4o-mini gets the **question + the top-8 retrieved passages** — *not* the corpus. Each passage is one line `step {n}: {text}`, under the instruction *'answer using only the provided context.'*

The **same top-8** is what `recall@8` scores (did the gold evidence land in that set?). So tuning θ to put the gold in the top-8 → the LLM is handed passages that contain the answer → better judged answer. That chain is what ρ=0.69 validates.

This is why the selective prompt is ~704 tokens flat: it's always ~8 passages. *Dump-all* is the `uncapped_context` path that sends everything instead.


## Why compare to dump-all and not just 'regular RAG'?

**You compare to both — dump-all is one of several baselines.** The roster: canonical θ, stock BM25, **tuned BM25** (= regular sparse RAG, done fairly), tuned attention, dense/RAG memory, **HippoRAG**, **MemGPT/Letta**, and dump-all.

Regular RAG is basically a *special case* of your own method (selective embedding retrieval = θ with graph/recency weights near zero). So 'V4t vs RAG' is answered by the canonical→corpus-tuned lift and by the tuned-BM25 fair baseline (V4t **wins FB, ties QASPER, loses CUAD**).

Dump-all gets the spotlight because only it is the **accuracy ceiling** and the **long-context challenge** — beating a RAG baseline wouldn't answer 'do we even need retrieval when context windows are huge?'; refuting dump-all's scalability does.


## How the memory graph is built

A directed graph, built **incrementally** as documents are ingested. Two node types: **event** nodes (one per stored observation, holding its text + embedding + step) and **entity** nodes (hubs — a company, a dollar amount, a year).

Per incoming observation: (1) **storage gate** — keep it only if importance > θ_store; (2) create the event node; (3) **extract entities** by regex (proper nouns, money, years, section refs); (4) attach `mentions` / `mentioned_in` edges to each entity hub (in Stage 3, every entity becomes a hub); (5) with probability θ_temporal, draw a `temporal` edge from the previous event.

So the shape is a chain of event nodes hanging off shared entity hubs. θ_entity/θ_temporal/θ_decay shape this; `w_graph` later scores retrieval by it (but is inert on the Stage-3 corpora).


## Embeddings & entities (where the numbers come from)

**Embeddings:** a whole observation → one 384-dim vector via the pretrained, **frozen** `all-MiniLM-L6-v2` sentence encoder. Nothing in this thesis trains it; only cosine similarity is used downstream. (Legacy 31-word TF-IDF only for grid worlds.)

**Entities:** no neural NER — deterministic **regex/heuristics** for capitalized proper-noun phrases, dollar amounts, years, and section references; normalized, deduped, capped at 20 per observation.
