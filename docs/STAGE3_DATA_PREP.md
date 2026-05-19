# Stage 3 — Data Preparation: Six Long-Context Benchmarks

**Date:** May 2026
**Phase:** Stage 3 Phase 0 (data fetch + verification, pre-LLM)
**Status:** PASS — 6/6 benchmarks fetched, parsed, and verified usable offline

---

## Motivation

Stages 1 and 2 of the thesis evaluated parameterized graph memory (V4, V5,
V6) on **synthetic** environments — grid-world key-door tasks and the
hand-authored `DocumentQA` library (`FANTASY_LORE`, `MYSTERY_CASE`,
`SCIENCE_FACTS`). Those served as proof-of-concept harnesses where the
learnable θ vector could be tuned with CMA-ES under controlled retrieval
metrics.

Stage 3 is the *thesis-critical* stage: running a real LLM agent (GPT-4o
class) over **published, real-world long-context benchmarks** so that
results are comparable to the wider literature and not vulnerable to the
"you constructed the test set" critique. This phase prepares the data
that everything downstream — adapters, smoke tests, full runs, analysis,
thesis chapter — depends on.

The six benchmarks were chosen to span:

* **Fantasy / narrative** — NarrativeQA (full books + film scripts)
* **Science** — QASPER (NLP papers)
* **Law** — CUAD (commercial contracts)
* **Wiki-style** — HotpotQA (multi-hop Wikipedia)
* **Finance** — FinanceBench (SEC 10-K / 10-Q filings)
* **Long-term dialogue memory** — LongMemEval (multi-session chat history)

This is the canonical "memory pressure" test bed: each benchmark requires
locating relevant evidence inside a long context where naively dumping
everything into the LLM context window is either impossible (too long)
or wasteful (too expensive). Selective memory is the lever.

## What this phase delivered

* `scripts/prefetch_benchmarks.py` — single-command idempotent fetcher
  for all 6 benchmarks, with per-benchmark error handling and a merging
  manifest.
* `scripts/verify_benchmarks.py` — offline-only verifier that loads each
  benchmark, pulls 3 real samples, sweeps the full set for length
  distributions, and asserts every item has (question, answer, source
  content).
* `data/benchmarks/` — 36.9 GB on-disk cache (gitignored).
* `data/benchmarks/manifest.json` — what was fetched, when, sample
  schemas, item counts.
* `data/benchmarks/verification.json` — verification report (6/6 OK).
* `requirements.txt` — adds `datasets>=2.14`, `huggingface_hub>=0.20`.

## The six benchmarks — fetched and verified

| # | Benchmark | Source | Items (eval split) | Cache size | Source content (median chars) |
|---|---|---|---:|---:|---:|
| 1 | **HotpotQA** | HF `hotpotqa/hotpot_qa` (distractor) | 7,405 dev | ~few MB | 5,575 (p95 9,000) |
| 2 | **QASPER** | AI2 S3 tarball v0.3 | 281 dev + 416 test papers, 5,049 Qs | ~14 MB | 22 K–55 K per paper |
| 3 | **CUAD** | Zenodo v1 ZIP | 510 contracts × 41 Q-types = **20,910 QAs** | 101 MB | 33,143 (p95 161,626, max 338,211) |
| 4 | **NarrativeQA** | HF `deepmind/narrativeqa` | 3,461 val + 10,557 test | **~22 GB** | 236,825 (p95 852,731, max **1,199,816**) |
| 5 | **FinanceBench** | HF `PatronusAI/financebench` | 150 (canonical public eval) | ~few MB | 1,450 (max 12,124) — `evidence` excerpts |
| 6 | **LongMemEval** | HF `xiaowu0162/longmemeval-cleaned`, raw JSON via `hf_hub_download` | 500 oracle + 500 short + 500 medium | **~3 GB** (m_cleaned = 2.5 GB) | 2 sessions, 12–36 messages per item (p95 4 sessions) |

**Total on disk: 36.9 GB.**

### Sample-key sets (informs adapter design)

* **HotpotQA**: `{answer, context, id, level, question, supporting_facts, type}` — `context` is `{title: [...], sentences: [[...], ...]}`, 10 passages per item. `supporting_facts` gives sentence-level gold relevance.
* **QASPER**: `{paper_id → {title, abstract, full_text, qas}}` — `full_text` is `[{section_name, paragraphs: [...]}]`. Answers may be `free_form_answer`, `extractive_spans`, or `yes_no` boolean.
* **CUAD**: SQuAD v2 — `{data: [{title, paragraphs: [{context, qas: [{question, answers, is_impossible}]}]}]}`. **6,702 answerable, 14,208 `is_impossible=True`** *by design* — most contracts don't address most clauses; the no-answer items still test memory ("did the agent correctly determine this contract doesn't address X?").
* **NarrativeQA**: `{question.text, answers: [{text, tokens}, ...], document: {text, summary, ...}}` — 2 reference answers per question; documents are full books / film scripts.
* **FinanceBench**: `{answer, company, doc_link, doc_name, doc_period, doc_type, evidence, financebench_id, justification, question, question_reasoning, question_type, ...}`. **All 150/150** have non-empty question + answer + evidence.
* **LongMemEval**: `{question, answer, question_type, question_date, haystack_sessions, haystack_dates, haystack_session_ids, answer_session_ids}` — `haystack_sessions` is a list of dialogue sessions (each a list of `{role, content}`); `answer_session_ids` is **gold relevance** (which sessions contain the answer).

### Two gold-relevance signals we get for free

The thesis already reports retrieval **precision@k** and **recall@k** on
synthetic envs. On real benchmarks, two of the six expose ground-truth
relevance labels at the right granularity:

1. **HotpotQA** `supporting_facts` — per-question list of `(title, sent_idx)` tuples pointing to the gold sentences that contain the answer (typically 2 per question, across 2 of the 10 passages).
2. **LongMemEval** `answer_session_ids` — per-question list of session IDs that contain the answer to the question.

This is exactly analogous to the `relevant_paragraph_indices` field in
the existing `DocumentQA` evaluator. The same retrieval-metric code path
that produces the headline `precision = 0.94+` on synthetic
`MegaQuestRoom` should drop in unchanged on these two real benchmarks —
giving us memory-quality metrics with zero LLM cost.

For the other four benchmarks (QASPER, CUAD, NarrativeQA, FinanceBench),
retrieval relevance is *implicit* — there is a single gold answer, and
we measure correctness via LLM judge. The agent's choice of which
context to retrieve is judged by downstream answer quality.

## Three problems hit during fetch (all fixed)

### 1. HF `datasets>=3` dropped loading-script support

QASPER (`allenai/qasper`) and CUAD (`theatticusproject/cuad-qa`) both
ship as legacy "loading-script" datasets (a `qasper.py` / `cuad-qa.py`
file inside the repo that runs at load time). `datasets>=3` removed
support — `load_dataset(...)` now raises `RuntimeError: Dataset scripts
are no longer supported`. **Fix:** download canonical archives directly.

* QASPER: `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-{train-dev,test-and-evaluator}-v0.3.tgz`. URL extracted from the `qasper.py` loading script's `_DOWNLOAD_URL`.
* CUAD: `https://zenodo.org/records/4595826/files/CUAD_v1.zip`. Citeable archive; SQuAD v2 JSON inside.

### 2. LongMemEval — three URLs wrong before the right one

* `github.com/.../releases/.../longmemeval_data.tar.gz` → 404 (release tag changed).
* `raw.githubusercontent.com/.../data/longmemeval_*.json` → 404 (file paths in the repo no longer match).
* HF Hub file fetch of original `xiaowu0162/longmemeval` → 404 (repo is deprecated).

**Fix:** the author published a *cleaned* version at
`xiaowu0162/longmemeval-cleaned` with JSON blobs:
`longmemeval_oracle.json` (15 MB), `longmemeval_s_cleaned.json` (277 MB),
`longmemeval_m_cleaned.json` (2.5 GB).

### 3. `load_dataset` autoparsing on longmemeval-cleaned

The HF repo for `longmemeval-cleaned` has a known type mismatch in the
`answer` column — the dataset viewer is broken, and `load_dataset`
likewise raises `DatasetGenerationError`. Critically, `load_dataset`
generates **all** splits when called for one, so the broken `m_cleaned`
split kills the other two. **Fix:** bypass `load_dataset` entirely; use
`huggingface_hub.hf_hub_download` to fetch the three JSON files
directly. We do our own parsing (and accept that the 2.5 GB
`m_cleaned.json` requires streaming JSON for full iteration — file is
on disk; oracle split is the primary target anyway).

## Verification — 6/6 OK, offline

`scripts/verify_benchmarks.py` runs with `HF_DATASETS_OFFLINE=1` and
`HF_HUB_OFFLINE=1` to prove no benchmark needs the network at runtime.
For each benchmark it:

1. Loads the cached data (HF cache or local JSON).
2. Counts items in the canonical eval split.
3. Pulls 3 real samples (first / middle / last).
4. Asserts each has non-empty question, gold answer, source content.
5. Sweeps the full set (or a strided sub-sample of 50–100) for content-length distribution.

All 6 pass. Report saved to `data/benchmarks/verification.json`.

### Verifier highlights

* **NarrativeQA** documents in the val split range 47 K – **1,199,816 chars** (p50 236 K, p95 853 K). Single books *the size of a small novel* — exactly the long-context regime where memory selection matters.
* **CUAD** answerable/no-answer breakdown: **6,702 answerable, 14,208 `is_impossible=True`** across the 20,910 QAs. By-design (most contracts don't address most clauses), not a data problem.
* **HotpotQA** has a context-length p95 of 9,000 chars across the dev set — well within GPT-4o's 128 K context, so the test isn't "can the LLM read it all", it's "can the memory pick the 2 relevant passages out of 10".
* **LongMemEval** oracle: sessions-per-item p50 = 2, p95 = 4, max = 6 — concise multi-session episodes that test session-level retrieval.
* **FinanceBench** evidence: median 1,450 chars, max 12,124 — modest but each item has clean `(question, answer, evidence)` triplets.

## Why these six (and not others)

Considered but excluded:

* **MuSiQue** — strong multi-hop benchmark but largely overlaps HotpotQA in shape (Wikipedia, multi-hop reasoning). HotpotQA chosen for tighter integration with existing literature.
* **SCROLLS** — meta-benchmark wrapping NarrativeQA/QASPER/ContractNLI/GovReport. We pull NarrativeQA and QASPER directly from source; ContractNLI is NLI, not QA; GovReport is summarization, off-spec.
* **NeedleInAHaystack** — synthetic, position-of-fact retrieval probe. Useful for ablation but doesn't measure real reasoning over messy real text.
* **Loft / Long-RAG-Bench** — newer (2024+) but mostly subsumes the six above; would inflate scope.

The six chosen represent **disjoint domains** with **non-overlapping
benchmark provenance** and together cover both retrieval-precision
testing (HotpotQA, LongMemEval gold relevance) and answer-quality
testing (the other four with LLM judge).

## Files added in this phase

* `scripts/prefetch_benchmarks.py` (~480 lines) — six fetchers with manifest merge.
* `scripts/verify_benchmarks.py` (~340 lines) — six verifiers, offline mode.
* `data/benchmarks/manifest.json` — what was fetched, sample schemas.
* `data/benchmarks/verification.json` — 6/6 PASS verification report.
* `requirements.txt` — adds `datasets>=2.14`, `huggingface_hub>=0.20`.
* `.gitignore` — excludes `data/benchmarks/` and `.cache/huggingface/` (36.9 GB stays local).

## How to reproduce

```powershell
# One-time fetch (idempotent — re-runs only refetch missing ones)
python scripts/prefetch_benchmarks.py

# Fetch only specific benchmarks
python scripts/prefetch_benchmarks.py --only hotpotqa qasper
python scripts/prefetch_benchmarks.py --skip narrativeqa  # the 22 GB one

# Verify offline (no network calls)
python scripts/verify_benchmarks.py

# Inspect a single benchmark interactively
python -c "import json; m = json.load(open('data/benchmarks/manifest.json')); print(json.dumps(m['results']['hotpotqa'], indent=2))"
```

## Time / disk / network budget

* **Time:** ~2 minutes total when caches are warm; first fetch dominated by NarrativeQA (22 GB of books) and LongMemEval m_cleaned (2.5 GB) — both download in ~30–60 seconds on a fast connection.
* **Disk:** 36.9 GB. Predominantly NarrativeQA (full book text). All gitignored.
* **Network:** Each fetcher caches; reruns are no-op for already-cached benchmarks.

## What this unlocks (Stage 3 Phase 1+)

With data verified and stable, the next phase is the *adapter* layer —
six modules in `environment/benchmarks/` that translate each benchmark's
native shape into the existing `DocumentQA`-compatible
`(title, paragraphs, qa_pairs)` contract that the rest of the Stage 3
pipeline (LLM agent + memory + judge) expects. After adapters:

1. Smoke test: load each benchmark, run 1 V4 episode end-to-end (no LLM).
2. API smoke: 1–2 real OpenAI calls per benchmark (~$0.10 total).
3. Full Stage 3 runs: 3 configs × 6 benchmarks (~$60–70 budget).
4. Analysis + figures + thesis chapter draft.
