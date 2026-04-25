# Dependencies and Optional Components

This project is pure Python but some experiments rely on heavier ML stacks. This
document explains which pieces are **required**, which are **optional**, and how
to deal with known compatibility issues.

## Core dependencies (required)

These are sufficient to run the grid-world experiments, GraphMemory variants,
benchmark, ablation, transfer, sensitivity, and most visualizations:

- `networkx` — graph construction for GraphMemory
- `numpy`, `scipy`, `scikit-learn` — embeddings, ES/CMA-ES, statistics
- `matplotlib` — static figures in `docs/figures/`
- `pyyaml` — experiment configs

They are captured in `requirements.txt`.

## Sentence-transformers (now the default embedding)

As of the April 2026 PoC hardening pass (Phase 3 / S1), the **default**
embedding backend is ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim).
The legacy 31-token TF-IDF embedding remains available via the
``EMBEDDING_BACKEND=tfidf`` env var.

Selection precedence (high → low):

1. ``EMBEDDING_BACKEND`` env var (case-insensitive: ``sentence-transformers``
   / ``minilm`` or ``tfidf``).
2. Default: try sentence-transformers; if the model can't load, fall back to
   TF-IDF and emit a one-time warning.

Under the default backend, RAGMemory and GraphMemoryV4/V5 share the same
embedding function (`memory.embedding.embed_observation`). The neural
controllers (`memory.neural_controller*`) explicitly stay on TF-IDF for
their own input feature so their parameter count (5,674 / 1,962) is
independent of the storage embedding choice.

Reproducibility note: changing the backend invalidates every numeric
result that touches embeddings. The active backend is recorded in the
per-run manifest (`results/manifest.py`) so legacy reproductions are
identifiable.

### Sentence-transformers — historical fallback path

The `RAGMemory` system in `memory/rag_memory.py` originally guarded its
sentence-transformers usage behind a try/except. This guard is still in
place — RAGMemory works with or without the model installed.

- If `sentence-transformers` cannot be imported *or* the model fails to load
  due to a Keras / `tf-keras` / Transformers mismatch, `RAGMemory` now falls
  back to the lightweight TF‑IDF embedder in `memory/embedding.py`.
- The flag `RAGMemory.using_sentence_transformers` reports whether the real
  model is active.

Known issue:

- With Keras 3, some versions of `transformers` expect a separate `tf-keras`
  package. If you need the full RAG behaviour, install a compatible stack, for
  example:

  ```bash
  pip install "tensorflow<2.16" "keras<3" "transformers<5" sentence-transformers
  ```

  or follow the latest `sentence-transformers` installation guide.

If you do **not** care about dense RAG, you can either:

- Leave `sentence-transformers` uninstalled (RAGMemory will use TF‑IDF), or
- Skip `RAGMemory` entirely when running global experiments (see below).

## Skipping RAGMemory in global runs

To avoid any risk of heavy model downloads or dependency conflicts in global
experiments, you can skip `RAGMemory` at the script level:

- **Benchmark** (`run_benchmark.py`):

  ```powershell
  python run_benchmark.py RAGMemory
  ```

  Any additional positional arguments are interpreted as system names to skip.

- **DocumentQA memory-quality** (`run_document_qa_memory.py`):

  ```powershell
  python run_document_qa_memory.py RAGMemory
  ```

  Again, positional arguments are treated as system names to skip.

## LLM experiments (DocumentQA + LLM)

The DocumentQA + LLM path uses the OpenAI Python client via `agent/llm_agent.py`.

- You need a valid OpenAI API key in your environment for real LLM runs.
- If the API is unavailable, `LLMAgent` falls back to a heuristic answerer so
  the pipeline can still be exercised (with meaningless QA scores).

The key LLM experiment is configured in `experiments/document_qa_llm.yaml` and
run via:

```powershell
python runner.py --config experiments/document_qa_llm.yaml
```

This experiment is optional for most development; the grid-world experiments and
all GraphMemory analyses do **not** require LLM access.

