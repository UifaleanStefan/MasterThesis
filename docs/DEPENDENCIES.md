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

## Sentence-transformers and RAGMemory

The `RAGMemory` system in `memory/rag_memory.py` uses
`sentence-transformers` (`all-MiniLM-L6-v2`) for dense embeddings. This is
**optional** and can be disabled without affecting the rest of the thesis.

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

