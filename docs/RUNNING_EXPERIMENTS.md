# Running Experiments — Commands and Order

This document lists the **exact commands** to regenerate all result files and figures, in the recommended order. All paths are relative to the project root. Use PowerShell on Windows; on Unix use `python` and adjust paths as needed.

---

## 1. Prerequisites

- Python 3.10+ with dependencies from `requirements.txt`.
- Optional: OpenAI API key for DocumentQA+LLM and NeuralControllerV2 (see `docs/DEPENDENCIES.md`).
- To skip RAGMemory when `sentence_transformers` is broken: pass `RAGMemory` as a positional argument to `run_benchmark.py` and `run_document_qa_memory.py`.

---

## 2. Regenerating All Result Files (Order)

Run in this order so downstream scripts have the JSON they need.

| Step | Command | Output | Approx. runtime |
|------|---------|--------|------------------|
| 1 | `python run_graphmemory_v4_cmaes.py` | `results/graphmemory_v4_cmaes_results.json` | ~15–30 min (30 gens × 50 eps) |
| 1 (quick) | `python run_graphmemory_v4_cmaes.py --quick` | same | ~2 min (5 gens × 20 eps) |
| 2 | `python run_benchmark.py` | `results/benchmark_results.json` | ~2–5 min (12×4 envs, 50 eps) |
| 2 (skip RAG) | `python run_benchmark.py RAGMemory` | same | same |
| 3 | `python run_ablation.py` | `results/ablation_results.json` | ~5–10 min |
| 4 | `python run_transfer.py` | `results/transfer_results.json` | ~2–5 min |
| 5 | `python run_sensitivity.py` | `results/sensitivity_results.json` | ~10–15 min |
| 6 | `python run_document_qa_memory.py` | `results/document_qa_memory_results.json` | ~2–5 min |
| 6 (skip RAG) | `python run_document_qa_memory.py RAGMemory` | same | same |
| 7 (optional) | `python run_neural_controller_v2.py` | `results/neural_controller_v2_results.json` | ~30+ min |

**Note:** Transfer and sensitivity use V4 params from step 1; run step 1 first (or use existing `graphmemory_v4_cmaes_results.json`).

---

## 3. Regenerating Figures

- **Publication figures (from JSON):**  
  `python generate_thesis_figures.py`  
  Requires: `benchmark_results.json`, `graphmemory_v4_cmaes_results.json`, `ablation_results.json`, `transfer_results.json`, `sensitivity_results.json`, `neural_controller_v2_results.json`.  
  If any file is missing, the script exits with a message listing the missing files and the command to generate each. Use `--allow-missing` to generate only figures for which data exists.

- **Fig 6 & 7 (live runs):**  
  `python regen_figs.py`  
  Produces grid trajectory and per-episode metrics from live agent runs (not from JSON).

- **Benchmark fig5 variants:**  
  `python regen_benchmark_figs.py`  
  Requires `results/benchmark_results.json` (run `run_benchmark.py` first).

- **Regenerate all (thesis + benchmark + extended):**  
  `python regen_all_figures.py`  
  (See Phase 3 of the analysis plan; this script runs thesis figures with `--allow-missing`, then benchmark figs if data exists, then extended figures with real data when available.)

---

## 4. Smoke Tests

To verify the pipeline without full runs:

```powershell
python run_smoke_tests.py
```

Runs one episode per (env, memory) for a small subset, one DocumentQA memory eval, and the DocumentQA+LLM code path (with LLM fallback if no API key). Use `--no-runner` to skip the runner branch.

---

## 5. DocumentQA + LLM (Real Cost)

Requires OpenAI API key. Single run:

```powershell
python runner.py --config experiments/document_qa_llm.yaml
```

This uses the DocumentQA+LLM path: reading phase stores paragraphs in memory; QA phase retrieves top-k per question, calls the LLM, and records cost and score. Results are saved to the SQLite database under `results/`.

---

## 6. Dashboard (Interactive)

After generating result JSONs and (optionally) figures:

```powershell
streamlit run dashboard/app.py
```

See AGENTS.md and the Phase 6 plan for dashboard tabs (Benchmark, Ablation, Transfer, Sensitivity, DocumentQA memory, Figures, Playground).
