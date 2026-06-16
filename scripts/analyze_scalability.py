"""Scalability analysis: dump-all vs selective prompt size as the CUAD corpus grows.

Demonstrates (no LLM needed) that the full-context "dump-all" baseline's prompt
grows O(N) and exceeds the answerer's context window at small N, while selective
retrieval (k=8) stays O(1). Writes results/stage3/scalability_cuad.json and
docs/figures/fig_stage3_scalability.png.
"""
import json
from pathlib import Path
import numpy as np
import tiktoken
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CONTEXT_LIMIT = 128_000          # gpt-4o-mini context window (tokens)
OVERHEAD = 200                   # system prompt + question, approx
enc = tiktoken.get_encoding("cl100k_base")

# --- dump-all: cumulative contract tokens ---------------------------------
cuad = json.loads((ROOT / "data/benchmarks/cuad/CUAD_v1/CUAD_v1.json").read_text("utf-8"))
per_contract = [len(enc.encode(c["paragraphs"][0]["context"])) for c in cuad["data"]]
cum = np.cumsum(per_contract)                     # tokens for first N contracts
n_contracts = len(per_contract)

# --- selective: measured mean k=8 prompt from the real corpus-tuned run ----
qab = json.loads((ROOT / "results/stage3/corpus_traces/cuad__v4t-corpus-tuned/qa_batch.json").read_text("utf-8"))
sel_tokens = [e["prompt_tokens_estimated"] for e in qab if e.get("prompt_tokens_estimated")]
sel_mean = float(np.mean(sel_tokens))
sel_p95 = float(np.percentile(sel_tokens, 95))

# --- find the contract count where dump-all exceeds the context window -----
dump_at = cum + OVERHEAD
n_star = int(np.argmax(dump_at > CONTEXT_LIMIT)) + 1   # first N over the limit
dump_at_510 = int(dump_at[-1])

out = {
    "context_limit_tokens": CONTEXT_LIMIT,
    "n_contracts": n_contracts,
    "mean_tokens_per_contract": float(np.mean(per_contract)),
    "dumpall_tokens_at_N": {str(n): int(dump_at[n - 1]) for n in [1, 10, 13, 50, 100, 300, 510]},
    "dumpall_exceeds_context_at_N": n_star,
    "dumpall_tokens_at_510": dump_at_510,
    "dumpall_overflow_factor_at_510": round(dump_at_510 / CONTEXT_LIMIT, 1),
    "selective_prompt_tokens_mean": round(sel_mean, 1),
    "selective_prompt_tokens_p95": round(sel_p95, 1),
    "selective_independent_of_N": True,
}
(ROOT / "results/stage3/scalability_cuad.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

# --- figure ----------------------------------------------------------------
BLUE, RED, GREEN, GRAY = "#2563EB", "#DC2626", "#16A34A", "#6B7280"
fig, ax = plt.subplots(figsize=(8, 5))
N = np.arange(1, n_contracts + 1)
ax.plot(N, dump_at, color=RED, lw=2.2, label="Dump-all (full context)")
ax.plot(N, np.full_like(N, sel_mean, dtype=float), color=GREEN, lw=2.2,
        label=f"Selective retrieval, $k=8$ (mean {sel_mean:,.0f} tok)")
ax.axhline(CONTEXT_LIMIT, color=GRAY, ls="--", lw=1.5)
ax.text(n_contracts, CONTEXT_LIMIT * 1.15, "gpt-4o-mini 128K context limit",
        ha="right", va="bottom", fontsize=9, color=GRAY)
ax.axvline(n_star, color=RED, ls=":", lw=1.2, alpha=0.7)
ax.annotate(f"dump-all exceeds\ncontext at N$\\approx${n_star} contracts",
            xy=(n_star, CONTEXT_LIMIT), xytext=(n_star + 30, CONTEXT_LIMIT * 0.18),
            fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
ax.set_yscale("log")
ax.set_xlabel("Corpus size (number of CUAD contracts ingested)")
ax.set_ylabel("Answer-prompt size (tokens, log scale)")
ax.set_title("Scalability: dump-all prompt grows $O(N)$ and overflows the context;\n"
             f"selective memory stays $O(1)$ (at $N=510$ dump-all needs "
             f"{dump_at_510/1e6:.1f}M tokens, $\\approx{out['dumpall_overflow_factor_at_510']:.0f}\\times$ the limit)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, which="both", alpha=0.25)
out_png = ROOT / "docs/figures/fig_stage3_scalability.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"  [OK] {out_png}")
