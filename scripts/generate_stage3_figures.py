"""Generate the three Stage-3 thesis figures into docs/figures/:
  fig_stage3_lift.png          — six-benchmark canonical vs corpus-tuned batch lift
  fig_stage3_online_batch.png  — CUAD online-vs-batch (the recency effect)
  fig_stage3_cost.png          — FinanceBench accuracy vs cost (dump-all parity)
Numbers are the committed authoritative aggregates (THESIS.md / CUAD-50 /
stage3_finbench_corpus.json).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
BLUE, GREEN, RED, GRAY, ORANGE = "#2563EB", "#16A34A", "#DC2626", "#6B7280", "#F59E0B"

# ---- 1. six-benchmark batch lift -----------------------------------------
benches = ["FinanceBench", "HotpotQA", "QASPER", "CUAD", "LongMemEval", "NarrativeQA"]
canon = [0.243, 0.215, 0.250, 0.028, 0.165, 0.400]
tuned = [0.645, 0.755, 0.415, 0.172, 0.330, 0.400]
note = ["", "", "n.s.", "", "", "undef."]
x = np.arange(len(benches)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, canon, w, color=GRAY, label="canonical $\\theta$ (grid-tuned)")
ax.bar(x + w/2, tuned, w, color=GREEN, label="corpus-tuned $\\theta$")
for i, (c, t, nt) in enumerate(zip(canon, tuned, note)):
    lift = t - c
    ax.text(i + w/2, t + 0.015, f"+{lift:.2f}" if lift > 0 else "0.00",
            ha="center", va="bottom", fontsize=8.5,
            color=GREEN if lift > 0 else GRAY, fontweight="bold")
    if nt:
        ax.text(i, -0.055, nt, ha="center", va="top", fontsize=8, color=RED, style="italic")
ax.set_xticks(x); ax.set_xticklabels([b.replace("Bench", "\nBench").replace("MemEval", "\nMemEval").replace("tiveQA", "tive\nQA") for b in benches], fontsize=8.5)
ax.set_ylabel("End-of-corpus (batch) judge mean")
ax.set_ylim(-0.02, 0.85)
ax.set_title("Stage 3: end-of-corpus accuracy lift from corpus-cumulative tuning\n"
             "(survives held-out on 4/6; QASPER n.s.; NarrativeQA undefined-by-construction)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, axis="y", alpha=0.3)
fig.savefig(FIG / "fig_stage3_lift.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[OK] fig_stage3_lift.png")

# ---- 2. CUAD online vs batch ---------------------------------------------
configs = ["V4$_t$ canonical", "V4$_t$ corpus-tuned"]
online = [0.295, 0.390]; batch = [0.028, 0.172]
x = np.arange(len(configs)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - w/2, online, w, color=BLUE, label="online (source doc recent)")
ax.bar(x + w/2, batch, w, color=ORANGE, label="batch (end-of-corpus)")
for i, (o, b) in enumerate(zip(online, batch)):
    ax.text(i - w/2, o + 0.008, f"{o:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, b + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel("Judge mean ($n{=}644$, 50 contracts)")
ax.set_title("CUAD online vs.\\ batch: the recency effect\n"
             "(canonical collapses end-of-corpus; corpus-tuned $\\theta$ recovers)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 0.45)
fig.savefig(FIG / "fig_stage3_online_batch.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[OK] fig_stage3_online_batch.png")

# ---- 3. FinanceBench accuracy vs cost (dump-all parity) ------------------
sys_names = ["corpus-tuned", "attention", "BM25", "dump-all"]
acc = [0.645, 0.657, 0.503, 0.607]      # FB batch judge means
cost = [0.15, 0.15, 0.17, 2.68]          # $ per 150 questions
colors = [GREEN, ORANGE, BLUE, RED]
fig, ax = plt.subplots(figsize=(8, 5))
for nm, a, c, col in zip(sys_names, acc, cost, colors):
    ax.scatter(c, a, s=140, color=col, edgecolors="white", zorder=3)
    ax.annotate(nm, (c, a), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Cost (USD per 150 questions, log scale)")
ax.set_ylabel("FinanceBench batch judge mean")
ax.set_title("Accuracy vs.\\ cost on FinanceBench: dump-all is accuracy-tied but\n"
             "$\\approx$18$\\times$ more expensive than corpus-tuned selective memory")
ax.grid(True, which="both", alpha=0.25)
ax.set_ylim(0.45, 0.72)
fig.savefig(FIG / "fig_stage3_cost.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("[OK] fig_stage3_cost.png")
