"""
generate_thesis_figures.py

Generates all thesis publication-quality figures from real experimental data.
Produces: fig_master_benchmark, fig_ablation_ranked, fig_transfer_annotated,
          fig_sensitivity_annotated, fig_neural_analysis, and regenerates
          fig11_pareto and fig13_memory_size with real data.

Usage:
    python generate_thesis_figures.py
    python generate_thesis_figures.py --allow-missing   # generate only figures for available JSONs
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGS_DIR    = os.path.join(os.path.dirname(__file__), "docs", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Which result file(s) each figure needs (for missing-file check)
FIGURE_DEPS = {
    "fig_master_benchmark": ["benchmark_results.json", "graphmemory_v4_cmaes_results.json"],
    "fig_ablation_ranked": ["ablation_results.json"],
    "fig_transfer_annotated": ["transfer_results.json"],
    "fig_sensitivity_annotated": ["sensitivity_results.json"],
    "fig_neural_analysis": ["neural_controller_v2_results.json"],
    "fig_neural_v2_curves": ["neural_controller_v2_results.json"],
    "fig_neural_transfer": ["neural_controller_v2_results.json", "transfer_results.json"],
    "fig11_pareto": ["benchmark_results.json", "graphmemory_v4_cmaes_results.json"],
    "fig13_memory_size": ["benchmark_results.json", "graphmemory_v4_cmaes_results.json"],
    "fig_story_precision_reward": ["benchmark_results.json"],
}
COMMANDS = {
    "run_benchmark.py": "benchmark_results.json",
    "run_graphmemory_v4_cmaes.py": "graphmemory_v4_cmaes_results.json",
    "run_ablation.py": "ablation_results.json",
    "run_transfer.py": "transfer_results.json",
    "run_sensitivity.py": "sensitivity_results.json",
    "run_neural_controller_v2.py": "neural_controller_v2_results.json",
}


def check_required_files(allow_missing: bool) -> dict[str, bool]:
    """Return {figure_name: can_run}. If not allow_missing and any required file is missing, print and exit 1."""
    missing_files: set[str] = set()
    for deps in FIGURE_DEPS.values():
        for f in deps:
            if not os.path.isfile(os.path.join(RESULTS_DIR, f)):
                missing_files.add(f)
    can_run = {}
    for fig, deps in FIGURE_DEPS.items():
        can_run[fig] = all(os.path.isfile(os.path.join(RESULTS_DIR, f)) for f in deps)
    if missing_files and not allow_missing:
        print("Missing result files (run these commands first):", file=sys.stderr)
        for f in sorted(missing_files):
            cmd = next((c for c, out in COMMANDS.items() if out == f), None)
            print(f"  {f}  <- python {cmd}", file=sys.stderr)
        print("Use --allow-missing to generate only figures for available data.", file=sys.stderr)
        sys.exit(1)
    return can_run


def load(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)

# ── colour palette ──────────────────────────────────────────────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
RED    = "#DC2626"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
GOLD   = "#D97706"
TEAL   = "#0D9488"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A — Master Benchmark Summary
# ═══════════════════════════════════════════════════════════════════════════
def fig_master_benchmark():
    bench  = load("benchmark_results.json")
    v4res  = load("graphmemory_v4_cmaes_results.json")

    multihop = bench["MultiHop-KeyDoor"]

    # Add V4 entry to multihop data
    multihop["GraphMemoryV4"] = {
        "mean_reward":        v4res["v4"]["eval"]["mean_reward"],
        "mean_tokens":        v4res["v4"]["eval"]["mean_tokens"],
        "mean_memory_size":   v4res["v4"]["eval"]["mean_memory_size"],
        "retrieval_precision": v4res["v4"]["eval"]["mean_precision"],
        "efficiency":         v4res["v4"]["eval"]["efficiency"],
    }
    multihop["GraphMemoryV1"] = {
        "mean_reward":        v4res["v1_baseline"]["eval"]["mean_reward"],
        "mean_tokens":        v4res["v1_baseline"]["eval"]["mean_tokens"],
        "mean_memory_size":   v4res["v1_baseline"]["eval"]["mean_memory_size"],
        "retrieval_precision": v4res["v1_baseline"]["eval"]["mean_precision"],
        "efficiency":         v4res["v1_baseline"]["eval"]["efficiency"],
    }

    systems = sorted(multihop.keys(), key=lambda s: multihop[s]["mean_reward"], reverse=True)
    rewards  = [multihop[s]["mean_reward"]      for s in systems]
    tokens   = [multihop[s]["mean_tokens"]       for s in systems]
    memsizes = [multihop[s]["mean_memory_size"]  for s in systems]
    prec     = [multihop[s].get("retrieval_precision") or 0.0 for s in systems]

    v4_idx  = systems.index("GraphMemoryV4")
    colors  = [GREEN if i == v4_idx else (ORANGE if s == "GraphMemoryV1" else BLUE)
               for i, s in enumerate(systems)]

    # V4 CMA-ES learning curve
    v4_hist = v4res["v4"]["opt_history"]
    v1_hist = v4res["v1_baseline"]["opt_history"]
    v4_gens = [h["generation"]    for h in v4_hist]
    v4_fit  = [h["best_fitness"]  for h in v4_hist]
    v1_gens = [h["generation"]    for h in v1_hist]
    v1_fit  = [h["best_fitness"]  for h in v1_hist]

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── top-left: reward bar chart ──
    bars = ax1.barh(range(len(systems)), rewards, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(systems)))
    ax1.set_yticklabels(systems, fontsize=8)
    ax1.set_xlabel("Mean Reward (MultiHopKeyDoor)")
    ax1.set_title("All Memory Systems — Reward Ranking")
    ax1.axvline(rewards[v4_idx], color=GREEN, linestyle="--", linewidth=1.2, alpha=0.6)
    for i, (bar, r) in enumerate(zip(bars, rewards)):
        ax1.text(r + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{r:.3f}", va="center", fontsize=7.5,
                 color=GREEN if i == v4_idx else "black", fontweight="bold" if i == v4_idx else "normal")
    ax1.set_xlim(0, max(rewards) * 1.25)
    patch_v4 = mpatches.Patch(color=GREEN,  label="GraphMemoryV4 (ours)")
    patch_v1 = mpatches.Patch(color=ORANGE, label="GraphMemoryV1 (baseline)")
    patch_ot = mpatches.Patch(color=BLUE,   label="Other systems")
    ax1.legend(handles=[patch_v4, patch_v1, patch_ot], loc="lower right", fontsize=7)

    # ── top-right: reward vs precision scatter ──
    scatter_colors = []
    type_map = {
        "GraphMemoryV4": GREEN, "GraphMemoryV5": TEAL, "GraphMemoryV1": ORANGE,
        "SemanticMemory": PURPLE, "RAGMemory": TEAL,
        "EpisodicSemantic": PURPLE, "HierarchicalMemory": GRAY,
        "CausalMemory": GRAY, "SummaryMemory": GRAY,
        "AttentionMemory": BLUE, "WorkingMemory(7)": BLUE,
        "FlatWindow(50)": BLUE, "GraphMemory+Theta": BLUE,
    }
    for s in systems:
        scatter_colors.append(type_map.get(s, BLUE))

    prec_vals = [p if p > 0 else None for p in prec]
    for i, (s, r, p, c) in enumerate(zip(systems, rewards, prec_vals, scatter_colors)):
        if p is not None:
            ax2.scatter(p, r, color=c, s=80, zorder=3, edgecolors="white", linewidth=0.8)
            ax2.annotate(s, (p, r), textcoords="offset points", xytext=(5, 2), fontsize=7)
    ax2.set_xlabel("Retrieval Precision")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Reward vs Retrieval Precision")
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.01, max(rewards) * 1.2)
    ax2.grid(True, alpha=0.3)

    # ── bottom-left: CMA-ES learning curves ──
    ax3.plot(v4_gens, v4_fit, color=GREEN,  linewidth=2.0, marker="o", markersize=3, label="GraphMemoryV4 (10D)")
    ax3.plot(v1_gens, v1_fit, color=ORANGE, linewidth=2.0, marker="s", markersize=3, label="GraphMemoryV1 (3D)")
    ax3.axhline(v4res["v4"]["eval"]["mean_reward"],        color=GREEN,  linestyle="--", alpha=0.6, linewidth=1.2, label=f"V4 eval: {v4res['v4']['eval']['mean_reward']:.3f}")
    ax3.axhline(v4res["v1_baseline"]["eval"]["mean_reward"], color=ORANGE, linestyle="--", alpha=0.6, linewidth=1.2, label=f"V1 eval: {v4res['v1_baseline']['eval']['mean_reward']:.3f}")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Best Fitness (mean reward)")
    ax3.set_title("CMA-ES Optimization — Learning Curves")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    # annotate jump
    jump_gen = next((h["generation"] for h in v4_hist if h["best_fitness"] >= 0.2), None)
    if jump_gen:
        ax3.axvline(jump_gen, color=GREEN, linestyle=":", alpha=0.5)
        ax3.text(jump_gen + 0.3, min(v4_fit) + 0.005, f"Jump\ngen {jump_gen}", fontsize=7.5, color=GREEN)

    # ── bottom-right: memory size comparison ──
    size_systems = ["GraphMemoryV1\n(V1 baseline)", "GraphMemoryV4\n(V4 optimized)"]
    size_vals    = [v4res["v1_baseline"]["eval"]["mean_memory_size"],
                    v4res["v4"]["eval"]["mean_memory_size"]]
    # add a few benchmark systems for context
    for s in ["FlatWindow(50)", "SemanticMemory", "WorkingMemory(7)"]:
        if s in multihop:
            size_systems.append(s)
            size_vals.append(multihop[s]["mean_memory_size"])
    bar_colors = [ORANGE, GREEN] + [BLUE] * (len(size_systems) - 2)
    bars4 = ax4.bar(range(len(size_systems)), size_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax4.set_xticks(range(len(size_systems)))
    ax4.set_xticklabels(size_systems, fontsize=8)
    ax4.set_ylabel("Mean Memory Size (events)")
    ax4.set_title("Memory Footprint Comparison")
    for bar, v in zip(bars4, size_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
    reduction = size_vals[0] / size_vals[1]
    ax4.text(0.5, 0.92, f"V4 uses {reduction:.0f}x fewer events than V1",
             transform=ax4.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor=GREEN, alpha=0.9))
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("GraphMemoryV4 — Master Benchmark Summary", fontsize=13, fontweight="bold", y=1.01)
    out = os.path.join(FIGS_DIR, "fig_master_benchmark.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B — Ablation Importance Ranking
# ═══════════════════════════════════════════════════════════════════════════
def fig_ablation_ranked():
    data = load("ablation_results.json")
    results = data["results"]

    # exclude "full" (reference)
    configs = {k: v for k, v in results.items() if k != "full"}
    full_reward = results["full"]["mean_reward"]
    full_prec   = results["full"]["mean_precision"]

    names = list(configs.keys())
    reward_deg = [configs[n]["degradation"] * 100 for n in names]
    prec_deg   = [
        (full_prec - configs[n]["mean_precision"]) / (full_prec + 1e-9) * 100
        for n in names
    ]

    # sort by reward degradation descending
    order = sorted(range(len(names)), key=lambda i: reward_deg[i], reverse=True)
    names_s    = [names[i]      for i in order]
    reward_s   = [reward_deg[i] for i in order]
    prec_s     = [prec_deg[i]   for i in order]

    def bar_color(deg):
        if deg > 50:  return RED
        if deg > 10:  return ORANGE
        return GREEN

    r_colors = [bar_color(d) for d in reward_s]
    p_colors = [bar_color(d) for d in prec_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    y = range(len(names_s))
    ax1.barh(y, reward_s, color=r_colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names_s, fontsize=9)
    ax1.set_xlabel("Reward Degradation (%)")
    ax1.set_title("Reward Degradation When Feature Removed")
    ax1.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(ax1.patches, reward_s):
        x_pos = val + 0.5 if val >= 0 else val - 0.5
        ha = "left" if val >= 0 else "right"
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", ha=ha, fontsize=8)
    ax1.set_xlim(min(min(reward_s) - 10, -15), max(reward_s) + 15)

    ax2.barh(y, prec_s, color=p_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Precision Degradation (%)")
    ax2.set_title("Precision Degradation When Feature Removed")
    ax2.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(ax2.patches, prec_s):
        x_pos = val + 0.5 if val >= 0 else val - 0.5
        ha = "left" if val >= 0 else "right"
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", ha=ha, fontsize=8)
    ax2.set_xlim(min(min(prec_s) - 10, -15), max(prec_s) + 15)

    # annotation
    ax1.text(0.97, 0.03,
             "theta_novel is the\nload-bearing pillar",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor=RED, alpha=0.9))

    patch_r = mpatches.Patch(color=RED,    label=">50% degradation (critical)")
    patch_o = mpatches.Patch(color=ORANGE, label="10-50% degradation (important)")
    patch_g = mpatches.Patch(color=GREEN,  label="<10% degradation (minor)")
    ax1.legend(handles=[patch_r, patch_o, patch_g], loc="lower left", fontsize=8)

    fig.suptitle("GraphMemoryV4 — Ablation Study: Feature Importance Ranking",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig_ablation_ranked.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE C — Transfer Heatmap with Interpretation
# ═══════════════════════════════════════════════════════════════════════════
def fig_transfer_annotated():
    data   = load("transfer_results.json")
    matrix = data["matrix"]["MultiHop_V4_zeroshot"]

    envs = ["MultiHopKeyDoor", "GoalRoom", "HardKeyDoor", "MegaQuestRoom"]
    rewards = [matrix[e]["mean_reward"] for e in envs]
    tokens  = [matrix[e]["mean_tokens"] for e in envs]

    annotations = {
        "MultiHopKeyDoor": "In-distribution\n(training env)",
        "GoalRoom":        "Strong transfer\n(simpler task)",
        "HardKeyDoor":     "Moderate transfer\n(similar difficulty)",
        "MegaQuestRoom":   "Complete failure\n(OOD: 10x harder)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── left: heatmap ──
    reward_arr = np.array(rewards).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("rg", [RED, ORANGE, GREEN])
    im = ax1.imshow(reward_arr, cmap=cmap, vmin=0, vmax=max(rewards) * 1.1,
                    aspect="auto")
    ax1.set_xticks(range(len(envs)))
    ax1.set_xticklabels([e.replace("KeyDoor", "\nKeyDoor") for e in envs], fontsize=9)
    ax1.set_yticks([0])
    ax1.set_yticklabels(["V4 theta\n(MultiHop)"], fontsize=9)
    ax1.set_title("Zero-Shot Transfer — Mean Reward")
    plt.colorbar(im, ax=ax1, label="Mean Reward", shrink=0.6)

    for j, (env, r) in enumerate(zip(envs, rewards)):
        ax1.text(j, 0, f"{r:.3f}\n\n{annotations[env]}",
                 ha="center", va="center", fontsize=8.5,
                 color="white" if r < 0.15 else "black", fontweight="bold")

    # ── right: token cost bar ──
    bar_colors = [GREEN if e != "MegaQuestRoom" else RED for e in envs]
    bars = ax2.bar(range(len(envs)), tokens, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(envs)))
    ax2.set_xticklabels([e.replace("KeyDoor", "\nKeyDoor") for e in envs], fontsize=9)
    ax2.set_ylabel("Mean Retrieval Tokens")
    ax2.set_title("Token Cost per Environment")
    for bar, t in zip(bars, tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{t:.0f}", ha="center", fontsize=8.5)
    ratio = tokens[-1] / tokens[0]
    ax2.text(0.97, 0.95, f"MegaQuestRoom\n{ratio:.1f}x more tokens\nthan MultiHop",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED, alpha=0.9))
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("GraphMemoryV4 — Zero-Shot Transfer Analysis",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig_transfer_annotated.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE D — Sensitivity Landscape with Annotations
# ═══════════════════════════════════════════════════════════════════════════
def fig_sensitivity_annotated():
    data = load("sensitivity_results.json")

    dim1_vals = data["dim1_values"]
    dim2_vals = data["dim2_values"]
    reward_grid = np.array(data["reward_grid"])
    prec_grid   = np.array(data["precision_grid"])

    learned_d1 = data["learned_dim1"]
    learned_d2 = data["learned_dim2"]
    best_reward = data["best_reward"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3)

    cmap_r = "YlOrRd"
    cmap_p = "Blues"

    for ax, grid, cmap, title, label in [
        (ax1, reward_grid, cmap_r, "Reward Landscape\n(theta_novel × w_recency)", "Mean Reward"),
        (ax2, prec_grid,   cmap_p, "Precision Landscape\n(theta_novel × w_recency)", "Retrieval Precision"),
    ]:
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                       extent=[min(dim2_vals), max(dim2_vals), min(dim1_vals), max(dim1_vals)])
        plt.colorbar(im, ax=ax, label=label, shrink=0.8)

        # contour lines
        cs = ax.contour(dim2_vals, dim1_vals, grid, levels=5,
                        colors="white", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        # mark learned optimum
        ax.scatter([learned_d2], [learned_d1], color=GOLD, s=120, zorder=5,
                   marker="*", edgecolors="black", linewidth=0.8)
        ax.annotate(f"Learned\noptimum\n({learned_d1:.2f}, {learned_d2:.2f})",
                    (learned_d2, learned_d1),
                    textcoords="offset points", xytext=(12, -20),
                    fontsize=8, color=GOLD, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))

        ax.set_xlabel("w_recency")
        ax.set_ylabel("theta_novel")
        ax.set_title(title)

    # annotation: sharp peak vs broad plateau (data-driven)
    analysis = data.get("analysis", {})
    is_sharp = analysis.get("is_sharp_peak", False)
    if is_sharp:
        ann_text = "Sharp peak —\noptimizer must find\nprecise values"
        ann_color = RED
    else:
        ann_text = "Broad plateau —\nsystem is robust"
        ann_color = GOLD
    ax1.text(0.03, 0.97, ann_text,
             transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor=ann_color, alpha=0.9))

    fig.suptitle("GraphMemoryV4 — Sensitivity Analysis: theta_novel × w_recency",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig_sensitivity_annotated.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE E — Neural Controller Training Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig_neural_analysis():
    data = load("neural_controller_v2_results.json")

    history = data["training"]["history"]
    gens    = [h["generation"]   for h in history]
    fits    = [h["best_fitness"] for h in history]

    neural_multihop = data["eval_multihop"]["mean_reward"]
    v4_scalar       = data["v4_scalar_comparison"]["mean_reward"]
    v4_prec         = data["v4_scalar_comparison"]["mean_precision"]
    neural_prec     = data["eval_multihop"]["mean_precision"]

    # find plateau start
    plateau_start = None
    for i in range(1, len(fits)):
        if fits[i] == fits[i-1]:
            plateau_start = gens[i-1]
            break

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── left: training curve ──
    ax1.plot(gens, fits, color=PURPLE, linewidth=2.0, marker="o", markersize=3)
    ax1.axhline(v4_scalar, color=GREEN, linestyle="--", linewidth=1.5,
                label=f"V4 scalar: {v4_scalar:.3f}")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness (mean reward)")
    ax1.set_title("NeuralControllerV2Small\nCMA-ES Training Curve")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    if plateau_start:
        ax1.axvspan(plateau_start, max(gens), alpha=0.08, color=GRAY)
        ax1.text(plateau_start + 0.5, min(fits) + 0.002,
                 f"Plateau from gen {plateau_start}", fontsize=8, color=GRAY)
    ax1.text(0.97, 0.05,
             f"Final: {fits[-1]:.3f}\nvs V4: {v4_scalar:.3f}",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F3FF", edgecolor=PURPLE, alpha=0.9))

    # ── middle: comparison bar chart (dual y-axis: reward vs precision on different scales) ──
    systems_c = ["NeuralV2Small\n(1962 params)", "V4 Scalar\n(10 params)", "V1 Baseline\n(3 params)"]
    rewards_c = [neural_multihop, v4_scalar, 0.1017]
    precs_c   = [neural_prec,     v4_prec,   0.6321]
    bar_colors_c = [PURPLE, GREEN, ORANGE]

    x = np.arange(len(systems_c))
    w = 0.35
    ax2_left = ax2
    ax2_right = ax2.twinx()
    b1 = ax2_left.bar(x - w/2, rewards_c, w, color=bar_colors_c, alpha=0.85, label="Reward", edgecolor="white")
    b2 = ax2_right.bar(x + w/2, precs_c,   w, color=bar_colors_c, alpha=0.45, label="Precision", edgecolor="white", hatch="//")
    ax2_left.set_ylabel("Mean Reward", color=PURPLE)
    ax2_right.set_ylabel("Retrieval Precision", color=GRAY)
    ax2_left.tick_params(axis="y", labelcolor=PURPLE)
    ax2_right.tick_params(axis="y", labelcolor=GRAY)
    ax2_left.set_ylim(0, max(rewards_c) * 1.4)
    ax2_right.set_ylim(0, 1.15)
    ax2_left.set_xticks(x)
    ax2_left.set_xticklabels(systems_c, fontsize=8.5)
    ax2_left.set_title("Performance Comparison\n(MultiHopKeyDoor)")
    ax2_left.legend(loc="upper left", fontsize=8)
    ax2_right.legend(loc="upper right", fontsize=8)
    ax2_left.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(b1, rewards_c):
        ax2_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b2, precs_c):
        ax2_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    # Data-driven annotation: neural can match or exceed V4 with enough budget
    if neural_multihop >= v4_scalar:
        ax2.text(0.5, 0.97,
                 "Neural matches or exceeds\nscalar V4 with sufficient\ntraining budget (200 gens)",
                 transform=ax2.transAxes, ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor=GREEN, alpha=0.9))
    else:
        pct = (v4_scalar - neural_multihop) / (v4_scalar + 1e-9) * 100
        ax2.text(0.5, 0.97,
                 f"Neural is {pct:.0f}% worse than\nscalar V4 — expressivity\nvs trainability tradeoff",
                 transform=ax2.transAxes, ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor=RED, alpha=0.9))

    # ── right: transfer comparison (MultiHop vs MegaQuest, Neural vs V4) ──
    v4_megaquest = 0.0
    try:
        trans = load("transfer_results.json")
        v4_megaquest = trans["matrix"]["MultiHop_V4_zeroshot"]["MegaQuestRoom"]["mean_reward"]
    except (KeyError, TypeError, FileNotFoundError):
        pass
    envs_r = ["MultiHopKeyDoor", "MegaQuestRoom"]
    neural_vals = [neural_multihop, (data.get("eval_megaquest") or {}).get("mean_reward", 0.0)]
    v4_vals = [v4_scalar, v4_megaquest]
    x_r = np.arange(len(envs_r))
    w_r = 0.35
    b_r1 = ax3.bar(x_r - w_r/2, neural_vals, w_r, color=PURPLE, alpha=0.85, label="NeuralV2Small", edgecolor="white")
    b_r2 = ax3.bar(x_r + w_r/2, v4_vals, w_r, color=GREEN, alpha=0.85, label="V4 Scalar", edgecolor="white")
    ax3.set_xticks(x_r)
    ax3.set_xticklabels([e.replace("KeyDoor", "\nKeyDoor") for e in envs_r], fontsize=8.5)
    ax3.set_ylabel("Mean Reward")
    ax3.set_title("Zero-Shot Transfer\n(Neural and V4 trained on MultiHop)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(b_r1, neural_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b_r2, v4_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax3.text(0.5, 0.03, "Both fail on MegaQuestRoom (OOD);\ntask-dependent memory",
             transform=ax3.transAxes, ha="center", fontsize=7.5, color=GRAY,
             style="italic")

    fig.suptitle("NeuralMemoryControllerV2Small — Training Analysis & Comparison",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig_neural_analysis.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE — Neural V2 learning curve (from JSON, no re-run)
# ═══════════════════════════════════════════════════════════════════════════
def fig_neural_v2_curves():
    """Regenerate 2-panel learning curve from neural_controller_v2_results.json."""
    data = load("neural_controller_v2_results.json")
    history = data["training"]["history"]
    gens = [h["generation"] for h in history]
    best = [h["best_fitness"] for h in history]
    sigmas = [h["sigma"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(gens, best, "b-o", markersize=4, linewidth=1.5, label="Best fitness")
    ax1.set_ylabel("Mean Reward (best candidate)", fontsize=11)
    ax1.set_title("NeuralMemoryControllerV2Small — CMA-ES Training\n"
                  "(50->32->10 MLP, 1,962 params, MultiHopKeyDoor)", fontsize=12)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=9)
    ax2.plot(gens, sigmas, "r-", linewidth=1.5, label="Sigma (step size)")
    ax2.set_ylabel("CMA-ES Sigma", fontsize=11)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "fig_neural_v2_curves.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE — Neural vs V4 transfer comparison (dedicated)
# ═══════════════════════════════════════════════════════════════════════════
def fig_neural_transfer():
    """Dedicated transfer figure: Neural vs V4 on MultiHop and MegaQuest."""
    data = load("neural_controller_v2_results.json")
    trans = load("transfer_results.json")
    neural_mh = data["eval_multihop"]["mean_reward"]
    neural_mq = (data.get("eval_megaquest") or {}).get("mean_reward", 0.0)
    v4_mh = data["v4_scalar_comparison"]["mean_reward"]
    v4_mq = trans["matrix"]["MultiHop_V4_zeroshot"]["MegaQuestRoom"]["mean_reward"]

    envs = ["MultiHopKeyDoor", "MegaQuestRoom"]
    neural_vals = [neural_mh, neural_mq]
    v4_vals = [v4_mh, v4_mq]
    x = np.arange(len(envs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, neural_vals, w, color=PURPLE, alpha=0.85, label="NeuralV2Small", edgecolor="white")
    b2 = ax.bar(x + w/2, v4_vals, w, color=GREEN, alpha=0.85, label="V4 Scalar", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("KeyDoor", "\nKeyDoor") for e in envs], fontsize=10)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Zero-Shot Transfer — Neural vs V4\n(both trained on MultiHopKeyDoor)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(b1, neural_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(b2, v4_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("NeuralMemoryControllerV2Small — Transfer Comparison", fontsize=12, fontweight="bold", y=1.02)
    out = os.path.join(FIGS_DIR, "fig_neural_transfer.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE fig11 — Pareto Front (real data)
# ═══════════════════════════════════════════════════════════════════════════
def fig11_pareto():
    bench  = load("benchmark_results.json")
    v4res  = load("graphmemory_v4_cmaes_results.json")

    multihop = bench["MultiHop-KeyDoor"]
    all_systems = dict(multihop)
    all_systems["GraphMemoryV4"] = {
        "mean_reward":      v4res["v4"]["eval"]["mean_reward"],
        "mean_tokens":      v4res["v4"]["eval"]["mean_tokens"],
        "mean_memory_size": v4res["v4"]["eval"]["mean_memory_size"],
    }
    all_systems["GraphMemoryV1"] = {
        "mean_reward":      v4res["v1_baseline"]["eval"]["mean_reward"],
        "mean_tokens":      v4res["v1_baseline"]["eval"]["mean_tokens"],
        "mean_memory_size": v4res["v1_baseline"]["eval"]["mean_memory_size"],
    }

    names   = list(all_systems.keys())
    rewards = [all_systems[s]["mean_reward"] for s in names]
    tokens  = [all_systems[s]["mean_tokens"]  for s in names]

    type_colors = {
        "GraphMemoryV4": GREEN, "GraphMemoryV5": TEAL, "GraphMemoryV1": ORANGE,
        "SemanticMemory": PURPLE, "RAGMemory": TEAL,
        "EpisodicSemantic": PURPLE, "HierarchicalMemory": GRAY,
        "CausalMemory": GRAY, "SummaryMemory": GRAY,
        "AttentionMemory": BLUE, "WorkingMemory(7)": BLUE,
        "FlatWindow(50)": BLUE, "GraphMemory+Theta": BLUE,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r, t in zip(names, rewards, tokens):
        c = type_colors.get(name, BLUE)
        ax.scatter(t, r, color=c, s=100, zorder=3, edgecolors="white", linewidth=0.8)
        ax.annotate(name, (t, r), textcoords="offset points", xytext=(5, 3), fontsize=7.5)

    # compute Pareto front (maximize reward, minimize tokens)
    pareto = []
    for i, (r, t) in enumerate(zip(rewards, tokens)):
        dominated = False
        for j, (r2, t2) in enumerate(zip(rewards, tokens)):
            if i != j and r2 >= r and t2 <= t and (r2 > r or t2 < t):
                dominated = True
                break
        if not dominated:
            pareto.append((t, r, names[i]))
    pareto.sort()
    if len(pareto) > 1:
        px, py = zip(*[(p[0], p[1]) for p in pareto])
        ax.plot(px, py, color=RED, linewidth=1.5, linestyle="--", alpha=0.7, label="Pareto front")

    ax.set_xlabel("Mean Retrieval Tokens (proxy for LLM cost)")
    ax.set_ylabel("Mean Reward (MultiHopKeyDoor)")
    ax.set_title("Pareto Front — Reward vs Token Cost\n(top-left preferred: high reward, low cost)")
    ax.grid(True, alpha=0.3)
    patch_v4 = mpatches.Patch(color=GREEN,  label="GraphMemoryV4")
    patch_v1 = mpatches.Patch(color=ORANGE, label="GraphMemoryV1")
    patch_ot = mpatches.Patch(color=BLUE,   label="Other systems")
    ax.legend(handles=[patch_v4, patch_v1, patch_ot,
                        mpatches.Patch(color=RED, label="Pareto front")],
              fontsize=8)

    out = os.path.join(FIGS_DIR, "fig11_pareto.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE fig13 — Memory Size Comparison (real data)
# ═══════════════════════════════════════════════════════════════════════════
def fig13_memory_size():
    bench  = load("benchmark_results.json")
    v4res  = load("graphmemory_v4_cmaes_results.json")

    multihop = bench["MultiHop-KeyDoor"]
    all_systems = dict(multihop)
    all_systems["GraphMemoryV4"] = {
        "mean_reward":      v4res["v4"]["eval"]["mean_reward"],
        "mean_tokens":      v4res["v4"]["eval"]["mean_tokens"],
        "mean_memory_size": v4res["v4"]["eval"]["mean_memory_size"],
    }
    all_systems["GraphMemoryV1"] = {
        "mean_reward":      v4res["v1_baseline"]["eval"]["mean_reward"],
        "mean_tokens":      v4res["v1_baseline"]["eval"]["mean_tokens"],
        "mean_memory_size": v4res["v1_baseline"]["eval"]["mean_memory_size"],
    }

    names    = list(all_systems.keys())
    sizes    = [all_systems[s]["mean_memory_size"] for s in names]
    rewards  = [all_systems[s]["mean_reward"]      for s in names]

    order = sorted(range(len(names)), key=lambda i: sizes[i], reverse=True)
    names_s   = [names[i]   for i in order]
    sizes_s   = [sizes[i]   for i in order]
    rewards_s = [rewards[i] for i in order]

    type_colors = {
        "GraphMemoryV4": GREEN, "GraphMemoryV5": TEAL, "GraphMemoryV1": ORANGE,
    }
    bar_colors = [type_colors.get(n, BLUE) for n in names_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(wspace=0.35)

    bars = ax1.barh(range(len(names_s)), sizes_s, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(names_s)))
    ax1.set_yticklabels(names_s, fontsize=8.5)
    ax1.set_xlabel("Mean Memory Size (events stored)")
    ax1.set_title("Memory Footprint — All Systems")
    for bar, val in zip(bars, sizes_s):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}", va="center", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="x")

    # reward vs memory size scatter
    for name, size, reward in zip(names, sizes, rewards):
        c = type_colors.get(name, BLUE)
        ax2.scatter(size, reward, color=c, s=90, zorder=3, edgecolors="white", linewidth=0.8)
        ax2.annotate(name, (size, reward), textcoords="offset points", xytext=(4, 3), fontsize=7.5)
    ax2.set_xlabel("Mean Memory Size (events)")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Memory Efficiency — Reward vs Size")
    ax2.grid(True, alpha=0.3)

    v4_size   = all_systems["GraphMemoryV4"]["mean_memory_size"]
    v4_reward = all_systems["GraphMemoryV4"]["mean_reward"]
    v1_size   = all_systems["GraphMemoryV1"]["mean_memory_size"]
    v1_reward = all_systems["GraphMemoryV1"]["mean_reward"]
    ax2.annotate("",
                 xy=(v4_size, v4_reward), xytext=(v1_size, v1_reward),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
    ax2.text((v1_size + v4_size)/2, (v1_reward + v4_reward)/2 + 0.005,
             f"{v1_size/v4_size:.0f}x smaller\n+{(v4_reward-v1_reward):.3f} reward",
             ha="center", fontsize=8.5, color=GREEN, fontweight="bold")

    patch_v4 = mpatches.Patch(color=GREEN,  label="GraphMemoryV4 (ours)")
    patch_v1 = mpatches.Patch(color=ORANGE, label="GraphMemoryV1 (baseline)")
    patch_ot = mpatches.Patch(color=BLUE,   label="Other systems")
    ax2.legend(handles=[patch_v4, patch_v1, patch_ot], fontsize=8)

    fig.suptitle("Memory Size Comparison — All Systems on MultiHopKeyDoor",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig13_memory_size.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE — Story: Precision vs Reward by Environment
# ═══════════════════════════════════════════════════════════════════════════
def fig_story_precision_reward():
    """One scatter per environment: precision vs reward. Shows precision gates success on MultiHop."""
    bench = load("benchmark_results.json")
    envs = list(bench.keys())
    n_envs = len(envs)
    if n_envs == 0:
        return
    cols = min(2, n_envs)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n_envs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, env_name in enumerate(envs):
        ax = axes[idx]
        data = bench[env_name]
        systems = [s for s in data.keys() if "error" not in data[s]]
        rewards = [data[s].get("mean_reward", 0) for s in systems]
        precs = [data[s].get("retrieval_precision") for s in systems]
        x = [p if p is not None else -0.05 for p in precs]
        colors = [GREEN if "V4" in s else (TEAL if "V5" in s else (ORANGE if "V1" in s else BLUE)) for s in systems]
        for s, xi, r, c in zip(systems, x, rewards, colors):
            ax.scatter(xi, r, color=c, s=80, zorder=3, edgecolors="white")
            ax.annotate(s[:18], (xi, r), textcoords="offset points", xytext=(4, 2), fontsize=6)
        ax.set_xlabel("Retrieval precision")
        ax.set_ylabel("Mean reward")
        has_na = any(p is None for p in precs)
        title = f"{env_name}" + (" (precision N/A: no hint steps)" if has_na else "")
        ax.set_title(title)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, max(rewards) * 1.15 if rewards else 1)
        ax.grid(True, alpha=0.3)
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Precision vs Reward by Environment — Precision Gates Success on MultiHop",
                 fontsize=12, fontweight="bold")
    out = os.path.join(FIGS_DIR, "fig_story_precision_reward.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate thesis figures from results/*.json")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Generate only figures whose result files exist; do not exit on missing files")
    args = parser.parse_args()

    can_run = check_required_files(allow_missing=args.allow_missing)
    print("Generating thesis figures...")
    print()

    if can_run.get("fig_master_benchmark"):
        print("Figure A — Master Benchmark Summary")
        fig_master_benchmark()
    else:
        print("Figure A — Master Benchmark Summary (skipped, missing data)")

    if can_run.get("fig_ablation_ranked"):
        print("Figure B — Ablation Importance Ranking")
        fig_ablation_ranked()
    else:
        print("Figure B — Ablation Importance Ranking (skipped, missing data)")

    if can_run.get("fig_transfer_annotated"):
        print("Figure C — Transfer Heatmap with Interpretation")
        fig_transfer_annotated()
    else:
        print("Figure C — Transfer Heatmap (skipped, missing data)")

    if can_run.get("fig_sensitivity_annotated"):
        print("Figure D — Sensitivity Landscape with Annotations")
        fig_sensitivity_annotated()
    else:
        print("Figure D — Sensitivity (skipped, missing data)")

    if can_run.get("fig_neural_analysis"):
        print("Figure E — Neural Controller Training Analysis")
        fig_neural_analysis()
    else:
        print("Figure E — Neural Analysis (skipped, missing data)")

    if can_run.get("fig_neural_v2_curves"):
        print("Figure — Neural V2 Learning Curve (from JSON)")
        fig_neural_v2_curves()
    else:
        print("Figure — Neural V2 Curves (skipped, missing data)")

    if can_run.get("fig_neural_transfer"):
        print("Figure — Neural vs V4 Transfer")
        fig_neural_transfer()
    else:
        print("Figure — Neural Transfer (skipped, missing data)")

    if can_run.get("fig11_pareto"):
        print("Figure fig11 — Pareto Front (real data)")
        fig11_pareto()
    else:
        print("Figure fig11 — Pareto (skipped, missing data)")

    if can_run.get("fig13_memory_size"):
        print("Figure fig13 — Memory Size Comparison (real data)")
        fig13_memory_size()
    else:
        print("Figure fig13 — Memory Size (skipped, missing data)")

    if can_run.get("fig_story_precision_reward"):
        print("Figure — Story: Precision vs Reward by Environment")
        fig_story_precision_reward()
    else:
        print("Figure — Story precision/reward (skipped, missing data)")

    print()
    print("All figures saved to docs/figures/")
