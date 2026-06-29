"""Figure: descriptor-predicted-theta recovered-lift fraction, per benchmark
(leave-one-benchmark-out). Reads results/stage3/theta_predictability.json,
writes docs/figures/fig_theta_transfer.png + thesis/figures/."""
import json, collections, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results/stage3/theta_predictability.json").read_text())
by = collections.defaultdict(list)
for s in d["per_slice"]:
    if s["recovered_lift_fraction"] is not None:
        by[s["benchmark"]].append(s["recovered_lift_fraction"])
benches = sorted(by, key=lambda b: -sum(by[b]) / len(by[b]))
means = [sum(by[b]) / len(by[b]) for b in benches]
overall = d["mean_recovered_lift_fraction"]

fig, ax = plt.subplots(figsize=(7.5, 4.6))
colors = ["#2563EB" if m >= 0.7 else "#D97706" for m in means]
bars = ax.bar(range(len(benches)), means, color=colors, edgecolor="white")
for i, m in enumerate(means):
    ax.text(i, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(overall, color="#111827", ls="--", lw=1.2,
           label=f"overall mean {overall:.2f}  (95% CI "
                 f"[{d['recovered_fraction_ci']['ci_lower']:.2f}, "
                 f"{d['recovered_fraction_ci']['ci_upper']:.2f}])")
ax.axhline(1.0, color="#9CA3AF", ls=":", lw=1, label="full tuned-theta lift")
ax.set_xticks(range(len(benches)))
ax.set_xticklabels(benches, rotation=15)
ax.set_ylabel("recovered lift fraction\n(predicted vs. canonical, of tuned lift)")
ax.set_ylim(0, 1.12)
ax.set_title("Descriptor-predicted $\\theta$ (leave-one-benchmark-out):\n"
             "fraction of the per-task tuning lift recovered without tuning")
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
out = ROOT / "docs/figures/fig_theta_transfer.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
shutil.copy(out, ROOT / "thesis/figures/fig_theta_transfer.png")
print(f"[OK] wrote {out} and thesis/figures/ copy")
