"""Generate promotional figure: Pipeline iterative improvement showcase.

Shows two experiment cases side-by-side demonstrating how the AutoResearchClaw
pipeline progressively improves experimental methods through self-iteration.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BLUE = "#1565C0"
GREEN = "#2E7D32"
RED = "#C62828"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"
GRAY = "#757575"

# ── Data ─────────────────────────────────────────────────────────────────────

# Case 1: Continual Meta-Learning for Few-Shot Adaptation
case1_iters = [0, 1, 2, 3, 4]
case1_labels = [
    "Baseline\n(Initial Code)",
    "Iter 1\n(Deep Encoder\n+ Meta-SGD)",
    "Iter 2\n(Prototype Net\n— Regression)",
    "Iter 3\n(Linear Clf\n+ L2 Anchor)",
    "Iter 4\n(Converged)",
]
case1_error = [0.7411, 0.1883, 0.2249, 0.0663, 0.0656]
case1_accuracy = [100 * (1 - e) for e in case1_error]
# Marker styles: green=improved, red=regressed, gray=no change
case1_colors = [GRAY, GREEN, RED, GREEN, GRAY]
case1_improved = [None, True, False, True, None]

# Case 2: RLHF + Curriculum-Based Reward Shaping
case2_iters = [0, 1, 2, 3, 4]
case2_labels = [
    "Baseline\n(Vanilla PPO)",
    "Iter 1\n(No Change)",
    "Iter 2\n(+Reward Model\n+Curriculum)",
    "Iter 3\n(+Rank-Norm\n+Policy EMA)",
    "Iter 4\n(+Confidence\nGating)",
]
case2_error = [0.6443, 0.6443, 0.3843, 0.3696, 0.3344]
case2_alignment = [100 * (1 - e) for e in case2_error]
case2_colors = [GRAY, GRAY, GREEN, GREEN, GREEN]

# ── Figure ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ── Case 1: Meta-Learning ───────────────────────────────────────────────────

# Main line
ax1.plot(case1_iters, case1_accuracy, "o-", color=BLUE, linewidth=2.5,
         markersize=10, zorder=5, label="Few-Shot Accuracy")

# Colored markers for improvement status
for i, (x, y, c) in enumerate(zip(case1_iters, case1_accuracy, case1_colors)):
    ax1.scatter(x, y, s=120, color=c, zorder=6, edgecolors="white", linewidths=1.5)

# Annotate key improvements
ax1.annotate(
    "+55.3 pts\nDeep encoder\n+ context-gated replay",
    xy=(1, case1_accuracy[1]), xytext=(1.3, 55),
    fontsize=8.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    ha="left",
)
ax1.annotate(
    "Prototype net\ntoo simple",
    xy=(2, case1_accuracy[2]), xytext=(2.25, 65),
    fontsize=8, color=RED, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    ha="left",
)
ax1.annotate(
    "+15.9 pts\nLinear clf + L2 anchor\n+ cosine gating",
    xy=(3, case1_accuracy[3]), xytext=(2.5, 98),
    fontsize=8.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    ha="left",
)

# Reference line for "ideal" performance
ax1.axhline(y=100, color=ORANGE, linestyle=":", alpha=0.6, linewidth=1.5)
ax1.text(4.3, 99, "Oracle (100%)", fontsize=8, color=ORANGE, ha="right",
         fontstyle="italic", va="top")

# Shaded improvement region
ax1.fill_between(case1_iters, case1_accuracy, case1_accuracy[0],
                 where=[a >= case1_accuracy[0] for a in case1_accuracy],
                 alpha=0.08, color=BLUE)

ax1.set_xlabel("Self-Iteration Round", fontsize=12)
ax1.set_ylabel("Few-Shot Accuracy (%)", fontsize=12)
ax1.set_title("Case A: Continual Meta-Learning\nfor Few-Shot Adaptation", fontsize=13,
              fontweight="bold", pad=12)
ax1.set_ylim(15, 105)
ax1.set_xticks(case1_iters)
ax1.set_xticklabels(case1_labels, fontsize=7.5, ha="center")

# Summary box
summary1 = f"Baseline: {case1_accuracy[0]:.1f}%  →  Best: {case1_accuracy[3]:.1f}%\nImprovement: +{case1_accuracy[3]-case1_accuracy[0]:.1f} pts ({(case1_accuracy[3]-case1_accuracy[0])/case1_accuracy[0]*100:.0f}% rel.)"
ax1.text(0.02, 0.97, summary1, transform=ax1.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.9,
                   edgecolor=BLUE, linewidth=1.2))

# ── Case 2: RLHF ────────────────────────────────────────────────────────────

ax2.plot(case2_iters, case2_alignment, "s-", color=PURPLE, linewidth=2.5,
         markersize=10, zorder=5, label="Alignment Score")

for i, (x, y, c) in enumerate(zip(case2_iters, case2_alignment, case2_colors)):
    ax2.scatter(x, y, s=120, color=c, zorder=6, edgecolors="white", linewidths=1.5,
                marker="s")

# Annotate
ax2.annotate(
    "No improvement\n(minor code fix)",
    xy=(1, case2_alignment[1]), xytext=(1.3, 30),
    fontsize=8, color=GRAY, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
    ha="left",
)
ax2.annotate(
    "+26.0 pts\n+Learned reward model\n+Curriculum scheduling",
    xy=(2, case2_alignment[2]), xytext=(1.8, 75),
    fontsize=8.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    ha="left",
)
ax2.annotate(
    "+1.4 pts\n+Rank-norm\n+Policy EMA",
    xy=(3, case2_alignment[3]), xytext=(3.2, 73),
    fontsize=8, color=GREEN,
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    ha="left",
)
ax2.annotate(
    "+3.6 pts\n+Confidence gating\n+Mini-batch RM",
    xy=(4, case2_alignment[4]), xytext=(3.5, 80),
    fontsize=8.5, color=GREEN, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    ha="left",
)

# Shaded improvement
ax2.fill_between(case2_iters, case2_alignment, case2_alignment[0],
                 where=[a >= case2_alignment[0] for a in case2_alignment],
                 alpha=0.08, color=PURPLE)

ax2.set_xlabel("Self-Iteration Round", fontsize=12)
ax2.set_ylabel("LLM Alignment Score (%)", fontsize=12)
ax2.set_title("Case B: RLHF with Curriculum-Based\nReward Shaping for LLM Alignment", fontsize=13,
              fontweight="bold", pad=12)
ax2.set_ylim(15, 105)
ax2.set_xticks(case2_iters)
ax2.set_xticklabels(case2_labels, fontsize=7.5, ha="center")

summary2 = f"Baseline: {case2_alignment[0]:.1f}%  →  Best: {case2_alignment[4]:.1f}%\nImprovement: +{case2_alignment[4]-case2_alignment[0]:.1f} pts ({(case2_alignment[4]-case2_alignment[0])/case2_alignment[0]*100:.0f}% rel.)"
ax2.text(0.02, 0.97, summary2, transform=ax2.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5", alpha=0.9,
                   edgecolor=PURPLE, linewidth=1.2))

# ── Legend ───────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=GREEN, edgecolor="white", label="Improved"),
    mpatches.Patch(facecolor=RED, edgecolor="white", label="Regressed (auto-recovered)"),
    mpatches.Patch(facecolor=GRAY, edgecolor="white", label="No change / Baseline"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=10, frameon=True, fancybox=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

# ── Suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "AutoResearchClaw: Autonomous Self-Iterating Experiment Optimization",
    fontsize=15, fontweight="bold", y=1.02,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.98])

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "iteration_improvement_showcase.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

# Also save a PDF version for papers
pdf_path = out_dir / "iteration_improvement_showcase.pdf"
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
print(f"Saved: {pdf_path}")

plt.close(fig)
