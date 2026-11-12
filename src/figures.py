"""Figures: funnel-by-bin and pass@k curves. Style follows the diagram-design
skill tokens: paper background, muted series, coral accent, faint gridlines,
labeled axes, no decoration."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
ACCENT = "#eb6c36"
GRID = "rgba(45,49,66,0.08)"
BIN_LABEL = {"A": "5-10 heavy atoms", "B": "11-20", "C": "21-30"}
BIN_COLOR = {"A": ACCENT, "B": MUTED, "C": "#8fa1bd"}
STAGES = ["P0", "P1", "P2", "P3"]
STAGE_LABEL = {"P0": "P0 parse", "P1": "P1 formula", "P2": "P2 graph", "P3": "P3 geometry"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": "rgba(45,49,66,0.25)", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load():
    return json.loads(Path("results/summary.json").read_text())


def fig_funnel(d):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = ["A", "B", "C"]
    x = np.arange(len(STAGES))
    w = 0.26
    for bi, b in enumerate(bins):
        vals = [d["bins"][b]["funnel"][s.lower()]["frac_of_all"] for s in STAGES]
        bars = ax.bar(x + (bi - 1) * w, vals, w, label=BIN_LABEL[b],
                      color=BIN_COLOR[b], edgecolor=INK, linewidth=0.4, alpha=0.92)
        for xi, v in zip(x + (bi - 1) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
    ax.set_ylabel("fraction of samples surviving")
    ax.set_ylim(0, 1.05)
    ax.set_title("Check funnel by size bin (cumulative survival)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, title="size bin", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig_funnel.png", dpi=200)
    fig.savefig("results/fig_funnel.svg")


def fig_passk(d):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ks = np.arange(1, 9)
    for b in ["A", "B", "C"]:
        mean = [d["bins"][b]["pass_at_k"][str(k)]["mean"] for k in ks]
        lo = [d["bins"][b]["pass_at_k"][str(k)]["lo"] for k in ks]
        hi = [d["bins"][b]["pass_at_k"][str(k)]["hi"] for k in ks]
        ax.plot(ks, mean, "-o", color=BIN_COLOR[b], markersize=4, linewidth=1.4,
                label=BIN_LABEL[b])
        ax.fill_between(ks, lo, hi, color=BIN_COLOR[b], alpha=0.15, linewidth=0)
    ax.set_xlabel("k (samples per molecule)")
    ax.set_ylabel("pass@k")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.set_title("pass@k by size bin (checker as verifier)")
    ax.legend(frameon=False, fontsize=8.5, title="size bin", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig_passk.png", dpi=200)
    fig.savefig("results/fig_passk.svg")


if __name__ == "__main__":
    d = load()
    fig_funnel(d)
    fig_passk(d)
    print("wrote results/fig_funnel.* and results/fig_passk.*")
