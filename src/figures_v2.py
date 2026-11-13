"""v2 figure: cumulative success vs call budget per bin, three arms.
Same diagram-design tokens as v1 figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures import PAPER, INK, MUTED, ACCENT  # noqa: F401  (keeps rcParams)

ARM_STYLE = {"independent": ("#8fa1bd", "o", "independent resampling"),
             "binary": (MUTED, "s", "binary feedback"),
             "struct": (ACCENT, "^", "structured feedback")}
BIN_LABEL = {"A": "5-10 heavy atoms", "B": "11-20 heavy atoms"}


def main():
    df = pd.read_csv("results/v2/success_by_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, b in zip(axes, ["A", "B"]):
        for arm in ["independent", "binary", "struct"]:
            sub = df[(df["bin"] == b) & (df["arm"] == arm)].sort_values("calls")
            if sub.empty:
                continue
            c, m, lab = ARM_STYLE[arm]
            ax.plot(sub["calls"], sub["success"], f"-{m}", color=c, markersize=5,
                    linewidth=1.5, label=lab)
            ax.fill_between(sub["calls"], sub["lo"], sub["hi"], color=c,
                            alpha=0.15, linewidth=0)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("call budget (responses per molecule)")
        ax.set_title(f"bin {b}: {BIN_LABEL[b]}")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("cumulative fraction of molecules passed")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Repair loops vs independent resampling", fontsize=11, fontweight="bold")
    fig.tight_layout()
    Path("results/v2").mkdir(exist_ok=True)
    fig.savefig("results/v2/fig_v2_budget.png", dpi=200)
    fig.savefig("results/v2/fig_v2_budget.svg")
    print("wrote results/v2/fig_v2_budget.*")


if __name__ == "__main__":
    main()
