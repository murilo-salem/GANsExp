"""Generate a data-grounded convergence reference for paper Figure 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/2324_temporal_gan_loss_ablation/results/training_history.csv"
OUTPUT = ROOT / "artifacts/paper_figures/fig4_fold_convergence"

TARGETS = {
    "Biomassa": ("Biomass selection", "#2C7FB8"),
    "Produtividade": ("Yield selection", "#D95F0E"),
}


def summarize(history: pd.DataFrame) -> pd.DataFrame:
    selected = history[
        history["split"].eq("outer") & history["arm"].eq("full")
    ].copy()
    summary = (
        selected.groupby(["target", "outer_block", "epoch"], as_index=False)["l1"]
        .agg(mean="mean", sd="std", n="count")
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


def draw() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(SOURCE)
    summary = summarize(history)
    summary.to_csv(OUTPUT / "fold_convergence_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.75), sharex=True, sharey=True,
                             constrained_layout=True)
    for index, (ax, block) in enumerate(zip(axes, [1, 2, 3, 4])):
        for target, (label, color) in TARGETS.items():
            part = summary[
                (summary["target"] == target) & (summary["outer_block"] == block)
            ]
            x = part["epoch"].to_numpy()
            y = part["mean"].to_numpy()
            ci = part["ci95"].fillna(0).to_numpy()
            ax.plot(x, y, color=color, lw=1.35, label=label)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.14, linewidth=0)
        ax.set_title(f"Outer fold {block}")
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 200)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.text(-0.16, 1.04, f"({chr(97 + index)})", transform=ax.transAxes,
                fontweight="bold", va="bottom")
    axes[0].set_ylabel("Training reconstruction loss (L1)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside lower center")
    fig.savefig(OUTPUT / "fig4_fold_convergence_reference.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUTPUT / "fig4_fold_convergence_reference.svg", bbox_inches="tight")


if __name__ == "__main__":
    draw()
