"""Generate a data-grounded reference for paper Figure 3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/runs/2324_temporal_gan_loss_ablation/results"
OUTPUT = ROOT / "artifacts/paper_figures/fig3_loss_ablation"

ARM_ORDER = [
    "l1",
    "l1_gan",
    "l1_indices_texture",
    "l1_gan_feature_matching",
    "full",
]
LABELS = {
    "l1": "L1",
    "l1_gan": "L1 + GAN",
    "l1_indices_texture": "L1 + indices + texture",
    "l1_gan_feature_matching": "L1 + GAN + feature matching",
    "full": "Full objective",
}
COLORS = {
    "l1": "#3366A6",
    "l1_gan": "#DD7F2A",
    "l1_indices_texture": "#3B8E5A",
    "l1_gan_feature_matching": "#8A5AA6",
    "full": "#B84848",
}


def curve_summary(history: pd.DataFrame) -> pd.DataFrame:
    selected = history[history["split"].eq("outer")].copy()
    summary = (
        selected.groupby(["target", "arm", "epoch"], as_index=False)["l1"]
        .agg(mean="mean", sd="std", n="count")
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    per_run = (
        metrics.groupby(["target", "arm", "outer_block", "seed"], as_index=False)["l1_masked"]
        .mean()
    )
    summary = (
        per_run.groupby(["target", "arm"], as_index=False)["l1_masked"]
        .agg(mean="mean", sd="std", n="count")
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci95"] = 1.96 * summary["se"]
    return summary


def draw() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(SOURCE / "training_history.csv")
    metrics = pd.read_csv(SOURCE / "gan_image_metrics.csv")
    curves = curve_summary(history)
    heldout = metric_summary(metrics)
    curves.to_csv(OUTPUT / "curve_summary.csv", index=False)
    heldout.to_csv(OUTPUT / "heldout_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), constrained_layout=True)

    for ax, target, title in zip(
        axes[:2],
        ["Biomassa", "Produtividade"],
        ["Biomass-selected configurations", "Yield-selected configurations"],
    ):
        for arm in ARM_ORDER:
            part = curves[(curves.target == target) & (curves.arm == arm)]
            x = part.epoch.to_numpy()
            y = part["mean"].to_numpy()
            ci = part.ci95.fillna(0).to_numpy()
            ax.plot(x, y, lw=1.35, color=COLORS[arm], label=LABELS[arm])
            ax.fill_between(x, y - ci, y + ci, color=COLORS[arm], alpha=0.10, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training reconstruction loss (L1)")
        ax.set_xlim(1, 200)
        ax.grid(alpha=0.18, linewidth=0.5)

    ax = axes[2]
    x = np.arange(len(ARM_ORDER))
    width = 0.36
    for offset, target, color, label in [
        (-width / 2, "Biomassa", "#2C7FB8", "Biomass selection"),
        (width / 2, "Produtividade", "#D95F0E", "Yield selection"),
    ]:
        part = heldout.set_index(["target", "arm"]).loc[[(target, arm) for arm in ARM_ORDER]]
        ax.bar(x + offset, part["mean"], width, yerr=part["ci95"], capsize=2,
               color=color, alpha=0.82, label=label, linewidth=0)
    ax.set_title("Held-out synthetic-image error")
    ax.set_ylabel("Masked L1 error")
    ax.set_xticks(x, [LABELS[a] for a in ARM_ORDER], rotation=32, ha="right")
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.legend(frameon=False, loc="lower left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="outside lower center")
    for i, ax in enumerate(axes):
        ax.text(-0.14, 1.05, f"({chr(97 + i)})", transform=ax.transAxes,
                fontweight="bold", va="bottom")

    fig.savefig(OUTPUT / "fig3_loss_ablation_reference.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "fig3_loss_ablation_reference.svg", bbox_inches="tight")


if __name__ == "__main__":
    draw()
