"""Build a band-wise real-versus-synthetic R5 comparison for two N extremes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "artifacts/runs/2324_gan_example_images/results"
STAGE4 = ROOT / "artifacts/archive/legacy/pipeline/stage4_2324"
OUTPUT = ROOT / "artifacts/paper_figures/fig6_bandwise_comparison"
SAMPLES = [(132, 0), (128, 400)]


def ndvi(refl: np.ndarray) -> np.ndarray:
    red, nir = refl[..., 0], refl[..., 2]
    value = (nir - red) / (nir + red + 1e-8)
    value[np.all(refl == 0, axis=-1)] = np.nan
    return value


def false_color(refl: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    channels = [refl[..., 2], refl[..., 0], refl[..., 1]]
    rgb = np.stack([
        np.clip((channel - low) / (high - low + 1e-8), 0, 1)
        for channel, (low, high) in zip(channels, limits)
    ], axis=-1)
    rgb[np.all(refl == 0, axis=-1)] = 0
    return rgb


def draw() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(STAGE4 / "manifest.csv").set_index(["fid", "stage"])
    metrics = pd.read_csv(EXAMPLES / "metrics_pairs.csv").set_index("pair")
    data = []
    records = []
    for fid, dose in SAMPLES:
        source_path = STAGE4 / manifest.loc[(fid, "V13"), "npy"]
        real_path = STAGE4 / manifest.loc[(fid, "R5"), "npy"]
        pair = f"p{fid}_V13toR5_d{dose}_b4"
        fake_path = EXAMPLES / f"{pair}_fake.npy"
        source = np.load(source_path).astype(np.float32)
        real = np.load(real_path).astype(np.float32)
        fake = np.load(fake_path).astype(np.float32)
        data.append((fid, dose, source, real, fake))
        metric = metrics.loc[pair]
        records.append({
            "pair": pair,
            "fid": fid,
            "outer_block": 4,
            "nitrogen_kg_ha": dose,
            "l1": float(metric.l1),
            "abs_mean_ndvi_error": float(metric.ndvi_error),
            "selection_rule": "nitrogen-treatment extremes",
            "source": str(source_path.relative_to(ROOT)),
            "synthetic": str(fake_path.relative_to(ROOT)),
            "real_target": str(real_path.relative_to(ROOT)),
        })
    pd.DataFrame(records).to_csv(OUTPUT / "sample_manifest.csv", index=False)

    arrays = [array for _, _, source, real, fake in data for array in (source, real, fake)]
    display_order = [2, 0, 1]  # NIR, Red, RedEdge -> RGB
    limits = []
    for band in display_order:
        values = np.concatenate([a[..., band].ravel() for a in arrays])
        values = values[values > 0]
        limits.append(tuple(np.percentile(values, [2, 98])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(4, 6, figsize=(11.0, 7.6), constrained_layout=True)
    headers = ["V13 input (FCC)", "Red", "RedEdge", "NIR", "NDVI", "R5 FCC"]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontweight="bold")

    band_image = ndvi_image = None
    for sample_index, (fid, dose, source, real, fake) in enumerate(data):
        for local_row, (kind, target) in enumerate((("Real R5", real), ("Synthetic R5", fake))):
            row = 2 * sample_index + local_row
            if local_row == 0:
                axes[row, 0].axis("off")
            else:
                axes[row, 0].imshow(false_color(source, limits))
                axes[row, 0].annotate(
                    "", xy=(1.13, 0.5), xytext=(0.96, 0.5), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
                )
            for column, band in enumerate([0, 1, 2], start=1):
                band_image = axes[row, column].imshow(target[..., band], cmap="viridis",
                                                      vmin=0, vmax=1)
            ndvi_image = axes[row, 4].imshow(ndvi(target), cmap="viridis", vmin=0, vmax=1)
            axes[row, 5].imshow(false_color(target, limits))
            axes[row, 1].set_ylabel(
                f"{kind}\nN = {dose} kg ha⁻¹ · Plot {fid}",
                rotation=0, ha="right", va="center", labelpad=52,
            )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                if axis.axison:
                    for spine in axis.spines.values():
                        spine.set_linewidth(0.5)
                        spine.set_color("#555555")

    fig.colorbar(band_image, ax=axes[:, 1:4], location="bottom", shrink=0.65,
                 pad=0.02, label="Normalized reflectance")
    fig.colorbar(ndvi_image, ax=axes[:, 4], location="bottom", shrink=0.75,
                 pad=0.02, label="NDVI")
    fig.savefig(OUTPUT / "fig6_bandwise_comparison_reference.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "fig6_bandwise_comparison_reference.svg",
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    draw()
