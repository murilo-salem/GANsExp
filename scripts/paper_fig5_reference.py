"""Build a qualitative V13-to-R5 comparison from real held-out data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "artifacts/runs/2324_gan_example_images/results"
STAGE4 = ROOT / "artifacts/archive/legacy/pipeline/stage4_2324"
OUTPUT = ROOT / "artifacts/paper_figures/fig5_v13_r5_examples"


def false_color(refl: np.ndarray) -> np.ndarray:
    """Map [Red, RedEdge, NIR] to false-color [NIR, Red, RedEdge]."""
    channels = [refl[..., 2], refl[..., 0], refl[..., 1]]
    stretched = []
    for channel in channels:
        foreground = channel[channel > 0]
        values = foreground if foreground.size else channel.ravel()
        low, high = np.nanpercentile(values, [2, 98])
        stretched.append(np.clip((channel - low) / (high - low + 1e-8), 0, 1))
    rgb = np.stack(stretched, axis=-1)
    rgb[np.all(refl == 0, axis=-1)] = 0
    return rgb


def ndvi(refl: np.ndarray) -> np.ndarray:
    red, nir = refl[..., 0], refl[..., 2]
    value = (nir - red) / (nir + red + 1e-8)
    value[np.all(refl == 0, axis=-1)] = np.nan
    return value


def draw() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(EXAMPLES / "metrics_pairs.csv")
    rows = metrics[(metrics["veg"] == "V13") & (metrics["rep"] == "R5")].copy()
    rows["dose_n"] = rows["pair"].str.extract(r"_d(\d+)_").astype(int)
    rows = rows.sort_values("dose_n").reset_index(drop=True)

    manifest = pd.read_csv(STAGE4 / "manifest.csv")
    lookup = manifest.set_index(["fid", "stage"])["npy"].to_dict()
    records = []
    figure_rows = []
    for row in rows.itertuples(index=False):
        source_path = STAGE4 / lookup[(int(row.fid), "V13")]
        real_path = STAGE4 / lookup[(int(row.fid), "R5")]
        fake_path = EXAMPLES / f"{row.pair}_fake.npy"
        source = np.load(source_path).astype(np.float32)
        fake = np.load(fake_path).astype(np.float32)
        real = np.load(real_path).astype(np.float32)
        figure_rows.append((source, fake, real))
        records.append({
            "pair": row.pair,
            "fid": int(row.fid),
            "outer_block": int(row.bloco),
            "nitrogen_kg_ha": int(row.dose_n),
            "l1": float(row.l1),
            "abs_mean_ndvi_error": float(row.ndvi_error),
            "source": str(source_path.relative_to(ROOT)),
            "synthetic": str(fake_path.relative_to(ROOT)),
            "real_target": str(real_path.relative_to(ROOT)),
        })
    pd.DataFrame(records).to_csv(OUTPUT / "sample_manifest.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(len(figure_rows), 6, figsize=(10.8, 10.2),
                             constrained_layout=True)
    titles = [
        "V13 input",
        "Synthetic R5",
        "Real R5",
        "Synthetic NDVI",
        "Real NDVI",
        "|NDVI error|",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="bold")

    ndvi_image = None
    error_image = None
    for index, ((source, fake, real), record) in enumerate(zip(figure_rows, records)):
        source_ndvi, fake_ndvi, real_ndvi = ndvi(source), ndvi(fake), ndvi(real)
        panels = [false_color(source), false_color(fake), false_color(real)]
        for column, panel in enumerate(panels):
            axes[index, column].imshow(panel)
        ndvi_image = axes[index, 3].imshow(fake_ndvi, cmap="viridis", vmin=0, vmax=1)
        axes[index, 4].imshow(real_ndvi, cmap="viridis", vmin=0, vmax=1)
        error_image = axes[index, 5].imshow(
            np.abs(fake_ndvi - real_ndvi), cmap="magma", vmin=0, vmax=0.35
        )
        axes[index, 0].set_ylabel(
            f"N = {record['nitrogen_kg_ha']} kg ha⁻¹\nPlot {record['fid']}",
            rotation=0, ha="right", va="center", labelpad=34,
        )
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#555555")

    fig.colorbar(ndvi_image, ax=axes[:, 3:5], location="bottom", shrink=0.72,
                 pad=0.015, label="NDVI")
    fig.colorbar(error_image, ax=axes[:, 5], location="bottom", shrink=0.72,
                 pad=0.015, label="Absolute NDVI error")
    fig.savefig(OUTPUT / "fig5_v13_r5_examples_reference.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / "fig5_v13_r5_examples_reference.svg",
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    draw()
