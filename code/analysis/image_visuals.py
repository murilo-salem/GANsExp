#!/usr/bin/env python3
"""Visualização dos cubos hiperespectrais (voo 12.12) e das texturas GLCM.

Por cubo (<parcela><M|S>): RGB verdadeira, falsa-cor infravermelho, mapa de NDVI e
painel de textura GLCM (ASM/Contraste/Entropia/Correlação/Homogeneidade) em janelas 3/5/7.
Também monta galerias (todas as amostras) e comparações M vs S por parcela.

Reusa hyperspectral_features (read_hdr/read_bil/band_at/glcm_window_maps).

Uso:
  python3 image_visuals.py --input ../../data/2025-2026/hyperspectral_12dez --out out/visuals
  python3 image_visuals.py ... --samples 6        # renderiza individualmente só 6 cubos
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hyperspectral_features as H

RGB_WAVES = (660.0, 550.0, 470.0)      # R, G, B
FC_WAVES = (800.0, 660.0, 550.0)       # NIR, R, G (falsa-cor)


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def composite(cube, wl, waves):
    return np.stack([stretch(cube[H.band_at(wl, w)]) for w in waves], axis=-1)


def load_cube(bil):
    hdr = H.read_hdr(Path(str(bil) + ".hdr"))
    cube = H.read_bil(bil, hdr)
    return cube, hdr["wavelength"]


def ndvi_of(cube, wl, thr=0.3):
    red, nir = cube[H.band_at(wl, 670)], cube[H.band_at(wl, 800)]
    nd = H.safe_ratio(nir, red)
    return nd, nd > thr


def save_cube_pngs(stem, cube, wl, outdir):
    plt.imsave(outdir / f"{stem}_rgb.png", composite(cube, wl, RGB_WAVES))
    plt.imsave(outdir / f"{stem}_falsecolor.png", composite(cube, wl, FC_WAVES))
    nd, mask = ndvi_of(cube, wl)
    disp = np.where(mask, nd, np.nan)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(disp, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    ax.set_title(f"{stem} — NDVI (veg {mask.mean()*100:.0f}%)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(outdir / f"{stem}_ndvi.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def save_texture_panel(stem, cube, wl, outdir, windows=(3, 5, 7), levels=8):
    _, mask = ndvi_of(cube, wl)
    nir = cube[H.band_at(wl, 800)]
    fig, axes = plt.subplots(len(windows), len(H.GLCM_NAMES),
                             figsize=(4 * len(H.GLCM_NAMES), 3.2 * len(windows)))
    for r, w in enumerate(windows):
        maps, good = H.glcm_window_maps(nir, mask, w, levels=levels)
        for c, name in enumerate(H.GLCM_NAMES):
            ax = axes[r, c]
            im = ax.imshow(np.where(good, maps[name], np.nan), cmap="viridis")
            ax.set_title(f"{name} · w{w}", fontsize=9); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.035)
    fig.suptitle(f"{stem} — textura GLCM (banda NIR)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"{stem}_glcm.png", dpi=100, bbox_inches="tight"); plt.close(fig)


def gallery(thumbs, labels, path, title, cmap=None):
    n = len(thumbs)
    ncol = 8
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 1.9))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(thumbs[i], cmap=cmap, vmin=-0.1 if cmap else None, vmax=0.9 if cmap else None)
            ax.set_title(labels[i], fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out/visuals")
    ap.add_argument("--glob", default="*.bil")
    ap.add_argument("--samples", type=int, default=8,
                    help="nº de cubos renderizados individualmente (cubes/+texture/); 0 = todos")
    ap.add_argument("--step", type=int, default=4, help="downsample dos thumbnails da galeria")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("cubes", "texture", "pair_MvsS"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cubes = sorted((p for p in Path(args.input).glob(args.glob)
                    if p.stat().st_size > 0 and (p.parent / (p.name + ".hdr")).exists()),
                   key=lambda p: (int(re.match(r"(\d+)", p.stem).group(1)), p.stem))
    n_indiv = len(cubes) if args.samples == 0 else min(args.samples, len(cubes))
    print(f"{len(cubes)} cubos | individuais: {n_indiv} | galerias: todos")

    rgb_thumbs, ndvi_thumbs, labels = [], [], []
    rgb_by_parcela = {}
    for i, bil in enumerate(cubes):
        cube, wl = load_cube(bil)
        stem = bil.stem
        rgb = composite(cube, wl, RGB_WAVES)
        nd, mask = ndvi_of(cube, wl)
        rgb_thumbs.append(rgb[::args.step, ::args.step])
        ndvi_thumbs.append(np.where(mask, nd, np.nan)[::args.step, ::args.step])
        labels.append(stem)
        m = re.match(r"(\d+)([MS])", stem)
        if m:
            rgb_by_parcela.setdefault(int(m.group(1)), {})[m.group(2)] = rgb[::args.step, ::args.step]
        if i < n_indiv:
            save_cube_pngs(stem, cube, wl, out / "cubes")
            save_texture_panel(stem, cube, wl, out / "texture")
        print(f"  [{i+1:2d}/{len(cubes)}] {stem}")

    gallery(rgb_thumbs, labels, out / "gallery_rgb.png", "RGB verdadeira — todas as amostras")
    gallery(ndvi_thumbs, labels, out / "gallery_ndvi.png",
            "NDVI — todas as amostras", cmap="RdYlGn")

    # M vs S por parcela (quando ambos existem)
    npairs = 0
    for parc, dd in sorted(rgb_by_parcela.items()):
        if "M" in dd and "S" in dd:
            fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
            for ax, k in zip(axes, ("M", "S")):
                ax.imshow(dd[k]); ax.set_title(f"parcela {parc} — {k}", fontsize=9); ax.axis("off")
            fig.tight_layout()
            fig.savefig(out / "pair_MvsS" / f"parcela_{parc:02d}.png", dpi=110,
                        bbox_inches="tight"); plt.close(fig)
            npairs += 1

    print(f"\nOK -> {out}/  (cubes: {n_indiv}, texture: {n_indiv}, pares M/S: {npairs}, "
          f"galerias: gallery_rgb.png + gallery_ndvi.png)")


if __name__ == "__main__":
    main()
