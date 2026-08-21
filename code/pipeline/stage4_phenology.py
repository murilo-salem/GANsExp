#!/usr/bin/env python3
"""ETAPA 4 — Divisão por estágio fenológico.

Recorta cada parcela (shapefile) de cada ortomosaico RRENIR de uma safra, identifica o
estágio pelo nome do arquivo (RRENIR_<estagio>_YYYY_YYYY.tif) e organiza os recortes em
duas classes fenológicas:

    V6 / V8 / V11 / V13 / V18  -> VEGETATIVO
    R2 / R5                    -> REPRODUTIVO

Saídas (em --out):
    manifest.csv           uma linha por (estagio, parcela): fase, chave, NDVI/CHL médios, caminhos
    npy/<tag>.npy          reflectância float32 (H, W, 3) [Red, RedEdge, NIR], redimensionada
    vegetativo/<tag>.png   composição falsa-cor p/ inspeção visual
    reprodutivo/<tag>.png

`<tag>` = <estagio>__p<fid>_d<dose>_b<bloco>, estável entre estágios p/ parear na etapa 5.

Uso:
    python3 code/pipeline/stage4_phenology.py \
        --ortho-dir "data/raw/safra_2023_2024/orthomosaics" \
        --shapefile "data/raw/safra_2023_2024/geometry/Shape_parcelas23_24.shp" \
        --out artifacts/runs/2324_stage4_phenology/results --size 256
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from geo import Ortho, read_parcels, to_reflectance, rrenir_indices

# V-stages = vegetativo; R-stages = reprodutivo (fenologia do milho)
STAGE_RE = re.compile(r"RRENIR_(V\d+|R\d+|zero)_", re.IGNORECASE)


def stage_of(path: Path) -> str | None:
    m = STAGE_RE.search(path.name)
    return m.group(1).upper() if m else None


def phase_of(stage: str) -> str | None:
    s = stage.upper()
    if s.startswith("V"):
        return "vegetativo"
    if s.startswith("R"):
        return "reprodutivo"
    return None   # 'zero' (solo/emergência) fica de fora


def false_color(refl: np.ndarray) -> np.ndarray:
    """NIR-Red-RedEdge -> RGB uint8 (falsa-cor infravermelho) com stretch por percentil."""
    chans = [refl[..., 2], refl[..., 0], refl[..., 1]]
    out = []
    for c in chans:
        lo, hi = np.nanpercentile(c, [2, 98])
        out.append(np.clip((c - lo) / (hi - lo + 1e-8), 0, 1))
    return (np.stack(out, -1) * 255).astype(np.uint8)


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona (H,W,3) float p/ (size,size,3) via PIL (bilinear), preservando escala."""
    out = np.zeros((size, size, arr.shape[2]), np.float32)
    for b in range(arr.shape[2]):
        im = Image.fromarray(arr[..., b].astype(np.float32), mode="F").resize(
            (size, size), Image.BILINEAR)
        out[..., b] = np.asarray(im)
    return out


def parcel_mask(points: list, bbox: tuple[float, float, float, float], shape: tuple[int, int],
                size: int) -> np.ndarray:
    """Rasteriza a parcela no recorte; pixels externos não entram na GAN/GLCM."""
    xmin, ymin, xmax, ymax = bbox
    h, w = shape
    xy = [((x - xmin) / (xmax - xmin + 1e-12) * (w - 1),
           (ymax - y) / (ymax - ymin + 1e-12) * (h - 1)) for x, y in points]
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(xy, fill=255)
    return (np.asarray(im.resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho-dir", required=True)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--pad-m", type=float, default=0.0, help="margem em metros no recorte")
    ap.add_argument("--reflectance-scale", choices=("crop", "mosaic"), default="crop",
                    help="crop preserva legado; mosaic usa um único p99.5 por voo")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    (out / "vegetativo").mkdir(parents=True, exist_ok=True)
    (out / "reprodutivo").mkdir(parents=True, exist_ok=True)

    parcels = read_parcels(args.shapefile)
    tifs = sorted(Path(args.ortho_dir).glob("RRENIR_*.tif"))
    rows = []
    for tif in tifs:
        stage = stage_of(tif)
        phase = phase_of(stage) if stage else None
        if phase is None:
            print(f"[skip] {tif.name} (estágio '{stage}' fora das classes)")
            continue
        ortho = Ortho(tif)
        mosaic_scale = float(np.percentile(ortho.read(), 99.5)) if args.reflectance_scale == "mosaic" else None
        print(f"[{stage:4s} -> {phase:11s}] {tif.name}  ({ortho.width}x{ortho.height})")
        for p in parcels:
            crop = ortho.crop_bbox(p.bbox, pad_m=args.pad_m)
            if mosaic_scale:
                refl = np.clip(crop.astype(np.float32) / max(mosaic_scale, 1.0), 0, 1)
            else:
                refl = to_reflectance(crop)
            # A bbox com margem é a referência do raster de máscara.
            xmin, ymin, xmax, ymax = p.bbox
            padded = (xmin-args.pad_m, ymin-args.pad_m, xmax+args.pad_m, ymax+args.pad_m)
            mask = parcel_mask(p.points, padded, crop.shape[:2], args.size)
            refl = resize(refl, args.size) * mask[..., None]
            idx = rrenir_indices(refl)
            tag = f"{stage}__p{p.fid:02d}_d{p.dose_n}_b{p.bloco}"
            np.save(out / "npy" / f"{tag}.npy", refl.astype(np.float32))
            np.save(out / "mask" / f"{tag}.npy", mask)
            Image.fromarray(false_color(refl)).save(out / phase / f"{tag}.png")
            rows.append(dict(
                safra=tif.name.split("_")[-2] + "_" + tif.name.split("_")[-1].split(".")[0],
                stage=stage, phase=phase, fid=p.fid, dose_n=p.dose_n, bloco=p.bloco,
                ndvi_mean=round(float(np.nanmean(idx["NDVI"])), 4),
                ndre_mean=round(float(np.nanmean(idx["NDRE"])), 4),
                cire_mean=round(float(np.nanmean(idx["CIrededge"])), 4),
                npy=str((out / "npy" / f"{tag}.npy").relative_to(out)),
                mask=str((out / "mask" / f"{tag}.npy").relative_to(out)),
                reflectance_scale=args.reflectance_scale,
                mosaic_p995=mosaic_scale,
                png=str((out / phase / f"{tag}.png").relative_to(out)),
                tag=tag,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"\n== manifest: {len(df)} recortes -> {out/'manifest.csv'}")
    print(df.groupby(["phase", "stage"]).size().to_string())


if __name__ == "__main__":
    main()
