#!/usr/bin/env python3
"""Recorta V10, V13 e R1 da safra 2025/26 para a GAN fenológica.

Os voos têm schemas distintos (V10: B,R,RE,NIR; V13/R1: B,G,R,RE,NIR).
A saída sempre usa o núcleo comum [Red, RedEdge, NIR].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geo import Ortho, read_parcels, rrenir_indices, to_reflectance
from season2526 import normalize_workbook
from stage4_phenology import false_color, phase_of, resize

BANDS = {"V10": (1, 2, 3), "V13": (2, 3, 4), "R1": (2, 3, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", required=True)
    ap.add_argument("--v13", required=True)
    ap.add_argument("--r1", required=True)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--pad-m", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("npy", "vegetativo", "reprodutivo"):
        (out / d).mkdir(parents=True, exist_ok=True)
    field = normalize_workbook(args.workbook).set_index("parcela")
    parcels = {p.fid: p for p in read_parcels(args.shapefile)}
    missing = sorted(set(field.index) - set(parcels))
    if missing:
        raise ValueError(f"Parcelas da planilha ausentes no shapefile: {missing}")

    rows = []
    for stage, filename in (("V10", args.v10), ("V13", args.v13), ("R1", args.r1)):
        ortho = Ortho(filename)
        indices = BANDS[stage]
        if ortho.bands <= max(indices):
            raise ValueError(f"{stage}: esperado índice de banda {max(indices)}, mas o TIFF tem {ortho.bands}")
        phase = phase_of(stage)
        print(f"[{stage} -> {phase}] {Path(filename).name}: {ortho.width}x{ortho.height}x{ortho.bands}")
        for parcela, meta in field.iterrows():
            crop = ortho.crop_bbox(parcels[parcela].bbox, pad_m=args.pad_m)[..., indices]
            refl = resize(to_reflectance(crop), args.size)
            idx = rrenir_indices(refl)
            tag = f"{stage}__p{parcela:02d}_b{int(meta.bloco)}"
            npy = out / "npy" / f"{tag}.npy"
            png = out / phase / f"{tag}.png"
            np.save(npy, refl.astype(np.float32))
            Image.fromarray(false_color(refl)).save(png)
            rows.append({
                "safra": "2025_2026", "date": {"V10": "2025-12-12", "V13": "2025-12-23", "R1": "2026-01-14"}[stage],
                "stage": stage, "phase": phase, "fid": parcela, "parcela": parcela,
                "dose_n": meta.dose, "tratamento": meta.tratamento, "bloco": int(meta.bloco),
                "chl_total": meta.chl_total_r1 if stage == "R1" else meta.chl_total_v10,
                "biomassa_kg_ha": meta.biomassa_kg_ha_v10 if stage == "V10" else np.nan,
                "ndvi_mean": round(float(np.nanmean(idx["NDVI"])), 4),
                "ndre_mean": round(float(np.nanmean(idx["NDRE"])), 4),
                "cire_mean": round(float(np.nanmean(idx["CIrededge"])), 4),
                "npy": str(npy.relative_to(out)), "png": str(png.relative_to(out)), "tag": tag,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"{len(df)} recortes -> {out / 'manifest.csv'}")
    print(df.groupby(["phase", "stage"]).size().to_string())


if __name__ == "__main__":
    main()
