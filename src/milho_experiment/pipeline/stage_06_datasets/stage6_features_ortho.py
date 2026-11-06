#!/usr/bin/env python3
"""PLUG DO Y (etapas 6-7) — Features multiespectrais por parcela + alvo agronômico real.

Para a safra 23/24 (auto-consistente: orto RRENIR + tabela de campo + shapefile no mesmo CRS),
extrai por parcela/estágio um vetor de atributos no MESMO schema de code/analysis/features_*.csv
(índices espectrais, estatísticas, GLCM global + janelas 3/5/7) e junta o Y real
(Biomassa / Produtividade / CHL_total) da tabela normalizada. Gera dois CSVs prontos p/ o
plsr_kfold.py existente (join por 'sample'), substituindo o alvo sintético por dados reais.

Saídas:
    <out>/features_ortho_2324.csv   sample,parcela,medida,n_veg_pixels + índices/stats/GLCM
                                   + atributos físicos de campo (phys_CHL_total, phys_N_percent,
                                     phys_N_acumulado) — clorofila/N como X (diagrama etapa 6)
    <out>/targets_ortho_2324.csv    sample,Biomassa,Produtividade,CHL_total

Uso:
    python3 code/pipeline/stage6_features_ortho.py \
        --ortho-dir "data/raw/safra_2023_2024/orthomosaics" \
        --shapefile "data/raw/safra_2023_2024/geometry/Shape_parcelas23_24.shp" \
        --table "data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx" \
        --out artifacts/runs/2324_features_ortho/results
Depois:
    cd code/analysis && python3 plsr_kfold.py \
        --features ../pipeline/out/plsr_2324/features_ortho_2324.csv \
        --target-csv ../pipeline/out/plsr_2324/targets_ortho_2324.csv \
        --join sample --target-col Biomassa --block texture+physical --group parcela
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from milho_experiment.pipeline.stage_03_attributes.hyperspectral_features import glcm_features, glcm_window_features
from milho_experiment.pipeline.stage_03_attributes.geo import Ortho, read_parcels, to_reflectance, rrenir_indices

STAGE_RE = re.compile(r"RRENIR_(V\d+|R\d+)_", re.IGNORECASE)
TARGETS = ["Biomassa", "Produtividade", "CHL_total"]
# Atributos físicos de campo usados como X (diagrama etapa 6: "clorofila (físico)").
# Mapeia coluna da tabela -> nome sanitizado no CSV de features (prefixo phys_).
PHYS_COLS = {"CHL_total": "phys_CHL_total", "%N": "phys_N_percent", "N_ACUMULADO": "phys_N_acumulado"}


def parcel_features(refl: np.ndarray, ndvi_thr: float = 0.3) -> dict:
    """Índices espectrais + estatísticas + GLCM (banda NIR) para um recorte RRENIR."""
    idx = rrenir_indices(refl)
    mask = idx["NDVI"] > ndvi_thr
    if mask.sum() < 200:                       # parcela sem vegetação suficiente
        mask = np.ones(refl.shape[:2], bool)
    feat = {
        "n_veg_pixels": int(mask.sum()),
        # índices espectrais (média sobre a máscara de vegetação)
        "NDVI": float(np.nanmean(idx["NDVI"][mask])),
        "NDRE": float(np.nanmean(idx["NDRE"][mask])),
        "CIrededge": float(np.nanmean(idx["CIrededge"][mask])),
        "SAVI": float(np.nanmean(idx["SAVI"][mask])),
    }
    # estatísticas espectrais (sobre reflectância das 3 bandas, pixels de vegetação)
    vals = refl[mask]
    feat.update({
        "spec_mean": float(np.nanmean(vals)),
        "spec_std": float(np.nanstd(vals)),
        "spec_p10": float(np.nanpercentile(vals, 10)),
        "spec_p50": float(np.nanpercentile(vals, 50)),
        "spec_p90": float(np.nanpercentile(vals, 90)),
    })
    # textura GLCM sobre a banda NIR
    nir = refl[..., 2]
    feat.update(glcm_features(nir, mask))
    feat.update(glcm_window_features(nir, mask))
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho-dir", required=True)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    parcels = read_parcels(args.shapefile)
    tbl = pd.read_excel(args.table)
    tbl["Estagio"] = tbl["Estagio"].astype(str).str.upper()

    feat_rows, tgt_rows = [], []
    for tif in sorted(Path(args.ortho_dir).glob("RRENIR_*.tif")):
        m = STAGE_RE.search(tif.name)
        if not m:
            continue
        stage = m.group(1).upper()
        ortho = Ortho(tif)
        print(f"[{stage:4s}] {tif.name}")
        for p in parcels:
            refl = to_reflectance(ortho.crop_bbox(p.bbox))
            f = parcel_features(refl)
            sample = f"{stage}_d{p.dose_n}_b{p.bloco}"
            # Y da tabela por (Estagio, Dose_N, Bloco)
            q = tbl[(tbl.Estagio == stage) & (tbl.Dose_N == int(p.dose_n)) &
                    (tbl.Bloco == int(p.bloco))]
            # atributos físicos de campo como features (X) — clorofila/N (diagrama etapa 6)
            for src, dst in PHYS_COLS.items():
                f[dst] = float(q[src].iloc[0]) if len(q) and src in q else np.nan
            feat_rows.append(dict(sample=sample, parcela=p.fid, medida=stage, **f))
            row = dict(sample=sample)
            for t in TARGETS:
                row[t] = float(q[t].iloc[0]) if len(q) and t in q else np.nan
            tgt_rows.append(row)

    feats = pd.DataFrame(feat_rows)
    tgts = pd.DataFrame(tgt_rows)
    feats.to_csv(out / "features_ortho_2324.csv", index=False)
    tgts.to_csv(out / "targets_ortho_2324.csv", index=False)
    print(f"\n== {len(feats)} amostras (parcela×estágio) × {feats.shape[1]} cols")
    print(f"   features -> {out/'features_ortho_2324.csv'}")
    print(f"   targets  -> {out/'targets_ortho_2324.csv'}")
    print("   cobertura do Y:")
    for t in TARGETS:
        print(f"     {t:14s}: {tgts[t].notna().sum()}/{len(tgts)} não-nulos")
    print("   atributos físicos (X):")
    for src, dst in PHYS_COLS.items():
        print(f"     {dst:18s}: {feats[dst].notna().sum()}/{len(feats)} não-nulos")


if __name__ == "__main__":
    main()
