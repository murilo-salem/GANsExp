#!/usr/bin/env python3
"""PASSO 1 — Cubo 5-bandas por pixel (B1-B5) fundindo RGB + RRENIR.

É viável nas safras que têm RGB **e** RRENIR por estágio. Para cada estágio/parcela, recorta
os dois orthos (mesmo CRS, EPSG:31982), reamostra para um tamanho comum e empilha:

    B1 Azul    = RGB[...,2]      B2 Verde = RGB[...,1]
    B3 Vermelho= RRENIR[...,0]   B4 RedEdge = RRENIR[...,1]   B5 NIR = RRENIR[...,2]

(usa o vermelho do RRENIR — multiespectral calibrado; azul/verde do RGB são DN, ressalva
radiométrica documentada). Gera, por parcela/estágio:
    npy5/<tag>.npy                    stack 5-bandas (H,W,5) em [0,1]
    features_5band_<safra>.csv        índices (inclui GNDVI/EVI/VARI/TGI) + estatísticas
                                       + GLCM(NIR) + GLCM de luminância RGB
    manifest.csv

Uso:
    python3 code/pipeline/build_5band.py \
        --ortho-dir "data/raw/safra_2022_2023/orthomosaics" \
        --shapefile "data/raw/safra_2022_2023/geometry/Shape_parcelas_2223.shp" \
        --out artifacts/runs/2223_band5/results --safra 2223 --size 256
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .hyperspectral_features import glcm_features, glcm_window_features
from .geo import Ortho, read_parcels, to_reflectance, bands5_indices

STAGE_RE = re.compile(r"(V\d+|R\d+)", re.IGNORECASE)


def resize(arr: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size, arr.shape[2]), np.float32)
    for b in range(arr.shape[2]):
        im = Image.fromarray(arr[..., b].astype(np.float32), mode="F").resize(
            (size, size), Image.BILINEAR)
        out[..., b] = np.asarray(im)
    return out


def find_pairs(ortho_dir: Path) -> dict:
    """Casa RGB com RRENIR por estágio e retorna a ordem RGB dos canais.

    Os mosaicos históricos usam ``RGB_*`` e armazenam [R, G, B]. Os novos
    mosaicos 23/24 usam os nomes ``Ortho_BGR*``/``Orho_B_G_R*`` e armazenam
    [B, G, R]. A ordem é explicitada no retorno, nunca inferida pelo leitor
    GeoTIFF (os arquivos não trazem ColorInterpretation por banda).
    """
    def stage(p):
        m = STAGE_RE.search(p.stem)
        return m.group(1).upper() if m else None
    rgb = {}
    for p in ortho_dir.glob("*.tif"):
        name = p.name.lower()
        s = stage(p)
        if not s:
            continue
        if name.startswith("rgb"):
            rgb[s] = (p, (0, 1, 2))
        elif name.startswith(("ortho_bgr", "orho_b_g_r")):
            rgb[s] = (p, (2, 1, 0))
    rre = {stage(p): p for p in ortho_dir.glob("RRENIR_*.tif") if stage(p)}
    return {s: (*rgb[s], rre[s]) for s in sorted(set(rgb) & set(rre))}


def stack5(rgb_crop: np.ndarray, rgb_bands: tuple[int, int, int],
           rre_crop: np.ndarray, size: int) -> np.ndarray:
    """Empilha Azul, Verde, Red, RedEdge e NIR após normalização por recorte."""
    rgb = resize(to_reflectance(rgb_crop[..., rgb_bands]), size)  # [R,G,B]
    rre = resize(to_reflectance(rre_crop), size)     # [Red,RedEdge,NIR]
    return np.stack([rgb[..., 2], rgb[..., 1], rre[..., 0], rre[..., 1], rre[..., 2]], axis=-1)


def parcel_features_5b(refl5: np.ndarray, ndvi_thr: float = 0.3) -> dict:
    idx = bands5_indices(refl5)
    mask = idx["NDVI"] > ndvi_thr
    if mask.sum() < 200:
        mask = np.ones(refl5.shape[:2], bool)
    feat = {"n_veg_pixels": int(mask.sum())}
    for k, v in idx.items():
        feat[k] = float(np.nanmean(v[mask]))
    vals = refl5[mask]
    feat.update({"spec_mean": float(np.nanmean(vals)), "spec_std": float(np.nanstd(vals)),
                 "spec_p10": float(np.nanpercentile(vals, 10)),
                 "spec_p50": float(np.nanpercentile(vals, 50)),
                 "spec_p90": float(np.nanpercentile(vals, 90))})
    nir = refl5[..., 4]
    feat.update(glcm_features(nir, mask))
    feat.update(glcm_window_features(nir, mask))
    # Texturas RGB adicionais: luminância da câmera, preservando as texturas
    # RRENIR/NIR acima para manter comparabilidade com a série anterior.
    red, green, blue = refl5[..., 2], refl5[..., 1], refl5[..., 0]
    rgb_luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    feat.update({f"rgb_{name}": value
                 for name, value in glcm_features(rgb_luminance, mask).items()})
    feat.update({f"rgb_{name}": value
                 for name, value in glcm_window_features(rgb_luminance, mask).items()})
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho-dir", required=True)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--safra", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out); (out / "npy5").mkdir(parents=True, exist_ok=True)
    parcels = read_parcels(args.shapefile)
    pairs = find_pairs(Path(args.ortho_dir))
    print(f"pares RGB↔RRENIR por estágio: {list(pairs)}")

    feat_rows, man_rows = [], []
    for stage, (rgb_p, rgb_bands, rre_p) in pairs.items():
        rgb, rre = Ortho(rgb_p), Ortho(rre_p)
        print(f"[{stage:4s}] {rgb_p.name} (RGB={rgb_bands}) + {rre_p.name}")
        for p in parcels:
            refl5 = stack5(rgb.crop_bbox(p.bbox), rgb_bands, rre.crop_bbox(p.bbox), args.size)
            tag = f"{stage}_p{p.fid:02d}_d{p.dose_n}_b{p.bloco}"
            np.save(out / "npy5" / f"{tag}.npy", refl5.astype(np.float32))
            f = parcel_features_5b(refl5)
            feat_rows.append(dict(sample=f"{stage}_d{p.dose_n}_b{p.bloco}",
                                  parcela=p.fid, medida=stage, **f))
            man_rows.append(dict(safra=args.safra, stage=stage, fid=p.fid,
                                 dose_n=p.dose_n, bloco=p.bloco, tag=tag))
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(out / f"features_5band_{args.safra}.csv", index=False)
    pd.DataFrame(man_rows).to_csv(out / "manifest.csv", index=False)
    print(f"\n== {len(feats)} amostras × {feats.shape[1]} cols -> features_5band_{args.safra}.csv")
    print("   índices médios (checagem):")
    for c in ["NDVI", "GNDVI", "EVI", "VARI", "TGI"]:
        print(f"     {c:6s}: {feats[c].mean():.3f}  (NaN={feats[c].isna().sum()})")


if __name__ == "__main__":
    main()
