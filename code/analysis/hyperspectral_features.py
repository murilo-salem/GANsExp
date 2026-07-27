#!/usr/bin/env python3
"""Extração de atributos de cubos hiperespectrais Resonon Pika L (ENVI BIL, 300 bandas).

Para cada cubo (.bil) gera um vetor de atributos por amostra:
  - máscara de vegetação por NDVI
  - índices de vegetação (NDVI, GNDVI, NDRE, SAVI, EVI2, CIrededge)
  - espectro médio de reflectância por banda (b000..bNNN) sobre a máscara
  - estatísticas espectrais globais (média/desvio/percentis do espectro)
  - textura GLCM (ASM, contraste, entropia, correlação, homogeneidade) numa banda-chave

Não depende de rasterio/spectral/skimage — lê o header ENVI e o BIL direto com numpy.
Os rótulos M/S no nome (ex.: 21M, 21S) são duas medidas por parcela.

Uso:
  python3 hyperspectral_features.py \
      --input "../../data/2025-2026/hyperspectral_12dez" \
      --out   out/features_12dez.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------- ENVI I/O

_ENVI_DTYPE = {  # ENVI 'data type' -> numpy
    1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
    5: np.float64, 12: np.uint16, 13: np.uint32,
}


def read_hdr(hdr_path: Path) -> dict:
    """Parseia um header ENVI (.hdr) num dict; expande 'wavelength'."""
    txt = hdr_path.read_text(errors="ignore")
    # junta blocos { ... } numa linha só
    txt = re.sub(r"\{[^}]*\}", lambda m: m.group(0).replace("\n", " "), txt)
    hdr = {}
    for line in txt.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        hdr[k.strip().lower()] = v.strip()
    for k in ("lines", "samples", "bands", "data type", "byte order", "header offset"):
        if k in hdr:
            hdr[k] = int(hdr[k])
    if "wavelength" in hdr:
        nums = re.findall(r"[-+]?\d*\.?\d+", hdr["wavelength"])
        hdr["wavelength"] = np.array([float(x) for x in nums], dtype=float)
    return hdr


def read_bil(bil_path: Path, hdr: dict) -> np.ndarray:
    """Lê um cubo BIL -> array float32 (bands, lines, samples), escalado p/ reflectância."""
    L, S, B = hdr["lines"], hdr["samples"], hdr["bands"]
    dt = np.dtype(_ENVI_DTYPE[hdr["data type"]])
    dt = dt.newbyteorder("<" if hdr.get("byte order", 0) == 0 else ">")
    off = hdr.get("header offset", 0)
    raw = np.fromfile(bil_path, dtype=dt, offset=off, count=L * S * B)
    # BIL: por linha, todas as bandas -> (lines, bands, samples)
    cube = raw.reshape(L, B, S).transpose(1, 0, 2).astype(np.float32)
    scale = float(hdr.get("reflectance scale factor", 1) or 1)
    if scale and scale != 1:
        cube /= scale
    return cube  # (B, L, S)


# ---------------------------------------------------------------- helpers

def band_at(wl: np.ndarray, target: float) -> int:
    """Índice da banda cujo comprimento de onda é mais próximo de `target` (nm)."""
    return int(np.argmin(np.abs(wl - target)))


def safe_ratio(a, b):
    return (a - b) / (a + b + 1e-6)


# Conjunto completo de descritores de Haralick (ordem = pedido do usuário).
# 'homogeneity' fica como alias de IDM p/ compatibilidade com a versão em janela.
HARALICK_NAMES = [
    "asm", "contrast", "correlation", "variance", "idm", "sum_average",
    "sum_variance", "sum_entropy", "entropy", "diff_entropy", "imc1", "imc2",
    "mcc", "dissimilarity", "inertia", "cluster_shade", "cluster_prominence",
    "homogeneity",
]


def haralick_full(p: np.ndarray) -> dict:
    """Descritores de Haralick a partir de uma GLCM normalizada e simétrica p (Ng×Ng, Σp=1).

    Retorna dict com chaves 'glcm_<nome>' (ver HARALICK_NAMES): Angular Second Moment, Contrast,
    Correlation, Variance, Inverse Difference Moment, Sum Average, Sum Variance, Sum Entropy,
    Entropy, Difference Entropy, Information Measure of Correlation 1/2, Maximal Correlation
    Coefficient, Dissimilarity, Inertia, Cluster Shade, Cluster Prominence.
    """
    eps = 1e-12
    Ng = p.shape[0]
    p = p / (p.sum() + eps)
    i, j = np.indices(p.shape)
    idx = np.arange(Ng, dtype=np.float64)

    px = p.sum(axis=1)                    # marginal em i
    py = p.sum(axis=0)                    # marginal em j
    mux = float((idx * px).sum())
    muy = float((idx * py).sum())
    sigx = float(np.sqrt(((idx - mux) ** 2 * px).sum()))
    sigy = float(np.sqrt(((idx - muy) ** 2 * py).sum()))

    # distribuições soma (k=0..2Ng-2) e diferença (k=0..Ng-1)
    pxy_s = np.bincount((i + j).ravel(), weights=p.ravel(), minlength=2 * Ng - 1)
    pxy_d = np.bincount(np.abs(i - j).ravel(), weights=p.ravel(), minlength=Ng)
    ks = np.arange(len(pxy_s), dtype=np.float64)
    kd = np.arange(len(pxy_d), dtype=np.float64)

    def _H(a):
        a = a[a > 0]
        return float(-(a * np.log(a)).sum())

    HXY = _H(p)
    HX, HY = _H(px), _H(py)
    pxpy = np.outer(px, py)
    m1 = p > 0
    HXY1 = float(-(p[m1] * np.log(pxpy[m1] + eps)).sum())
    m2 = pxpy > 0
    HXY2 = float(-(pxpy[m2] * np.log(pxpy[m2] + eps)).sum())

    contrast = float(((i - j) ** 2 * p).sum())
    sum_entropy = _H(pxy_s)

    # Maximal Correlation Coefficient: 2º maior autovalor de Q, raiz quadrada
    try:
        Q = np.zeros((Ng, Ng), dtype=np.float64)
        for k in range(Ng):
            if py[k] > 0:
                col = p[:, k]
                Q += np.outer(col, col) / py[k]
        Q /= (px[:, None] + eps)
        ev = np.sort(np.real(np.linalg.eigvals(Q)))
        mcc = float(np.sqrt(max(0.0, ev[-2]))) if Ng >= 2 else np.nan
    except np.linalg.LinAlgError:
        mcc = np.nan

    feats = {
        "asm": float((p ** 2).sum()),
        "contrast": contrast,
        "correlation": float(((i * j * p).sum() - mux * muy) / (sigx * sigy + eps)),
        "variance": float(((i - mux) ** 2 * p).sum()),
        "idm": float((p / (1.0 + (i - j) ** 2)).sum()),
        "sum_average": float((ks * pxy_s).sum()),
        "sum_variance": float(((ks - sum_entropy) ** 2 * pxy_s).sum()),
        "sum_entropy": sum_entropy,
        "entropy": HXY,
        "diff_entropy": _H(pxy_d),
        "imc1": float((HXY - HXY1) / (max(HX, HY) + eps)),
        "imc2": float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (HXY2 - HXY))))),
        "mcc": mcc,
        "dissimilarity": float((np.abs(i - j) * p).sum()),
        "inertia": contrast,                       # inércia = contraste (mesma forma)
        "cluster_shade": float(((i + j - mux - muy) ** 3 * p).sum()),
        "cluster_prominence": float(((i + j - mux - muy) ** 4 * p).sum()),
        "homogeneity": float((p / (1.0 + np.abs(i - j))).sum()),
    }
    return {f"glcm_{k}": feats[k] for k in HARALICK_NAMES}


def glcm_features(img2d: np.ndarray, mask: np.ndarray, levels: int = 16):
    """GLCM simétrica (offsets (0,1) e (1,0)) sobre pixels mascarados -> Haralick completo."""
    nan = {f"glcm_{k}": np.nan for k in HARALICK_NAMES}
    v = img2d[mask]
    if v.size < 50:
        return nan
    lo, hi = np.percentile(v, [2, 98])
    q = np.clip(((img2d - lo) / (hi - lo + 1e-9) * (levels - 1)), 0, levels - 1).astype(np.int16)
    q[~mask] = -1  # ignora fundo
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in ((0, 1), (1, 0)):
        a = q[: q.shape[0] - dy, : q.shape[1] - dx]
        b = q[dy:, dx:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(glcm, (a[ok], b[ok]), 1)
    if glcm.sum() == 0:
        return nan
    glcm = glcm + glcm.T                # simetriza (padrão Haralick)
    return haralick_full(glcm / glcm.sum())


def _boxsum(a, w):
    """Soma numa janela w×w (uniform_filter * área)."""
    return uniform_filter(a.astype(np.float64), size=w, mode="constant", cval=0.0) * (w * w)


GLCM_NAMES = ("asm", "contrast", "entropy", "correlation", "homogeneity")


def _quantize(img, mask, levels):
    """Quantiza a imagem em `levels` níveis (contraste 2–98% sobre a máscara)."""
    v = img[mask]
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-9) * (levels - 1), 0, levels - 1).astype(np.int16)


def _glcm_offset_maps(q, mask, w, levels, dy, dx):
    """Mapas GLCM por pixel para UM offset (grade recortada). Retorna (maps, valid).

    Contraste/homogeneidade/correlação saem de somas-janela de funções de (a,b);
    ASM e entropia precisam do histograma por par (loop em levels²).
    """
    H, W = q.shape
    a = q[: H - dy, : W - dx].astype(np.float64)
    b = q[dy:, dx:].astype(np.float64)
    mv = mask[: H - dy, : W - dx] & mask[dy:, dx:]
    val = mv.astype(np.float64)
    N = _boxsum(val, w)
    Nz = np.maximum(N, 1.0)
    good = N > 0
    contrast = _boxsum(((a - b) ** 2) * val, w) / Nz
    homog = _boxsum((1.0 / (1.0 + np.abs(a - b))) * val, w) / Nz
    mua, mub = _boxsum(a * val, w) / Nz, _boxsum(b * val, w) / Nz
    va = _boxsum(a * a * val, w) / Nz - mua ** 2
    vb = _boxsum(b * b * val, w) / Nz - mub ** 2
    cov = _boxsum(a * b * val, w) / Nz - mua * mub
    corr = np.where((va > 1e-9) & (vb > 1e-9),
                    cov / np.sqrt(np.maximum(va * vb, 0.0) + 1e-12), 0.0)
    comb = (a * levels + b).astype(np.int32)
    asm_num = np.zeros_like(N)
    clogc = np.zeros_like(N)
    for k in range(levels * levels):
        ck = _boxsum(((comb == k) & mv), w)
        asm_num += ck * ck
        nz = ck > 0
        clogc[nz] += ck[nz] * np.log(ck[nz])
    maps = {"asm": asm_num / (Nz ** 2), "contrast": contrast,
            "entropy": np.log(Nz) - clogc / Nz, "correlation": corr, "homogeneity": homog}
    return maps, good & mv


def glcm_window_features(img, mask, windows=(3, 5, 7), levels=8, offsets=((0, 1), (1, 0))):
    """Textura GLCM por janela deslizante -> escalares (média sobre a máscara)."""
    v = img[mask]
    if v.size < 100:
        return {f"glcmw{w}_{n}": np.nan for w in windows for n in GLCM_NAMES}
    q = _quantize(img, mask, levels)
    out = {}
    for w in windows:
        acc = {n: [] for n in GLCM_NAMES}
        for dy, dx in offsets:
            maps, valid = _glcm_offset_maps(q, mask, w, levels, dy, dx)
            for n in GLCM_NAMES:
                sel = maps[n][valid]
                acc[n].append(float(sel.mean()) if sel.size else np.nan)
        for n, vals in acc.items():
            out[f"glcmw{w}_{n}"] = float(np.nanmean(vals))
    return out


def glcm_window_maps(img, mask, w, levels=8, offsets=((0, 1), (1, 0))):
    """Mapas GLCM por pixel (tamanho da imagem), média sobre offsets. Para visualização."""
    q = _quantize(img, mask, levels)
    H, W = img.shape
    acc = {n: np.zeros((H, W)) for n in GLCM_NAMES}
    cnt = np.zeros((H, W))
    for dy, dx in offsets:
        maps, valid = _glcm_offset_maps(q, mask, w, levels, dy, dx)
        sl = (slice(0, H - dy), slice(0, W - dx))
        cnt[sl] += valid
        for n in GLCM_NAMES:
            acc[n][sl] += np.where(valid, maps[n], 0.0)
    good = cnt > 0
    out = {n: np.where(good, acc[n] / np.maximum(cnt, 1), np.nan) for n in GLCM_NAMES}
    return out, good


# ---------------------------------------------------------------- por cubo

def features_for_cube(bil: Path, ndvi_thr: float, wl_bands: dict,
                      glcm_windows=(), glcm_levels=8) -> dict:
    hdr = read_hdr(bil.with_suffix(".bil.hdr") if (bil.parent / (bil.name + ".hdr")).exists()
                   else Path(str(bil) + ".hdr"))
    cube = read_bil(bil, hdr)          # (B, L, S)
    wl = hdr["wavelength"]
    iB = {name: band_at(wl, t) for name, t in wl_bands.items()}
    red, nir = cube[iB["red"]], cube[iB["nir"]]
    ndvi = safe_ratio(nir, red)
    mask = ndvi > ndvi_thr
    if mask.sum() < 100:               # fallback: usa tudo se máscara vazia
        mask = np.ones_like(ndvi, dtype=bool)

    green, rededge = cube[iB["green"]], cube[iB["rededge"]]
    mean_spec = cube[:, mask].mean(axis=1)          # (B,)
    r, n, g, re_ = (x[mask].mean() for x in (red, nir, green, rededge))

    m = re.match(r"(\d+)([MS]?)", bil.stem)
    feat = {
        "sample": bil.stem,
        "parcela": int(m.group(1)) if m else -1,
        "medida": m.group(2) if m else "",
        "n_veg_pixels": int(mask.sum()),
        # índices de vegetação
        "NDVI": float(safe_ratio(n, r)),
        "GNDVI": float(safe_ratio(n, g)),
        "NDRE": float(safe_ratio(n, re_)),
        "CIrededge": float(n / (re_ + 1e-6) - 1),
        "SAVI": float(1.5 * (n - r) / (n + r + 0.5)),
        "EVI2": float(2.5 * (n - r) / (n + 2.4 * r + 1)),
        # estatísticas do espectro
        "spec_mean": float(mean_spec.mean()),
        "spec_std": float(mean_spec.std()),
        "spec_p10": float(np.percentile(mean_spec, 10)),
        "spec_p50": float(np.percentile(mean_spec, 50)),
        "spec_p90": float(np.percentile(mean_spec, 90)),
    }
    feat.update(glcm_features(nir, mask))            # textura GLCM global (banda NIR)
    if glcm_windows:                                 # textura GLCM por janela deslizante
        feat.update(glcm_window_features(nir, mask, windows=tuple(glcm_windows),
                                         levels=glcm_levels))
    feat.update({f"b{k:03d}_{int(round(wl[k]))}nm": float(mean_spec[k])
                 for k in range(len(mean_spec))})    # espectro médio por banda
    # 1ª derivada espectral (Savitzky-Golay) — mesmo tipo de feature (d1_Band_*)
    # usado no PLSR de clorofila existente
    d1 = savgol_filter(mean_spec, window_length=7, polyorder=2, deriv=1)
    feat.update({f"d1_b{k:03d}_{int(round(wl[k]))}nm": float(d1[k])
                 for k in range(len(d1))})
    return feat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pasta com os .bil")
    ap.add_argument("--out", required=True, help="CSV de saída")
    ap.add_argument("--glob", default="*.bil")
    ap.add_argument("--ndvi-thr", type=float, default=0.3)
    ap.add_argument("--red", type=float, default=670.0)
    ap.add_argument("--nir", type=float, default=800.0)
    ap.add_argument("--green", type=float, default=550.0)
    ap.add_argument("--rededge", type=float, default=720.0)
    ap.add_argument("--glcm-windows", default="",
                    help="janelas GLCM deslizantes, ex.: '3,5,7' (vazio = desliga)")
    ap.add_argument("--glcm-levels", type=int, default=8)
    args = ap.parse_args()

    wl_bands = {"red": args.red, "nir": args.nir, "green": args.green, "rededge": args.rededge}
    gwin = tuple(int(x) for x in args.glcm_windows.split(",") if x.strip())
    cubes = sorted(p for p in Path(args.input).glob(args.glob) if p.stat().st_size > 0)
    print(f"{len(cubes)} cubos válidos em {args.input}"
          + (f" | GLCM janelas {gwin}" if gwin else ""))
    rows = []
    for i, bil in enumerate(cubes, 1):
        try:
            rows.append(features_for_cube(bil, args.ndvi_thr, wl_bands,
                                          glcm_windows=gwin, glcm_levels=args.glcm_levels))
            print(f"  [{i:2d}/{len(cubes)}] {bil.name}  ok")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:2d}/{len(cubes)}] {bil.name}  ERRO: {e}")
    df = pd.DataFrame(rows).sort_values(["parcela", "medida"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} amostras x {df.shape[1]} colunas -> {args.out}")
    cols = ["sample", "parcela", "medida", "n_veg_pixels",
            "NDVI", "NDRE", "GNDVI", "SAVI"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
