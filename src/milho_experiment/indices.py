"""Cálculo canônico de índices de vegetação e canais condicionais para a GAN.

Todas as funções recebem reflectância em [0, 1] e retornam arrays da mesma
geometria espacial (H, W) ou (H, W, C). Usar este módulo garante que os
índices vistos pelo gerador sejam numericamente consistentes com os extraídos
na pipeline agronômica (stage6_features_ortho.py, build_5band.py etc.).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


EPS = 1e-8


# ------------------------------------------------------------------ helpers

def _band_at(refl: np.ndarray, idx: int) -> np.ndarray:
    """Extrai banda de um array (H,W) ou (H,W,C)."""
    if refl.ndim == 2:
        return refl
    return refl[..., idx]


def norm01(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Normaliza x para [0, 1] usando percentis 2–98 ou limites fornecidos."""
    if lo is None or hi is None:
        lo, hi = float(np.nanpercentile(x, 2)), float(np.nanpercentile(x, 98))
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def norm01_global(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalização por limites globais (ex.: estatísticas de toda a safra/voo)."""
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ RRENIR

def rrenir_indices(refl: np.ndarray, eps: float = EPS) -> dict[str, np.ndarray]:
    """Índices para orto RRENIR (bandas: 0=Red, 1=RedEdge, 2=NIR)."""
    red, rededge, nir = (_band_at(refl, i) for i in range(3))
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    cire = nir / (rededge + eps) - 1.0
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + eps)
    return {"NDVI": ndvi, "NDRE": ndre, "CIrededge": cire, "SAVI": savi, "EVI2": evi2}


def chlorophyll_map(refl: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Mapa de clorofila (proxy CIrededge) normalizado para [0, 1]."""
    cire = rrenir_indices(refl, eps)["CIrededge"]
    return norm01(cire)


# ------------------------------------------------------------------ 5-bandas

def bands5_indices(refl5: np.ndarray, eps: float = EPS) -> dict[str, np.ndarray]:
    """Índices para stack 5-bandas B1–B5 (0=Azul, 1=Verde, 2=Vermelho, 3=RedEdge, 4=NIR)."""
    blue, green, red, rededge, nir = (_band_at(refl5, k) for k in range(5))
    return {
        "NDVI": (nir - red) / (nir + red + eps),
        "GNDVI": (nir - green) / (nir + green + eps),
        "NDRE": (nir - rededge) / (nir + rededge + eps),
        "CIrededge": nir / (rededge + eps) - 1.0,
        "SAVI": 1.5 * (nir - red) / (nir + red + 0.5 + eps),
        "EVI": 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps),
        "EVI2": 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + eps),
        "VARI": (green - red) / (green + red - blue + eps),
        "TGI": green - 0.39 * red - 0.61 * blue,
    }


# ------------------------------------------------------------------ canais GAN

# Nomes canônicos dos canais condicionais disponíveis para a GAN.
# A ordem importa: o primeiro é o canal de clorofila (sempre incluído no modo legacy).
# Textura: mapa GLCM por pixel (janela 5, banda NIR), médio sobre offsets.
GLCM_WINDOW = 5
GLCM_TEXTURE_NAMES = ("glcm_asm", "glcm_contrast", "glcm_entropy", "glcm_homogeneity")
GAN_ATTRIBUTE_NAMES = ("chlorophyll", "NDVI", "NDRE", "SAVI", "GNDVI", "EVI2",
                       *GLCM_TEXTURE_NAMES)


# ------------------------------------------------------------------ GLCM por pixel

def _boxsum(a, w):
    return uniform_filter(a.astype(np.float64), size=w, mode="constant", cval=0.0) * (w * w)


def _quantize(img, mask, levels):
    v = img[mask]
    lo, hi = np.nanpercentile(v, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-9) * (levels - 1), 0, levels - 1).astype(np.int16)


def _glcm_offset_maps(q, mask, w, levels, dy, dx):
    """Mapas GLCM por pixel para UM offset (grade recortada). Retorna (maps, valid)."""
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


def glcm_window_maps_quantized(q, mask, w=GLCM_WINDOW, levels=8,
                               offsets=((0, 1), (1, 0))) -> dict[str, np.ndarray]:
    """Mapas GLCM por pixel (mesma geometria da imagem), média sobre offsets.

    Retorna os 5 mapas (asm, contrast, entropy, correlation, homogeneity).
    """
    names = ("asm", "contrast", "entropy", "correlation", "homogeneity")
    H, W = q.shape
    acc = {n: np.zeros((H, W)) for n in names}
    cnt = np.zeros((H, W))
    for dy, dx in offsets:
        maps, valid = _glcm_offset_maps(q, mask, w, levels, dy, dx)
        sl = (slice(0, H - dy), slice(0, W - dx))
        cnt[sl] += valid
        for n in names:
            acc[n][sl] += np.where(valid, maps[n], 0.0)
    good = cnt > 0
    return {n: np.where(good, acc[n] / np.maximum(cnt, 1), np.nan) for n in names}


def texture_channels(refl: np.ndarray, mask: np.ndarray | None = None,
                     names: tuple[str, ...] = GLCM_TEXTURE_NAMES,
                     window: int = GLCM_WINDOW, levels: int = 8,
                     normalize: str = "image",
                     global_bounds: dict[str, tuple[float, float]] | None = None,
                     eps: float = EPS) -> np.ndarray:
    """Mapas GLCM por pixel (banda NIR do RRENIR) como canais condicionais.

    ``names`` deve conter chaves de GLCM_TEXTURE_NAMES (asm, contrast, entropy,
    homogeneity). A correlação não é incluída por padrão (pode ter valores
    negativos/instáveis; adicione explicitamente se necessário).
    """
    nir = refl[..., 2]
    if mask is None:
        ndvi = (nir - refl[..., 0]) / (nir + refl[..., 0] + eps)
        mask = ndvi > 0.3
        if mask.sum() < 100:
            mask = np.ones(refl.shape[:2], bool)
    q = _quantize(nir, mask, levels)
    maps = glcm_window_maps_quantized(q, mask, w=window, levels=levels)
    out = []
    for name in names:
        if name not in maps:
            raise ValueError(f"mapa de textura desconhecido: {name!r}")
        raw = maps[name]
        raw = np.where(np.isnan(raw), 0.0, raw)
        if normalize == "image":
            out.append(norm01(raw))
        elif normalize == "global":
            bounds = (global_bounds or {}).get(name)
            if bounds is None:
                raise ValueError(f"limite global ausente para {name!r}")
            out.append(norm01_global(raw, *bounds))
        elif normalize == "none":
            out.append(raw.astype(np.float32))
        else:
            raise ValueError(f"normalize deve ser 'image', 'global' ou 'none'; recebeu {normalize!r}")
    return np.stack(out, axis=-1)


def attribute_channels(
    refl: np.ndarray,
    names: tuple[str, ...] = ("chlorophyll", "NDVI", "NDRE", "SAVI"),
    normalize: str = "image",
    global_bounds: dict[str, tuple[float, float]] | None = None,
    eps: float = EPS,
) -> np.ndarray:
    """Canais de condicionamento da GAN a partir de reflectância RRENIR [0,1].

    Parâmetros
    ----------
    refl : np.ndarray
        Reflectância RRENIR (H, W, 3) = [Red, RedEdge, NIR] em [0, 1].
    names : tuple[str, ...]
        Quais canais retornar, em ordem. Padrão = 4 canais legados.
    normalize : {"image", "global", "none"}
        "image" -> percentis 2–98 de cada imagem (compatível com legacy);
        "global" -> limites fornecidos em ``global_bounds``;
        "none" -> sem normalização.
    global_bounds : dict[str, tuple[float, float]] | None
        Limites (lo, hi) por nome, usados quando normalize="global".
    eps : float
        Constante de estabilidade numérica.

    Retorna
    -------
    np.ndarray de shape (H, W, len(names)) em [0, 1] se normalizado.
    """
    idx = rrenir_indices(refl, eps)
    tex_mask = None

    def _raw(name: str) -> np.ndarray:
        if name == "chlorophyll":
            return idx["CIrededge"]
        if name.startswith("glcm_"):
            nonlocal tex_mask
            tex_mask = _texture_mask(refl, idx, tex_mask, eps)
            return _texture_map(refl, name, tex_mask, window=GLCM_WINDOW, eps=eps)
        if name in idx:
            return idx[name]
        raise ValueError(f"canal condicional desconhecido: {name!r}; "
                         f"válidos: {GAN_ATTRIBUTE_NAMES}")

    out = []
    for name in names:
        raw = _raw(name)
        if normalize == "image":
            out.append(norm01(raw))
        elif normalize == "global":
            bounds = (global_bounds or {}).get(name)
            if bounds is None:
                raise ValueError(f"limite global ausente para {name!r}")
            out.append(norm01_global(raw, *bounds))
        elif normalize == "none":
            out.append(raw.astype(np.float32))
        else:
            raise ValueError(f"normalize deve ser 'image', 'global' ou 'none'; recebeu {normalize!r}")

    return np.stack(out, axis=-1)


def _texture_mask(refl: np.ndarray, idx: dict, cached: np.ndarray | None,
                  eps: float) -> np.ndarray:
    """Máscara de vegetação para os mapas GLCM (cacheada entre chamadas)."""
    if cached is not None:
        return cached
    nir = refl[..., 2]
    ndvi = idx["NDVI"]
    mask = ndvi > 0.3
    if mask.sum() < 100:
        mask = np.ones(refl.shape[:2], bool)
    return mask


def _texture_map(refl: np.ndarray, name: str, mask: np.ndarray,
                 window: int = GLCM_WINDOW, eps: float = EPS) -> np.ndarray:
    """Retorna um mapa GLCM (banda NIR) por nome, ex.: 'glcm_asm'."""
    nir = refl[..., 2]
    q = _quantize(nir, mask, 8)
    maps = glcm_window_maps_quantized(q, mask, w=window, levels=8)
    key = name[len("glcm_"):]
    if key not in maps:
        raise ValueError(f"canal condicional desconhecido: {name!r}; "
                         f"válidos: {GAN_ATTRIBUTE_NAMES}")
    return np.where(np.isnan(maps[key]), 0.0, maps[key])


def attribute_channels_5band(
    refl5: np.ndarray,
    names: tuple[str, ...] = ("chlorophyll", "NDVI", "NDRE", "SAVI"),
    normalize: str = "image",
    global_bounds: dict[str, tuple[float, float]] | None = None,
    eps: float = EPS,
) -> np.ndarray:
    """Versão de attribute_channels para stack 5-bandas (usado quando disponível)."""
    idx = bands5_indices(refl5, eps)
    nir = refl5[..., 4]
    red = refl5[..., 2]
    tex_mask = None

    def _raw(name: str) -> np.ndarray:
        if name == "chlorophyll":
            return idx["CIrededge"]
        if name.startswith("glcm_"):
            nonlocal tex_mask
            if tex_mask is None:
                mask = idx["NDVI"] > 0.3
                if mask.sum() < 100:
                    mask = np.ones(refl5.shape[:2], bool)
                tex_mask = mask
            q = _quantize(nir, tex_mask, 8)
            maps = glcm_window_maps_quantized(q, tex_mask, w=GLCM_WINDOW, levels=8)
            key = name[len("glcm_"):]
            if key not in maps:
                raise ValueError(f"canal condicional desconhecido: {name!r}")
            return np.where(np.isnan(maps[key]), 0.0, maps[key])
        if name in idx:
            return idx[name]
        raise ValueError(f"canal condicional desconhecido: {name!r}; "
                         f"válidos: {GAN_ATTRIBUTE_NAMES}")

    out = []
    for name in names:
        raw = _raw(name)
        if normalize == "image":
            out.append(norm01(raw))
        elif normalize == "global":
            bounds = (global_bounds or {}).get(name)
            if bounds is None:
                raise ValueError(f"limite global ausente para {name!r}")
            out.append(norm01_global(raw, *bounds))
        elif normalize == "none":
            out.append(raw.astype(np.float32))
        else:
            raise ValueError(f"normalize deve ser 'image', 'global' ou 'none'; recebeu {normalize!r}")

    return np.stack(out, axis=-1)
