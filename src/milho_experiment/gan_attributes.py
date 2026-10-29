"""Catálogo completo de condicionamento calculável com Red/RedEdge/NIR.

Reutiliza os índices e os extratores GLCM existentes. Descritores globais são
escalares por imagem, transmitidos como planos constantes; texturas locais
mantêm a geometria dos mapas. Os aliases legados conservam seus valores.
"""
from __future__ import annotations

import numpy as np

from .indices import attribute_channels, norm01, norm01_global, rrenir_indices
from .pipeline.stage_03_attributes.hyperspectral_features import (
    GLCM_NAMES,
    HARALICK_NAMES,
    glcm_features,
    glcm_window_maps,
)


RRENIR_INDEX_NAMES = ("NDVI", "NDRE", "CIrededge", "SAVI", "EVI2")
GLCM_WINDOWS = (3, 5, 7)
GLOBAL_GLCM_CHANNELS = tuple(f"glcm_global_{name}" for name in HARALICK_NAMES)
LOCAL_GLCM_CHANNELS = tuple(
    f"glcmw{window}_{name}" for window in GLCM_WINDOWS for name in GLCM_NAMES
)
ALL_RRENIR_CANDIDATES = (*RRENIR_INDEX_NAMES, *GLOBAL_GLCM_CHANNELS, *LOCAL_GLCM_CHANNELS)
LEGACY_RRENIR_CANDIDATES = (
    "chlorophyll", "NDVI", "NDRE", "SAVI", "EVI2",
    "glcm_asm", "glcm_contrast", "glcm_entropy", "glcm_homogeneity",
)


def rrenir_attribute_channels(
    refl: np.ndarray,
    names: tuple[str, ...],
    normalize: str = "image",
    global_bounds: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Empilha os candidatos solicitados em H×W×C, na ordem recebida.

    GLCM global: 16 níveis, máscara NDVI > .3, fallback com <200 pixels,
    como ``parcel_features``. Mapas locais: 8 níveis, janelas 3/5/7.
    Para ``normalize='image'``, escalares globais usam a transformação fixa
    .5 + atan(sign(x)*log1p(abs(x)))/pi. Normalizá-los por percentis de seu
    plano constante apagaria seu valor. Nenhum limite é aprendido no teste.
    Descritores globais indefinidos recebem zero bruto (.5 codificado).
    """
    refl = np.asarray(refl)
    if refl.ndim != 3 or refl.shape[-1] != 3 or min(refl.shape[:2]) < 2:
        raise ValueError("esperado H×W×3 RRENIR com H,W >= 2")
    if not np.isfinite(refl).all():
        raise ValueError("reflectância contém valores não finitos")
    if normalize not in {"image", "global", "none"}:
        raise ValueError(f"normalização inválida: {normalize!r}")
    supported = set(ALL_RRENIR_CANDIDATES) | set(LEGACY_RRENIR_CANDIDATES)
    invalid = set(names) - supported
    if invalid:
        raise ValueError(f"canais indisponíveis em RRENIR: {sorted(invalid)}")
    if not names:
        return np.empty((*refl.shape[:2], 0), dtype=np.float32)

    indices = rrenir_indices(refl)
    mask = indices["NDVI"] > .3
    if mask.sum() < 200:
        mask = np.ones(refl.shape[:2], dtype=bool)
    global_features = None
    window_maps = {}
    result = []
    for name in names:
        if name in LEGACY_RRENIR_CANDIDATES or name == "CIrededge":
            # Conserva inclusive a máscara e a quantização dos checkpoints antigos.
            channel = attribute_channels(
                refl, names=(name,), normalize=normalize, global_bounds=global_bounds,
            )[..., 0]
            result.append(channel)
            continue
        if name in GLOBAL_GLCM_CHANNELS:
            if global_features is None:
                global_features = glcm_features(refl[..., 2], mask, levels=16)
            key = name.replace("glcm_global_", "glcm_", 1)
            value = float(global_features[key])
            value = value if np.isfinite(value) else 0.0
            raw = np.full(refl.shape[:2], value, dtype=np.float64)
            if normalize == "image":
                encoded = .5 + np.arctan(np.sign(value) * np.log1p(abs(value))) / np.pi
                channel = np.full(refl.shape[:2], encoded, dtype=np.float32)
                result.append(channel)
                continue
        else:
            window_token, metric = name.split("_", 1)
            window = int(window_token.removeprefix("glcmw"))
            if window not in window_maps:
                maps, _ = glcm_window_maps(refl[..., 2], mask, window, levels=8)
                window_maps[window] = maps
            raw = np.nan_to_num(window_maps[window][metric], nan=0.0, posinf=0.0, neginf=0.0)
        if normalize == "image":
            channel = norm01(raw)
        elif normalize == "global":
            bounds = (global_bounds or {}).get(name)
            if bounds is None:
                raise ValueError(f"limite global ausente para {name!r}")
            channel = norm01_global(raw, *bounds)
        else:
            channel = raw.astype(np.float32)
        result.append(channel)
    return np.stack(result, axis=-1)
