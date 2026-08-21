#!/usr/bin/env python3
"""Georreferenciamento leve para os ortomosaicos RRENIR (GeoTIFF) e as parcelas (shapefile).

Sem rasterio/gdal: lê o GeoTIFF com `tifffile` e o geotransform pelas tags GeoTIFF
(ModelPixelScale + ModelTiepoint), e o shapefile com `pyshp`. Todos os produtos deste
projeto estão em SIRGAS 2000 / UTM 22S (EPSG:31982), então parcelas e orto compartilham CRS
e o recorte por parcela é uma simples conversão UTM->pixel.

Uso como biblioteca:
    ortho = Ortho("data/raw/safra_2023_2024/orthomosaics/RRENIR_R2_2023_2024.tif")
    parcels = read_parcels("data/raw/safra_2023_2024/geometry/Shape_parcelas23_24.shp")
    crop = ortho.crop_bbox(parcels[0].bbox)     # (H, W, 3) uint16, bandas [Red, RedEdge, NIR]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


# ----------------------------------------------------------------- GeoTIFF

@dataclass
class GeoTransform:
    """Mapeia UTM (x, y) <-> pixel (col, row) para um raster 'pixel is area'."""
    sx: float          # tamanho do pixel em x (m)
    sy: float          # tamanho do pixel em y (m, positivo)
    x0: float          # UTM x da borda esquerda do pixel (0,0)
    y0: float          # UTM y da borda superior do pixel (0,0)

    def xy_to_colrow(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / self.sx, (self.y0 - y) / self.sy

    @classmethod
    def from_tags(cls, tags: dict) -> "GeoTransform":
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            raise ValueError("GeoTIFF sem ModelPixelScale/ModelTiepoint — não georreferenciado.")
        sx, sy = float(scale[0]), float(scale[1])
        # tie: (i, j, k, X, Y, Z) — pixel (i,j) corresponde a UTM (X,Y)
        i, j, _, X, Y, _ = tie[:6]
        x0 = X - i * sx
        y0 = Y + j * sy
        return cls(sx=sx, sy=sy, x0=x0, y0=y0)


class Ortho:
    """Ortomosaico GeoTIFF (leitura preguiçosa por janela via tifffile)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with tifffile.TiffFile(self.path) as t:
            page = t.pages[0]
            self.height, self.width = int(page.imagelength), int(page.imagewidth)
            self.bands = int(page.samplesperpixel)
            self.dtype = page.dtype
            tags = {tag.name: tag.value for tag in page.tags}
        self.gt = GeoTransform.from_tags(tags)
        self._arr = None                      # cache preguiçoso do raster inteiro

    def read(self) -> np.ndarray:
        """Lê o raster inteiro (H, W, bandas) uma vez e cacheia (orto pode ter GBs)."""
        if self._arr is None:
            self._arr = tifffile.imread(self.path)
        return self._arr

    def crop_bbox(self, bbox, pad_m: float = 0.0) -> np.ndarray:
        """Recorta a janela que cobre um bbox UTM (xmin, ymin, xmax, ymax).

        `pad_m` adiciona uma margem em metros. Retorna (h, w, bandas).
        Lê apenas a região necessária (tifffile aiff -> fatiamento na memmap).
        """
        xmin, ymin, xmax, ymax = bbox
        pad_x = pad_m / self.gt.sx
        pad_y = pad_m / self.gt.sy
        c0, r1 = self.gt.xy_to_colrow(xmin, ymin)   # canto inferior-esquerdo
        c1, r0 = self.gt.xy_to_colrow(xmax, ymax)   # canto superior-direito
        c0 = max(0, int(np.floor(c0 - pad_x)))
        c1 = min(self.width, int(np.ceil(c1 + pad_x)))
        r0 = max(0, int(np.floor(r0 - pad_y)))
        r1 = min(self.height, int(np.ceil(r1 + pad_y)))
        if c1 <= c0 or r1 <= r0:
            raise ValueError(f"bbox fora do raster {self.path.name}: {bbox}")
        window = self.read()[r0:r1, c0:c1]         # usa cache (lê o orto só 1x)
        if window.ndim == 2:
            window = window[..., None]
        return np.ascontiguousarray(window)


# ----------------------------------------------------------------- shapefile

@dataclass
class Parcel:
    fid: int
    dose_n: str
    bloco: int
    bbox: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax (UTM)
    points: list                              # anel do polígono (UTM)

    @property
    def key(self) -> tuple[str, int]:
        """Chave de junção com a tabela de campo: (Dose_N, Bloco)."""
        return (str(self.dose_n), int(self.bloco))


def read_parcels(shp_path: str | Path) -> list[Parcel]:
    import shapefile  # pyshp

    sf = shapefile.Reader(str(shp_path))
    field_names = [f[0] for f in sf.fields[1:]]
    out: list[Parcel] = []
    for rec, shp in zip(sf.records(), sf.shapes()):
        d = dict(zip(field_names, rec))
        out.append(
            Parcel(
                fid=int(d.get("fid", d.get("id", -1))),
                dose_n=str(d.get("Dose_N", "")),
                bloco=int(d.get("Bloco", -1)),
                bbox=tuple(shp.bbox),
                points=list(shp.points),
            )
        )
    return out


# ----------------------------------------------------------------- índices

def to_reflectance(crop: np.ndarray) -> np.ndarray:
    """uint16 -> float32 em [0,1] por normalização robusta (percentil 99.5)."""
    x = crop.astype(np.float32)
    hi = np.percentile(x, 99.5)
    if hi <= 0:
        hi = float(x.max()) or 1.0
    return np.clip(x / hi, 0.0, 1.0)


def rrenir_indices(refl: np.ndarray, eps: float = 1e-8) -> dict:
    """Índices para orto RRENIR (bandas: 0=Red, 1=RedEdge, 2=NIR)."""
    red, rededge, nir = refl[..., 0], refl[..., 1], refl[..., 2]
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    cire = nir / (rededge + eps) - 1.0            # CIrededge ~ proxy de clorofila
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    return {"NDVI": ndvi, "NDRE": ndre, "CIrededge": cire, "SAVI": savi}


def chlorophyll_map(refl: np.ndarray) -> np.ndarray:
    """Mapa de clorofila (proxy CIrededge) normalizado para [0,1]."""
    cire = rrenir_indices(refl)["CIrededge"]
    lo, hi = np.nanpercentile(cire, [2, 98])
    return np.clip((cire - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def bands5_indices(refl5: np.ndarray, eps: float = 1e-8) -> dict:
    """Índices para stack 5-bandas B1-B5 (0=Azul,1=Verde,2=Vermelho,3=RedEdge,4=NIR).

    Além dos de RRENIR, habilita índices que exigem azul/verde: GNDVI, EVI, VARI, TGI.
    Obs.: azul/verde vêm de câmera RGB (DN, não calibrada) — usar com essa ressalva.
    """
    blue, green, red, rededge, nir = (refl5[..., k] for k in range(5))
    return {
        "NDVI": (nir - red) / (nir + red + eps),
        "GNDVI": (nir - green) / (nir + green + eps),
        "NDRE": (nir - rededge) / (nir + rededge + eps),
        "CIrededge": nir / (rededge + eps) - 1.0,
        "SAVI": 1.5 * (nir - red) / (nir + red + 0.5 + eps),
        "EVI": 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps),
        "VARI": (green - red) / (green + red - blue + eps),
        "TGI": green - 0.39 * red - 0.61 * blue,   # Triangular Greenness Index (simplificado)
    }


if __name__ == "__main__":
    import sys

    base = Path("data/raw/safra_2023_2024")
    shp = base / "Shapefile/Shape_parcelas23_24.shp"
    tif = base / "Ortomosaicos/RRENIR_R2_2023_2024.tif"
    parcels = read_parcels(shp)
    ortho = Ortho(tif)
    print(f"orto {tif.name}: {ortho.width}x{ortho.height} {ortho.bands}b {ortho.dtype}")
    print(f"gt: sx={ortho.gt.sx:.5f} x0={ortho.gt.x0:.2f} y0={ortho.gt.y0:.2f}")
    print(f"parcelas: {len(parcels)}  chaves ex.: {[p.key for p in parcels[:5]]}")
    p = parcels[0]
    crop = ortho.crop_bbox(p.bbox)
    print(f"parcela fid={p.fid} key={p.key} -> crop {crop.shape} "
          f"min={crop.min()} max={crop.max()}")
    refl = to_reflectance(crop)
    idx = rrenir_indices(refl)
    print("NDVI médio:", round(float(np.nanmean(idx['NDVI'])), 3),
          "| CIrededge médio:", round(float(np.nanmean(idx['CIrededge'])), 3))
