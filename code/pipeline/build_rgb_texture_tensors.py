#!/usr/bin/env python3
"""Gera tensores RGB + índices + GLCM para os ortomosaicos com RGB físico.

Cada saída é um Zarr ``tensor`` H×W×22 em float32, acompanhado de ``valid_mask``.
Os canais são RGB, quatro índices RGB e cinco mapas GLCM em janelas 3, 5 e 7.
O processamento é feito em blocos com halo, logo não exige carregar o mosaico inteiro.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from hyperspectral_features import GLCM_NAMES, glcm_window_maps_quantized  # noqa: E402


WINDOWS = (3, 5, 7)
INDEX_NAMES = ("VARI", "ExG", "GLI", "TGI")
CHANNEL_NAMES = (
    "R", "G", "B", *INDEX_NAMES,
    *(f"glcmw{window}_{name}" for window in WINDOWS for name in GLCM_NAMES),
)
HALO = 4  # raio máximo de janela 7 (3) + deslocamento GLCM de 1 pixel


@dataclass(frozen=True)
class Source:
    season: str
    stage: str
    date: str
    path: Path | None
    rgb_bands: tuple[int, int, int] | None = None
    exclusion: str | None = None


def discover_sources(data_dir: Path) -> list[Source]:
    """Lista a cobertura definida para a receita RGB física, incluindo exclusões."""
    p22 = data_dir / "raw/safra_2022_2023/orthomosaics"
    p23 = data_dir / "raw/safra_2023_2024/orthomosaics"
    p25 = data_dir / "raw/safra_2025_2026/orthomosaics"
    sources = [
        Source("2022_2023", "V8", "", p22 / "rgb_v8_2022_2023.tif", (0, 1, 2)),
        Source("2022_2023", "V11", "", p22 / "RGB_v11_2022_2023.tif", (0, 1, 2)),
        Source("2022_2023", "V18", "", p22 / "RGB_v18_2022_2023.tif", (0, 1, 2)),
        Source("2022_2023", "R2", "", p22 / "RGB_R2_2022_2023.tif", (0, 1, 2)),
        Source("2022_2023", "R5", "", p22 / "RGB_R5_2022_2023.tif", (0, 1, 2)),
        Source("2025_2026", "V13", "2025-12-23",
               p25 / "23.12.2025_voo regular (30m)/Milho_23.12.2025_comsensor_vooregular_mosaico_modificado.tif",
               (0, 1, 2)),
        Source("2025_2026", "R1", "2026-01-14",
               p25 / "14.01.2026/milho_14_01_2026_com_sensor_mosaico_modificado.tif", (0, 1, 2)),
    ]
    sources.extend(
        Source("2023_2024", stage, "", None, exclusion="RGB físico ausente")
        for stage in ("V6", "V8", "V13", "R2", "R5")
    )
    sources.append(Source("2025_2026", "V10", "2025-12-12", None,
                          exclusion="banda verde ausente; não há RGB físico"))
    return sources


def rgb_indices(rgb: np.ndarray) -> np.ndarray:
    """Retorna VARI, ExG, GLI e TGI para RGB normalizado em [0, 1]."""
    red, green, blue = (rgb[..., i] for i in range(3))
    eps = np.float32(1e-8)
    return np.stack((
        (green - red) / (green + red - blue + eps),
        2 * green - red - blue,
        (2 * green - red - blue) / (2 * green + red + blue + eps),
        green - 0.39 * red - 0.61 * blue,
    ), axis=-1).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def quantize(img: np.ndarray, lo: float, hi: float, levels: int = 8) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo + 1e-9) * (levels - 1), 0, levels - 1).astype(np.int16)


class TiffReader:
    """Leitor fatiável de GeoTIFF, inclusive para imagens tiled sem memmap."""

    def __init__(self, path: Path):
        self.path = path
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            self.shape = (int(page.imagelength), int(page.imagewidth), int(page.samplesperpixel))
            self.dtype = page.dtype
            self.tags = {tag.name: tag.value for tag in page.tags}
        self._store = tifffile.imread(path, aszarr=True)
        opened = zarr.open(self._store, mode="r")
        # TIFF tiled com pirâmide abre como grupo; o nível 0 é a resolução total.
        self.array = opened["0"] if isinstance(opened, zarr.Group) and "0" in opened else opened

    def read(self, y0: int, y1: int, x0: int, x1: int, bands: tuple[int, int, int]) -> np.ndarray:
        return np.asarray(self.array[y0:y1, x0:x1, list(bands)])

    def close(self) -> None:
        self._store.close()


def percentile_bounds(reader: TiffReader, bands: tuple[int, int, int], chunk: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Estima percentis globais por amostragem regular, sem carregar o mosaico inteiro."""
    height, width, _ = reader.shape
    samples = []
    stride = max(1, int(np.ceil(max(height, width) / 2048)))
    for y in range(0, height, chunk):
        for x in range(0, width, chunk):
            raw = reader.read(y, min(y + chunk, height), x, min(x + chunk, width), bands)
            samples.append(raw[::stride, ::stride].reshape(-1, 3))
    values = np.concatenate(samples, axis=0).astype(np.float32)
    valid = np.any(values > 0, axis=1)
    if not valid.any():
        raise ValueError(f"{reader.path}: não há pixels RGB válidos")
    p995 = np.percentile(values[valid], 99.5, axis=0).astype(np.float32)
    p995[p995 <= 0] = 1
    lum = luminance(np.clip(values[valid] / p995, 0, 1))
    return p995, tuple(float(x) for x in np.percentile(lum, [2, 98]))


def write_source(source: Source, out_dir: Path, chunk: int) -> dict:
    assert source.path is not None and source.rgb_bands is not None
    if not source.path.is_file():
        raise FileNotFoundError(source.path)
    reader = TiffReader(source.path)
    try:
        height, width, _ = reader.shape
        scales, (lum_lo, lum_hi) = percentile_bounds(reader, source.rgb_bands, chunk)
        path = out_dir / "tensors" / source.season / f"{source.stage}.zarr"
        group = zarr.open_group(path, mode="w")
        compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5)
        tensor = group.create_array("tensor", shape=(height, width, len(CHANNEL_NAMES)),
                                    chunks=(min(chunk, height), min(chunk, width), len(CHANNEL_NAMES)),
                                    dtype="f4", compressors=compressor)
        valid_out = group.create_array("valid_mask", shape=(height, width),
                                       chunks=(min(chunk, height), min(chunk, width)), dtype="b1",
                                       compressors=compressor)
        for y0 in range(0, height, chunk):
            for x0 in range(0, width, chunk):
                y1, x1 = min(y0 + chunk, height), min(x0 + chunk, width)
                hy0, hx0 = max(0, y0 - HALO), max(0, x0 - HALO)
                hy1, hx1 = min(height, y1 + HALO), min(width, x1 + HALO)
                raw = reader.read(hy0, hy1, hx0, hx1, source.rgb_bands).astype(np.float32)
                rgb = np.clip(raw / scales, 0, 1)
                valid = np.any(raw > 0, axis=-1)
                lum = luminance(rgb)
                q = quantize(lum, lum_lo, lum_hi)
                channels = [rgb, rgb_indices(rgb)]
                for window in WINDOWS:
                    maps, _ = glcm_window_maps_quantized(q, valid, window)
                    channels.append(np.stack([maps[name] for name in GLCM_NAMES], axis=-1).astype(np.float32))
                full = np.concatenate(channels, axis=-1)
                full[~valid] = np.nan
                sy0, sx0 = y0 - hy0, x0 - hx0
                tensor[y0:y1, x0:x1] = full[sy0:sy0 + (y1-y0), sx0:sx0 + (x1-x0)]
                valid_out[y0:y1, x0:x1] = valid[sy0:sy0 + (y1-y0), sx0:sx0 + (x1-x0)]
        group.attrs.update({
            "season": source.season, "stage": source.stage, "date": source.date,
            "source": str(source.path), "shape": [height, width, len(CHANNEL_NAMES)],
            "channel_names": list(CHANNEL_NAMES), "rgb_scales_p99_5": scales.tolist(),
            "luminance_quantiles": [lum_lo, lum_hi], "glcm_windows": list(WINDOWS),
            "geotiff_tags": json.dumps({k: str(v) for k, v in reader.tags.items()
                                         if k in ("ModelPixelScaleTag", "ModelTiepointTag", "GeoKeyDirectoryTag")}),
        })
        return {"season": source.season, "stage": source.stage, "date": source.date,
                "status": "generated", "source": str(source.path), "zarr": str(path.relative_to(out_dir)),
                "height": height, "width": width, "channels": len(CHANNEL_NAMES), "exclusion": ""}
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.chunk <= 2 * HALO:
        raise ValueError(f"--chunk deve ser maior que {2 * HALO}")
    rows = []
    for source in discover_sources(args.data_dir):
        if source.exclusion:
            rows.append({"season": source.season, "stage": source.stage, "date": source.date,
                         "status": "excluded", "source": "", "zarr": "", "height": "", "width": "",
                         "channels": 0, "exclusion": source.exclusion})
            continue
        if args.dry_run:
            print(f"[dry-run] {source.season}/{source.stage}: {source.path}")
            continue
        print(f"[{source.season}/{source.stage}] {source.path.name}")
        rows.append(write_source(source, args.out, args.chunk))
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out / "manifest.csv", index=False)
        print(f"{len(rows)} registros -> {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
