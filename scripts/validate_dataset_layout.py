#!/usr/bin/env python3
"""Valida a árvore canônica e o catálogo do dataset ABC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "dataset_registry" / "assets.csv"
SHAPE_SUFFIXES = {".shp", ".shx", ".dbf", ".prj", ".cpg"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ABC_DATA_DIR", ROOT / "data")))
    parser.add_argument("--hash", action="store_true")
    parser.add_argument("--read-images", action="store_true")
    args = parser.parse_args()
    data_root = args.data_dir.expanduser().resolve()
    errors: list[str] = []
    if any(data_root.glob("Safra*")):
        errors.append("foram encontrados diretórios legados Safra* na raiz de data/")
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        assets = list(csv.DictReader(stream))
    listed = {row["relative_path"] for row in assets}
    actual = {path.relative_to(data_root).as_posix() for path in data_root.rglob("*") if path.is_file()}
    errors.extend(f"sem catálogo: {path}" for path in sorted(actual - listed))
    errors.extend(f"ausente no disco: {path}" for path in sorted(listed - actual))
    for row in assets:
        path = data_root / row["relative_path"]
        if not path.exists() or row["status"] == "quarantined":
            continue
        if path.suffix.lower() == ".bil" and not path.with_name(path.name + ".hdr").is_file():
            errors.append(f"BIL sem HDR: {row['relative_path']}")
        if path.name.endswith(".bil.hdr") and not path.with_suffix("").is_file():
            errors.append(f"HDR sem BIL: {row['relative_path']}")
        if path.suffix.lower() in SHAPE_SUFFIXES:
            missing = [suffix for suffix in SHAPE_SUFFIXES if not path.with_suffix(suffix).is_file()]
            if missing:
                errors.append(f"shapefile incompleto: {row['relative_path']} ({', '.join(missing)})")
        if args.hash and row["sha256"] and digest(path) != row["sha256"]:
            errors.append(f"checksum divergente: {row['relative_path']}")
    if args.read_images:
        import tifffile
        for row in assets:
            path = data_root / row["relative_path"]
            if row["status"] == "usable" and path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    with tifffile.TiffFile(path) as image:
                        _ = image.series[0]
                except Exception as exc:  # pragma: no cover - message depends on TIFF library
                    errors.append(f"GeoTIFF ilegível: {row['relative_path']} ({exc})")
    if errors:
        raise SystemExit("Validação falhou:\n- " + "\n- ".join(errors))
    print(f"Validação aprovada: {len(assets)} arquivos catalogados.")


if __name__ == "__main__":
    main()
