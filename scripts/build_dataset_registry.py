#!/usr/bin/env python3
"""Gera o catálogo versionado dos binários do dataset ABC."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "dataset_registry"
FIELDS = [
    "asset_id", "relative_path", "season", "domain", "acquisition_date", "stage",
    "sample_id", "asset_group", "status", "quarantine_reason", "size_bytes", "sha256",
]
DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
STAGE_RE = re.compile(r"(?:^|[_-])(V(?:6|8|10|11|13|18)|R[125])(?:[_-]|$)", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^(\d{1,2}[MS])(?:[.-]|$)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_quarantine() -> dict[str, str]:
    with (REGISTRY / "quarantine.csv").open(newline="", encoding="utf-8") as stream:
        return {row["relative_path"]: row["reason"] for row in csv.DictReader(stream)}


def asset_group(relative: Path) -> str:
    name = relative.name
    if name.endswith(".bil.hdr"):
        name = name[:-4]
    elif relative.suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
        name = relative.stem
    elif name.endswith(".tif.aux.xml"):
        name = name[:-8]
    return relative.with_name(name).with_suffix("")


def describe(relative: Path, quarantined: dict[str, str], include_hash: bool, data_root: Path) -> dict[str, str]:
    parts = relative.parts
    tree = parts[0] if parts else ""
    season = parts[1] if len(parts) > 1 and tree in {"raw", "quarantine"} else ""
    domain = parts[2] if len(parts) > 2 and tree in {"raw", "quarantine"} else ""
    matches = [part for part in parts if DATE_RE.fullmatch(part)]
    stage_match = STAGE_RE.search(relative.name)
    sample_match = SAMPLE_RE.match(relative.name)
    rel_text = relative.as_posix()
    path = data_root / relative
    return {
        "asset_id": re.sub(r"[^a-z0-9]+", "-", rel_text.lower()).strip("-"),
        "relative_path": rel_text,
        "season": season,
        "domain": domain,
        "acquisition_date": matches[-1] if matches else "",
        "stage": stage_match.group(1).upper() if stage_match else "",
        "sample_id": sample_match.group(1).upper() if sample_match else "",
        "asset_group": asset_group(relative).as_posix(),
        "status": "quarantined" if tree == "quarantine" else "usable",
        "quarantine_reason": quarantined.get(rel_text, ""),
        "size_bytes": str(path.stat().st_size),
        "sha256": sha256(path) if include_hash else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ABC_DATA_DIR", ROOT / "data")))
    parser.add_argument("--hash", action="store_true", help="calcula SHA-256 para cada arquivo")
    args = parser.parse_args()
    data_root = args.data_dir.expanduser().resolve()
    quarantined = load_quarantine()
    rows = [describe(path.relative_to(data_root), quarantined, args.hash, data_root)
            for path in sorted(data_root.rglob("*")) if path.is_file()]
    REGISTRY.mkdir(exist_ok=True)
    with (REGISTRY / "assets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} arquivos catalogados em {REGISTRY / 'assets.csv'}")


if __name__ == "__main__":
    main()
