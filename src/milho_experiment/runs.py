"""Manifesto mínimo, legível e rastreável para cada execução oficial."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .runtime import project_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: Path, config_path: Path, inputs: Iterable[Path], command: list[str]) -> Path:
    """Grava hashes de configuração/entradas sem copiar os dados grandes."""
    root = project_root(config_path)
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    payload = {
        "created_at": datetime.now(UTC).isoformat(), "git_revision": revision,
        "command": command, "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs if path.is_file()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return manifest
