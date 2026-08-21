"""Configuração e caminhos portáveis para os CLIs do experimento."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def project_root(start: Path | None = None) -> Path:
    """Localiza a raiz pelo ``pyproject.toml`` sem assumir máquina ou diretório atual."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("raiz do projeto não encontrada: pyproject.toml ausente")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    artifacts: Path
    vendor_gan: Path
    archive: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> "ProjectPaths":
        root = project_root(root)
        data = Path(os.environ.get("ABC_DATA_DIR", root / "data")).expanduser().resolve()
        artifacts = Path(os.environ.get("ABC_ARTIFACTS_DIR", root / "artifacts")).expanduser().resolve()
        vendor = root / "pytorch-CycleGAN-and-pix2pix"
        archive = Path(os.environ.get("ABC_ARCHIVE_DIR", root.parent / "ABC-archive")).expanduser().resolve()
        return cls(root=root, data=data, artifacts=artifacts, vendor_gan=vendor, archive=archive)

    def expand(self, value: str) -> Path:
        """Expande os marcadores usados em TOML para caminhos absolutos seguros."""
        mapping = {
            "{root}": str(self.root), "{data}": str(self.data),
            "{artifacts}": str(self.artifacts), "{vendor_gan}": str(self.vendor_gan),
            "{archive}": str(self.archive),
        }
        for marker, replacement in mapping.items():
            value = value.replace(marker, replacement)
        return Path(value).expanduser().resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for required in ("experiment", "execution"):
        if required not in config:
            raise ValueError(f"configuração inválida: seção [{required}] ausente em {path}")
    if "entrypoint" not in config["execution"]:
        raise ValueError(f"configuração inválida: execution.entrypoint ausente em {path}")
    return config
