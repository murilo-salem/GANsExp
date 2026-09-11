"""Boundary for the externally managed Pix2Pix checkout."""
from __future__ import annotations

import sys
from pathlib import Path

from .runtime import ProjectPaths


def checkout() -> Path:
    """Return the configured Pix2Pix checkout or fail with an actionable error."""
    path = ProjectPaths.discover().vendor_gan
    if not (path / "models" / "networks.py").is_file():
        raise RuntimeError(
            "checkout Pix2Pix ausente; defina ABC_PIX2PIX_DIR ou restaure "
            f"{path}"
        )
    return path


def enable_imports() -> Path:
    """Expose the external checkout only to callers that need its modules."""
    path = checkout()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
