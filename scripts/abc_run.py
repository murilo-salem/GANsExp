#!/usr/bin/env python3
"""Executa um experimento TOML e grava manifesto em artifacts/runs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Permite executar o CLI diretamente de um checkout, sem exigir instalação prévia.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from milho_experiment.runs import write_manifest
from milho_experiment.runtime import ProjectPaths, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    paths, config = ProjectPaths.discover(), load_config(args.config)
    execution = config["execution"]
    entrypoint = paths.expand(execution["entrypoint"])
    arguments = [str(paths.expand(str(arg))) if "{" in str(arg) else str(arg)
                 for arg in execution.get("arguments", [])]
    inputs = [paths.expand(value) for value in config.get("inputs", {}).values()]
    missing = [str(path) for path in inputs if not path.exists()]
    if args.check_inputs and missing:
        raise SystemExit("entradas ausentes:\n- " + "\n- ".join(missing))
    run_id = config["experiment"].get("run_id", args.config.stem)
    output_dir = paths.artifacts / "runs" / run_id
    command = [sys.executable, str(entrypoint), *arguments]
    if args.dry_run:
        print(" ".join(command))
        print(f"manifesto: {output_dir / 'manifest.json'}")
        return
    if not entrypoint.is_file():
        raise SystemExit(f"entrypoint ausente: {entrypoint}")
    manifest = write_manifest(output_dir, args.config, inputs, command)
    print(f"manifesto: {manifest}")
    subprocess.run(command, check=True, cwd=paths.root)


if __name__ == "__main__":
    main()
