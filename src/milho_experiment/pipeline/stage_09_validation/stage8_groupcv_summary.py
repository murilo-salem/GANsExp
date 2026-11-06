#!/usr/bin/env python3
"""Agrega resultados ``stage8_scenarios.py`` de uma validação GAN leave-one-block-out.

Cada diretório deve conter ``scenarios_<alvo>_parcelas.csv`` produzido por um modelo GAN
treinado sem o bloco correspondente. A métrica final usa uma previsão por parcela.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(y, pred):
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "R2": float(1 - np.sum(err ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(err))),
        "RPD": float(np.std(y, ddof=1) / (rmse + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", action="append", required=True,
                    help="diretório de um fold; repita para cada bloco")
    ap.add_argument("--target", default="Produtividade")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for directory in args.fold_dir:
        path = Path(directory) / f"scenarios_{args.target}_parcelas.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"fid", "y_meas", "y_pred_real", "y_pred_synth", "l1_img"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{path}: colunas ausentes {sorted(missing)}")
        frame["fold_dir"] = str(directory)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.fid.duplicated().any():
        raise ValueError(f"parcela aparece em mais de um fold: {data[data.fid.duplicated()].fid.tolist()}")

    real = metrics(data.y_meas.to_numpy(), data.y_pred_real.to_numpy())
    synth = metrics(data.y_meas.to_numpy(), data.y_pred_synth.to_numpy())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / f"oof_{args.target}_parcelas.csv", index=False)
    report = [
        f"=== GAN GROUP-CV — {args.target} ===",
        f"Parcelas OOF: {len(data)} | folds: {len(frames)} | unidade independente: parcela",
        f"L1 GAN médio por parcela: {data.l1_img.mean():.4f}", "",
        f"REPRODUTIVO REAL     -> R²={real['R2']:.3f} RMSE={real['RMSE']:.2f} MAE={real['MAE']:.2f} RPD={real['RPD']:.2f}",
        f"REPRODUTIVO SINTÉTICO-> R²={synth['R2']:.3f} RMSE={synth['RMSE']:.2f} MAE={synth['MAE']:.2f} RPD={synth['RPD']:.2f}",
    ]
    (out / f"report_{args.target}.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
