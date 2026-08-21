#!/usr/bin/env python3
"""ETAPA 5 (parte 1) — Monta os pares alinhados vegetativo -> reprodutivo p/ o Pix2Pix.

Usa os recortes por parcela da etapa 4 (manifest.csv + npy/). Para cada parcela, forma pares
(estágio vegetativo, estágio reprodutivo) — por padrão {V8,V13} x {R2,R5} — e escreve o dataroot
no formato esperado pelo dataset condicional do pix2pix:

    <dataroot>/train/input/<par>.npy    reflectância vegetativa (H,W,3) [Red,RedEdge,NIR]
    <dataroot>/train/target/<par>.npy   reflectância reprodutiva (H,W,3)
    <dataroot>/val/input , <dataroot>/val/target
    <dataroot>/pairs.csv                manifest dos pares

O canal de clorofila (mapa CIrededge) é derivado da entrada dentro do dataset, em tempo de
treino — não precisa materializar aqui. Split treino/val é por PARCELA (evita vazamento).

Uso:
    python3 code/pipeline/stage5_make_pairs.py \
        --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
        --out    artifacts/runs/2324_pheno_gan/dataset \
        --veg V8 V13 --rep R2 R5 --val-blocos 4
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", required=True, help="dir de saída da etapa 4 (com manifest.csv)")
    ap.add_argument("--out", required=True, help="dataroot do pix2pix a criar")
    ap.add_argument("--veg", nargs="+", default=["V8", "V13"])
    ap.add_argument("--rep", nargs="+", default=["R2", "R5"])
    ap.add_argument("--val-blocos", type=int, default=4,
                    help="parcelas com Bloco>=este valor vão p/ validação")
    ap.add_argument("--val-bloco", type=int,
                    help="bloco exato de validação; substitui --val-blocos e habilita CV por bloco")
    args = ap.parse_args()

    s4 = Path(args.stage4)
    man = pd.read_csv(s4 / "manifest.csv")
    veg = set(v.upper() for v in args.veg)
    rep = set(r.upper() for r in args.rep)

    out = Path(args.out)
    for split in ("train", "val"):
        for sub in ("input", "target"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    # indexa npy por (fid, stage)
    by = {(int(r.fid), r.stage.upper()): r for r in man.itertuples()}
    fids = sorted(man.fid.unique())

    rows = []
    for fid in fids:
        # bloco da parcela (constante entre estágios)
        sub = man[man.fid == fid].iloc[0]
        split = "val" if (int(sub.bloco) == args.val_bloco if args.val_bloco is not None
                           else int(sub.bloco) >= args.val_blocos) else "train"
        for vs in sorted(veg):
            for rs in sorted(rep):
                a = by.get((fid, vs)); b = by.get((fid, rs))
                if a is None or b is None:
                    continue
                # Metadados de campo podem conter '/', espaços e acentos (ex.: "90 kg N/ha").
                # O identificador do par também é um nome de arquivo, portanto precisa ser portátil.
                dose_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(sub.dose_n)).strip("-")
                pair = f"p{fid:02d}_{vs}to{rs}_d{dose_tag}_b{sub.bloco}"
                shutil.copy(s4 / a.npy, out / split / "input" / f"{pair}.npy")
                shutil.copy(s4 / b.npy, out / split / "target" / f"{pair}.npy")
                rows.append(dict(pair=pair, split=split, fid=fid, dose_n=sub.dose_n,
                                 bloco=sub.bloco, veg=vs, rep=rs))
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    print(f"== {len(df)} pares -> {out}")
    print(df.groupby(["split"]).size().to_string())
    print("por combinação veg->rep:")
    print(df.groupby(["veg", "rep", "split"]).size().to_string())


if __name__ == "__main__":
    main()
