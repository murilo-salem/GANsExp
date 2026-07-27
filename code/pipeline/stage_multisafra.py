#!/usr/bin/env python3
"""PASSO 4 — Generalização multi-safra: PLSR em 22/23 e validação cruzada entre safras.

(a) Junta o Y (parametros_2223_normalizado) às features 5-bandas de 22/23 e roda PLSR+GroupKFold.
(b) Validação cruzada: treina o PLSR numa safra e testa na outra, usando o bloco de TEXTURA
    (descritores GLCM — schema comum entre as features 3-bandas do 23/24 e 5-bandas do 22/23),
    medindo a transferência entre safras.

Uso:
    python3 code/pipeline/stage_multisafra.py \
        --feat2223 code/pipeline/out/band5_2223/features_5band_2223.csv \
        --table2223 "data/Safra2022a2023/tabela de dados/parametros_2223_normalizado.xlsx" \
        --feat2324 code/pipeline/out/plsr_2324/features_ortho_2324.csv \
        --tgt2324  code/pipeline/out/plsr_2324/targets_ortho_2324.csv \
        --target Biomassa --out code/pipeline/out/multisafra
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "analysis"))
from plsr_kfold import metrics, select_block  # noqa: E402


def build_targets_2223(feat: pd.DataFrame, table: str, target: str) -> pd.Series:
    tbl = pd.read_excel(table)
    tbl["Estagio"] = tbl["Estagio"].astype(str).str.upper()
    ymap = {(r.Estagio, str(int(r.Dose_N)), int(r.Bloco)): float(getattr(r, target))
            for r in tbl.itertuples() if pd.notna(getattr(r, target))}
    y = []
    for r in feat.itertuples():
        stage, dose, bloco = _parse_sample(r.sample)
        y.append(ymap.get((stage, dose, bloco), np.nan))
    return pd.Series(y, index=feat.index, name=target)


def _parse_sample(s: str):
    # sample = "<STAGE>_d<dose>_b<bloco>"
    stage, dose, bloco = s.split("_d")[0], s.split("_d")[1].split("_b")[0], s.split("_b")[1]
    return stage.upper(), str(int(dose)), int(bloco)


def run_groupkfold(X, y, groups, target, folds=5, max_comp=8):
    mask = y.notna() & X.notna().all(axis=1)
    X, y, groups = X[mask].values, y[mask].values, np.asarray(groups)[mask.values]
    n_splits = min(folds, len(np.unique(groups)))
    kf = GroupKFold(n_splits=n_splits)
    best = (1, -np.inf)
    mc = min(max_comp, X.shape[1], len(y) - n_splits - 1)
    for nc in range(1, mc + 1):
        yhat = cross_val_predict(make_pipeline(StandardScaler(), PLSRegression(nc)),
                                 X, y, cv=kf, groups=groups)
        r2 = metrics(y, yhat)["R2"]
        if r2 > best[1]:
            best = (nc, r2)
    yhat = cross_val_predict(make_pipeline(StandardScaler(), PLSRegression(best[0])),
                             X, y, cv=kf, groups=groups)
    m = metrics(y, yhat)
    return best[0], n_splits, len(y), m


def cross_safra(Xtr, ytr, Xte, yte):
    """Treina numa safra (texture) e testa na outra. Alinha colunas comuns."""
    cols = [c for c in Xtr.columns if c in Xte.columns]
    tr = Xtr[cols].notna().all(axis=1) & ytr.notna()
    te = Xte[cols].notna().all(axis=1) & yte.notna()
    pipe = make_pipeline(StandardScaler(), PLSRegression(min(6, len(cols))))
    pipe.fit(Xtr[cols][tr].values, ytr[tr].values)
    yhat = pipe.predict(Xte[cols][te].values).ravel()
    return len(cols), metrics(yte[te].values, yhat), np.corrcoef(yhat, yte[te].values)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat2223", required=True)
    ap.add_argument("--table2223", required=True)
    ap.add_argument("--feat2324", required=True)
    ap.add_argument("--tgt2324", required=True)
    ap.add_argument("--target", default="Biomassa")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tgt = args.target

    # ---- features + Y de cada safra ----
    f23 = pd.read_csv(args.feat2223)
    y23 = build_targets_2223(f23, args.table2223, tgt)
    f24 = pd.read_csv(args.feat2324)
    t24 = pd.read_csv(args.tgt2324)
    m24 = f24.merge(t24[["sample", tgt]], on="sample", how="inner")
    y24 = m24[tgt]

    lines = [f"=== PASSO 4 — MULTI-SAFRA (alvo={tgt}, bloco=texture) ==="]

    # ---- (a) within-safra GroupKFold ----
    for name, F, Y in [("2022/23", f23, y23), ("2023/24", m24, y24)]:
        Xt = select_block(F, "texture")
        nc, ns, n, mt = run_groupkfold(Xt, Y, F["parcela"], tgt)
        lines.append(f"[within {name}] n={n} GroupKFold({ns}) nc={nc}: "
                     f"R²={mt['R2']:.3f} RMSE={mt['RMSE']:.1f} RPD={mt['RPD']:.2f}")

    # ---- (b) cross-safra (texture, colunas comuns) ----
    X23, X24 = select_block(f23, "texture"), select_block(m24, "texture")
    for a, b, Xa, Ya, Xb, Yb in [("2023/24→2022/23", "", X24, y24, X23, y23),
                                  ("2022/23→2023/24", "", X23, y23, X24, y24)]:
        k, mt, r = cross_safra(Xa, Ya, Xb, Yb)
        lines.append(f"[cross {a}] {k} feats: R²={mt['R2']:.3f} RMSE={mt['RMSE']:.1f} "
                     f"corr={r:.3f}")

    report = "\n".join(lines)
    print(report)
    (out / f"multisafra_{tgt}.txt").write_text(report + "\n")
    print(f"\n-> {out}/multisafra_{tgt}.txt")


if __name__ == "__main__":
    main()
