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


def run_pooled(f23, y23, m24, y24, target, folds=5, max_comp=8):
    """Modelo único treinado nas DUAS safras juntas (colunas de textura comuns +
    dummy de safra), GroupKFold por safra+parcela. Aumenta n e a robustez; a dummy
    permite offset por safra (ajuda a calibração absoluta), tudo dentro do CV."""
    X23, X24 = select_block(f23, "texture"), select_block(m24, "texture")
    cols = [c for c in X23.columns if c in X24.columns]
    a = X23[cols].copy(); a["season"] = 0.0
    a["_grp"] = "2223_" + f23["parcela"].astype(str); a["_y"] = y23.values
    b = X24[cols].copy(); b["season"] = 1.0
    b["_grp"] = "2324_" + m24["parcela"].astype(str); b["_y"] = y24.values
    pool = pd.concat([a, b], ignore_index=True)
    feat_cols = cols + ["season"]
    X = pool[feat_cols]
    y = pool["_y"]
    return run_groupkfold(X, y, pool["_grp"], target, folds=folds, max_comp=max_comp)


def cross_safra(Xtr, ytr, Xte, yte, domain_adapt=False):
    """Treina numa safra (texture) e testa na outra. Alinha colunas comuns.

    domain_adapt: adaptação de domínio não-supervisionada por padronização por-domínio
    (z-score com a média/desvio de CADA safra). Alinha o 1º/2º momentos das features
    entre safras (variante "lite" do CORAL/feature-standardization DA) sem usar rótulos
    da safra-alvo. Corrige o shift de escala de GSD/iluminação/sensor no espaço de X;
    a calibração ABSOLUTA de y ainda depende de as distribuições do alvo coincidirem
    (por isso reportamos R² e corr separados — corr = transferência de ranking)."""
    cols = [c for c in Xtr.columns if c in Xte.columns]
    tr = Xtr[cols].notna().all(axis=1) & ytr.notna()
    te = Xte[cols].notna().all(axis=1) & yte.notna()
    Xtr_, ytr_ = Xtr[cols][tr].values, ytr[tr].values
    Xte_, yte_ = Xte[cols][te].values, yte[te].values
    nc = min(6, len(cols))
    if domain_adapt:
        # padroniza cada domínio pela SUA própria média/desvio (alinhamento de momentos)
        Xtr_z = StandardScaler().fit_transform(Xtr_)
        Xte_z = StandardScaler().fit_transform(Xte_)
        pls = PLSRegression(nc).fit(Xtr_z, ytr_)
        yhat = pls.predict(Xte_z).ravel()
    else:
        pipe = make_pipeline(StandardScaler(), PLSRegression(nc))
        pipe.fit(Xtr_, ytr_)
        yhat = pipe.predict(Xte_).ravel()
    return len(cols), metrics(yte_, yhat), np.corrcoef(yhat, yte_)[0, 1]


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

    # ---- (b) cross-safra (texture, colunas comuns): baseline vs adaptação de domínio ----
    X23, X24 = select_block(f23, "texture"), select_block(m24, "texture")
    lines.append("")
    lines.append("-- Cross-safra: baseline (sem DA) vs adaptação de domínio (z-score por safra) --")
    for name, Xa, Ya, Xb, Yb in [("2023/24→2022/23", X24, y24, X23, y23),
                                  ("2022/23→2023/24", X23, y23, X24, y24)]:
        k, m0, r0 = cross_safra(Xa, Ya, Xb, Yb, domain_adapt=False)
        _, m1, r1 = cross_safra(Xa, Ya, Xb, Yb, domain_adapt=True)
        lines.append(f"[cross {name}] {k} feats")
        lines.append(f"    baseline : R²={m0['R2']:.3f} RMSE={m0['RMSE']:.1f} corr={r0:.3f}")
        lines.append(f"    +DA      : R²={m1['R2']:.3f} RMSE={m1['RMSE']:.1f} corr={r1:.3f}  "
                     f"(Δcorr={r1 - r0:+.3f})")

    # ---- (c) modelo POOLED (duas safras juntas + dummy de safra) ----
    nc, ns, n, mp = run_pooled(f23, y23, m24, y24, tgt)
    lines.append("")
    lines.append(f"-- Pooled (22/23 + 23/24, texture+dummy safra) --")
    lines.append(f"[pooled] n={n} GroupKFold({ns}) nc={nc}: "
                 f"R²={mp['R2']:.3f} RMSE={mp['RMSE']:.1f} RPD={mp['RPD']:.2f}")

    report = "\n".join(lines)
    print(report)
    (out / f"multisafra_{tgt}.txt").write_text(report + "\n")
    print(f"\n-> {out}/multisafra_{tgt}.txt")


if __name__ == "__main__":
    main()
