#!/usr/bin/env python3
"""Seleção de bandas/atributos por VIP (Variable Importance in Projection) + forward K-Fold.

Espelha o passo de "otimização de bandas" do PLSR de clorofila existente, mas escalável a
300 bandas (força bruta é inviável): usa VIP para ranquear e seleção sequencial (forward)
guiada pelo ranking, escolhendo o subconjunto com melhor R² em K-Fold CV.

Saídas:
  out/vip_scores.csv          — VIP de cada atributo (com comprimento de onda quando aplicável)
  out/band_selection.txt      — melhor subconjunto, nc, métricas e curva R² × nº de bandas

Exemplos:
  python3 band_selection.py --features out/features_12dez.csv --synthetic --block derivative
  python3 band_selection.py --features out/features_12dez.csv \
      --target-csv campo.csv --target-col biomassa --block spectrum --max-features 20
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .plsr_kfold import select_block, metrics, optimize_components


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """VIP por variável de um PLSRegression já ajustado (alvo único)."""
    t = pls.x_scores_          # (n, A)
    w = pls.x_weights_         # (p, A)
    q = pls.y_loadings_        # (1, A)
    p, A = w.shape
    ss = (q[0] ** 2) * np.sum(t ** 2, axis=0)          # SS explicada de y por componente (A,)
    wn = w / (np.linalg.norm(w, axis=0) + 1e-12)       # pesos normalizados por componente
    return np.sqrt(p * ((wn ** 2) @ ss) / (ss.sum() + 1e-12))   # (p,)


def wavelength_of(col: str):
    m = re.search(r"(\d+)nm", col)
    return int(m.group(1)) if m else None


def cv_r2(X, y, nc, kf):
    pipe = make_pipeline(StandardScaler(), PLSRegression(n_components=nc))
    return metrics(y, cross_val_predict(pipe, X, y, cv=kf))


def load_target(df, args, ap):
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        base = select_block(df, "derivative")
        w = rng.normal(size=base.shape[1])
        y = base.values @ w
        y = 100 + 40 * (y - y.mean()) / (y.std() + 1e-9) + rng.normal(0, 5, len(y))
        return df, pd.Series(y, index=df.index), "alvo_sintetico (SMOKE-TEST)"
    if not (args.target_csv and args.target_col):
        ap.error("sem --synthetic é preciso --target-csv e --target-col")
    tgt = pd.read_csv(args.target_csv)
    merged = df.merge(tgt[[args.join, args.target_col]], on=args.join, how="inner")
    return merged, merged[args.target_col], args.target_col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--block", default="derivative",
                    choices=["spectrum", "derivative", "indices", "texture", "all"])
    ap.add_argument("--target-csv")
    ap.add_argument("--target-col")
    ap.add_argument("--join", default="parcela")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-comp", type=int, default=10)
    ap.add_argument("--max-features", type=int, default=25,
                    help="tamanho máximo do subconjunto testado na seleção forward")
    ap.add_argument("--vip-thr", type=float, default=1.0,
                    help="limiar de VIP para o subconjunto 'VIP>thr' reportado à parte")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/band_selection.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df, y, target_name = load_target(df, args, ap)
    Xdf = select_block(df, args.block)
    keep = y.notna() & Xdf.notna().all(axis=1)
    Xdf, y = Xdf[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    feats = list(Xdf.columns)
    X, yv = Xdf.values, y.values
    n = len(yv)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # 1) VIP a partir de um PLSR full (nc ótimo por CV)
    nc_full, _ = optimize_components(X, yv, kf, min(args.max_comp, X.shape[1], n - args.folds - 1))
    pls = make_pipeline(StandardScaler(), PLSRegression(n_components=nc_full)).fit(X, yv)
    vip = vip_scores(pls.named_steps["plsregression"])
    order = np.argsort(vip)[::-1]

    vip_df = pd.DataFrame({
        "feature": feats,
        "wavelength_nm": [wavelength_of(f) for f in feats],
        "vip": vip,
    }).sort_values("vip", ascending=False).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vip_df.to_csv(Path(args.out).with_name("vip_scores.csv"), index=False)

    # 2) seleção forward guiada por VIP: top-k features, escolhe melhor R² CV
    max_k = min(args.max_features, len(feats), n - args.folds - 1)
    curve, best = [], (0, -np.inf, None, None)
    for k in range(1, max_k + 1):
        idx = order[:k]
        Xk = X[:, idx]
        nc = min(args.max_comp, k, n - args.folds - 1)
        best_nc, best_r2 = 1, -np.inf
        for c in range(1, nc + 1):
            r2 = cv_r2(Xk, yv, c, kf)["R2"]
            if r2 > best_r2:
                best_nc, best_r2 = c, r2
        curve.append((k, best_r2, best_nc))
        if best_r2 > best[1]:
            best = (k, best_r2, best_nc, idx.copy())

    k_best, r2_best, nc_best, idx_best = best
    mt = cv_r2(X[:, idx_best], yv, nc_best, kf)
    sel = [feats[i] for i in idx_best]

    # 3) subconjunto por limiar VIP>thr (à parte)
    idx_thr = np.where(vip > args.vip_thr)[0]
    thr_line = "  (nenhuma feature acima do limiar)"
    if len(idx_thr):
        nc_thr = min(args.max_comp, len(idx_thr), n - args.folds - 1)
        mt_thr = cv_r2(X[:, idx_thr], yv, nc_thr, kf)
        thr_line = (f"  {len(idx_thr)} features, nc={nc_thr} -> R²={mt_thr['R2']:.3f} "
                    f"RMSE={mt_thr['RMSE']:.3f} RPD={mt_thr['RPD']:.3f}")

    lines = [
        "=== SELEÇÃO DE BANDAS / VIP ==========================================",
        f"Alvo: {target_name}",
        f"Bloco: '{args.block}' ({len(feats)} atributos)  |  amostras: {n}  |  folds: {args.folds}",
        f"PLSR full: nc={nc_full}",
        "",
        f"MELHOR SUBCONJUNTO (forward por VIP): {k_best} atributos, nc={nc_best}",
        f"  R²={mt['R2']:.4f}  RMSE={mt['RMSE']:.4f}  MAE={mt['MAE']:.4f}  RPD={mt['RPD']:.4f}",
        "  selecionados:",
    ]
    for f in sel:
        wl = wavelength_of(f)
        lines.append(f"    {f}" + (f"  ({wl} nm)" if wl else ""))
    lines += ["", f"SUBCONJUNTO VIP>{args.vip_thr}:", thr_line,
              "", "TOP-15 VIP:"]
    for _, r in vip_df.head(15).iterrows():
        wl = f"  ({int(r.wavelength_nm)} nm)" if pd.notna(r.wavelength_nm) else ""
        lines.append(f"  {r.vip:5.2f}  {r.feature}{wl}")
    lines += ["", "Curva R² × nº de atributos (forward, CV):"]
    lines += [f"  k={k:2d} nc={c:2d}: R²={r2:.3f}" for k, r2, c in curve]
    report = "\n".join(lines)

    print(report)
    Path(args.out).write_text(report + "\n")
    print(f"\n-> {args.out}  (+ vip_scores.csv)")


if __name__ == "__main__":
    main()
