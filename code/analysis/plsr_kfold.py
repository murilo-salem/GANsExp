#!/usr/bin/env python3
"""PLSR + validação K-Fold para estimar parâmetros agronômicos a partir dos atributos hiper.

Fluxo:
  X = atributos (CSV do hyperspectral_features.py / stage6_features_ortho.py), bloco
      combinável com '+': spectrum | derivative | indices | texture | physical | all
      (physical = atributos de campo clorofila/N — phys_CHL_total, phys_N_percent,
       phys_N_acumulado; realiza o "clorofila (físico)" do diagrama etapa 6)
  y = alvo por amostra (altura/biomassa/produtividade/clorofila), vindo de um CSV de campo
      com colunas [parcela|sample, <alvo>]. Enquanto o Y agronômico real não chega,
      use --synthetic para um smoke-test do pipeline.

Otimiza n_components por K-Fold CV (R² médio) e reporta R²/RMSE/MAE/RPD por fold e agregado.
Guarda anti-vazamento: o alvo (e sua contraparte phys_<alvo>) é sempre removido de X.

Exemplos:
  # smoke-test (alvo sintético a partir das features) — só valida a mecânica
  python3 plsr_kfold.py --features out/features_12dez.csv --synthetic --block derivative

  # uso real — textura + clorofila/N físicos (diagrama etapa 6)
  python3 plsr_kfold.py --features out/features_ortho_2324.csv \
      --target-csv out/targets_ortho_2324.csv --target-col Biomassa \
      --join sample --block texture+physical --group parcela
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_COLS = ["sample", "parcela", "medida", "n_veg_pixels"]
INDEX_COLS = ["NDVI", "GNDVI", "NDRE", "CIrededge", "SAVI", "EVI2"]
STAT_COLS = ["spec_mean", "spec_std", "spec_p10", "spec_p50", "spec_p90"]


BLOCKS = ("spectrum", "derivative", "indices", "texture", "physical", "all")


def select_block(df: pd.DataFrame, block: str) -> pd.DataFrame:
    """Seleciona colunas do bloco; aceita combinações com '+' (ex.: texture+physical)."""
    is_spec = lambda c: c.startswith("b") and c[1:4].isdigit()
    is_deriv = lambda c: c.startswith("d1_")
    is_tex = lambda c: c.startswith("glcm")        # glcm_* (global) e glcmwN_* (janela)
    is_phys = lambda c: c.startswith("phys_")      # phys_CHL_total, phys_N_percent, ...
    pool = {
        "spectrum":   [c for c in df.columns if is_spec(c)],
        "derivative": [c for c in df.columns if is_deriv(c)],
        "texture":    [c for c in df.columns if is_tex(c)],
        "indices":    [c for c in df.columns if c in INDEX_COLS + STAT_COLS],
        "physical":   [c for c in df.columns if is_phys(c)],
        "all":        [c for c in df.columns if c not in ID_COLS],
    }
    cols = []
    for part in block.split("+"):
        part = part.strip()
        if part not in pool:
            raise ValueError(f"bloco desconhecido '{part}' (válido: {list(BLOCKS)})")
        for c in pool[part]:
            if c not in cols:
                cols.append(c)
    if not cols:
        raise ValueError(f"bloco '{block}' não selecionou colunas neste CSV")
    return df[cols]


def metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": 1 - ss_res / (ss_tot + 1e-12),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(err))),
        "RPD": float(np.std(y_true, ddof=1) / (rmse + 1e-12)),
    }


def optimize_components(X, y, kf, max_comp, groups=None):
    """Escolhe n_components pelo maior R² médio em cross_val_predict."""
    best = (1, -np.inf)
    curve = []
    for nc in range(1, max_comp + 1):
        pipe = make_pipeline(StandardScaler(), PLSRegression(n_components=nc))
        yhat = cross_val_predict(pipe, X, y, cv=kf, groups=groups)
        r2 = metrics(y, yhat)["R2"]
        curve.append((nc, r2))
        if r2 > best[1]:
            best = (nc, r2)
    return best[0], curve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="CSV de atributos (hyperspectral_features.py)")
    ap.add_argument("--block", default="derivative",
                    help="bloco de atributos, combinável com '+': " + " | ".join(BLOCKS))
    ap.add_argument("--target-csv", help="CSV de campo com o alvo")
    ap.add_argument("--target-col", help="nome da coluna-alvo em --target-csv")
    ap.add_argument("--join", default="parcela", help="chave de junção (parcela|sample)")
    ap.add_argument("--group", default=None,
                    help="coluna de agrupamento p/ GroupKFold (ex.: parcela) — evita vazamento "
                         "quando o alvo se repete entre amostras da mesma unidade")
    ap.add_argument("--synthetic", action="store_true",
                    help="gera alvo sintético a partir das features (smoke-test)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-comp", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/plsr_report.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.features)

    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        base = select_block(df, "derivative")
        w = rng.normal(size=base.shape[1])
        y = (base.values @ w)
        y = 100 + 40 * (y - y.mean()) / (y.std() + 1e-9) + rng.normal(0, 5, len(y))
        y = pd.Series(y, index=df.index, name="alvo_sintetico")
        target_name = "alvo_sintetico (SMOKE-TEST)"
    else:
        if not (args.target_csv and args.target_col):
            ap.error("sem --synthetic é preciso --target-csv e --target-col")
        tgt = pd.read_csv(args.target_csv)
        merged = df.merge(tgt[[args.join, args.target_col]], on=args.join, how="inner")
        df, y = merged, merged[args.target_col]
        target_name = args.target_col

    X = select_block(df, args.block)
    # Guarda anti-vazamento: nunca deixar o alvo (nem sua contraparte phys_<alvo>) em X.
    # Isto fecha o "R²≈1.0 espúrio" do bloco 'all' (o merge adiciona o alvo ao df).
    leaked = []
    if not args.synthetic:
        for cand in (args.target_col, f"phys_{args.target_col}"):
            if cand in X.columns:
                leaked.append(cand)
        if leaked:
            X = X.drop(columns=leaked)
    mask = y.notna() & X.notna().all(axis=1)
    groups = None
    if args.group:
        if args.group not in df.columns:
            ap.error(f"--group '{args.group}' não está nas colunas das features")
        groups = df[args.group][mask].reset_index(drop=True).values
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    n = len(y)

    if args.group:
        n_splits = min(args.folds, len(np.unique(groups)))
        kf = GroupKFold(n_splits=n_splits)
        split_args = dict(groups=groups)
        val_kind = f"GroupKFold por '{args.group}' ({n_splits} grupos/folds)"
    else:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_args = {}
        val_kind = f"KFold aleatório ({args.folds} folds)"
    max_comp = min(args.max_comp, X.shape[1], n - args.folds - 1)

    nc, curve = optimize_components(X.values, y.values, kf, max_comp, groups=groups)

    # predição CV com o melhor nc + métricas por fold
    pipe = make_pipeline(StandardScaler(), PLSRegression(n_components=nc))
    yhat = cross_val_predict(pipe, X.values, y.values, cv=kf, groups=groups)
    overall = metrics(y.values, yhat)
    per_fold = []
    for k, (_, te) in enumerate(kf.split(X, y, **split_args), 1):
        per_fold.append((k, metrics(y.values[te], yhat[te])))

    lines = [
        "=== PLSR + K-FOLD =====================================================",
        f"Alvo: {target_name}",
        f"Atributos: bloco '{args.block}'  ({X.shape[1]} colunas)",
        f"Validação: {val_kind}",
        f"Amostras: {n}  |  seed: {args.seed}",
        f"n_components ótimo (CV): {nc}  (máx testado: {max_comp})",
    ]
    if leaked:
        lines.append(f"Anti-vazamento: removido de X -> {leaked}")
    lines += [
        "",
        "MÉTRICAS AGREGADAS (predição K-Fold out-of-fold):",
        f"  R²   = {overall['R2']:.4f}",
        f"  RMSE = {overall['RMSE']:.4f}",
        f"  MAE  = {overall['MAE']:.4f}",
        f"  RPD  = {overall['RPD']:.4f}",
        "",
        "POR FOLD:",
    ]
    for k, mt in per_fold:
        lines.append(f"  Fold {k}: R²={mt['R2']:.3f}  RMSE={mt['RMSE']:.3f}  "
                     f"MAE={mt['MAE']:.3f}  RPD={mt['RPD']:.3f}")
    lines += ["", "Curva R² × n_components (CV):"]
    lines += [f"  nc={nc_:2d}: R²={r2:.3f}" for nc_, r2 in curve]
    report = "\n".join(lines)

    print(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    pd.DataFrame({"y_true": y.values, "y_pred_cv": yhat}).to_csv(
        Path(args.out).with_suffix(".pred.csv"), index=False)
    print(f"\n-> {args.out}  (+ .pred.csv)")


if __name__ == "__main__":
    main()
