#!/usr/bin/env python3
"""Validação do valor preditivo da GAN fenológica (CV por fold, multi-seed).

Para cada combinação de atributos condicionais ("número de objetos"), a GAN é
retreinada uma vez por bloco externo (4 folds de 6 parcelas) e por semente;
imagens sintéticas OOF são geradas para as 24 parcelas. Um modelo downstream é
ajustado somente nos blocos de treino e avalia o R² OOF por parcela em cinco
cenários:

    veg          V8/V13 REAIS                  (baseline pré-colheita)
    real         R2/R5 REAIS                   (teto; inviável pré-colheita)
    fake         R2/R5 SINTÉTICOS (GAN)        (a GAN pura prediz sozinha?)
    fusion_veg   vegetativo + sintético        (a GAN agrega ao que temos hoje?)
    fusion_real  reprodutivo real + sintético  (a GAN complementa o reprodutivo?)

Diferença de R² (fusion_veg − veg) é o indicador central de "valor da GAN";
o critério de sucesso do projeto é ΔR² ≥ +0.05. Métricas de imagem (L1, ΔNDVI)
são agregadas por parcela. Bootstrap IC95% é calculado sobre as 24 predições OOF.

Subcomandos:
    prepare-folds   cria dataroots por bloco a partir do stage4
    sweep           orquestra train+infer+downstream para todas as configurações/seeds

Uso:
    python3 code/pipeline/stage5_attribute_search.py sweep \
        --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
        --table "data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx" \
        --target Biomassa --epochs 200 --seeds 7 11 23 \
        --out artifacts/runs/2324_attr_search/results
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PIX2PIX = REPO / "pytorch-CycleGAN-and-pix2pix"
sys.path[:0] = [str(PIX2PIX), str(REPO / "src"), str(REPO / "code" / "pipeline"),
                str(REPO / "code" / "analysis")]

from data.phenology_dataset import DEFAULT_ATTR_NAMES  # noqa: E402
from milho_experiment.indices import attribute_channels  # noqa: E402
from models.networks import define_G  # noqa: E402
from stage6_features_ortho import parcel_features  # noqa: E402

STAGE5_MAKE_PAIRS = REPO / "code" / "pipeline" / "stage5_make_pairs.py"
STAGE5_INFER = REPO / "code" / "pipeline" / "stage5_infer.py"

# ------------------------------------------------------------------ grade de configurações

ATTRIBUTE_CONFIGS = [
    ("chl", ("chlorophyll",)),
    ("chl_ndvi", ("chlorophyll", "NDVI")),
    ("chl_ndvi_ndre", ("chlorophyll", "NDVI", "NDRE")),
    ("chl_ndvi_ndre_savi", DEFAULT_ATTR_NAMES),          # 4 canais legados
    ("chl_ndvi_ndre_savi_evi2", ("chlorophyll", "NDVI", "NDRE", "SAVI", "EVI2")),
    ("chl_ndvi_tex1", ("chlorophyll", "NDVI", "glcm_asm")),
    ("chl_ndvi_tex2", ("chlorophyll", "NDVI", "glcm_asm", "glcm_contrast")),
    ("chl_ndvi_tex3", ("chlorophyll", "NDVI", "glcm_asm", "glcm_contrast", "glcm_entropy")),
]

BLOCKS = (1, 2, 3, 4)
MODELS = ("plsr", "elasticnet", "extratrees")


def available_configs() -> list[tuple[str, tuple[str, ...]]]:
    return ATTRIBUTE_CONFIGS


# ------------------------------------------------------------------ prepare-folds

def prepare_folds(stage4: Path, out_root: Path, veg: list[str] = ("V8", "V13"),
                  rep: list[str] = ("R2", "R5")):
    """Cria um dataroot por bloco: train = demais blocos, val = bloco ``b``."""
    out_root.mkdir(parents=True, exist_ok=True)
    for b in BLOCKS:
        droot = out_root / f"block{b}"
        if (droot / "pairs.csv").is_file():
            print(f"[prepare] bloco {b}: já existe, pulando")
            continue
        cmd = [
            sys.executable, str(STAGE5_MAKE_PAIRS),
            "--stage4", str(stage4),
            "--out", str(droot),
            "--veg", *veg, "--rep", *rep,
            "--val-bloco", str(b),
        ]
        subprocess.run(cmd, check=True, cwd=REPO)
        print(f"[prepare] bloco {b}: {len(list((droot / 'train' / 'input').glob('*.npy')))} "
              f"treino / {len(list((droot / 'val' / 'input').glob('*.npy')))} val")


# ------------------------------------------------------------------ train/infer fold

def train_fold(dataroot: Path, cfg_out: Path, attr_names: tuple[str, ...],
               epochs: int, batch_size: int, seed: int, save_freq: int | None = None,
               no_html: bool = True):
    ckpt_dir = cfg_out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = cfg_out.name
    final_ckpt = ckpt_dir / name / f"{epochs}_net_G.pth"
    if final_ckpt.is_file():
        print(f"[train] {name}: checkpoint já existe, pulando ({final_ckpt.name})")
        return final_ckpt
    cmd = [
        sys.executable, str(PIX2PIX / "train.py"),
        "--dataroot", str(dataroot),
        "--name", name,
        "--model", "pix2pix",
        "--dataset_mode", "phenology",
        "--direction", "AtoB",
        "--load_size", "286", "--crop_size", "256",
        "--batch_size", str(batch_size),
        "--n_epochs", str(epochs), "--n_epochs_decay", "0",
        "--seed", str(seed),
        "--gan_attr_names", ",".join(attr_names),
        "--gan_attr_normalize", "image",
        "--checkpoints_dir", str(ckpt_dir),
    ]
    if save_freq:
        cmd += ["--save_epoch_freq", str(save_freq)]
    if no_html:
        cmd.append("--no_html")
    env = {**dict(os.environ), "WANDB_MODE": "disabled"}
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return final_ckpt


def infer_fold(dataroot: Path, ckpt: Path, cfg_out: Path, attr_names: tuple[str, ...],
               phase: str = "val"):
    infer_dir = cfg_out / "infer"
    if (infer_dir / "metrics_pairs.csv").is_file():
        print(f"[infer] {cfg_out.name}: resultados já existem, pulando")
        return infer_dir
    cmd = [
        sys.executable, str(STAGE5_INFER),
        "--dataroot", str(dataroot),
        "--phase", phase,
        "--ckpt", str(ckpt),
        "--out", str(infer_dir),
        "--gan-attr-names", ",".join(attr_names),
        "--gan-attr-normalize", "image",
        "--pairs", str(dataroot / "pairs.csv"),
    ]
    subprocess.run(cmd, check=True, cwd=REPO)
    return infer_dir


# ------------------------------------------------------------------ downstream

def _metrics(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {"r2": float(r2_score(y, p)), "rmse": rmse,
            "mae": float(mean_absolute_error(y, p)),
            "rpd": float(np.std(y, ddof=1) / (rmse + 1e-12))}


def _bootstrap_r2(y, p, rng, n=1000):
    vals = []
    for _ in range(n):
        take = rng.integers(0, len(y), len(y))
        if np.unique(y[take]).size > 1:
            vals.append(_metrics(y[take], p[take])["r2"])
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _make_model(kind, seed, p, n):
    ks = sorted({min(k, p) for k in (3, 5, 8)})
    common = [("impute", SimpleImputer(strategy="median")), ("select", SelectKBest(f_regression))]
    if kind == "plsr":
        pipe = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] +
                        [("model", PLSRegression())])
        grid = [{"select__k": [k], "model__n_components": [c]}
                for k in ks for c in range(1, min(3, k, n - 2) + 1)]
    elif kind == "elasticnet":
        pipe = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] +
                        [("model", ElasticNet(max_iter=20000, random_state=seed))])
        grid = [{"select__k": [k], "model__alpha": [a], "model__l1_ratio": [r]}
                for k, a, r in itertools.product(ks, (0.01, 0.1, 1.0), (0.1, 0.5, 0.9))]
    else:
        pipe = Pipeline(common + [("model", ExtraTreesRegressor(n_estimators=300,
                                                               random_state=seed, n_jobs=-1))])
        grid = [{"select__k": [k], "model__min_samples_leaf": [leaf], "model__max_features": [mf]}
                for k, leaf, mf in itertools.product(ks, (1, 4), (1.0,))]
    return pipe, grid


def _feature_vector(refl: np.ndarray, cols: list[str]) -> np.ndarray:
    f = parcel_features(refl)
    return np.array([f[c] for c in cols], dtype=float)


def _feature_cols(refl: np.ndarray) -> list[str]:
    f = parcel_features(refl)
    return sorted(k for k in f if k.startswith("glcm") or k in {"NDVI", "NDRE", "CIrededge", "SAVI"})


def _load_fold_features(folds_root: Path, cfg_out: Path, table: Path, target: str,
                        seed: int) -> pd.DataFrame:
    """Carrega features veg/real/fake por par de validação e agrega por parcela."""
    tbl = pd.read_excel(table)
    tbl["Estagio"] = tbl["Estagio"].astype(str).str.upper()
    ymap = {}
    for r in tbl.itertuples():
        if pd.notna(getattr(r, target)):
            ymap[(int(r.Bloco), str(int(r.Dose_N)))] = float(getattr(r, target))

    rows, cols = [], None
    for b in BLOCKS:
        droot = folds_root / f"block{b}"
        infer_dir = cfg_out / f"block{b}" / "infer"
        pairs = pd.read_csv(droot / "pairs.csv")
        metrics = pd.read_csv(infer_dir / "metrics_pairs.csv")
        df = metrics.merge(pairs[["pair", "dose_n"]], on="pair", how="left",
                           suffixes=("", "_pairs"))
        for _, r in df.iterrows():
            key = (int(r.bloco), str(int(r.dose_n)))
            if key not in ymap:
                continue
            veg = np.load(droot / "val" / "input" / f"{r.pair}.npy").astype(np.float32)
            real = np.load(droot / "val" / "target" / f"{r.pair}.npy").astype(np.float32)
            fake = np.load(infer_dir / f"{r.pair}_fake.npy").astype(np.float32)
            if cols is None:
                cols = _feature_cols(real)
            rows.append({
                "fid": int(r.fid), "block": int(r.bloco), "y": ymap[key],
                **{f"veg_{c}": v for c, v in zip(cols, _feature_vector(veg, cols))},
                **{f"real_{c}": v for c, v in zip(cols, _feature_vector(real, cols))},
                **{f"fake_{c}": v for c, v in zip(cols, _feature_vector(fake, cols))},
            })
    if not rows:
        raise RuntimeError(f"sem parcelas com alvo {target!r} no downstream")
    data = pd.DataFrame(rows)
    feat_cols = [c for c in data.columns if c not in {"fid", "block", "y"}]
    agg = data.groupby("fid", as_index=False).agg(
        {"block": "first", "y": "first", **{c: "mean" for c in feat_cols}})
    return agg.sort_values("fid").reset_index(drop=True)


def _evaluate_scenario(agg: pd.DataFrame, feats: list[str], seed: int,
                       bootstrap: int = 1000) -> dict:
    """R²/RMSE OOF por bloco para um cenário + melhor modelo (média sobre folds)."""
    rng = np.random.default_rng(seed)
    best = None
    for kind in MODELS:
        preds, ytrue = [], []
        for fold in sorted(agg.block.unique()):
            tr, te = agg.block != fold, agg.block == fold
            if len(feats) == 0 or int(tr.sum()) < 3:
                continue
            pipe, grid = _make_model(kind, seed + fold, len(feats), int(tr.sum()))
            search = GridSearchCV(pipe, grid, cv=GroupKFold(3), scoring="r2", n_jobs=-1)
            search.fit(agg.loc[tr, feats], agg.loc[tr, "y"], groups=agg.loc[tr, "block"])
            preds.extend(search.predict(agg.loc[te, feats]).ravel())
            ytrue.extend(agg.loc[te, "y"].tolist())
        if not preds:
            continue
        y, p = np.array(ytrue), np.array(preds)
        m = _metrics(y, p)
        lo, hi = _bootstrap_r2(y, p, rng, bootstrap)
        row = {"scenario": None, "model": kind, **m, "r2_lo": lo, "r2_hi": hi}
        if best is None or row["r2"] > best["r2"]:
            best = row
    return best


def downstream(cfg_out: Path, folds_root: Path, table: Path, target: str,
               seed: int = 42, bootstrap: int = 1000) -> pd.DataFrame:
    """Ajusta modelos downstream sobre features OOF por parcela (5 cenários)."""
    agg = _load_fold_features(folds_root, cfg_out, table, target, seed)
    cols = sorted({c.split("_", 1)[1] for c in agg.columns
                   if c.startswith(("veg_", "real_", "fake_"))})
    scenarios = {
        "veg": [f"veg_{c}" for c in cols],
        "real": [f"real_{c}" for c in cols],
        "fake": [f"fake_{c}" for c in cols],
        "fusion_veg": [f"veg_{c}" for c in cols] + [f"fake_{c}" for c in cols],
        "fusion_real": [f"real_{c}" for c in cols] + [f"fake_{c}" for c in cols],
    }
    rows = []
    for scenario, feats in scenarios.items():
        best = _evaluate_scenario(agg, feats, seed, bootstrap)
        if best is None:
            continue
        best = dict(best)
        best["scenario"] = scenario
        rows.append(best)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CLI

def _sweep(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds_root = out / "folds"
    prepare_folds(Path(args.stage4), folds_root, veg=args.veg, rep=args.rep)
    seeds = args.seeds

    all_summary = []
    for cfg_name, attr_names in available_configs():
        print(f"\n===== Configuração: {cfg_name} -> {attr_names} =====")
        for seed in seeds:
            cfg_out = out / cfg_name / f"seed{seed}"
            cfg_out.mkdir(parents=True, exist_ok=True)
            l1s, ndvis = [], []
            for b in BLOCKS:
                block_out = cfg_out / f"block{b}"
                ckpt = train_fold(folds_root / f"block{b}", block_out, attr_names,
                                  epochs=args.epochs, batch_size=args.batch_size,
                                  seed=seed, save_freq=args.save_freq)
                infer_fold(folds_root / f"block{b}", ckpt, block_out, attr_names, phase="val")
                mp = pd.read_csv(block_out / "infer" / "metrics_parcelas.csv")
                l1s.append(mp.l1.mean())
                ndvis.append(mp.ndvi_error.mean())
            down = downstream(cfg_out, folds_root, Path(args.table), args.target,
                              seed=seed, bootstrap=args.bootstrap)
            down.to_csv(cfg_out / "downstream.csv", index=False)
            for _, r in down.iterrows():
                all_summary.append({
                    "cfg": cfg_name,
                    "attr_names": ",".join(attr_names),
                    "n_cond": len(attr_names),
                    "seed": seed,
                    "l1": float(np.mean(l1s)),
                    "ndvi_error": float(np.mean(ndvis)),
                    "scenario": r["scenario"],
                    "model": r["model"],
                    "r2": r["r2"], "rmse": r["rmse"],
                    "r2_lo": r["r2_lo"], "r2_hi": r["r2_hi"],
                })
            s = pd.DataFrame(all_summary)
            s.to_csv(out / "summary.csv", index=False)

    # Agrega por (cfg, scenario): média e desvio do R² entre seeds
    s = pd.DataFrame(all_summary)
    if not s.empty:
        agg = s.groupby(["cfg", "attr_names", "n_cond", "scenario", "model"], as_index=False).agg(
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            r2_lo=("r2_lo", "mean"), r2_hi=("r2_hi", "mean"),
            rmse_mean=("rmse", "mean"), l1=("l1", "mean"), ndvi_error=("ndvi_error", "mean"),
            seeds=("seed", "count"))
        agg.to_csv(out / "summary_aggregated.csv", index=False)
        # Tabela chave por config: melhor cenário fusion_veg e veg baseline
        pivot = agg[agg.scenario.isin(["veg", "fusion_veg"])].pivot_table(
            index=["cfg", "attr_names", "n_cond"], columns="scenario", values="r2_mean")
        pivot = pivot.sort_values("fusion_veg", ascending=False)
        pivot.to_csv(out / "gan_value.csv")
        print("\n=== ΔR² (fusion_veg − veg) por configuração (média sobre seeds) ===")
        for idx, row in pivot.iterrows():
            dv = row.get("fusion_veg", np.nan) - row.get("veg", np.nan)
            print(f"{idx[0]:24s} n_cond={idx[2]:d}  veg={row.get('veg', np.nan):.3f}  "
                  f"fusion_veg={row.get('fusion_veg', np.nan):.3f}  ΔR²={dv:+.3f}")

    (out / "meta.json").write_text(json.dumps({
        "stage4": str(args.stage4), "table": str(args.table), "target": args.target,
        "epochs": args.epochs, "seeds": seeds, "bootstrap": args.bootstrap,
        "configs": [{"name": n, "attrs": list(a)} for n, a in available_configs()],
    }, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-folds")
    p.add_argument("--stage4", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--veg", nargs="+", default=["V8", "V13"])
    p.add_argument("--rep", nargs="+", default=["R2", "R5"])

    p = sub.add_parser("sweep")
    p.add_argument("--stage4", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--target", default="Biomassa")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--veg", nargs="+", default=["V8", "V13"])
    p.add_argument("--rep", nargs="+", default=["R2", "R5"])
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.command == "prepare-folds":
        prepare_folds(Path(args.stage4), Path(args.out), veg=args.veg, rep=args.rep)
    elif args.command == "sweep":
        _sweep(args)


if __name__ == "__main__":
    main()