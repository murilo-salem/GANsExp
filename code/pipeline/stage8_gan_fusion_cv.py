#!/usr/bin/env python3
"""Produtividade OOF com fusão de vegetativo real, GAN e dose por bloco.

Cada checkpoint deve ter sido treinado sem o bloco de validação correspondente. Para cada
parcela, o script usa V8/V13 reais e as duas imagens reprodutivas geradas para formar uma
única linha. A seleção PLSR/ElasticNet/ExtraTrees ocorre somente nos três blocos de treino.
"""
from __future__ import annotations

import argparse
import itertools
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "pytorch-CycleGAN-and-pix2pix"
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parents[0] / "analysis")]
from data.phenology_dataset import N_COND, attribute_channels  # noqa: E402
from models.networks import define_G  # noqa: E402
from stage6_features_ortho import parcel_features  # noqa: E402


def load_g(path: str, device: torch.device):
    model = define_G(3 + N_COND, 3, 64, "unet_256", norm="batch")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def synth(model, refl, device):
    x = np.concatenate([refl, attribute_channels(refl)], axis=-1)
    t = torch.from_numpy(x * 2 - 1).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return np.clip((model(t)[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)


def vector(refl, columns):
    f = parcel_features(refl)
    return np.array([f[c] for c in columns], dtype=float)


def make_model(kind, seed, p, n):
    ks = sorted({min(k, p) for k in (3, 5, 8)})
    common = [("impute", SimpleImputer(strategy="median")), ("select", SelectKBest(f_regression))]
    if kind == "plsr":
        pipe = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] + [("model", PLSRegression())])
        grid = [{"select__k": [k], "model__n_components": [c]}
                for k in ks for c in range(1, min(3, k, n - 2) + 1)]
    elif kind == "elasticnet":
        pipe = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] +
                        [("model", ElasticNet(max_iter=20000, random_state=seed))])
        grid = [{"select__k": [k], "model__alpha": [a], "model__l1_ratio": [r]}
                for k, a, r in itertools.product(ks, (0.01, 0.1, 1.0), (0.1, 0.5, 0.9))]
    else:
        pipe = Pipeline(common + [("model", ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1))])
        grid = [{"select__k": [k], "model__min_samples_leaf": [leaf], "model__max_features": [mf]}
                for k, leaf, mf in itertools.product(ks, (1, 2, 4), (0.5, 1.0))]
    return pipe, grid


def metrics(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {"r2": float(r2_score(y, p)), "rmse": rmse, "mae": float(mean_absolute_error(y, p)),
            "rpd": float(np.std(y, ddof=1) / rmse), "bias": float(np.mean(p - y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", required=True); ap.add_argument("--table", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="diretório contendo pheno_2324_cv_bN/2000_net_G.pth")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(Path(args.stage4) / "manifest.csv"); man.stage = man.stage.str.upper()
    npy = {(int(r.fid), r.stage): Path(args.stage4) / r.npy for r in man.itertuples()}
    tbl = pd.read_excel(args.table); tbl.Estagio = tbl.Estagio.astype(str).str.upper()
    ymap = {(str(int(r.Dose_N)), int(r.Bloco)): float(r.Produtividade) for r in tbl.itertuples()
            if r.Estagio == "R2" and pd.notna(r.Produtividade)}
    sample = np.load(next(iter(npy.values())))
    cols = sorted(k for k in parcel_features(sample) if k.startswith("glcm") or k in {"NDVI", "NDRE", "CIrededge", "SAVI"})
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for block in sorted(man.bloco.unique()):
        ckpt = Path(args.ckpt_dir) / f"pheno_2324_cv_b{block}" / "2000_net_G.pth"
        model = load_g(str(ckpt), dev)
        for fid, g in man.groupby("fid"):
            dose, parcel_block = str(g.dose_n.iloc[0]), int(g.bloco.iloc[0])
            # O checkpoint deste laço foi treinado sem ``block``.  Gerar as
            # demais parcelas aqui criaria quatro versões da mesma observação
            # e faria o ajuste posterior reaproveitar o bloco de teste.
            if parcel_block != int(block):
                continue
            if (dose, parcel_block) not in ymap:
                continue
            veg = [np.load(npy[(int(fid), stage)]) for stage in ("V8", "V13")]
            v = np.mean([vector(x, cols) for x in veg], axis=0)
            s = np.mean([vector(synth(model, x, dev), cols) for x in veg], axis=0)
            row = {"fid": int(fid), "block": parcel_block, "y": ymap[(dose, parcel_block)],
                   "fold": int(block), "dose": float(dose)}
            row.update({f"veg_{c}": z for c, z in zip(cols, v)})
            row.update({f"gan_{c}": z for c, z in zip(cols, s)})
            rows.append(row)
    data = pd.DataFrame(rows)
    scenarios = {"veg_dose": [c for c in data if c.startswith("veg_")] + ["dose"],
                 "gan_dose": [c for c in data if c.startswith("gan_")] + ["dose"],
                 "fusion_dose": [c for c in data if c.startswith(("veg_", "gan_"))] + ["dose"]}
    all_pred, summary = [], []
    for scenario, features in scenarios.items():
        for kind in ("plsr", "elasticnet", "extratrees"):
            preds, inners = [], []
            for fold in sorted(data.block.unique()):
                tr, te = data.block != fold, data.block == fold
                pipe, grid = make_model(kind, args.seed + fold, len(features), int(tr.sum()))
                search = GridSearchCV(pipe, grid, cv=GroupKFold(3), scoring="r2", n_jobs=1)
                search.fit(data.loc[tr, features], data.loc[tr, "y"], groups=data.loc[tr, "block"])
                p = search.predict(data.loc[te, features]).ravel()
                q = data.loc[te, ["fid", "block", "y"]].copy(); q["pred"] = p
                q["scenario"], q["model"], q["inner_r2"] = scenario, kind, search.best_score_
                preds.append(q); inners.append(search.best_score_)
            q = pd.concat(preds); mt = metrics(q.y, q.pred)
            summary.append({"scenario": scenario, "model": kind, "mean_inner_r2": np.mean(inners), **mt})
            all_pred.append(q)
    pd.concat(all_pred).to_csv(out / "oof_predictions.csv", index=False)
    s = pd.DataFrame(summary).sort_values(["scenario", "mean_inner_r2"], ascending=[True, False])
    s.to_csv(out / "summary.csv", index=False); print(s.to_string(index=False))


if __name__ == "__main__": main()
