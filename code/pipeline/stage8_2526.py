#!/usr/bin/env python3
"""Cenário 2025/26: V10/V13 -> GAN -> R1, com alvo real CHL_total por parcela.

O split vem de ``pairs.csv``. As duas transições de uma mesma parcela são
agregadas antes de qualquer métrica: a unidade independente é a parcela.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "pytorch-CycleGAN-and-pix2pix"
sys.path[:0] = [str(REPO), str(HERE), str(HERE.parents[0] / "analysis"),
                str(HERE.parents[1] / "src")]
from data.phenology_dataset import DEFAULT_ATTR_NAMES  # noqa: E402
from models.networks import define_G  # noqa: E402
from plsr_kfold import metrics  # noqa: E402
from stage6_features_ortho import parcel_features  # noqa: E402
from milho_experiment.indices import attribute_channels  # noqa: E402


def load_generator(path, device, ngf=64, netg="unet_256", norm="batch",
                   attr_names=DEFAULT_ATTR_NAMES):
    model = define_G(3 + len(attr_names), 3, ngf, netg, norm=norm)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def synth(model, refl, device, attr_names=DEFAULT_ATTR_NAMES, attr_normalize="image"):
    x = np.concatenate([refl, attribute_channels(refl, names=attr_names,
                                                  normalize=attr_normalize)], axis=-1)
    t = torch.from_numpy(x * 2 - 1).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        return np.clip((model(t)[0].permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)


def texture_vector(refl, columns):
    values = parcel_features(refl)
    return np.array([values[c] for c in columns], dtype=np.float64)


def bootstrap_r2(y, yhat, rng, n=1000):
    vals = []
    for _ in range(n):
        take = rng.integers(0, len(y), len(y))
        if np.unique(y[take]).size > 1:
            vals.append(metrics(y[take], yhat[take])["R2"])
    return tuple(np.quantile(vals, [0.025, 0.975])) if vals else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nc", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gan-attr-names", type=str, default=",".join(DEFAULT_ATTR_NAMES),
                    help="nomes dos canais condicionais usados no treino")
    ap.add_argument("--gan-attr-normalize", type=str, default="image",
                    choices=["image", "global", "none"])
    args = ap.parse_args()
    attr_names = tuple(n.strip() for n in args.gan_attr_names.split(",") if n.strip())

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(args.stage4) / "manifest.csv")
    pairs = pd.read_csv(args.pairs)
    npy = {(int(r.fid), r.stage): Path(args.stage4) / r.npy for r in manifest.itertuples()}
    r1 = manifest[manifest.stage == "R1"].set_index("fid")
    if r1.chl_total.isna().any():
        raise ValueError("CHL_total R1 ausente no manifest")

    example = np.load(next(iter(npy.values())))
    columns = sorted(k for k in parcel_features(example) if k.startswith("glcm"))
    train_ids = sorted(pairs.loc[pairs.split == "train", "fid"].unique())
    val_pairs = pairs[pairs.split == "val"].copy()
    val_ids = sorted(val_pairs.fid.unique())
    if set(train_ids) & set(val_ids):
        raise ValueError("Vazamento: parcela aparece em treino e validação")

    xtrain = np.vstack([texture_vector(np.load(npy[(fid, "R1")]), columns) for fid in train_ids])
    ytrain = r1.loc[train_ids, "chl_total"].to_numpy(float)
    nc = min(args.nc, xtrain.shape[1], len(train_ids) - 1)
    model_y = make_pipeline(StandardScaler(), PLSRegression(n_components=nc)).fit(xtrain, ytrain)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = load_generator(args.ckpt, device, attr_names=attr_names)

    rows = []
    for p in val_pairs.itertuples():
        veg, real = np.load(npy[(p.fid, p.veg)]), np.load(npy[(p.fid, p.rep)])
        fake = synth(generator, veg, device, attr_names=attr_names,
                     attr_normalize=args.gan_attr_normalize)
        rows.append({
            "pair": p.pair, "fid": p.fid, "veg": p.veg, "rep": p.rep,
            "y_meas": float(r1.loc[p.fid, "chl_total"]),
            "y_pred_real": float(model_y.predict(texture_vector(real, columns)[None])[0]),
            "y_pred_synth": float(model_y.predict(texture_vector(fake, columns)[None])[0]),
            "l1_img": float(np.mean(np.abs(fake - real))),
        })
    by_pair = pd.DataFrame(rows)
    by_parcel = by_pair.groupby("fid", as_index=False).agg(
        y_meas=("y_meas", "first"), y_pred_real=("y_pred_real", "mean"),
        y_pred_synth=("y_pred_synth", "mean"), l1_img=("l1_img", "mean"), pairs=("pair", "count"))
    by_pair.to_csv(out / "scenarios_chl_pairs.csv", index=False)
    by_parcel.to_csv(out / "scenarios_chl_parcelas.csv", index=False)
    rng = np.random.default_rng(args.seed)
    m_real = metrics(by_parcel.y_meas.values, by_parcel.y_pred_real.values)
    m_syn = metrics(by_parcel.y_meas.values, by_parcel.y_pred_synth.values)
    ci_real = bootstrap_r2(by_parcel.y_meas.values, by_parcel.y_pred_real.values, rng, args.bootstrap)
    ci_syn = bootstrap_r2(by_parcel.y_meas.values, by_parcel.y_pred_synth.values, rng, args.bootstrap)
    report = "\n".join([
        "=== ETAPA 8 2025/26 — CENÁRIO V10/V13 -> R1 (CHL_total) ===",
        f"Treino: {len(train_ids)} parcelas | validação: {len(by_parcel)} parcelas ({len(by_pair)} pares) | nc={nc}",
        f"R1 real: R²={m_real['R2']:.3f} RMSE={m_real['RMSE']:.2f} | IC95% bootstrap R²=[{ci_real[0]:.3f}, {ci_real[1]:.3f}]",
        f"R1 sintético: R²={m_syn['R2']:.3f} RMSE={m_syn['RMSE']:.2f} | IC95% bootstrap R²=[{ci_syn[0]:.3f}, {ci_syn[1]:.3f}]",
        f"L1 médio por parcela={by_parcel.l1_img.mean():.4f}",
        "Unidade independente: parcela; pares V10/V13 foram agregados antes das métricas.",
    ])
    (out / "report_chl_total.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
