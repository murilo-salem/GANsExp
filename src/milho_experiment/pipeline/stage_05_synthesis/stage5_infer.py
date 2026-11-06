#!/usr/bin/env python3
"""ETAPA 5 (parte 2) — Inferência: gera o estágio REPRODUTIVO sintético a partir do vegetativo.

Carrega o gerador Pix2Pix treinado (input_nc=4: 3 bandas + clorofila) e, para cada par de
validação, produz o reprodutivo sintético. Salva um painel PNG [veg | sintético | real] e mede
o erro (L1 e erro de NDVI médio) contra o reprodutivo real — validação quantitativa da GAN.

Uso:
    python3 code/pipeline/stage5_infer.py \
        --dataroot artifacts/archive/legacy/gan/datasets/pheno_2324 --phase val \
        --ckpt artifacts/archive/legacy/gan/checkpoints/pheno_2324/latest_net_G.pth \
        --out artifacts/runs/2324_stage5_infer/results
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from milho_experiment.pix2pix import enable_imports
enable_imports()
from models.networks import define_G                       # noqa: E402
from data.phenology_dataset import DEFAULT_ATTR_NAMES       # noqa: E402
from milho_experiment.indices import attribute_channels      # noqa: E402


def false_color(refl: np.ndarray) -> np.ndarray:
    """(H,W,3)[Red,RE,NIR] em [0,1] -> RGB falsa-cor uint8 (NIR,Red,RE) com stretch."""
    chans = [refl[..., 2], refl[..., 0], refl[..., 1]]
    out = []
    for c in chans:
        lo, hi = np.nanpercentile(c, [2, 98])
        out.append(np.clip((c - lo) / (hi - lo + 1e-8), 0, 1))
    return (np.stack(out, -1) * 255).astype(np.uint8)


def ndvi(refl):
    r, n = refl[..., 0], refl[..., 2]
    return (n - r) / (n + r + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--phase", default="val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--netG", default="unet_256")
    ap.add_argument("--norm", default="batch")
    ap.add_argument("--pairs", help="pairs.csv; quando informado, agrega métricas por parcela")
    ap.add_argument("--gan-attr-names", type=str, default=",".join(DEFAULT_ATTR_NAMES),
                    help="nomes dos canais condicionais usados no treino, separados por vírgula")
    ap.add_argument("--gan-attr-normalize", type=str, default="image",
                    choices=["image", "global", "none"],
                    help="modo de normalização dos canais condicionais usado no treino")
    args = ap.parse_args()

    attr_names = tuple(n.strip() for n in args.gan_attr_names.split(",") if n.strip())
    n_cond = len(attr_names)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G = define_G(3 + n_cond, 3, args.ngf, args.netG, norm=args.norm)
    state = torch.load(args.ckpt, map_location=dev, weights_only=True)
    G.load_state_dict(state)
    G.to(dev).eval()

    in_dir = Path(args.dataroot) / args.phase / "input"
    tg_dir = Path(args.dataroot) / args.phase / "target"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    for p in sorted(in_dir.glob("*.npy")):
        veg = np.load(p).astype(np.float32)                 # (H,W,3) [0,1]
        rep = np.load(tg_dir / p.name).astype(np.float32)
        A = np.concatenate([veg, attribute_channels(veg, names=attr_names,
                                                          normalize=args.gan_attr_normalize)], -1)
        t = torch.from_numpy(A * 2 - 1).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            fake = G(t)[0].permute(1, 2, 0).cpu().numpy()    # [-1,1]
        fake = np.clip((fake + 1) / 2, 0, 1)                 # -> [0,1]

        pair_rows.append({
            "pair": p.stem,
            "l1": float(np.mean(np.abs(fake - rep))),
            "ndvi_error": float(np.abs(np.mean(ndvi(fake)) - np.mean(ndvi(rep)))),
        })

        panel = np.concatenate([false_color(veg), false_color(fake), false_color(rep)], axis=1)
        Image.fromarray(panel).save(out / f"{p.stem}.png")
        np.save(out / f"{p.stem}_fake.npy", fake.astype(np.float32))

    metrics = pd.DataFrame(pair_rows)
    if args.pairs:
        pairs = pd.read_csv(args.pairs)
        metrics = metrics.merge(pairs[["pair", "fid", "bloco", "veg", "rep"]], on="pair",
                                how="left", validate="one_to_one")
    metrics.to_csv(out / "metrics_pairs.csv", index=False)
    print(f"== inferência: {len(metrics)} pares  (painel = veg | sintético | real)")
    print(f"   L1 médio (sintético vs real)   = {metrics.l1.mean():.4f}")
    print(f"   |ΔNDVI| médio (sintético-real)  = {metrics.ndvi_error.mean():.4f}")
    if "fid" in metrics and metrics.fid.notna().any():
        by_parcel = metrics.groupby("fid", as_index=False)[["l1", "ndvi_error"]].mean()
        by_parcel.to_csv(out / "metrics_parcelas.csv", index=False)
        print(f"   por parcela independente: {len(by_parcel)} | L1={by_parcel.l1.mean():.4f} | "
              f"|ΔNDVI|={by_parcel.ndvi_error.mean():.4f}")
    print(f"   painéis + .npy -> {out}")


if __name__ == "__main__":
    main()
