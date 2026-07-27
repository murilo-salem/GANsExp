#!/usr/bin/env python3
"""ETAPA 5 (parte 2) — Inferência: gera o estágio REPRODUTIVO sintético a partir do vegetativo.

Carrega o gerador Pix2Pix treinado (input_nc=4: 3 bandas + clorofila) e, para cada par de
validação, produz o reprodutivo sintético. Salva um painel PNG [veg | sintético | real] e mede
o erro (L1 e erro de NDVI médio) contra o reprodutivo real — validação quantitativa da GAN.

Uso:
    python3 code/pipeline/stage5_infer.py \
        --dataroot pytorch-CycleGAN-and-pix2pix/datasets/pheno_2324 --phase val \
        --ckpt pytorch-CycleGAN-and-pix2pix/checkpoints/pheno_2324/latest_net_G.pth \
        --out code/pipeline/out/stage5_infer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2] / "pytorch-CycleGAN-and-pix2pix"
sys.path.insert(0, str(REPO))
from models.networks import define_G                       # noqa: E402
from data.phenology_dataset import attribute_channels, N_COND  # noqa: E402


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
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G = define_G(3 + N_COND, 3, args.ngf, args.netG, norm=args.norm)
    state = torch.load(args.ckpt, map_location=dev, weights_only=True)
    G.load_state_dict(state)
    G.to(dev).eval()

    in_dir = Path(args.dataroot) / args.phase / "input"
    tg_dir = Path(args.dataroot) / args.phase / "target"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    l1s, ndvi_errs = [], []
    for p in sorted(in_dir.glob("*.npy")):
        veg = np.load(p).astype(np.float32)                 # (H,W,3) [0,1]
        rep = np.load(tg_dir / p.name).astype(np.float32)
        A = np.concatenate([veg, attribute_channels(veg)], -1)
        t = torch.from_numpy(A * 2 - 1).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            fake = G(t)[0].permute(1, 2, 0).cpu().numpy()    # [-1,1]
        fake = np.clip((fake + 1) / 2, 0, 1)                 # -> [0,1]

        l1s.append(float(np.mean(np.abs(fake - rep))))
        ndvi_errs.append(float(np.abs(np.mean(ndvi(fake)) - np.mean(ndvi(rep)))))

        panel = np.concatenate([false_color(veg), false_color(fake), false_color(rep)], axis=1)
        Image.fromarray(panel).save(out / f"{p.stem}.png")
        np.save(out / f"{p.stem}_fake.npy", fake.astype(np.float32))

    print(f"== inferência: {len(l1s)} pares  (painel = veg | sintético | real)")
    print(f"   L1 médio (sintético vs real)   = {np.mean(l1s):.4f}")
    print(f"   |ΔNDVI| médio (sintético-real)  = {np.mean(ndvi_errs):.4f}")
    print(f"   painéis + .npy -> {out}")


if __name__ == "__main__":
    main()
