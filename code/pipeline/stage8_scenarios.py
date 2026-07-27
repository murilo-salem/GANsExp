#!/usr/bin/env python3
"""ETAPA 8 — Cenários futuros (projeções): veg real -> GAN -> reprodutivo sintético -> PLSR.

Amarra a pipeline ponta-a-ponta (capstone do diagrama):
  1. PLSR é treinado nos atributos do estágio REPRODUTIVO REAL (parcelas de treino) -> Y agronômico.
  2. Para cada parcela de validação: pega o estágio VEGETATIVO REAL, gera o REPRODUTIVO SINTÉTICO
     com o gerador Pix2Pix treinado, extrai atributos do sintético e prediz o Y com o PLSR.
  3. Compara: predição a partir do (i) sintético vs (ii) reprodutivo REAL vs (iii) Y medido —
     mede o quanto o dado sintético preserva a capacidade preditiva ("validação com dados reais").

Reusa: gerador de stage5_infer (define_G + attribute_channels), parcel_features de
stage6_features_ortho, e o alvo da tabela normalizada.

Uso:
    python3 code/pipeline/stage8_scenarios.py \
        --stage4 code/pipeline/out/stage4_2324 \
        --pairs pytorch-CycleGAN-and-pix2pix/datasets/pheno_2324/pairs.csv \
        --ckpt pytorch-CycleGAN-and-pix2pix/checkpoints/pheno_2324/latest_net_G.pth \
        --table "data/Safra2023a2024/Tabela de dados/parametros_2324_normalizado.xlsx" \
        --target Biomassa --out code/pipeline/out/stage8_2324
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "pytorch-CycleGAN-and-pix2pix"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "analysis"))
from models.networks import define_G                        # noqa: E402
from data.phenology_dataset import attribute_channels, N_COND  # noqa: E402
from stage6_features_ortho import parcel_features            # noqa: E402
from plsr_kfold import metrics                               # noqa: E402

TEX = lambda d: {k: v for k, v in d.items() if k.startswith("glcm")}


def load_generator(ckpt, dev, ngf=64, netG="unet_256", norm="batch"):
    G = define_G(3 + N_COND, 3, ngf, netG, norm=norm)
    G.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    return G.to(dev).eval()


def synth_reproductive(G, veg, dev):
    """veg (H,W,3)[0,1] -> reprodutivo sintético (H,W,3)[0,1]."""
    A = np.concatenate([veg, attribute_channels(veg)], -1)
    t = torch.from_numpy(A * 2 - 1).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        fake = G(t)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip((fake + 1) / 2, 0, 1)


def feat_vec(refl, cols):
    f = TEX(parcel_features(refl))
    return np.array([f[c] for c in cols], np.float64)


def false_color(refl):
    ch = [refl[..., 2], refl[..., 0], refl[..., 1]]
    o = [np.clip((c - np.nanpercentile(c, 2)) /
                 (np.nanpercentile(c, 98) - np.nanpercentile(c, 2) + 1e-8), 0, 1) for c in ch]
    return (np.stack(o, -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--target", default="Biomassa")
    ap.add_argument("--nc", type=int, default=6, help="componentes do PLSR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); (out / "paineis").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    man = pd.read_csv(Path(args.stage4) / "manifest.csv")
    man["stage"] = man["stage"].str.upper()
    npy = {(int(r.fid), r.stage): Path(args.stage4) / r.npy for r in man.itertuples()}

    tbl = pd.read_excel(args.table); tbl["Estagio"] = tbl["Estagio"].astype(str).str.upper()
    ycol = args.target
    ymap = {(str(int(r.Dose_N)), int(r.Bloco), r.Estagio): float(getattr(r, ycol))
            for r in tbl.itertuples() if pd.notna(getattr(r, ycol))}

    # colunas de textura (fixa a ordem a partir de uma amostra)
    sample_refl = np.load(next(iter(npy.values())))
    cols = sorted(TEX(parcel_features(sample_refl)).keys())

    pairs = pd.read_csv(args.pairs)
    rep_stages = sorted(pairs.rep.unique())

    # ---- 1) treina PLSR no REPRODUTIVO REAL de parcelas de TREINO -----------------
    Xtr, ytr = [], []
    for r in man[(man.phase == "reprodutivo")].itertuples():
        key = (str(r.dose_n), int(r.bloco), r.stage)
        # treino = parcelas fora do split de validação da GAN
        is_val = ((pairs.fid == r.fid) & (pairs.split == "val")).any()
        if is_val or key not in ymap:
            continue
        Xtr.append(feat_vec(np.load(Path(args.stage4) / r.npy), cols)); ytr.append(ymap[key])
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    plsr = make_pipeline(StandardScaler(), PLSRegression(n_components=min(args.nc, Xtr.shape[1])))
    plsr.fit(Xtr, ytr)
    print(f"PLSR treinado no reprodutivo real: {len(ytr)} amostras, alvo={ycol}, nc={args.nc}")

    # ---- 2) projeta nas parcelas de VALIDAÇÃO ------------------------------------
    G = load_generator(args.ckpt, dev)
    rows = []
    for pr in pairs[pairs.split == "val"].itertuples():
        veg = np.load(npy[(pr.fid, pr.veg)]); rep_real = np.load(npy[(pr.fid, pr.rep)])
        rep_syn = synth_reproductive(G, veg, dev)
        key = (str(pr.dose_n), int(pr.bloco), pr.rep)
        if key not in ymap:
            continue
        y_meas = ymap[key]
        y_syn = float(plsr.predict(feat_vec(rep_syn, cols)[None])[0])
        y_rl = float(plsr.predict(feat_vec(rep_real, cols)[None])[0])
        rows.append(dict(pair=pr.pair, fid=pr.fid, veg=pr.veg, rep=pr.rep,
                         y_meas=y_meas, y_pred_real=y_rl, y_pred_synth=y_syn,
                         l1_img=float(np.mean(np.abs(rep_syn - rep_real)))))
        panel = np.concatenate([false_color(veg), false_color(rep_syn),
                                false_color(rep_real)], axis=1)
        Image.fromarray(panel).save(out / "paineis" / f"{pr.pair}.png")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"scenarios_{ycol}.csv", index=False)

    m_syn = metrics(df.y_meas.values, df.y_pred_synth.values)
    m_rl = metrics(df.y_meas.values, df.y_pred_real.values)
    lines = [
        "=== ETAPA 8 — CENÁRIOS (projeção veg->GAN->reprodutivo sintético->PLSR) ===",
        f"Alvo: {ycol}  |  parcelas de validação: {len(df)}",
        f"Erro de imagem sintético vs real (L1): {df.l1_img.mean():.4f}",
        "",
        "Predição do Y medido a partir de:",
        f"  REPRODUTIVO REAL     -> R²={m_rl['R2']:.3f}  RMSE={m_rl['RMSE']:.2f}  RPD={m_rl['RPD']:.2f}",
        f"  REPRODUTIVO SINTÉTICO-> R²={m_syn['R2']:.3f}  RMSE={m_syn['RMSE']:.2f}  RPD={m_syn['RPD']:.2f}",
        "",
        f"Gap de predição (sintético vs real): "
        f"MAE={np.mean(np.abs(df.y_pred_synth - df.y_pred_real)):.2f}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out / f"report_{ycol}.txt").write_text(report + "\n")
    print(f"\n-> {out}/report_{ycol}.txt (+ scenarios_{ycol}.csv, paineis/)")


if __name__ == "__main__":
    main()
