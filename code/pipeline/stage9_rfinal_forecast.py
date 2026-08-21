#!/usr/bin/env python3
"""Projeção causal de R2 para R5 e biomassa entre safras.

O estágio separa explicitamente fonte (22/23, onde R2 e R5 são observados) e
alvo (23/24, onde somente R2 entra no treino).  Assim, R5 23/24 jamais pode
entrar por acidente na GAN ou no PLSR.  Há três adaptações sem alvo:

* ``stats_aug``: equalização robusta por banda + perturbação radiométrica;
* ``dann``: adversário de domínio no encoder da GAN temporal;
* ``cyclegan``: tradução não pareada de R2 alvo para o domínio fonte.

Fluxo típico (os dois ``stage4`` devem ter sido executados antes)::

  python code/pipeline/stage9_rfinal_forecast.py prepare --source-stage4 ... --target-stage4 ... --out ...
  python code/pipeline/stage9_rfinal_forecast.py train --dataset ... --method dann --out ...
  python code/pipeline/stage9_rfinal_forecast.py predict --dataset ... --method dann --checkpoint ... --out ...
  python code/pipeline/stage9_rfinal_forecast.py evaluate --dataset ... --predictions ... \
      --source-table ... --target-table ... --out ...

``evaluate`` é o único subcomando autorizado a abrir R5 23/24; ele o usa
somente para calcular métricas depois de os checkpoints terem sido definidos.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geo import rrenir_indices  # noqa: E402
from stage6_features_ortho import parcel_features  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cond_channels(x: np.ndarray) -> np.ndarray:
    """Os quatro canais de condição usados pela GAN fenológica existente."""
    red, rededge, nir = (x[..., i] for i in range(3))
    eps = 1e-8
    raw = [nir / (rededge + eps) - 1, (nir-red)/(nir+red+eps),
           (nir-rededge)/(nir+rededge+eps), 1.5*(nir-red)/(nir+red+0.5+eps)]
    out = []
    for item in raw:
        lo, hi = np.nanpercentile(item, [2, 98])
        out.append(np.clip((item-lo)/(hi-lo+eps), 0, 1))
    return np.stack(out, axis=-1).astype(np.float32)


def to_cond_tensor(x: np.ndarray) -> Tensor:
    a = np.concatenate([x, cond_channels(x)], axis=-1)
    return torch.from_numpy(a * 2 - 1).permute(2, 0, 1).contiguous()


def clean_stage(manifest: Path, stage: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    need = {"fid", "stage", "npy", "dose_n", "bloco"}
    if not need.issubset(df.columns):
        raise ValueError(f"manifesto inválido {manifest}; faltam {sorted(need-set(df.columns))}")
    df = df[df.stage.astype(str).str.upper() == stage].copy()
    if df.empty or df.fid.duplicated().any():
        raise ValueError(f"{manifest}: estágio {stage} ausente ou duplicado por parcela")
    df["path"] = [str((manifest.parent / p).resolve()) for p in df.npy]
    if not all(Path(p).is_file() for p in df.path):
        raise ValueError(f"{manifest}: há recortes .npy ausentes")
    return df.sort_values("fid").reset_index(drop=True)


def command_prepare(args: argparse.Namespace) -> None:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src_r2 = clean_stage(Path(args.source_stage4) / "manifest.csv", "R2")
    src_r5 = clean_stage(Path(args.source_stage4) / "manifest.csv", "R5")
    tgt_r2 = clean_stage(Path(args.target_stage4) / "manifest.csv", "R2")
    # R5 alvo fica em manifesto de avaliação separado, jamais no de treino.
    tgt_r5 = clean_stage(Path(args.target_stage4) / "manifest.csv", "R5")
    src = src_r2.merge(src_r5[["fid", "path"]], on="fid", suffixes=("_r2", "_r5"), validate="one_to_one")
    src["season"] = "2022_2023"; tgt_r2["season"] = "2023_2024"
    src.to_csv(out / "source_pairs_r2_r5.csv", index=False)
    tgt_r2.to_csv(out / "target_r2_unlabeled.csv", index=False)
    tgt_r5.to_csv(out / "TARGET_R5_HELDOUT.csv", index=False)
    policy = {
        "source_supervised": "2022_2023:R2->R5",
        "target_adaptation": "2023_2024:R2 only",
        "forbidden_before_evaluate": ["TARGET_R5_HELDOUT.csv", "R5 2023/24", "Biomassa 2023/24"],
    }
    (out / "leakage_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(f"fonte: {len(src)} pares R2->R5 | alvo não rotulado: {len(tgt_r2)} R2")


@dataclass
class AffineNormalizer:
    scale: list[float]
    shift: list[float]

    @classmethod
    def fit(cls, source: list[np.ndarray], target: list[np.ndarray]) -> "AffineNormalizer":
        # Casa mediana e IQR por banda. Só R2, portanto é permitido no alvo.
        s = np.concatenate([a.reshape(-1, 3) for a in source])
        t = np.concatenate([a.reshape(-1, 3) for a in target])
        sm, tm = np.median(s, 0), np.median(t, 0)
        si = np.percentile(s, 75, 0)-np.percentile(s, 25, 0)
        ti = np.percentile(t, 75, 0)-np.percentile(t, 25, 0)
        scale = si / np.maximum(ti, 1e-5)
        return cls(scale.tolist(), (sm-scale*tm).tolist())

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x*np.asarray(self.scale)+np.asarray(self.shift), 0, 1).astype(np.float32)


class PairDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, augment: bool = False):
        self.rows, self.augment = rows.reset_index(drop=True), augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x, y = np.load(r.path_r2).astype(np.float32), np.load(r.path_r5).astype(np.float32)
        if self.augment:
            # Aumento de estilo somente na entrada; preserva R5 como alvo físico.
            gain = np.random.uniform(.92, 1.08, size=(1, 1, 3)); bias = np.random.uniform(-.025, .025, size=(1, 1, 3))
            x = np.clip(x*gain+bias, 0, 1).astype(np.float32)
            if np.random.random() < .5: x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
        return to_cond_tensor(x), torch.from_numpy(y*2-1).permute(2, 0, 1).contiguous()


class UnlabeledDataset(Dataset):
    def __init__(self, rows: pd.DataFrame): self.rows = rows.reset_index(drop=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return to_cond_tensor(np.load(self.rows.iloc[i].path).astype(np.float32))


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.alpha*grad, None


class TemporalGenerator(nn.Module):
    """U-Net compacto com bottleneck exposto para a adaptação DANN."""
    def __init__(self, channels: int = 7):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(channels, 32, 4, 2, 1), nn.LeakyReLU(.2))
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(.2))
        self.e3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(.2))
        self.mid = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.InstanceNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU())
        self.d2 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.ReLU())
        self.d1 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    def encode(self, x):
        a, b, c = self.e1(x), self.e2(self.e1(x)), None
        # evita compartilhar aleatoriamente a avaliação de e1 entre ramos
        c = self.e3(b); return a, b, c + self.mid(c)
    def forward(self, x):
        a, b, z = self.encode(x); q = self.d3(z); q = self.d2(torch.cat([q, b], 1)); return self.d1(torch.cat([q, a], 1))
    def features(self, x): return self.encode(x)[2].mean((2, 3))


class PatchDiscriminator(nn.Module):
    def __init__(self, channels: int):
        super().__init__(); self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(.2),
            nn.Conv2d(64, 1, 4, 1, 1))
    def forward(self, x): return self.net(x)


class DomainHead(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)


def gan_losses(d: nn.Module, x: Tensor, y: Tensor, fake: Tensor, bce: nn.Module) -> tuple[Tensor, Tensor]:
    real = d(torch.cat([x, y], 1)); generated = d(torch.cat([x, fake.detach()], 1))
    dl = .5*(bce(real, torch.ones_like(real))+bce(generated, torch.zeros_like(generated)))
    gl = bce(d(torch.cat([x, fake], 1)), torch.ones_like(real))
    return dl, gl


def train_temporal(dataset: Path, method: str, out: Path, epochs: int, batch_size: int, seed: int, lambda_domain: float, validation_block: int) -> None:
    seed_all(seed); out.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(dataset / "source_pairs_r2_r5.csv"); tgt = pd.read_csv(dataset / "target_r2_unlabeled.csv")
    # Um bloco inteiro de 22/23 é reservado para escolher o checkpoint.  R5
    # 23/24 continua fora deste processo.
    train_src, valid_src = src[src.bloco != validation_block], src[src.bloco == validation_block]
    if train_src.empty or valid_src.empty:
        raise ValueError(f"bloco de validação {validation_block} não separa pares fonte")
    # O normalizador é usado no baseline e como pré-condicionamento nas variantes adversariais.
    norm = AffineNormalizer.fit([np.load(p) for p in src.path_r2], [np.load(p) for p in tgt.path])
    (out / "normalizer.json").write_text(json.dumps(asdict(norm), indent=2) + "\n")
    # CycleGAN é treinada antes da GAN temporal e produz R2 alvo no domínio da fonte.
    mapper = None
    if method == "cyclegan":
        mapper = train_cyclegan(src, tgt, out, epochs=max(1, epochs // 2), batch_size=batch_size, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(PairDataset(train_src, augment=method == "stats_aug"), batch_size=batch_size, shuffle=True, drop_last=len(train_src) > batch_size)
    target_loader = DataLoader(UnlabeledDataset(tgt), batch_size=batch_size, shuffle=True, drop_last=True)
    target_iter = iter(target_loader)
    g, d = TemporalGenerator().to(device), PatchDiscriminator(10).to(device)
    head = DomainHead().to(device) if method == "dann" else None
    opt_g = torch.optim.Adam(list(g.parameters()) + ([] if head is None else list(head.parameters())), lr=2e-4, betas=(.5, .999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(.5, .999)); bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    best_l1, best_epoch, best_state = np.inf, 1, None
    for epoch in range(1, epochs+1):
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if mapper is not None:  # mapa alvo->fonte não se aplica à fonte supervisionada
                pass
            fake = g(x); dl, _ = gan_losses(d, x, y, fake, bce)
            opt_d.zero_grad(); dl.backward(); opt_d.step()
            generated = d(torch.cat([x, fake], 1))
            total = bce(generated, torch.ones_like(generated)) + 100*l1(fake, y)
            if head is not None:
                try: tx = next(target_iter)
                except StopIteration: target_iter = iter(target_loader); tx = next(target_iter)
                tx = tx.to(device)
                alpha = min(1.0, epoch / max(1, epochs//3))
                fs, ft = g.features(x), g.features(tx)
                logits = head(ReverseGradient.apply(torch.cat([fs, ft]), alpha))
                labels = torch.cat([torch.zeros(len(fs), 1, device=device), torch.ones(len(ft), 1, device=device)])
                total = total + lambda_domain*bce(logits, labels)
            opt_g.zero_grad(); total.backward(); opt_g.step(); losses.append(float(total.detach().cpu()))
        g.eval()
        with torch.no_grad():
            val_l1 = np.mean([torch.mean(torch.abs(g(to_cond_tensor(np.load(r.path_r2).astype(np.float32))[None].to(device)) - torch.from_numpy(np.load(r.path_r5).astype(np.float32)*2-1).permute(2,0,1)[None].to(device))).item() for r in valid_src.itertuples()])
        g.train()
        if val_l1 < best_l1:
            best_l1, best_epoch, best_state = float(val_l1), epoch, copy.deepcopy(g.state_dict())
        print(f"[{method}] época {epoch}/{epochs}: perda_G={np.mean(losses):.4f} val_L1={val_l1:.4f}")
    # O checkpoint entregue é treinado novamente com toda 22/23, pelo número de
    # épocas escolhido acima. Isso cumpre o uso integral da safra fonte sem usar
    # R5 23/24 para a escolha do treinamento.
    g.load_state_dict(best_state)
    final_loader = DataLoader(PairDataset(src, augment=method == "stats_aug"), batch_size=batch_size, shuffle=True, drop_last=len(src) > batch_size)
    final_d = PatchDiscriminator(10).to(device)
    final_head = DomainHead().to(device) if method == "dann" else None
    final_opt_g = torch.optim.Adam(list(g.parameters()) + ([] if final_head is None else list(final_head.parameters())), lr=2e-4, betas=(.5, .999))
    final_opt_d = torch.optim.Adam(final_d.parameters(), lr=2e-4, betas=(.5, .999)); final_target_iter = iter(target_loader)
    for epoch in range(1, best_epoch+1):
        for x, y in final_loader:
            x, y = x.to(device), y.to(device); fake = g(x); dl, _ = gan_losses(final_d, x, y, fake, bce)
            final_opt_d.zero_grad(); dl.backward(); final_opt_d.step(); generated = final_d(torch.cat([x, fake], 1))
            total = bce(generated, torch.ones_like(generated)) + 100*l1(fake, y)
            if final_head is not None:
                try: tx = next(final_target_iter)
                except StopIteration: final_target_iter = iter(target_loader); tx = next(final_target_iter)
                fs, ft = g.features(x), g.features(tx.to(device)); alpha = min(1.0, epoch/max(1, best_epoch//3))
                logits = final_head(ReverseGradient.apply(torch.cat([fs, ft]), alpha))
                labels = torch.cat([torch.zeros(len(fs), 1, device=device), torch.ones(len(ft), 1, device=device)])
                total = total + lambda_domain*bce(logits, labels)
            final_opt_g.zero_grad(); total.backward(); final_opt_g.step()
    torch.save(g.state_dict(), out / "temporal_generator.pt")
    (out / "training.json").write_text(json.dumps({"method": method, "epochs": epochs, "seed": seed, "lambda_domain": lambda_domain, "validation_block": validation_block, "best_source_val_l1": best_l1, "selected_epoch": best_epoch, "final_training": "all source R2->R5"}, indent=2)+"\n")


class ImageGenerator(nn.Module):
    """Gerador residual leve para o CycleGAN R2 alvo→fonte."""
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(3, 32, 7, 1, 3), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 3, 7, 1, 3), nn.Tanh())
    def forward(self, x): return self.net(x)


def image_loader(paths: list[str], batch: int) -> DataLoader:
    class D(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return torch.from_numpy(np.load(paths[i]).astype(np.float32)*2-1).permute(2,0,1)
    return DataLoader(D(), batch_size=batch, shuffle=True, drop_last=True)


def train_cyclegan(src: pd.DataFrame, tgt: pd.DataFrame, out: Path, epochs: int, batch_size: int, seed: int) -> ImageGenerator:
    """CycleGAN não pareada, restrita a R2 das duas safras."""
    seed_all(seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_ts, g_st = ImageGenerator().to(dev), ImageGenerator().to(dev)
    d_s, d_t = PatchDiscriminator(3).to(dev), PatchDiscriminator(3).to(dev)
    og = torch.optim.Adam(list(g_ts.parameters())+list(g_st.parameters()), lr=2e-4, betas=(.5,.999)); od = torch.optim.Adam(list(d_s.parameters())+list(d_t.parameters()), lr=2e-4, betas=(.5,.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss(); si, ti = iter(image_loader(src.path_r2.tolist(), batch_size)), iter(image_loader(tgt.path.tolist(), batch_size))
    for _ in range(epochs):
        try: s = next(si)
        except StopIteration: si = iter(image_loader(src.path_r2.tolist(), batch_size)); s = next(si)
        try: t = next(ti)
        except StopIteration: ti = iter(image_loader(tgt.path.tolist(), batch_size)); t = next(ti)
        s, t = s.to(dev), t.to(dev); ts, st = g_ts(t), g_st(s)
        dl = sum(.5*(bce(d(real), torch.ones_like(d(real)))+bce(d(fake.detach()), torch.zeros_like(d(fake)))) for d,real,fake in [(d_s,s,ts),(d_t,t,st)])
        od.zero_grad(); dl.backward(); od.step()
        adv = bce(d_s(ts), torch.ones_like(d_s(ts)))+bce(d_t(st), torch.ones_like(d_t(st)))
        cyc = l1(g_st(ts), t)+l1(g_ts(st), s); og.zero_grad(); (adv+10*cyc).backward(); og.step()
    torch.save(g_ts.state_dict(), out / "cyclegan_target_to_source.pt")
    return g_ts


def command_train(args: argparse.Namespace) -> None:
    if args.method not in {"stats_aug", "dann", "cyclegan"}: raise ValueError("método inválido")
    train_temporal(Path(args.dataset), args.method, Path(args.out), args.epochs, args.batch_size, args.seed, args.lambda_domain, args.validation_block)


def load_generator(path: Path, device: torch.device) -> TemporalGenerator:
    g = TemporalGenerator().to(device); g.load_state_dict(torch.load(path / "temporal_generator.pt", map_location=device, weights_only=True)); return g.eval()


def false_color(x: np.ndarray) -> np.ndarray:
    """Composição NIR/Red/RedEdge para inspeção, sem alterar o .npy científico."""
    rgb = []
    for band in (x[..., 2], x[..., 0], x[..., 1]):
        lo, hi = np.nanpercentile(band, [2, 98])
        rgb.append(np.clip((band-lo)/(hi-lo+1e-8), 0, 1))
    return (np.stack(rgb, -1)*255).astype(np.uint8)


def command_predict(args: argparse.Namespace) -> None:
    dataset, ckpt, out = Path(args.dataset), Path(args.checkpoint), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(dataset / "target_r2_unlabeled.csv"); norm = AffineNormalizer(**json.loads((ckpt / "normalizer.json").read_text()))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); g = load_generator(ckpt, dev)
    mapper = None
    if args.method == "cyclegan":
        mapper = ImageGenerator().to(dev); mapper.load_state_dict(torch.load(ckpt / "cyclegan_target_to_source.pt", map_location=dev, weights_only=True)); mapper.eval()
    rows = []
    for r in target.itertuples():
        x = np.load(r.path).astype(np.float32)
        if mapper is not None:
            with torch.no_grad(): x = ((mapper(torch.from_numpy(x*2-1).permute(2,0,1)[None].to(dev))[0].permute(1,2,0).cpu().numpy()+1)/2).clip(0,1)
        else: x = norm.apply(x)
        with torch.no_grad(): fake = g(to_cond_tensor(x)[None].to(dev))[0].permute(1,2,0).cpu().numpy()
        fake = np.clip((fake+1)/2, 0, 1).astype(np.float32)
        name = f"p{int(r.fid):03d}_R5_synthetic.npy"; np.save(out / name, fake)
        png = name.replace(".npy", ".png"); Image.fromarray(false_color(fake)).save(out / png)
        rows.append({"fid": r.fid, "dose_n": r.dose_n, "bloco": r.bloco, "method": args.method, "prediction": name, "preview": png})
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    print(f"{len(rows)} patches R5 sintéticos -> {out}")


def field_targets(table: Path, stage: str) -> dict[tuple[str, int], float]:
    df = pd.read_excel(table); df.Estagio = df.Estagio.astype(str).str.upper(); df = df[df.Estagio == stage]
    return {(str(int(r.Dose_N)), int(r.Bloco)): float(r.Biomassa) for r in df.itertuples() if pd.notna(r.Biomassa)}


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e=y-p; return {"R2": float(1-(e@e)/(((y-y.mean())@(y-y.mean()))+1e-12)), "RMSE": float(np.sqrt(np.mean(e*e))), "MAE": float(np.mean(abs(e))), "bias": float(np.mean(p-y))}


def group_calibration_oof(data: pd.DataFrame, predictor: str, target: str = "biomassa_real",
                          group: str = "bloco") -> tuple[np.ndarray, list[dict]]:
    """Calibra ``target = a + b*predictor`` sem usar o bloco de teste.

    A função é propositalmente univariada: com 18 parcelas de calibração por
    fold, uma calibração afim corrige o deslocamento de escala sem transformar
    o protocolo em uma seleção de muitos atributos no alvo.
    """
    needed = {predictor, target, group}
    if not needed.issubset(data.columns) or data[list(needed)].isna().any().any():
        raise ValueError(f"dados de calibração incompletos; exigidas: {sorted(needed)}")
    groups = np.asarray(data[group])
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("calibração OOF exige ao menos três blocos independentes")
    y = data[target].to_numpy(float); x = data[[predictor]].to_numpy(float)
    oof = np.full(len(data), np.nan); folds = []
    for held in sorted(unique):
        train, test = groups != held, groups == held
        model = LinearRegression().fit(x[train], y[train])
        oof[test] = model.predict(x[test])
        folds.append({"heldout_block": int(held), "n_calibration": int(train.sum()),
                      "n_test": int(test.sum()), "intercept": float(model.intercept_),
                      "slope": float(model.coef_[0]), **metric(y[test], oof[test])})
    return oof, folds


def command_calibrate(args: argparse.Namespace) -> None:
    """Relata calibração local R5 por bloco para um método GAN já fixado."""
    source, out = Path(args.predictions), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    if args.method != "stats_aug":
        raise ValueError("o protocolo oficial fixa stats_aug; não selecione método com as predições OOF")
    calibrated, folds = group_calibration_oof(data, "biomassa_predita")
    oracle, oracle_folds = group_calibration_oof(data, "biomassa_oraculo_r5_real")
    raw = metric(data.biomassa_real.to_numpy(float), data.biomassa_predita.to_numpy(float))
    result = data.copy()
    result["biomassa_calibrada_oof"] = calibrated
    result["biomassa_oraculo_calibrada_oof"] = oracle
    result.to_csv(out / "oof_predictions.csv", index=False)
    y = data.biomassa_real.to_numpy(float)
    summary = {
        "protocol": "calibracao_local_R5_groupkfold_por_bloco",
        "method_fixed_before_calibration": args.method,
        "calibration_formula": "Biomassa_R5 = intercept + slope * Biomassa_PLSR_sintetica",
        "n_parcels": int(len(data)), "n_blocks": int(data.bloco.nunique()),
        "raw_synthetic": {**raw, "pearson_r": float(np.corrcoef(y, data.biomassa_predita)[0, 1]),
                          "spearman_r": float(pd.Series(y).corr(data.biomassa_predita, method="spearman"))},
        "calibrated_synthetic_oof": {**metric(y, calibrated), "pearson_r": float(np.corrcoef(y, calibrated)[0, 1]),
                                       "spearman_r": float(pd.Series(y).corr(pd.Series(calibrated), method="spearman")), "folds": folds},
        "calibrated_oracle_real_r5_oof": {**metric(y, oracle), "pearson_r": float(np.corrcoef(y, oracle)[0, 1]),
                                            "folds": oracle_folds},
        "interpretation": "Calibração local: cada predição OOF usa biomassa R5 de outros blocos; não é transferência cross-safra puramente remota.",
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    calibrated_metrics = summary["calibrated_synthetic_oof"]
    md = "\n".join([
        "# Biomassa R5 sintética — calibração local OOF", "",
        f"Método GAN fixado antes da calibração: `{args.method}`.",
        "", "| Cenário | R² | r | RMSE | MAE |", "|---|---:|---:|---:|---:|",
        f"| PLSR sintético bruto | {raw['R2']:.3f} | {summary['raw_synthetic']['pearson_r']:.3f} | {raw['RMSE']:.1f} | {raw['MAE']:.1f} |",
        f"| PLSR sintético + calibração OOF | {calibrated_metrics['R2']:.3f} | {calibrated_metrics['pearson_r']:.3f} | {calibrated_metrics['RMSE']:.1f} | {calibrated_metrics['MAE']:.1f} |",
        f"| R5 real + PLSR + calibração OOF | {summary['calibrated_oracle_real_r5_oof']['R2']:.3f} | {summary['calibrated_oracle_real_r5_oof']['pearson_r']:.3f} | {summary['calibrated_oracle_real_r5_oof']['RMSE']:.1f} | {summary['calibrated_oracle_real_r5_oof']['MAE']:.1f} |",
        "", "Cada fold calibra em três blocos (18 parcelas) e prevê o bloco retido (6 parcelas).",
        "O resultado é válido para calibração local R5; não representa previsão puramente remota em safra sem biomassa de calibração.", "",
    ])
    (out / "report.md").write_text(md)
    print(json.dumps({"R2_oof": calibrated_metrics["R2"], "pearson_r_oof": calibrated_metrics["pearson_r"],
                      "RMSE_oof": calibrated_metrics["RMSE"], "out": str(out)}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    dataset, pred_dir, out = Path(args.dataset), Path(args.predictions), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(dataset / "source_pairs_r2_r5.csv"); held = pd.read_csv(dataset / "TARGET_R5_HELDOUT.csv"); pred = pd.read_csv(pred_dir / "predictions.csv")
    ysrc, ytgt = field_targets(Path(args.source_table), "R5"), field_targets(Path(args.target_table), "R5")
    def key(r): return (str(int(float(r.dose_n))), int(r.bloco))
    # Schema fixado no R5 real fonte; nenhum atributo físico entra no PLSR.
    source_features = pd.DataFrame([parcel_features(np.load(r.path_r5)) for r in src.itertuples()])
    cols = list(source_features.columns); X = source_features[cols].replace([np.inf,-np.inf],np.nan).fillna(0).values
    y = np.array([ysrc[key(r)] for r in src.itertuples()]); groups = src.fid.values
    nfold = min(5, len(np.unique(groups))); maxc = min(15, X.shape[1], len(y)-len(y)//nfold-1)
    best = (1, -np.inf)
    for nc in range(1, maxc+1):
        p = cross_val_predict(make_pipeline(StandardScaler(), PLSRegression(n_components=nc)), X, y, cv=GroupKFold(nfold), groups=groups).ravel(); score=metric(y,p)["R2"]
        if score > best[1]: best=(nc,score)
    plsr = make_pipeline(StandardScaler(), PLSRegression(n_components=best[0])).fit(X,y)
    real = {int(r.fid): r.path for r in held.itertuples()}; rows=[]
    for r in pred.itertuples():
        fake=np.load(pred_dir/r.prediction); f=pd.DataFrame([parcel_features(fake)])[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        observed=np.load(real[int(r.fid)]); rf=pd.DataFrame([parcel_features(observed)])[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        yi=ytgt[key(r)]; idx_fake=rrenir_indices(fake); idx_real=rrenir_indices(observed)
        ff, rr = parcel_features(fake), parcel_features(observed)
        texture_delta = np.mean([abs(ff[c]-rr[c]) for c in cols if c.startswith("glcm")])
        rows.append({"fid":r.fid,"dose_n":r.dose_n,"bloco":r.bloco,"biomassa_real":yi,"biomassa_predita":float(plsr.predict(f.values).ravel()[0]),"biomassa_oraculo_r5_real":float(plsr.predict(rf.values).ravel()[0]),"l1":float(np.mean(abs(fake-observed))), "texture_mae":float(texture_delta), **{f"abs_{k}":float(abs(np.mean(idx_fake[k])-np.mean(idx_real[k]))) for k in idx_fake}})
    result=pd.DataFrame(rows); result.to_csv(out/"predicoes_parcela.csv",index=False)
    summary={"method": pred.method.iloc[0], "plsr_source_R5_groupcv_r2":best[1], "plsr_components":best[0], "synthetic":metric(result.biomassa_real.values,result.biomassa_predita.values), "oracle_real_R5":metric(result.biomassa_real.values,result.biomassa_oraculo_r5_real.values), "image":{"L1":float(result.l1.mean()), "texture_MAE":float(result.texture_mae.mean()), **{c:float(result[c].mean()) for c in result if c.startswith("abs_")}}}
    (out/"report.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare"); a.add_argument("--source-stage4",required=True); a.add_argument("--target-stage4",required=True); a.add_argument("--out",required=True); a.set_defaults(func=command_prepare)
    a=sub.add_parser("train"); a.add_argument("--dataset",required=True); a.add_argument("--method",required=True); a.add_argument("--out",required=True); a.add_argument("--epochs",type=int,default=200); a.add_argument("--batch-size",type=int,default=2); a.add_argument("--seed",type=int,default=42); a.add_argument("--lambda-domain",type=float,default=.1); a.add_argument("--validation-block",type=int,default=4); a.set_defaults(func=command_train)
    a=sub.add_parser("predict"); a.add_argument("--dataset",required=True); a.add_argument("--method",required=True); a.add_argument("--checkpoint",required=True); a.add_argument("--out",required=True); a.set_defaults(func=command_predict)
    a=sub.add_parser("evaluate"); a.add_argument("--dataset",required=True); a.add_argument("--predictions",required=True); a.add_argument("--source-table",required=True); a.add_argument("--target-table",required=True); a.add_argument("--out",required=True); a.set_defaults(func=command_evaluate)
    a=sub.add_parser("calibrate"); a.add_argument("--predictions", required=True, help="predicoes_parcela.csv da avaliação stats_aug"); a.add_argument("--method", default="stats_aug"); a.add_argument("--out", required=True); a.set_defaults(func=command_calibrate)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
