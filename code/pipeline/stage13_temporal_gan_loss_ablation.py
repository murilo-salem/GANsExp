#!/usr/bin/env python3
"""Ablação aninhada de funções de perda da GAN temporal para R5.

Mantém dados, U-Net, PatchGAN e avaliação downstream do stage12. A entrada é
sempre o histórico RRENIR, sem índices ou texturas como canais adicionais.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "pytorch-CycleGAN-and-pix2pix"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code" / "pipeline"), str(VENDOR)]

from milho_experiment.gan_losses import (  # noqa: E402
    calibrate_scales,
    discriminator_features,
    feature_matching_loss,
    gradient_loss,
    masked_l1,
    multiscale_texture_loss,
    parcel_mask,
    vegetation_index_loss,
)
from milho_experiment.temporal import (  # noqa: E402
    TemporalRecord,
    assert_disjoint,
    build_records,
    expected_input_channels,
    load_record,
    temporal_input,
)
from stage12_temporal_gan_value import (  # noqa: E402
    SCENARIOS,
    _glcm_feature_error,
    _seed_everything,
    feature_bundle,
    fit_predict,
    scenario_matrices,
)


ARM_NAMES = (
    "l1",
    "l1_gan",
    "l1_indices_texture",
    "l1_gan_feature_matching",
    "full",
)
LOSS_VERSION = 1
PRIMARY_SCENARIOS = {
    "Biomassa": ("augment_hybrid", "augment_real"),
    "Produtividade": ("forecast_fusion", "forecast_history"),
}
CHECKPOINT_POLICIES = ("all", "rotating", "none")


@dataclass(frozen=True)
class LossSpec:
    arm: str
    alpha: float = 0.0
    lambda_fm: float = 0.0

    def __post_init__(self):
        if self.arm not in ARM_NAMES:
            raise ValueError(f"braço inválido: {self.arm}")
        if self.alpha < 0 or self.lambda_fm < 0:
            raise ValueError("pesos de loss não podem ser negativos")

    @property
    def uses_gan(self) -> bool:
        return self.arm in {"l1_gan", "l1_gan_feature_matching", "full"}

    @property
    def uses_feature_matching(self) -> bool:
        return self.arm in {"l1_gan_feature_matching", "full"}

    @property
    def auxiliary_components(self) -> tuple[str, ...]:
        if self.arm == "l1_indices_texture":
            return ("indices", "texture")
        if self.arm == "full":
            return ("indices", "texture", "gradient")
        return ()


def _device(requested: str):
    import torch
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível")
    return torch.device(requested)


class TemporalMaskedDataset:
    """Pares temporais com máscara transformada junto das imagens."""

    def __init__(self, records: Sequence[TemporalRecord], load_size: int,
                 crop_size: int, seed: int):
        import torch
        self.rows = []
        for record in records:
            history, target = load_record(record)
            source = temporal_input(history, ())
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).float()
            mask = parcel_mask(target_tensor[None])[0]
            self.rows.append((torch.from_numpy(source).permute(2, 0, 1).float(),
                              target_tensor, mask))
        self.load_size = load_size
        self.crop_size = crop_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        import torch.nn.functional as F
        source, target, mask = (item.clone() for item in self.rows[index])
        if self.load_size != source.shape[-1]:
            source = F.interpolate(source[None], size=(self.load_size, self.load_size),
                                   mode="bilinear", align_corners=False)[0]
            target = F.interpolate(target[None], size=(self.load_size, self.load_size),
                                   mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None], size=(self.load_size, self.load_size),
                                 mode="nearest")[0]
        if self.load_size > self.crop_size:
            high = self.load_size - self.crop_size + 1
            y = int(torch.randint(high, (1,), generator=self.generator))
            x = int(torch.randint(high, (1,), generator=self.generator))
            source = source[:, y:y + self.crop_size, x:x + self.crop_size]
            target = target[:, y:y + self.crop_size, x:x + self.crop_size]
            mask = mask[:, y:y + self.crop_size, x:x + self.crop_size]
        if bool(torch.randint(2, (1,), generator=self.generator)):
            source, target, mask = source.flip(-1), target.flip(-1), mask.flip(-1)
        return source * 2 - 1, target * 2 - 1, mask


def _new_generator(input_nc: int, device):
    from models.networks import define_G
    return define_G(input_nc, 3, 64, "unet_256", norm="batch").to(device)


def _fingerprint(spec: LossSpec, records: Sequence[TemporalRecord], seed: int,
                 batch_size: int) -> str:
    files = []
    for record in records:
        paths = (*record.history_paths, record.target_path)
        files.append({"fid": record.fid, "paths": [str(path) for path in paths],
                      "sizes": [path.stat().st_size for path in paths],
                      "mtimes": [path.stat().st_mtime_ns for path in paths]})
    payload = {"loss_version": LOSS_VERSION, "spec": asdict(spec), "files": files,
               "seed": seed, "batch_size": batch_size, "architecture": "unet_256_ngf64",
               "discriminator": "basic_ndf64", "optimizer": "adam_2e-4_b0.5_0.999",
               "input_nc": expected_input_channels(records[0].history_stages, ())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _checkpoint_path(root: Path, outer: int, split: str, target: str,
                     spec: LossSpec, epochs: int, seed: int) -> Path:
    weights = f"a{spec.alpha:g}_fm{spec.lambda_fm:g}"
    return root / f"outer{outer}" / target.lower() / split / spec.arm / weights / \
        f"e{epochs}_s{seed}.pth"


def _rng_state(dataset, loader_generator):
    import torch
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dataset": dataset.generator.get_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, dataset, loader_generator):
    import torch
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    dataset.generator.set_state(state["dataset"])
    loader_generator.set_state(state["loader"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_to(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _calibrate(generator, records: Sequence[TemporalRecord], components: Sequence[str], device):
    import torch
    if not components:
        return {}
    values = {name: [] for name in components}
    reference = []
    generator.eval()
    with torch.no_grad():
        for record in records:
            history, target_np = load_record(record)
            source_np = temporal_input(history, ())
            source = torch.from_numpy(source_np * 2 - 1).permute(2, 0, 1)[None].to(device)
            target = torch.from_numpy(target_np).permute(2, 0, 1)[None].to(device)
            mask = parcel_mask(target)
            fake = ((generator(source) + 1) / 2).clamp(0, 1)
            reference.append(float(masked_l1(fake, target, mask)))
            batch_scales = calibrate_scales(fake, target, mask, components)
            for name in components:
                magnitude = reference[-1] / max(batch_scales[name], 1e-12)
                values[name].append(magnitude)
    ref = float(np.median(reference))
    generator.train()
    return {name: ref / max(float(np.median(raw)), 1e-6) for name, raw in values.items()}


def _save_checkpoint(path: Path, *, generator, discriminator, opt_g, opt_d,
                     epoch: int, epochs: int, fingerprint: str, spec: LossSpec,
                     scales: dict[str, float], history: list[dict], dataset,
                     loader_generator) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict() if opt_d is not None else None,
        "epoch": epoch,
        "epochs_requested": epochs,
        "fingerprint": fingerprint,
        "loss_spec": asdict(spec),
        "scales": scales,
        "history": history,
        "rng": _rng_state(dataset, loader_generator),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_to_load(policy: str, checkpoint: Path, resume_path: Path) -> Path | None:
    """Seleciona somente checkpoints compatíveis com a política de retenção."""
    if policy == "all" and checkpoint.is_file():
        return checkpoint
    if policy in {"all", "rotating"} and resume_path.is_file():
        return resume_path
    return None


def _finish_checkpoint(policy: str, checkpoint: Path, resume_path: Path,
                       save_final) -> None:
    """Persiste o modelo final apenas no modo legado e remove o checkpoint rotativo."""
    if policy == "all":
        save_final(checkpoint)
    else:
        checkpoint.unlink(missing_ok=True)
    resume_path.unlink(missing_ok=True)


def train_generator(records: Sequence[TemporalRecord], spec: LossSpec, *, epochs: int,
                    seed: int, batch_size: int, device, checkpoint: Path,
                    checkpoint_every: int = 10, checkpoint_policy: str = "all"):
    """Treina/restaura um braço e devolve gerador, histórico e escalas."""
    import torch
    from models.networks import GANLoss, define_D
    from torch.utils.data import DataLoader

    if not records:
        raise ValueError("fold de treino vazio")
    if checkpoint_policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"política de checkpoint inválida: {checkpoint_policy}")
    _seed_everything(seed)
    input_nc = expected_input_channels(records[0].history_stages, ())
    generator = _new_generator(input_nc, device)
    discriminator = (define_D(input_nc + 3, 64, "basic", norm="batch").to(device)
                     if spec.uses_gan else None)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = (torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
             if discriminator is not None else None)
    gan_loss = GANLoss("vanilla").to(device) if discriminator is not None else None
    dataset = TemporalMaskedDataset(records, 286, 256, seed)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
                        generator=loader_generator, num_workers=0)
    fingerprint = _fingerprint(spec, records, seed, batch_size)
    resume_path = checkpoint.with_suffix(".resume.pth")
    start_epoch, history, scales = 0, [], None

    load_path = _checkpoint_to_load(checkpoint_policy, checkpoint, resume_path)
    if load_path is not None:
        # RNG de CPU precisa continuar sendo ByteTensor de CPU; os pesos são
        # copiados para os módulos já alocados no device logo abaixo.
        payload = torch.load(load_path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError(f"checkpoint incompatível: {load_path}")
        generator.load_state_dict(payload["generator"])
        if discriminator is not None:
            discriminator.load_state_dict(payload["discriminator"])
            opt_d.load_state_dict(payload["optimizer_d"])
        opt_g.load_state_dict(payload["optimizer_g"])
        _optimizer_to(opt_g, device)
        _optimizer_to(opt_d, device)
        start_epoch = int(payload["epoch"])
        history = list(payload.get("history", []))
        scales = dict(payload.get("scales", {}))
        _restore_rng(payload.get("rng"), dataset, loader_generator)
        if start_epoch >= epochs:
            if checkpoint_policy == "rotating":
                resume_path.unlink(missing_ok=True)
            return generator.eval(), history, scales

    if scales is None:
        scales = _calibrate(generator, records, spec.auxiliary_components, device)
    component_functions = {
        "indices": vegetation_index_loss,
        "texture": multiscale_texture_loss,
        "gradient": gradient_loss,
    }
    generator.train()
    if discriminator is not None:
        discriminator.train()

    for epoch in range(start_epoch, epochs):
        totals: dict[str, float] = {}
        batches = 0
        for source, target_pm, mask in loader:
            source, target_pm, mask = source.to(device), target_pm.to(device), mask.to(device)
            target = ((target_pm + 1) / 2).clamp(0, 1)
            fake_pm = generator(source)

            loss_d = torch.zeros((), device=device)
            if discriminator is not None:
                opt_d.zero_grad(set_to_none=True)
                pred_fake = discriminator(torch.cat([source, fake_pm.detach()], 1))
                pred_real = discriminator(torch.cat([source, target_pm], 1))
                loss_d = 0.5 * (gan_loss(pred_fake, False) + gan_loss(pred_real, True))
                loss_d.backward()
                opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_pm = generator(source)
            fake = ((fake_pm + 1) / 2).clamp(0, 1)
            loss_l1 = masked_l1(fake, target, mask)
            auxiliary_raw = {name: component_functions[name](fake, target, mask)
                             for name in spec.auxiliary_components}
            if auxiliary_raw:
                auxiliary = torch.stack([scales[name] * value
                                         for name, value in auxiliary_raw.items()]).mean()
            else:
                auxiliary = torch.zeros((), device=device)
            loss_adv = torch.zeros((), device=device)
            loss_fm = torch.zeros((), device=device)
            if discriminator is not None:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                pair_fake = torch.cat([source, fake_pm], 1)
                if spec.uses_feature_matching:
                    pred_fake, features_fake = discriminator_features(discriminator, pair_fake)
                    with torch.no_grad():
                        _, features_real = discriminator_features(
                            discriminator, torch.cat([source, target_pm], 1))
                    loss_fm = feature_matching_loss(features_fake, features_real)
                else:
                    pred_fake = discriminator(pair_fake)
                loss_adv = gan_loss(pred_fake, True)
            loss_g = 100.0 * (loss_l1 + spec.alpha * auxiliary) + loss_adv + \
                spec.lambda_fm * loss_fm
            loss_g.backward()
            opt_g.step()
            if discriminator is not None:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

            observed = {"loss_g": loss_g, "loss_d": loss_d, "l1": loss_l1,
                        "adversarial": loss_adv, "feature_matching": loss_fm,
                        "auxiliary": auxiliary, **auxiliary_raw}
            for name, value in observed.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            batches += 1

        row = {"epoch": epoch + 1, **{name: value / batches for name, value in totals.items()}}
        if not all(np.isfinite(value) for key, value in row.items() if key != "epoch"):
            raise FloatingPointError(f"loss não finita na época {epoch + 1}: {row}")
        history.append(row)
        if (checkpoint_policy != "none" and checkpoint_every > 0 and
                (epoch + 1) % checkpoint_every == 0):
            _save_checkpoint(resume_path, generator=generator, discriminator=discriminator,
                             opt_g=opt_g, opt_d=opt_d, epoch=epoch + 1, epochs=epochs,
                             fingerprint=fingerprint, spec=spec, scales=scales,
                             history=history, dataset=dataset,
                             loader_generator=loader_generator)
            print(f"[train] {spec.arm} seed={seed} epoch={epoch + 1}/{epochs} "
                  f"checkpoint={resume_path}", flush=True)

    def save_final(path: Path) -> None:
        _save_checkpoint(path, generator=generator, discriminator=discriminator,
                         opt_g=opt_g, opt_d=opt_d, epoch=epochs, epochs=epochs,
                         fingerprint=fingerprint, spec=spec, scales=scales, history=history,
                         dataset=dataset, loader_generator=loader_generator)

    _finish_checkpoint(checkpoint_policy, checkpoint, resume_path, save_final)
    return generator.eval(), history, scales


def generate(generator, records: Sequence[TemporalRecord], device) -> dict[int, np.ndarray]:
    import torch
    result = {}
    for record in records:
        history, _ = load_record(record)
        source = temporal_input(history, ())
        tensor = torch.from_numpy(source * 2 - 1).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            fake = generator(tensor)[0].permute(1, 2, 0).cpu().numpy()
        result[record.fid] = np.clip((fake + 1) / 2, 0, 1).astype(np.float32)
    return result


def _primary_pair(target: str) -> tuple[str, str]:
    try:
        return PRIMARY_SCENARIOS[target]
    except KeyError as error:
        raise ValueError(f"alvo sem contraste primário: {target}") from error


def inner_score(records: Sequence[TemporalRecord], target: str, spec: LossSpec, *,
                outer_block: int, epochs: int, seed: int, batch_size: int, device,
                cache: Path, n_jobs: int, checkpoint_every: int,
                checkpoint_policy: str) -> float:
    candidate_scenario, baseline_scenario = _primary_pair(target)
    y_all, candidate_all, baseline_all = [], [], []
    for val_block in sorted({record.block for record in records}):
        train = [record for record in records if record.block != val_block]
        val = [record for record in records if record.block == val_block]
        assert_disjoint(train, val)
        checkpoint = _checkpoint_path(cache, outer_block, f"inner{val_block}", target,
                                      spec, epochs, seed)
        generator, _, _ = train_generator(
            train, spec, epochs=epochs, seed=seed, batch_size=batch_size, device=device,
            checkpoint=checkpoint, checkpoint_every=checkpoint_every,
            checkpoint_policy=checkpoint_policy)
        fake_train, fake_val = generate(generator, train, device), generate(generator, val, device)
        train_bundle = feature_bundle(train, fake_train)
        val_bundle = feature_bundle(val, fake_val)
        for scenario, destination in ((candidate_scenario, candidate_all),
                                      (baseline_scenario, baseline_all)):
            matrices = scenario_matrices(scenario, train, val, train_bundle, val_bundle,
                                         target, include_dose=False)
            prediction, _ = fit_predict(*matrices, seed=seed + val_block, n_jobs=n_jobs)
            destination.extend(prediction.tolist())
        y_all.extend(record.target(target) for record in val)
    return float(r2_score(y_all, candidate_all) - r2_score(y_all, baseline_all))


def select_specs(records: Sequence[TemporalRecord], target: str, *, outer_block: int,
                 args, device, cache: Path, progress=None):
    rows = []
    alpha_scores = []
    for alpha in args.alpha_grid:
        spec = LossSpec("l1_indices_texture", alpha=alpha)
        score = inner_score(records, target, spec, outer_block=outer_block,
                            epochs=args.screen_epochs, seed=args.screen_seed,
                            batch_size=args.batch_size, device=device, cache=cache,
                            n_jobs=args.n_jobs, checkpoint_every=args.checkpoint_every,
                            checkpoint_policy=args.checkpoint_policy)
        alpha_scores.append((score, -alpha, spec))
        rows.append({"arm": spec.arm, "alpha": alpha, "lambda_fm": 0.0,
                     "delta_r2": score})
        if progress is not None:
            progress("screen", target, outer_block, spec.arm, args.screen_seed)
    fm_scores = []
    for weight in args.fm_grid:
        spec = LossSpec("l1_gan_feature_matching", lambda_fm=weight)
        score = inner_score(records, target, spec, outer_block=outer_block,
                            epochs=args.screen_epochs, seed=args.screen_seed,
                            batch_size=args.batch_size, device=device, cache=cache,
                            n_jobs=args.n_jobs, checkpoint_every=args.checkpoint_every,
                            checkpoint_policy=args.checkpoint_policy)
        fm_scores.append((score, -weight, spec))
        rows.append({"arm": spec.arm, "alpha": 0.0, "lambda_fm": weight,
                     "delta_r2": score})
        if progress is not None:
            progress("screen", target, outer_block, spec.arm, args.screen_seed)
    selected_alpha = max(alpha_scores)[2]
    selected_fm = max(fm_scores)[2]
    specs = {
        "l1": LossSpec("l1"),
        "l1_gan": LossSpec("l1_gan"),
        selected_alpha.arm: selected_alpha,
        selected_fm.arm: selected_fm,
        "full": LossSpec("full", alpha=selected_alpha.alpha,
                         lambda_fm=selected_fm.lambda_fm),
    }
    for row in rows:
        chosen = specs[row["arm"]]
        row["selected"] = bool(row["alpha"] == chosen.alpha and
                               row["lambda_fm"] == chosen.lambda_fm)
    rows.extend([
        {"arm": "l1", "alpha": 0.0, "lambda_fm": 0.0,
         "delta_r2": np.nan, "selected": True},
        {"arm": "l1_gan", "alpha": 0.0, "lambda_fm": 0.0,
         "delta_r2": np.nan, "selected": True},
        {"arm": "full", "alpha": specs["full"].alpha,
         "lambda_fm": specs["full"].lambda_fm,
         "delta_r2": np.nan, "selected": True},
    ])
    return specs, rows


def crossfit_fakes(records: Sequence[TemporalRecord], target: str, spec: LossSpec, *,
                   outer_block: int, epochs: int, seed: int, batch_size: int, device,
                   cache: Path, checkpoint_every: int, checkpoint_policy: str,
                   histories: list[dict]):
    fakes = {}
    cache_target = "shared" if spec.arm in {"l1", "l1_gan"} else target
    for val_block in sorted({record.block for record in records}):
        train = [record for record in records if record.block != val_block]
        val = [record for record in records if record.block == val_block]
        checkpoint = _checkpoint_path(cache, outer_block, f"final{val_block}",
                                      cache_target, spec, epochs, seed)
        generator, history, scales = train_generator(
            train, spec, epochs=epochs, seed=seed, batch_size=batch_size, device=device,
            checkpoint=checkpoint, checkpoint_every=checkpoint_every,
            checkpoint_policy=checkpoint_policy)
        for row in history:
            histories.append({"target": target, "outer_block": outer_block,
                              "split": f"final{val_block}", "seed": seed,
                              **asdict(spec), "scales": json.dumps(scales, sort_keys=True), **row})
        fakes.update(generate(generator, val, device))
    if set(fakes) != {record.fid for record in records}:
        raise AssertionError("cross-fitting não gerou uma imagem por parcela de treino")
    return fakes


def _numpy_image_metrics(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    import torch
    real_t = torch.from_numpy(real).permute(2, 0, 1)[None]
    fake_t = torch.from_numpy(fake).permute(2, 0, 1)[None]
    mask = parcel_mask(real_t)
    valid = mask[0, 0].numpy().astype(bool)
    real_ndvi = (real[..., 2] - real[..., 0]) / (real[..., 2] + real[..., 0] + 1e-6)
    fake_ndvi = (fake[..., 2] - fake[..., 0]) / (fake[..., 2] + fake[..., 0] + 1e-6)
    real_ndre = (real[..., 2] - real[..., 1]) / (real[..., 2] + real[..., 1] + 1e-6)
    fake_ndre = (fake[..., 2] - fake[..., 1]) / (fake[..., 2] + fake[..., 1] + 1e-6)
    return {
        "l1_global": float(np.mean(np.abs(fake - real))),
        "l1_masked": float(np.mean(np.abs(fake[valid] - real[valid]))),
        "ndvi_mae": float(np.mean(np.abs(fake_ndvi[valid] - real_ndvi[valid]))),
        "ndre_mae": float(np.mean(np.abs(fake_ndre[valid] - real_ndre[valid]))),
        "texture_multiscale": float(multiscale_texture_loss(fake_t, real_t, mask)),
        "gradient_mae": float(gradient_loss(fake_t, real_t, mask)),
        "glcm_feature_mae": _glcm_feature_error(real, fake),
    }


def _write(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_progress(out: Path, payload: dict):
    temporary = out / "progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(out / "progress.json")


def _bootstrap(y, baseline, candidate, blocks, n_bootstrap, seed=2026):
    rng = np.random.default_rng(seed)
    positions = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    deltas = []
    for _ in range(n_bootstrap):
        take = np.concatenate([rng.choice(pos, len(pos), replace=True) for pos in positions])
        if np.unique(y[take]).size > 1:
            deltas.append(r2_score(y[take], candidate[take]) - r2_score(y[take], baseline[take]))
    values = np.asarray(deltas, dtype=float)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975)), \
        float((1 + np.count_nonzero(values <= 0)) / (len(values) + 1))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summarize(prediction_rows: list[dict], out: Path, bootstrap: int):
    predictions = pd.DataFrame(prediction_rows)
    identity = ["target", "rs_mode", "arm", "scenario", "seed", "fid"]
    if predictions.duplicated(identity).any():
        raise AssertionError("predições OOF duplicadas")
    counts = predictions.groupby(identity[:-1]).fid.nunique()
    if not (counts == 24).all():
        raise AssertionError(f"predições OOF incompletas: {sorted(counts.unique())}")

    per_seed_rows = []
    for key, frame in predictions.groupby(["target", "rs_mode", "arm", "scenario", "seed"]):
        y, pred = frame.y.to_numpy(), frame.prediction.to_numpy()
        per_seed_rows.append({"target": key[0], "rs_mode": key[1], "arm": key[2],
                              "scenario": key[3], "seed": key[4], "n": len(frame),
                              "r2": r2_score(y, pred),
                              "rmse": np.sqrt(mean_squared_error(y, pred)),
                              "mae": mean_absolute_error(y, pred)})
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    per_seed.groupby(["target", "rs_mode", "arm", "scenario"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        seeds=("seed", "nunique"),
    ).to_csv(out / "metrics_seed_summary.csv", index=False)

    group = ["target", "rs_mode", "arm", "scenario", "outer_block", "fid", "y"]
    ensemble = predictions.groupby(group, as_index=False).agg(
        prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
        seeds=("seed", "nunique"))
    ensemble.to_csv(out / "oof_predictions_ensemble.csv", index=False)
    metric_rows = []
    for key, frame in ensemble.groupby(["target", "rs_mode", "arm", "scenario"]):
        metric_rows.append({"target": key[0], "rs_mode": key[1], "arm": key[2],
                            "scenario": key[3], "n": len(frame),
                            "r2": r2_score(frame.y, frame.prediction),
                            "rmse": np.sqrt(mean_squared_error(frame.y, frame.prediction)),
                            "mae": mean_absolute_error(frame.y, frame.prediction)})
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)

    within_rows = []
    standard = (("augment_synthetic_vs_real", "augment_synthetic", "augment_real"),
                ("augment_hybrid_vs_real", "augment_hybrid", "augment_real"),
                ("forecast_synthetic_vs_history", "forecast_synthetic", "forecast_history"),
                ("forecast_fusion_vs_history", "forecast_fusion", "forecast_history"),
                ("combined_vs_history", "combined", "forecast_history"),
                ("combined_vs_fusion", "combined", "forecast_fusion"))
    for (target, rs_mode, arm), frame in ensemble.groupby(["target", "rs_mode", "arm"]):
        wide = frame.pivot(index=["outer_block", "fid", "y"], columns="scenario",
                           values="prediction").reset_index()
        for name, candidate_name, baseline_name in standard:
            y = wide.y.to_numpy()
            base = wide[baseline_name].to_numpy()
            candidate = wide[candidate_name].to_numpy()
            blocks = wide.outer_block.to_numpy()
            base_r2, candidate_r2 = r2_score(y, base), r2_score(y, candidate)
            lo, hi, p = _bootstrap(y, base, candidate, blocks, bootstrap)
            within_rows.append({"target": target, "rs_mode": rs_mode, "arm": arm,
                                "contrast": name, "baseline": baseline_name,
                                "candidate": candidate_name, "baseline_r2": base_r2,
                                "candidate_r2": candidate_r2,
                                "delta_r2": candidate_r2 - base_r2,
                                "delta_r2_lo": lo, "delta_r2_hi": hi,
                                "p_one_sided_bootstrap": p,
                                "improvement_supported": bool(candidate_r2 > base_r2 and lo > 0),
                                "absolute_useful": bool(candidate_r2 > 0)})
    pd.DataFrame(within_rows).to_csv(out / "contrasts.csv", index=False)

    loss_rows = []
    for (target, rs_mode, scenario), frame in ensemble.groupby(["target", "rs_mode", "scenario"]):
        wide = frame.pivot(index=["outer_block", "fid", "y"], columns="arm",
                           values="prediction").reset_index()
        if "l1_gan" not in wide:
            continue
        y, base, blocks = wide.y.to_numpy(), wide.l1_gan.to_numpy(), wide.outer_block.to_numpy()
        for arm in ARM_NAMES:
            if arm == "l1_gan" or arm not in wide:
                continue
            candidate = wide[arm].to_numpy()
            lo, hi, p = _bootstrap(y, base, candidate, blocks, bootstrap)
            loss_rows.append({"target": target, "rs_mode": rs_mode, "scenario": scenario,
                              "candidate_arm": arm, "baseline_arm": "l1_gan",
                              "baseline_r2": r2_score(y, base),
                              "candidate_r2": r2_score(y, candidate),
                              "delta_r2": r2_score(y, candidate) - r2_score(y, base),
                              "delta_r2_lo": lo, "delta_r2_hi": hi,
                              "p_one_sided_bootstrap": p,
                              "improvement_supported": bool(r2_score(y, candidate) > r2_score(y, base)
                                                            and lo > 0)})
    pd.DataFrame(loss_rows).to_csv(out / "loss_contrasts.csv", index=False)

    primary = pd.DataFrame(within_rows)
    wanted = []
    for target, (candidate, baseline) in PRIMARY_SCENARIOS.items():
        rows = primary[(primary.target == target) & (primary.rs_mode == "rs_puro") &
                       (primary.arm == "full") & (primary.candidate == candidate) &
                       (primary.baseline == baseline)]
        if len(rows) == 1:
            row = rows.iloc[0].to_dict()
            seed_frame = per_seed[(per_seed.target == target) &
                                  (per_seed.rs_mode == "rs_puro") &
                                  (per_seed.arm == "full") &
                                  (per_seed.scenario.isin([candidate, baseline]))]
            seed_wide = seed_frame.pivot(index="seed", columns="scenario", values="r2")
            seed_deltas = seed_wide[candidate] - seed_wide[baseline]
            row["positive_seeds"] = int((seed_deltas > 0).sum())
            row["seeds_total"] = int(len(seed_deltas))
            row["all_seeds_positive"] = bool(len(seed_deltas) > 0 and
                                               (seed_deltas > 0).all())
            wanted.append(row)
    if wanted:
        adjusted = _holm_adjust([row["p_one_sided_bootstrap"] for row in wanted])
        for row, p_adjusted in zip(wanted, adjusted):
            row["p_holm"] = p_adjusted
            row["confirmatory_supported"] = bool(row["delta_r2"] > 0 and
                                                   row["delta_r2_lo"] > 0 and
                                                   row["candidate_r2"] > 0 and
                                                   row["all_seeds_positive"] and
                                                   p_adjusted < 0.05)
    pd.DataFrame(wanted).to_csv(out / "primary_contrasts.csv", index=False)


def run(args):
    if args.horizon != "R5":
        raise ValueError("esta ablação foi pré-registrada somente para R5")
    if len(args.arms) != len(ARM_NAMES) or set(args.arms) != set(ARM_NAMES):
        raise ValueError(f"execução oficial requer os cinco braços: {ARM_NAMES}")
    if (not args.alpha_grid or not args.fm_grid or
            len(set(args.alpha_grid)) != len(args.alpha_grid) or
            len(set(args.fm_grid)) != len(args.fm_grid) or
            any(value <= 0 for value in (*args.alpha_grid, *args.fm_grid))):
        raise ValueError("grids de pesos devem conter valores positivos e únicos")
    if args.screen_epochs <= 0 or args.final_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("épocas e batch size devem ser positivos")
    if args.checkpoint_policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"política de checkpoint inválida: {args.checkpoint_policy}")
    if len(set(args.final_seeds)) != len(args.final_seeds):
        raise ValueError("seeds finais devem ser únicas")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "checkpoints"
    device = _device(args.device)
    records = build_records(args.stage4, args.table, "R5")
    blocks = sorted({record.block for record in records})
    counts = pd.Series([record.block for record in records]).value_counts().to_dict()
    if len(records) != 24 or blocks != [1, 2, 3, 4] or set(counts.values()) != {6}:
        raise ValueError(f"protocolo requer 24 parcelas, seis por bloco; recebeu {counts}")
    metadata = {"stage4": str(Path(args.stage4).resolve()),
                "table": str(Path(args.table).resolve()), "horizon": "R5",
                "targets": args.targets, "arms": args.arms,
                "screen_epochs": args.screen_epochs, "final_epochs": args.final_epochs,
                "screen_seed": args.screen_seed, "final_seeds": args.final_seeds,
                "alpha_grid": args.alpha_grid, "fm_grid": args.fm_grid,
                "checkpoint_policy": args.checkpoint_policy,
                "checkpoint_every": args.checkpoint_every,
                "input_channels": "RRENIR only", "mask": "target any band > 1e-6",
                "primary_scenarios": PRIMARY_SCENARIOS,
                "architecture": "unet_256 + single-scale PatchGAN"}
    (out / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    _write([{"fid": r.fid, "block": r.block, "dose_n": r.dose_n,
             "history_stages": ",".join(r.history_stages), "target_stage": r.target_stage,
             "biomass": r.biomass, "productivity": r.productivity} for r in records],
           out / "records_R5.csv")

    predictions, images, selections, histories = [], [], [], []
    screen_units = 0 if args.skip_screen else len(args.targets) * len(blocks) * (
        len(args.alpha_grid) + len(args.fm_grid))
    final_units = len(args.targets) * len(blocks) * len(args.arms) * len(args.final_seeds)
    progress_state = {"completed_units": 0, "total_units": screen_units + final_units,
                      "percent": 0.0, "phase": "starting"}

    def advance(phase, target, outer_block, arm, seed):
        progress_state["completed_units"] += 1
        progress_state.update({"phase": phase, "target": target,
                               "outer_block": outer_block, "arm": arm, "seed": seed,
                               "percent": round(100 * progress_state["completed_units"] /
                                                progress_state["total_units"], 2)})
        _write_progress(out, progress_state)
        print(f"[progress] {progress_state['percent']:.2f}% phase={phase} target={target} "
              f"outer={outer_block} arm={arm} seed={seed}", flush=True)

    _write_progress(out, progress_state)
    for target in args.targets:
        for outer_block in blocks:
            train = [record for record in records if record.block != outer_block]
            test = [record for record in records if record.block == outer_block]
            assert_disjoint(train, test)
            if args.skip_screen:
                specs = {"l1": LossSpec("l1"), "l1_gan": LossSpec("l1_gan"),
                         "l1_indices_texture": LossSpec("l1_indices_texture", args.alpha_grid[0]),
                         "l1_gan_feature_matching": LossSpec(
                             "l1_gan_feature_matching", lambda_fm=args.fm_grid[0]),
                         "full": LossSpec("full", args.alpha_grid[0], args.fm_grid[0])}
                trace = [{"arm": arm, "alpha": spec.alpha, "lambda_fm": spec.lambda_fm,
                          "delta_r2": np.nan, "selected": True}
                         for arm, spec in specs.items()]
            else:
                specs, trace = select_specs(train, target, outer_block=outer_block,
                                            args=args, device=device, cache=cache / "screen",
                                            progress=advance)
            for row in trace:
                selections.append({"target": target, "outer_block": outer_block, **row})
            _write(selections, out / "loss_selection.csv")

            for arm in args.arms:
                spec = specs[arm]
                for seed in args.final_seeds:
                    fake_train = crossfit_fakes(
                        train, target, spec, outer_block=outer_block,
                        epochs=args.final_epochs, seed=seed, batch_size=args.batch_size,
                        device=device, cache=cache / "final",
                        checkpoint_every=args.checkpoint_every,
                        checkpoint_policy=args.checkpoint_policy, histories=histories)
                    cache_target = "shared" if arm in {"l1", "l1_gan"} else target
                    checkpoint = _checkpoint_path(cache / "final", outer_block, "outer",
                                                  cache_target, spec, args.final_epochs, seed)
                    generator, history, scales = train_generator(
                        train, spec, epochs=args.final_epochs, seed=seed,
                        batch_size=args.batch_size, device=device, checkpoint=checkpoint,
                        checkpoint_every=args.checkpoint_every,
                        checkpoint_policy=args.checkpoint_policy)
                    for row in history:
                        histories.append({"target": target, "outer_block": outer_block,
                                          "split": "outer", "seed": seed, **asdict(spec),
                                          "scales": json.dumps(scales, sort_keys=True), **row})
                    fake_test = generate(generator, test, device)
                    train_bundle = feature_bundle(train, fake_train)
                    test_bundle = feature_bundle(test, fake_test)
                    for record in test:
                        _, real = load_record(record)
                        images.append({"target": target, "outer_block": outer_block,
                                       "fid": record.fid, "seed": seed, **asdict(spec),
                                       **_numpy_image_metrics(real, fake_test[record.fid])})
                    for rs_mode in args.rs_modes:
                        for scenario in SCENARIOS:
                            matrices = scenario_matrices(
                                scenario, train, test, train_bundle, test_bundle, target,
                                include_dose=rs_mode == "rs_dose")
                            pred, model = fit_predict(*matrices, seed=seed + outer_block,
                                                      n_jobs=args.n_jobs)
                            for record, value in zip(test, pred):
                                predictions.append({"target": target, "rs_mode": rs_mode,
                                                    "arm": arm, "scenario": scenario,
                                                    "outer_block": outer_block,
                                                    "fid": record.fid, "seed": seed,
                                                    "alpha": spec.alpha,
                                                    "lambda_fm": spec.lambda_fm,
                                                    "model": model,
                                                    "y": record.target(target),
                                                    "prediction": float(value)})
                    _write(predictions, out / "oof_predictions.csv")
                    _write(images, out / "gan_image_metrics.csv")
                    _write(histories, out / "training_history.csv")
                    advance("final", target, outer_block, arm, seed)
    summarize(predictions, out, args.bootstrap)
    progress_state.update({"phase": "complete", "percent": 100.0})
    _write_progress(out, progress_state)


def prepare(args):
    records = build_records(args.stage4, args.table, "R5")
    print(f"R5: {len(records)} parcelas completas; blocos="
          f"{pd.Series([r.block for r in records]).value_counts().sort_index().to_dict()}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--stage4", required=True)
        cmd.add_argument("--table", required=True)
        cmd.add_argument("--horizon", default="R5", choices=["R5"])
        if name == "run":
            cmd.add_argument("--out", required=True)
            cmd.add_argument("--targets", nargs="+", choices=list(PRIMARY_SCENARIOS),
                             default=list(PRIMARY_SCENARIOS))
            cmd.add_argument("--arms", nargs="+", choices=list(ARM_NAMES),
                             default=list(ARM_NAMES))
            cmd.add_argument("--rs-modes", nargs="+", choices=["rs_puro", "rs_dose"],
                             default=["rs_puro", "rs_dose"])
            cmd.add_argument("--screen-epochs", type=int, default=50)
            cmd.add_argument("--final-epochs", type=int, default=200)
            cmd.add_argument("--screen-seed", type=int, default=7)
            cmd.add_argument("--final-seeds", type=int, nargs="+", default=[7, 11, 23])
            cmd.add_argument("--alpha-grid", type=float, nargs="+", default=[0.1, 0.3, 1.0])
            cmd.add_argument("--fm-grid", type=float, nargs="+", default=[1.0, 5.0, 10.0])
            cmd.add_argument("--batch-size", type=int, default=4)
            cmd.add_argument("--bootstrap", type=int, default=10_000)
            cmd.add_argument("--checkpoint-every", type=int, default=10)
            cmd.add_argument("--checkpoint-policy", choices=CHECKPOINT_POLICIES, default="all")
            cmd.add_argument("--device", default="auto")
            cmd.add_argument("--n-jobs", type=int, default=1)
            cmd.add_argument("--skip-screen", action="store_true",
                             help="usa os menores pesos; destinado somente a smoke tests")
    return ap


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
