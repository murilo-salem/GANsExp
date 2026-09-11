#!/usr/bin/env python3
"""Busca controlada de arquiteturas R2/V8+R2 -> R5 sem checkpoints.

``selection`` faz leave-one-block-out exclusivamente em 2022/23. ``final``
exige vencedores explícitos e é a única rota que abre R5 e os alvos de 23/24.
Pesos neurais nunca são escritos em disco.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code" / "pipeline")]

from milho_experiment.architecture_models import (  # noqa: E402
    discriminator_scales,
    make_discriminators,
    make_generator,
)
from milho_experiment.temporal import paired_stratified_bootstrap_delta  # noqa: E402
from stage6_features_ortho import parcel_features  # noqa: E402


TRACK_MODELS = {
    "A": ("identity", "affine", "unet_l1", "unet_light", "resnet9",
          "attention_unet", "multiscale", "wgangp", "oracle_real"),
    "B": ("history", "convlstm_lite", "simvp_lite", "oracle_real"),
}
DEFAULT_TRACK_MODELS = {
    "A": tuple(name for name in TRACK_MODELS["A"] if name != "wgangp"),
    "B": TRACK_MODELS["B"],
}
NEURAL_MODELS = set(TRACK_MODELS["A"]) | set(TRACK_MODELS["B"])
NEURAL_MODELS -= {"identity", "affine", "history", "oracle_real"}
ADVERSARIAL_MODELS = {"unet_light", "resnet9", "attention_unet", "multiscale", "wgangp"}
BASELINES = {"A": "identity", "B": "history"}


@dataclass(frozen=True)
class SearchRecord:
    fid: int
    block: int
    dose_n: float
    history_paths: tuple[Path, ...]
    target_path: Path
    mask_path: Path | None
    weight: float
    biomass: float
    productivity: float

    def target(self, name: str) -> float:
        return self.biomass if name.lower() == "biomassa" else self.productivity


def _dose_key(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def _targets(table: str | Path) -> dict[tuple[int, str], tuple[float, float]]:
    frame = pd.read_excel(table)
    frame = frame[frame.Estagio.astype(str).str.upper() == "R5"].copy()
    if frame.duplicated(["Bloco", "Dose_N"]).any():
        raise ValueError("alvos R5 duplicados por bloco/dose")
    return {(int(row.Bloco), _dose_key(row.Dose_N)):
            (float(row.Biomassa), float(row.Produtividade))
            for row in frame.itertuples()
            if np.isfinite(row.Biomassa) and np.isfinite(row.Produtividade)}


def _stage_paths(stage4: str | Path) -> dict[tuple[int, str], Path]:
    root = Path(stage4)
    frame = pd.read_csv(root / "manifest.csv").copy()
    frame["stage"] = frame.stage.astype(str).str.upper()
    if frame.duplicated(["fid", "stage"]).any():
        raise ValueError("manifesto possui mais de uma imagem por parcela/estágio")
    return {(int(row.fid), str(row.stage)): (root / str(row.npy)).resolve()
            for row in frame.itertuples()}


def load_source_records(dataset: str | Path, stage4: str | Path,
                        table: str | Path, track: str) -> list[SearchRecord]:
    pairs = pd.read_csv(Path(dataset) / "source_pairs_registered.csv")
    paths = _stage_paths(stage4)
    targets = _targets(table)
    records = []
    for row in pairs.itertuples():
        key = (int(row.bloco), _dose_key(row.dose_n))
        if key not in targets:
            continue
        histories = (Path(row.r2),) if track == "A" else (
            paths[(int(row.fid), "V8")], Path(row.r2))
        records.append(SearchRecord(
            fid=int(row.fid), block=int(row.bloco), dose_n=float(row.dose_n),
            history_paths=histories, target_path=Path(row.r5_registered),
            mask_path=Path(row.mask), weight=float(row.quality_weight),
            biomass=targets[key][0], productivity=targets[key][1]))
    validate_records(records, expected=48, per_block=12)
    return records


def load_target_records(stage4: str | Path, table: str | Path,
                        track: str) -> list[SearchRecord]:
    root = Path(stage4)
    manifest = pd.read_csv(root / "manifest.csv").copy()
    manifest["stage"] = manifest.stage.astype(str).str.upper()
    paths = _stage_paths(root)
    targets = _targets(table)
    r2 = manifest[manifest.stage == "R2"]
    records = []
    for row in r2.itertuples():
        key = (int(row.bloco), _dose_key(row.dose_n))
        if key not in targets:
            continue
        histories = (paths[(int(row.fid), "R2")],) if track == "A" else (
            paths[(int(row.fid), "V8")], paths[(int(row.fid), "R2")])
        mask_value = getattr(row, "mask", None)
        records.append(SearchRecord(
            fid=int(row.fid), block=int(row.bloco), dose_n=float(row.dose_n),
            history_paths=histories, target_path=paths[(int(row.fid), "R5")],
            mask_path=(root / str(mask_value)).resolve() if isinstance(mask_value, str) else None,
            weight=1.0, biomass=targets[key][0], productivity=targets[key][1]))
    validate_records(records, expected=24, per_block=6)
    return records


def validate_records(records: Sequence[SearchRecord], *, expected: int,
                     per_block: int) -> None:
    counts = pd.Series([record.block for record in records]).value_counts().to_dict()
    if len(records) != expected or sorted(counts) != [1, 2, 3, 4] or set(counts.values()) != {per_block}:
        raise ValueError(f"protocolo requer n={expected}, {per_block}/bloco; recebeu {counts}")
    ids = [record.fid for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("cada parcela deve aparecer exatamente uma vez")
    missing = [str(path) for record in records
               for path in (*record.history_paths, record.target_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"arquivos ausentes: {missing[:5]}")


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset:
    def __init__(self, records: Sequence[SearchRecord], crop_size: int, seed: int):
        import torch
        self.rows = []
        for record in records:
            history = np.concatenate([np.load(path).astype(np.float32)
                                      for path in record.history_paths], axis=-1)
            target = np.load(record.target_path).astype(np.float32)
            mask = (np.load(record.mask_path).astype(np.float32) if record.mask_path
                    and record.mask_path.is_file() else np.ones(target.shape[:2], np.float32))
            self.rows.append((torch.from_numpy(history).permute(2, 0, 1),
                              torch.from_numpy(target).permute(2, 0, 1),
                              torch.from_numpy(mask)[None], float(record.weight)))
        self.crop_size = crop_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        source, target, mask, weight = (value.clone() if hasattr(value, "clone") else value
                                         for value in self.rows[index])
        height, width = source.shape[-2:]
        crop = min(self.crop_size, height, width)
        if height > crop:
            y = int(torch.randint(height - crop + 1, (1,), generator=self.generator))
        else:
            y = 0
        if width > crop:
            x = int(torch.randint(width - crop + 1, (1,), generator=self.generator))
        else:
            x = 0
        source = source[:, y:y + crop, x:x + crop]
        target = target[:, y:y + crop, x:x + crop]
        mask = mask[:, y:y + crop, x:x + crop]
        if bool(torch.randint(2, (1,), generator=self.generator)):
            source, target, mask = source.flip(-1), target.flip(-1), mask.flip(-1)
        return source * 2 - 1, target * 2 - 1, mask, torch.tensor(weight)


class AffineGenerator:
    def __init__(self, coefficients: np.ndarray):
        self.coefficients = coefficients

    @classmethod
    def fit(cls, records: Sequence[SearchRecord], seed: int, max_pixels: int = 5000):
        rng = np.random.default_rng(seed)
        xs = [[] for _ in range(3)]
        ys = [[] for _ in range(3)]
        for record in records:
            source = np.load(record.history_paths[-1])
            target = np.load(record.target_path)
            mask = np.load(record.mask_path) > 0 if record.mask_path else np.ones(source.shape[:2], bool)
            positions = np.flatnonzero(mask)
            if len(positions) > max_pixels:
                positions = rng.choice(positions, max_pixels, replace=False)
            for channel in range(3):
                xs[channel].append(source[..., channel].ravel()[positions])
                ys[channel].append(target[..., channel].ravel()[positions])
        coefficients = []
        for channel in range(3):
            x = np.concatenate(xs[channel])
            y = np.concatenate(ys[channel])
            coefficients.append(np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y,
                                                rcond=None)[0])
        return cls(np.asarray(coefficients))

    def predict(self, histories: Sequence[np.ndarray]) -> np.ndarray:
        source = histories[-1]
        result = np.empty_like(source)
        for channel, (slope, intercept) in enumerate(self.coefficients):
            result[..., channel] = slope * source[..., channel] + intercept
        return np.clip(result, 0, 1)


def train_generator(records: Sequence[SearchRecord], name: str, *, epochs: int,
                    seed: int, batch_size: int, crop_size: int, width: int, device):
    """Treina integralmente em memória e devolve histórico escalar, nunca pesos."""
    import torch
    from torch.utils.data import DataLoader

    _seed(seed)
    input_channels = 3 * len(records[0].history_paths)
    generator = make_generator(name, input_channels, width).to(device)
    adversarial = name in ADVERSARIAL_MODELS
    discriminators = (make_discriminators(name, input_channels, width).to(device)
                      if adversarial else None)
    opt_g = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_d = (torch.optim.Adam(discriminators.parameters(), 2e-4, betas=(0.5, 0.999))
             if discriminators is not None else None)
    bce = torch.nn.BCEWithLogitsLoss()
    dataset = PairDataset(records, crop_size, seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history = []

    def gradient_penalty(discriminator, real_pair, fake_pair):
        alpha = torch.rand(real_pair.shape[0], 1, 1, 1, device=device)
        mixed = (alpha * real_pair + (1 - alpha) * fake_pair).requires_grad_(True)
        score = discriminator(mixed)
        gradient = torch.autograd.grad(score.sum(), mixed, create_graph=True)[0]
        return ((gradient.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

    for epoch in range(epochs):
        totals = {"l1": 0.0, "g_adv": 0.0, "d": 0.0, "fm": 0.0}
        batches = 0
        generator.train()
        for source, target, mask, sample_weight in loader:
            source, target = source.to(device), target.to(device)
            mask = mask.to(device)
            sample_weight = sample_weight.to(device)
            fake = generator(source)
            loss_d = torch.zeros((), device=device)
            if discriminators is not None:
                opt_d.zero_grad(set_to_none=True)
                if name == "wgangp":
                    real_pair = torch.cat([source, target], 1)
                    fake_pair = torch.cat([source, fake.detach()], 1)
                    terms = []
                    for index, discriminator in enumerate(discriminators):
                        if index:
                            real_pair = torch.nn.functional.avg_pool2d(real_pair, 2)
                            fake_pair = torch.nn.functional.avg_pool2d(fake_pair, 2)
                        terms.append(discriminator(fake_pair).mean() -
                                     discriminator(real_pair).mean() +
                                     10.0 * gradient_penalty(discriminator, real_pair, fake_pair))
                    loss_d = sum(terms) / len(terms)
                else:
                    real_scores = discriminator_scales(source, target, discriminators)
                    fake_scores = discriminator_scales(source, fake.detach(), discriminators)
                    loss_d = sum((bce(real, torch.ones_like(real)) +
                                  bce(synth, torch.zeros_like(synth))) * 0.5
                                 for real, synth in zip(real_scores, fake_scores)) / len(real_scores)
                loss_d.backward()
                opt_d.step()
            opt_g.zero_grad(set_to_none=True)
            fake = generator(source)
            per_sample = ((fake - target).abs() * mask).sum((1, 2, 3)) / \
                (mask.sum((1, 2, 3)) * 3).clamp_min(1)
            loss_l1 = (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
            loss_adv = torch.zeros((), device=device)
            loss_fm = torch.zeros((), device=device)
            if discriminators is not None:
                fake_outputs = discriminator_scales(source, fake, discriminators,
                                                     return_features=True)
                if name == "wgangp":
                    loss_adv = -sum(score.mean() for score, _ in fake_outputs) / len(fake_outputs)
                else:
                    loss_adv = sum(bce(score, torch.ones_like(score))
                                   for score, _ in fake_outputs) / len(fake_outputs)
                if name == "multiscale":
                    with torch.no_grad():
                        real_outputs = discriminator_scales(source, target, discriminators,
                                                            return_features=True)
                    terms = [(a - b).abs().mean()
                             for (_, generated), (_, real) in zip(fake_outputs, real_outputs)
                             for a, b in zip(generated, real)]
                    loss_fm = torch.stack(terms).mean()
            loss_g = 100.0 * loss_l1 + loss_adv + 10.0 * loss_fm
            if not torch.isfinite(loss_g + loss_d):
                raise FloatingPointError(f"loss não finita em {name}, época {epoch + 1}")
            loss_g.backward()
            opt_g.step()
            totals["l1"] += float(loss_l1.detach())
            totals["g_adv"] += float(loss_adv.detach())
            totals["d"] += float(loss_d.detach())
            totals["fm"] += float(loss_fm.detach())
            batches += 1
        history.append({"epoch": epoch + 1,
                        **{key: value / max(batches, 1) for key, value in totals.items()}})
    return generator.eval(), history


def generate(model, records: Sequence[SearchRecord], device) -> dict[int, np.ndarray]:
    import torch
    results = {}
    for record in records:
        histories = [np.load(path).astype(np.float32) for path in record.history_paths]
        if model == "identity":
            result = histories[-1]
        elif isinstance(model, AffineGenerator):
            result = model.predict(histories)
        else:
            source = np.concatenate(histories, -1)
            tensor = torch.from_numpy(source * 2 - 1).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                output = model(tensor)[0].permute(1, 2, 0).cpu().numpy()
            result = np.clip((output + 1) / 2, 0, 1)
        mask = (np.load(record.mask_path) > 0 if record.mask_path and record.mask_path.is_file()
                else np.ones(result.shape[:2], bool))
        result = result * mask[..., None]
        if not np.isfinite(result).all():
            raise FloatingPointError(f"saída inválida para parcela {record.fid}")
        results[record.fid] = result.astype(np.float32)
    return results


def _vector(image: np.ndarray) -> np.ndarray:
    values = parcel_features(image)
    return np.asarray([values[key] for key in sorted(values)], dtype=float)


def _representation_features(track: str, record: SearchRecord,
                             synthetic: dict[int, np.ndarray] | None) -> np.ndarray:
    history = [_vector(np.load(path)) for path in record.history_paths]
    if track == "A":
        return _vector(synthetic[record.fid]) if synthetic is not None else history[-1]
    return np.concatenate(history + ([_vector(synthetic[record.fid])]
                                     if synthetic is not None else []))


def evaluation_units(track: str, records: Sequence[SearchRecord],
                     synthetic: dict[int, np.ndarray] | None, target: str,
                     scenario: str, feature_cache: dict[int, np.ndarray] | None = None):
    """Agrega os dois recortes de cada bloco+dose em uma unidade de campo."""
    if scenario not in {"rs_only", "rs_plus_dose", "dose_only"}:
        raise ValueError(f"cenário desconhecido: {scenario}")
    grouped: dict[tuple[int, str], list[tuple[SearchRecord, np.ndarray]]] = {}
    for record in records:
        features = (np.asarray([record.dose_n], dtype=float) if scenario == "dose_only"
                    else feature_cache[record.fid] if feature_cache is not None
                    else _representation_features(track, record, synthetic))
        grouped.setdefault((record.block, _dose_key(record.dose_n)), []).append((record, features))
    matrices, outcomes, units = [], [], []
    for (block, dose_key), rows in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        values = np.asarray([record.target(target) for record, _ in rows], dtype=float)
        if not np.allclose(values, values[0]):
            raise ValueError(f"alvos divergentes dentro da unidade bloco={block}, dose={dose_key}")
        features = np.mean([item[1] for item in rows], axis=0)
        dose = float(rows[0][0].dose_n)
        if scenario == "rs_plus_dose":
            features = np.concatenate([features, [dose]])
        matrices.append(features)
        outcomes.append(values[0])
        units.append({"unit_id": f"b{block}_d{dose_key}", "block": block,
                      "dose_n": dose, "n_fids": len(rows)})
    return np.stack(matrices), np.asarray(outcomes), units


def extra_trees(seed: int, n_features: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_regression, k=min(8, n_features))),
        ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                      max_features=1.0, random_state=seed, n_jobs=-1)),
    ])


def downstream_predict(train_x, train_y, test_x, seed: int) -> np.ndarray:
    model = extra_trees(seed, train_x.shape[1])
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=float)


def image_metrics(record: SearchRecord, synthetic: np.ndarray) -> dict[str, float]:
    real = np.load(record.target_path).astype(np.float32)
    mask = (np.load(record.mask_path) > 0 if record.mask_path and record.mask_path.is_file()
            else np.ones(real.shape[:2], bool))
    a, b = real[mask], synthetic[mask]
    mse = float(np.mean((a - b) ** 2))
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(np.mean((a - mu_a) * (b - mu_b)))
    ssim = ((2 * mu_a * mu_b + 0.01 ** 2) * (2 * covariance + 0.03 ** 2) /
            ((mu_a ** 2 + mu_b ** 2 + 0.01 ** 2) *
             (var_a + var_b + 0.03 ** 2)))
    dot = np.sum(a * b, axis=1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    sam = float(np.mean(np.arccos(np.clip(dot / np.maximum(norm, 1e-8), -1, 1))))
    return {"mae_image": float(np.mean(np.abs(a - b))), "psnr": psnr,
            "ssim": float(ssim), "sam": sam, "invalid_fraction": 0.0}


def _save_sample(path: Path, record: SearchRecord, synthetic: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    source = np.load(record.history_paths[-1])
    target = np.load(record.target_path)
    panel = np.concatenate([source[..., [2, 1, 0]], synthetic[..., [2, 1, 0]],
                            target[..., [2, 1, 0]]], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(panel / max(float(np.quantile(panel, 0.99)), 1e-6), 0, 1))


def _append(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _existing_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _metrics(y, prediction) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, prediction)))
    return {"r2": float(r2_score(y, prediction)), "rmse": rmse,
            "mae": float(mean_absolute_error(y, prediction))}


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize(predictions: pd.DataFrame, images: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (track, target, scenario), group in predictions[
            predictions.scenario != "dose_only"].groupby(["track", "target", "scenario"]):
        baseline_name = BASELINES[track]
        candidates = [name for name in group.model.unique() if name != baseline_name]
        dose_group = predictions[(predictions.track == track) &
                                 (predictions.target == target) &
                                 (predictions.scenario == "dose_only") &
                                 (predictions.model == "dose_only")]
        dose = dose_group.pivot(index="unit_id", columns="seed", values="prediction")
        target_rows = []
        pvalues = []
        for name in candidates:
            candidate = group[group.model == name].pivot(index="unit_id", columns="seed", values="prediction")
            baseline = group[group.model == baseline_name].pivot(index="unit_id", columns="seed", values="prediction")
            common = candidate.index.intersection(baseline.index).intersection(dose.index)
            candidate_mean = candidate.loc[common].mean(axis=1)
            baseline_mean = baseline.loc[common].mean(axis=1)
            dose_mean = dose.loc[common].mean(axis=1)
            truth = group.drop_duplicates("unit_id").set_index("unit_id").loc[common, "y"]
            blocks = group.drop_duplicates("unit_id").set_index("unit_id").loc[common, "block"]
            base_metrics = _metrics(truth, baseline_mean)
            cand_metrics = _metrics(truth, candidate_mean)
            dose_metrics = _metrics(truth, dose_mean)
            delta = cand_metrics["r2"] - base_metrics["r2"]
            low, high = paired_stratified_bootstrap_delta(
                truth, baseline_mean, candidate_mean, blocks,
                n_bootstrap=bootstrap, seed=2026)
            dose_low, dose_high = paired_stratified_bootstrap_delta(
                truth, dose_mean, candidate_mean, blocks,
                n_bootstrap=bootstrap, seed=2027)
            try:
                pvalue = float(wilcoxon((truth - baseline_mean) ** 2,
                                        (truth - candidate_mean) ** 2,
                                        alternative="greater").pvalue)
            except ValueError:
                pvalue = 1.0
            image_subset = images[(images.track == track) & (images.model == name)]
            target_rows.append({
                "track": track, "target": target, "scenario": scenario, "model": name,
                **cand_metrics, "baseline": baseline_name,
                "baseline_r2": base_metrics["r2"], "delta_r2": delta,
                "ci95_low": low, "ci95_high": high, "p_value": pvalue,
                "dose_only_r2": dose_metrics["r2"],
                "delta_vs_dose": cand_metrics["r2"] - dose_metrics["r2"],
                "dose_ci95_low": dose_low, "dose_ci95_high": dose_high,
                "seed_r2_std": float(group[group.model == name].groupby("seed").apply(
                    lambda frame: r2_score(frame.y, frame.prediction), include_groups=False).std(ddof=0)),
                "psnr": float(image_subset.psnr.mean()) if len(image_subset) else np.nan,
                "ssim": float(image_subset.ssim.mean()) if len(image_subset) else np.nan,
                "sam": float(image_subset.sam.mean()) if len(image_subset) else np.nan,
                "is_oracle": name == "oracle_real",
            })
            pvalues.append(pvalue)
        confirmatory = [index for index, row in enumerate(target_rows)
                        if not row["is_oracle"]]
        adjusted = [1.0] * len(target_rows)
        for index, value in zip(confirmatory,
                                holm_adjust([pvalues[index] for index in confirmatory])):
            adjusted[index] = value
        for row, value in zip(target_rows, adjusted):
            row["p_holm"] = value
            row["passes"] = bool(row["scenario"] == "rs_plus_dose" and
                                 not row["is_oracle"] and row["delta_r2"] >= 0.05 and
                                 row["ci95_low"] > 0 and row["dose_ci95_low"] > 0 and
                                 value < 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def _fit_representation(name: str, train: Sequence[SearchRecord], args, seed: int, device):
    if name == "identity":
        return "identity", []
    if name == "affine":
        return AffineGenerator.fit(train, seed), []
    return train_generator(train, name, epochs=args.epochs, seed=seed,
                           batch_size=args.batch_size, crop_size=args.crop_size,
                           width=args.width, device=device)


def run_selection(args) -> None:
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    prediction_rows = _existing_rows(out / "oof_predictions.csv")
    image_rows = _existing_rows(out / "image_metrics.csv")
    history_rows = _existing_rows(out / "training_history.csv")
    resolved_models = {}
    for track in args.tracks:
        selected = ([name for name in args.models if name in TRACK_MODELS[track]]
                    if args.models else list(DEFAULT_TRACK_MODELS[track]))
        if selected and BASELINES[track] not in selected:
            selected.insert(0, BASELINES[track])
        resolved_models[track] = selected
    metadata = {"phase": "selection", "protocol_version": 2,
                "selection_season": "2022/23",
                "final_season_locked": "2023/24", "tracks": args.tracks,
                "models": resolved_models, "seeds": args.seeds, "epochs": args.epochs,
                "checkpoint_policy": "none", "mask_generated_before_features": True,
                "experimental_unit": "block+dose", "scenarios": [
                    "rs_only", "dose_only", "rs_plus_dose"], "downstream": {
                    "kind": "ExtraTreesRegressor", "n_estimators": 500,
                    "min_samples_leaf": 2, "max_features": 1.0, "select_k": 8}}
    metadata_path = out / "meta.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("diretório de saída pertence a uma configuração diferente")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for track in args.tracks:
        records = load_source_records(args.dataset, args.source_stage4, args.source_table, track)
        pd.DataFrame([{"track": track, "fid": record.fid, "block": record.block,
                       "dose_n": record.dose_n,
                       "history_stages": "R2" if track == "A" else "V8,R2",
                       "target_stage": "R5", "biomass": record.biomass,
                       "productivity": record.productivity}
                      for record in records]).to_csv(out / f"records_{track}.csv", index=False)
        requested = resolved_models[track]
        if not requested:
            continue
        for seed in args.seeds:
            for block in (1, 2, 3, 4):
                train = [record for record in records if record.block != block]
                test = [record for record in records if record.block == block]
                if {record.fid for record in train} & {record.fid for record in test}:
                    raise AssertionError("vazamento de parcela no fold externo")
                test_unit_count = len({(record.block, _dose_key(record.dose_n)) for record in test})
                dose_completed = [row for row in prediction_rows
                                  if row["track"] == track and row["model"] == "dose_only"
                                  and int(row["seed"]) == seed and int(row["fold"]) == block]
                expected_dose = test_unit_count * len(args.targets)
                if len(dose_completed) != expected_dose:
                    prediction_rows = [row for row in prediction_rows
                                       if not (row["track"] == track and row["model"] == "dose_only"
                                               and int(row["seed"]) == seed
                                               and int(row["fold"]) == block)]
                    for target in args.targets:
                        train_x, train_y, _ = evaluation_units(
                            track, train, None, target, "dose_only")
                        test_x, test_y, test_units = evaluation_units(
                            track, test, None, target, "dose_only")
                        prediction = downstream_predict(train_x, train_y, test_x, seed)
                        for unit, truth, value in zip(test_units, test_y, prediction):
                            prediction_rows.append({"track": track, "target": target,
                                                    "scenario": "dose_only",
                                                    "model": "dose_only", "seed": seed,
                                                    "fold": block, **unit, "y": float(truth),
                                                    "prediction": float(value)})
                    _append(prediction_rows, out / "oof_predictions.csv")
                for name in requested:
                    completed = [row for row in prediction_rows
                                 if row["track"] == track and row["model"] == name
                                 and int(row["seed"]) == seed and int(row["fold"]) == block]
                    expected_predictions = test_unit_count * len(args.targets) * 2
                    if len(completed) == expected_predictions:
                        continue
                    # Uma combinação interrompida é refeita por inteiro; folds já
                    # concluídos continuam válidos sem exigir pesos serializados.
                    prediction_rows = [row for row in prediction_rows
                                       if not (row["track"] == track and row["model"] == name
                                               and int(row["seed"]) == seed
                                               and int(row["fold"]) == block)]
                    image_rows = [row for row in image_rows
                                  if not (row["track"] == track and row["model"] == name
                                          and int(row["seed"]) == seed
                                          and int(row["fold"]) == block)]
                    history_rows = [row for row in history_rows
                                    if not (row["track"] == track and row["model"] == name
                                            and int(row["seed"]) == seed
                                            and int(row["fold"]) == block)]
                    if name == "history":
                        train_fake = test_fake = None
                        history = []
                    elif name == "oracle_real":
                        train_fake = {record.fid: np.load(record.target_path).astype(np.float32)
                                      for record in train}
                        test_fake = {record.fid: np.load(record.target_path).astype(np.float32)
                                     for record in test}
                        history = []
                        for record in test:
                            image_rows.append({"track": track, "model": name, "seed": seed,
                                               "fold": block, "fid": record.fid,
                                               **image_metrics(record, test_fake[record.fid])})
                    else:
                        representation, history = _fit_representation(name, train, args, seed, device)
                        train_fake = generate(representation, train, device)
                        test_fake = generate(representation, test, device)
                        for record in test:
                            image_rows.append({"track": track, "model": name, "seed": seed,
                                               "fold": block, "fid": record.fid,
                                               **image_metrics(record, test_fake[record.fid])})
                        if args.sample_images and block == 1:
                            for record in test[:args.sample_images]:
                                _save_sample(out / "samples" / track / name /
                                             f"s{seed}_p{record.fid}.png", record,
                                             test_fake[record.fid])
                    for row in history:
                        history_rows.append({"track": track, "model": name, "seed": seed,
                                             "fold": block, **row})
                    train_feature_cache = {
                        record.fid: _representation_features(track, record, train_fake)
                        for record in train}
                    test_feature_cache = {
                        record.fid: _representation_features(track, record, test_fake)
                        for record in test}
                    for target in args.targets:
                        for scenario in ("rs_only", "rs_plus_dose"):
                            train_x, train_y, _ = evaluation_units(
                                track, train, train_fake, target, scenario,
                                feature_cache=train_feature_cache)
                            test_x, test_y, test_units = evaluation_units(
                                track, test, test_fake, target, scenario,
                                feature_cache=test_feature_cache)
                            prediction = downstream_predict(train_x, train_y, test_x, seed)
                            for unit, truth, value in zip(test_units, test_y, prediction):
                                prediction_rows.append({"track": track, "target": target,
                                                        "scenario": scenario, "model": name,
                                                        "seed": seed, "fold": block, **unit,
                                                        "y": float(truth),
                                                        "prediction": float(value)})
                    _append(prediction_rows, out / "oof_predictions.csv")
                    _append(image_rows, out / "image_metrics.csv")
                    _append(history_rows, out / "training_history.csv")
                    if name in NEURAL_MODELS:
                        del representation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
    summary = summarize(pd.DataFrame(prediction_rows), pd.DataFrame(image_rows), args.bootstrap)
    summary.to_csv(out / "summary.csv", index=False)
    winners = {}
    if len(summary):
        biomass = summary[(summary.target.str.lower() == "biomassa") &
                          (summary.scenario == "rs_plus_dose")]
        for track, group in biomass.groupby("track"):
            eligible = group[group.passes]
            if len(eligible):
                winner = eligible.sort_values(["delta_r2", "rmse"], ascending=[False, True]).iloc[0]
                winners[track] = str(winner.model)
    (out / "winners.json").write_text(json.dumps(winners, indent=2) + "\n")
    assert_no_weights(out)


def run_final(args) -> None:
    import torch
    if not args.target_stage4 or not args.target_table:
        raise ValueError("final requer --target-stage4 e --target-table")
    winners = json.loads(Path(args.winners).read_text())
    if not winners:
        raise ValueError("a seleção não produziu vencedor confirmatório")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    rows = []
    for track, winner in winners.items():
        if winner not in TRACK_MODELS[track] or winner == BASELINES[track]:
            raise ValueError(f"vencedor inválido para trilha {track}: {winner}")
        source = load_source_records(args.dataset, args.source_stage4, args.source_table, track)
        target_records = load_target_records(args.target_stage4, args.target_table, track)
        for seed in args.seeds:
            for target_name in args.targets:
                train_x, train_y, _ = evaluation_units(
                    track, source, None, target_name, "dose_only")
                test_x, test_y, test_units = evaluation_units(
                    track, target_records, None, target_name, "dose_only")
                prediction = downstream_predict(train_x, train_y, test_x, seed)
                for unit, truth, value in zip(test_units, test_y, prediction):
                    rows.append({"track": track, "target": target_name,
                                 "scenario": "dose_only", "model": "dose_only",
                                 "seed": seed, **unit, "y": float(truth),
                                 "prediction": float(value)})
            baseline = BASELINES[track]
            variants = [baseline, winner]
            for name in variants:
                if name == "history":
                    source_fake = target_fake = None
                else:
                    representation, _ = _fit_representation(name, source, args, seed, device)
                    source_fake = generate(representation, source, device)
                    target_fake = generate(representation, target_records, device)
                source_feature_cache = {
                    record.fid: _representation_features(track, record, source_fake)
                    for record in source}
                target_feature_cache = {
                    record.fid: _representation_features(track, record, target_fake)
                    for record in target_records}
                for target_name in args.targets:
                    for scenario in ("rs_only", "rs_plus_dose"):
                        train_x, train_y, _ = evaluation_units(
                            track, source, source_fake, target_name, scenario,
                            feature_cache=source_feature_cache)
                        test_x, test_y, test_units = evaluation_units(
                            track, target_records, target_fake, target_name, scenario,
                            feature_cache=target_feature_cache)
                        prediction = downstream_predict(train_x, train_y, test_x, seed)
                        for unit, truth, value in zip(test_units, test_y, prediction):
                            rows.append({"track": track, "target": target_name,
                                         "scenario": scenario, "model": name,
                                         "seed": seed, **unit, "y": float(truth),
                                         "prediction": float(value)})
                if name in NEURAL_MODELS:
                    del representation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        _append(rows, out / "final_predictions.csv")
    frame = pd.DataFrame(rows)
    summaries = []
    for (track, target_name, scenario, model), group in frame.groupby(
            ["track", "target", "scenario", "model"]):
        ensemble = group.pivot(index="unit_id", columns="seed", values="prediction").mean(axis=1)
        truth = group.drop_duplicates("unit_id").set_index("unit_id").loc[ensemble.index, "y"]
        summaries.append({"track": track, "target": target_name,
                          "scenario": scenario, "model": model,
                          **_metrics(truth, ensemble)})
    pd.DataFrame(summaries).to_csv(out / "final_summary.csv", index=False)
    comparison = summarize(
        frame.assign(fold=frame["block"]),
        pd.DataFrame(columns=["track", "model", "psnr", "ssim", "sam"]),
        args.bootstrap,
    )
    comparison.to_csv(out / "final_comparison.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"phase": "final", "winners": winners,
                                                "checkpoint_policy": "none"}, indent=2) + "\n")
    assert_no_weights(out)


def assert_no_weights(root: str | Path) -> None:
    forbidden = [path for path in Path(root).rglob("*")
                 if path.is_file() and path.suffix.lower() in {".pt", ".pth"}]
    if forbidden:
        raise AssertionError(f"artefatos de pesos proibidos: {forbidden[:5]}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", choices=("selection", "final"))
    result.add_argument("--dataset", required=True)
    result.add_argument("--source-stage4", required=True)
    result.add_argument("--source-table", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--target-stage4")
    result.add_argument("--target-table")
    result.add_argument("--winners")
    result.add_argument("--tracks", nargs="+", choices=("A", "B"), default=["A", "B"])
    result.add_argument("--models", nargs="+", default=[])
    result.add_argument("--targets", nargs="+", choices=("Biomassa", "Produtividade"),
                        default=["Biomassa", "Produtividade"])
    result.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--crop-size", type=int, default=128)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--bootstrap", type=int, default=10_000)
    result.add_argument("--sample-images", type=int, default=2)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    unknown = set(args.models) - set().union(*map(set, TRACK_MODELS.values()))
    if unknown:
        raise ValueError(f"modelos desconhecidos: {sorted(unknown)}")
    if args.epochs < 1 or args.width < 8 or args.crop_size < 16 or args.crop_size % 8:
        raise ValueError("epochs>=1, width>=8 e crop-size>=16 divisível por 8 são obrigatórios")
    if args.phase == "selection":
        run_selection(args)
    else:
        if not args.winners:
            raise ValueError("final requer --winners gerado pela seleção")
        run_final(args)


if __name__ == "__main__":
    main()
