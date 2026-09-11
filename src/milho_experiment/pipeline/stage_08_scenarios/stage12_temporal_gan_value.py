#!/usr/bin/env python3
"""CV aninhada do valor preditivo de GANs fenológicas temporais.

Horizontes oficiais de 2023/24:

* V6 + V8 -> V13
* V6 + V8 + V13 + R2 -> R5

Para cada fold externo, os canais auxiliares da GAN são escolhidos somente nos
três blocos de treino. A avaliação downstream separa aumento de dados, fusão
de atributos e o cenário combinado. Nenhuma afirmação de ganho usa parcelas
vistas pela GAN ou pelo regressor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from milho_experiment.pix2pix import enable_imports
from milho_experiment.pipeline.stage_06_datasets.stage6_features_ortho import parcel_features
enable_imports()
from milho_experiment.temporal import (
    AUX_CANDIDATES,
    HORIZONS,
    TemporalRecord,
    assert_disjoint,
    build_records,
    expected_input_channels,
    load_record,
    paired_stratified_bootstrap_delta,
    temporal_input,
)


SCENARIOS = (
    "augment_real",
    "augment_synthetic",
    "augment_hybrid",
    "forecast_history",
    "forecast_synthetic",
    "forecast_fusion",
    "combined",
)


def _slug(names: Sequence[str]) -> str:
    if not names:
        return "rrenir"
    readable = "_".join(name.lower().replace("glcm_", "g") for name in names)
    digest = hashlib.sha1(",".join(names).encode()).hexdigest()[:8]
    return f"{readable[:80]}_{digest}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _device(requested: str):
    import torch
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível")
    return torch.device(requested)


class _TemporalTorchDataset:
    """Dataset em memória; atributos GLCM são calculados uma vez por treino."""

    def __init__(self, records: Sequence[TemporalRecord], aux_names: Sequence[str],
                 load_size: int, crop_size: int, seed: int):
        import torch
        from torch.utils.data import Dataset

        class DatasetImpl(Dataset):
            def __init__(inner):
                inner.rows = []
                for record in records:
                    history, target = load_record(record)
                    source = temporal_input(history, aux_names)
                    inner.rows.append((torch.from_numpy(source).permute(2, 0, 1),
                                       torch.from_numpy(target).permute(2, 0, 1)))
                inner.generator = torch.Generator().manual_seed(seed)

            def __len__(inner):
                return len(inner.rows)

            def __getitem__(inner, index):
                import torch.nn.functional as functional

                source, target = inner.rows[index]
                source = source.float()
                target = target.float()
                if load_size != source.shape[-1]:
                    source = functional.interpolate(source[None], size=(load_size, load_size),
                                                    mode="bilinear", align_corners=False)[0]
                    target = functional.interpolate(target[None], size=(load_size, load_size),
                                                    mode="bilinear", align_corners=False)[0]
                if load_size > crop_size:
                    high = load_size - crop_size + 1
                    y = int(torch.randint(high, (1,), generator=inner.generator))
                    x = int(torch.randint(high, (1,), generator=inner.generator))
                    source = source[:, y:y + crop_size, x:x + crop_size]
                    target = target[:, y:y + crop_size, x:x + crop_size]
                if bool(torch.randint(2, (1,), generator=inner.generator)):
                    source = source.flip(-1)
                    target = target.flip(-1)
                return source * 2 - 1, target * 2 - 1

        self.dataset = DatasetImpl()


def _new_generator(input_nc: int, device):
    from models.networks import define_G
    return define_G(input_nc, 3, 64, "unet_256", norm="batch").to(device)


def train_gan(records: Sequence[TemporalRecord], aux_names: Sequence[str], *,
              epochs: int, seed: int, batch_size: int, device, checkpoint: Path):
    """Treina ou restaura um Pix2Pix temporal para um fold/configuração."""
    import torch
    from models.networks import GANLoss, define_D
    from torch.utils.data import DataLoader

    if not records:
        raise ValueError("fold de treino da GAN vazio")
    _seed_everything(seed)
    input_nc = expected_input_channels(records[0].history_stages, aux_names)
    generator = _new_generator(input_nc, device)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        generator.load_state_dict(state)
        return generator.eval()

    dataset = _TemporalTorchDataset(records, aux_names, 286, 256, seed).dataset
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
                        generator=loader_generator, num_workers=0)
    discriminator = define_D(input_nc + 3, 64, "basic", norm="batch").to(device)
    gan_loss = GANLoss("vanilla").to(device)
    l1_loss = torch.nn.L1Loss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()
    for _ in range(epochs):
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            fake = generator(source)
            opt_d.zero_grad(set_to_none=True)
            loss_d = 0.5 * (
                gan_loss(discriminator(torch.cat([source, fake.detach()], 1)), False) +
                gan_loss(discriminator(torch.cat([source, target], 1)), True)
            )
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake = generator(source)
            loss_g = (gan_loss(discriminator(torch.cat([source, fake], 1)), True) +
                      100.0 * l1_loss(fake, target))
            loss_g.backward()
            opt_g.step()

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": generator.state_dict(), "input_nc": input_nc,
                "aux_names": list(aux_names), "epochs": epochs, "seed": seed}, checkpoint)
    return generator.eval()


def generate(generator, records: Sequence[TemporalRecord], aux_names: Sequence[str],
             device) -> dict[int, np.ndarray]:
    import torch

    result = {}
    for record in records:
        history, _ = load_record(record)
        source = temporal_input(history, aux_names)
        tensor = torch.from_numpy(source * 2 - 1).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            fake = generator(tensor)[0].permute(1, 2, 0).cpu().numpy()
        result[record.fid] = np.clip((fake + 1) / 2, 0, 1).astype(np.float32)
    return result


def _feature_vector(image: np.ndarray) -> np.ndarray:
    values = parcel_features(image)
    return np.asarray([values[key] for key in sorted(values)], dtype=float)


def _glcm_feature_error(real: np.ndarray, fake: np.ndarray) -> float:
    real_features = parcel_features(real)
    fake_features = parcel_features(fake)
    keys = sorted(key for key in real_features if key.startswith("glcm"))
    if not keys:
        return np.nan
    return float(np.mean([abs(float(real_features[key]) - float(fake_features[key]))
                          for key in keys]))


def feature_bundle(records: Sequence[TemporalRecord],
                   fakes: dict[int, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    bundle = {}
    for record in records:
        history, target = load_record(record)
        history_features = np.concatenate([_feature_vector(image) for image in history])
        bundle[record.fid] = {
            "H": history_features,
            "T": _feature_vector(target),
            "S": _feature_vector(fakes[record.fid]),
        }
    return bundle


def _append_dose(matrix: np.ndarray, doses: Sequence[object], enabled: bool) -> np.ndarray:
    if not enabled:
        return matrix
    numeric = np.asarray([float(value) for value in doses], dtype=float)[:, None]
    return np.concatenate([matrix, numeric], axis=1)


def scenario_matrices(scenario: str, train_records: Sequence[TemporalRecord],
                      test_records: Sequence[TemporalRecord], train_bundle, test_bundle,
                      target: str, include_dose: bool):
    def stack(records, bundle, key):
        if key == "HS":
            return np.stack([np.concatenate([bundle[r.fid]["H"], bundle[r.fid]["S"]])
                             for r in records])
        if key == "HT":
            return np.stack([np.concatenate([bundle[r.fid]["H"], bundle[r.fid]["T"]])
                             for r in records])
        return np.stack([bundle[r.fid][key] for r in records])

    y_train = np.asarray([record.target(target) for record in train_records])
    groups = np.asarray([record.block for record in train_records])
    train_doses = [record.dose_n for record in train_records]
    test_doses = [record.dose_n for record in test_records]

    if scenario == "augment_real":
        x_train, x_test = stack(train_records, train_bundle, "T"), stack(test_records, test_bundle, "T")
    elif scenario == "augment_synthetic":
        x_train, x_test = stack(train_records, train_bundle, "S"), stack(test_records, test_bundle, "T")
    elif scenario == "augment_hybrid":
        real = stack(train_records, train_bundle, "T")
        synth = stack(train_records, train_bundle, "S")
        x_train = np.concatenate([real, synth])
        x_test = stack(test_records, test_bundle, "T")
        y_train = np.tile(y_train, 2)
        groups = np.tile(groups, 2)
        train_doses = train_doses * 2
    elif scenario == "forecast_history":
        x_train, x_test = stack(train_records, train_bundle, "H"), stack(test_records, test_bundle, "H")
    elif scenario == "forecast_synthetic":
        x_train, x_test = stack(train_records, train_bundle, "S"), stack(test_records, test_bundle, "S")
    elif scenario == "forecast_fusion":
        x_train, x_test = stack(train_records, train_bundle, "HS"), stack(test_records, test_bundle, "HS")
    elif scenario == "combined":
        real = stack(train_records, train_bundle, "HT")
        synth = stack(train_records, train_bundle, "HS")
        x_train = np.concatenate([real, synth])
        x_test = stack(test_records, test_bundle, "HS")
        y_train = np.tile(y_train, 2)
        groups = np.tile(groups, 2)
        train_doses = train_doses * 2
    else:
        raise ValueError(f"cenário desconhecido: {scenario}")

    x_train = _append_dose(x_train, train_doses, include_dose)
    x_test = _append_dose(x_test, test_doses, include_dose)
    return x_train, y_train, groups, x_test


def _model_searches(seed: int, n_features: int, n_samples: int):
    ks = sorted({min(k, n_features) for k in (3, 5, 8)})
    common = [("impute", SimpleImputer(strategy="median")),
              ("select", SelectKBest(f_regression))]
    plsr = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] +
                    [("model", PLSRegression())])
    max_components = max(1, min(3, n_samples - 2))
    plsr_grid = [{"select__k": [k], "model__n_components": list(range(1, min(k, max_components) + 1))}
                 for k in ks]

    elastic = Pipeline(common[:1] + [("scale", StandardScaler())] + common[1:] +
                       [("model", ElasticNet(max_iter=20_000, random_state=seed))])
    elastic_grid = {"select__k": ks, "model__alpha": [0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.1, 0.5, 0.9]}

    trees = Pipeline(common + [("model", ExtraTreesRegressor(
        n_estimators=300, random_state=seed, n_jobs=1))])
    trees_grid = {"select__k": ks, "model__min_samples_leaf": [1, 4],
                  "model__max_features": [1.0]}
    return (("plsr", plsr, plsr_grid), ("elasticnet", elastic, elastic_grid),
            ("extratrees", trees, trees_grid))


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
                x_test: np.ndarray, *, seed: int, n_jobs: int) -> tuple[np.ndarray, str]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("seleção downstream requer ao menos dois blocos de treino")
    cv = GroupKFold(min(3, len(unique_groups)))
    best = None
    for name, estimator, grid in _model_searches(seed, x_train.shape[1], len(x_train)):
        search = GridSearchCV(estimator, grid, cv=cv, scoring="r2", n_jobs=n_jobs,
                              error_score=np.nan, refit=True)
        search.fit(x_train, y_train, groups=groups)
        score = float(search.best_score_) if np.isfinite(search.best_score_) else -math.inf
        if best is None or score > best[0]:
            best = (score, name, search.best_estimator_)
    if best is None:
        raise RuntimeError("nenhum modelo downstream pôde ser ajustado")
    return np.asarray(best[2].predict(x_test)).ravel(), best[1]


def _checkpoint(root: Path, horizon: str, outer: int, inner: int | str,
                attrs: Sequence[str], epochs: int, seed: int) -> Path:
    return root / horizon / f"outer{outer}" / f"inner{inner}" / _slug(attrs) / \
        f"e{epochs}_s{seed}_net_G.pth"


def screen_score(records: Sequence[TemporalRecord], target: str, attrs: Sequence[str], *,
                 horizon: str, outer_block: int, epochs: int, seed: int,
                 batch_size: int, device, cache: Path, n_jobs: int) -> float:
    """ΔR² interno do cenário combinado contra o histórico sem sintético."""
    y_all, history_all, combined_all = [], [], []
    train_blocks = sorted({record.block for record in records})
    for val_block in train_blocks:
        train = [record for record in records if record.block != val_block]
        val = [record for record in records if record.block == val_block]
        assert_disjoint(train, val)
        ckpt = _checkpoint(cache, horizon, outer_block, val_block, attrs, epochs, seed)
        generator = train_gan(train, attrs, epochs=epochs, seed=seed,
                              batch_size=batch_size, device=device, checkpoint=ckpt)
        fake_train = generate(generator, train, attrs, device)
        fake_val = generate(generator, val, attrs, device)
        train_bundle = feature_bundle(train, fake_train)
        val_bundle = feature_bundle(val, fake_val)
        for scenario, destination in (("forecast_history", history_all),
                                      ("combined", combined_all)):
            matrices = scenario_matrices(scenario, train, val, train_bundle, val_bundle,
                                         target, include_dose=False)
            prediction, _ = fit_predict(*matrices, seed=seed + val_block, n_jobs=n_jobs)
            destination.extend(prediction.tolist())
        y_all.extend(record.target(target) for record in val)
    return float(r2_score(y_all, combined_all) - r2_score(y_all, history_all))


def progressive_select(records: Sequence[TemporalRecord], target: str, *, horizon: str,
                       outer_block: int, max_aux: int, epochs: int, seed: int,
                       batch_size: int, device, cache: Path, n_jobs: int):
    selected: tuple[str, ...] = ()
    current = screen_score(records, target, selected, horizon=horizon,
                           outer_block=outer_block, epochs=epochs, seed=seed,
                           batch_size=batch_size, device=device, cache=cache, n_jobs=n_jobs)
    trace = [{"step": 0, "candidate": "", "selected": "", "delta_r2": current,
              "accepted": True}]
    for step in range(1, max_aux + 1):
        candidates = []
        for name in AUX_CANDIDATES:
            if name in selected:
                continue
            attrs = (*selected, name)
            score = screen_score(records, target, attrs, horizon=horizon,
                                 outer_block=outer_block, epochs=epochs, seed=seed,
                                 batch_size=batch_size, device=device, cache=cache, n_jobs=n_jobs)
            candidates.append((score, name, attrs))
            trace.append({"step": step, "candidate": name, "selected": ",".join(attrs),
                          "delta_r2": score, "accepted": False})
        best_score, best_name, best_attrs = max(candidates, key=lambda row: (row[0], row[1]))
        if best_score <= current:
            break
        selected, current = best_attrs, best_score
        for row in reversed(trace):
            if row["step"] == step and row["candidate"] == best_name:
                row["accepted"] = True
                break
    return selected, trace


def crossfit_train_fakes(records: Sequence[TemporalRecord], attrs: Sequence[str], *,
                         horizon: str, outer_block: int, epochs: int, seed: int,
                         batch_size: int, device, cache: Path) -> dict[int, np.ndarray]:
    fakes = {}
    for val_block in sorted({record.block for record in records}):
        train = [record for record in records if record.block != val_block]
        val = [record for record in records if record.block == val_block]
        ckpt = _checkpoint(cache, horizon, outer_block, f"final{val_block}", attrs, epochs, seed)
        generator = train_gan(train, attrs, epochs=epochs, seed=seed,
                              batch_size=batch_size, device=device, checkpoint=ckpt)
        fakes.update(generate(generator, val, attrs, device))
    if set(fakes) != {record.fid for record in records}:
        raise AssertionError("cross-fitting não gerou exatamente uma imagem por parcela de treino")
    return fakes


def _write_incremental(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "checkpoints"
    device = _device(args.device)
    prediction_rows: list[dict] = []
    image_rows: list[dict] = []
    selection_rows: list[dict] = []
    if not 0 <= args.max_aux <= len(AUX_CANDIDATES):
        raise ValueError(f"max_aux deve estar entre 0 e {len(AUX_CANDIDATES)}")

    metadata = {
        "stage4": str(Path(args.stage4).resolve()),
        "table": str(Path(args.table).resolve()),
        "horizons": args.horizons,
        "targets": args.targets,
        "screen_epochs": args.screen_epochs,
        "final_epochs": args.final_epochs,
        "screen_seed": args.screen_seed,
        "final_seeds": args.final_seeds,
        "max_aux": args.max_aux,
        "candidate_channels": list(AUX_CANDIDATES),
        "selection_score": "inner combined minus forecast_history R2",
        "success": "delta_r2 > 0 and paired stratified bootstrap lower95 > 0",
    }
    (out / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    for horizon in args.horizons:
        records = build_records(args.stage4, args.table, horizon)
        record_rows = [{"horizon": horizon, "fid": r.fid, "block": r.block,
                        "dose_n": r.dose_n, "history_stages": ",".join(r.history_stages),
                        "target_stage": r.target_stage, "biomass": r.biomass,
                        "productivity": r.productivity} for r in records]
        pd.DataFrame(record_rows).to_csv(out / f"records_{horizon}.csv", index=False)
        blocks = sorted({record.block for record in records})
        block_counts = pd.Series([record.block for record in records]).value_counts().to_dict()
        if len(records) != 24 or blocks != [1, 2, 3, 4] or set(block_counts.values()) != {6}:
            raise ValueError("o protocolo 23/24 requer 24 parcelas, seis em cada bloco; "
                             f"recebeu n={len(records)}, contagens={block_counts}")

        for target in args.targets:
            for outer_block in blocks:
                train = [record for record in records if record.block != outer_block]
                test = [record for record in records if record.block == outer_block]
                assert_disjoint(train, test)
                if args.fixed_attrs is None:
                    attrs, trace = progressive_select(
                        train, target, horizon=horizon, outer_block=outer_block,
                        max_aux=args.max_aux, epochs=args.screen_epochs,
                        seed=args.screen_seed, batch_size=args.batch_size,
                        device=device, cache=cache / "screen", n_jobs=args.n_jobs)
                else:
                    attrs = tuple(args.fixed_attrs)
                    invalid = set(attrs) - set(AUX_CANDIDATES)
                    if invalid:
                        raise ValueError(f"canais fixos inválidos: {sorted(invalid)}")
                    trace = [{"step": 0, "candidate": "fixed", "selected": ",".join(attrs),
                              "delta_r2": np.nan, "accepted": True}]
                final_attrs = ",".join(attrs)
                for row in trace:
                    selection_rows.append({"horizon": horizon, "target": target,
                                           "outer_block": outer_block,
                                           "is_final": bool(row["accepted"] and
                                                            row["selected"] == final_attrs),
                                           **row})
                _write_incremental(selection_rows, out / "channel_selection.csv")

                for seed in args.final_seeds:
                    fake_train = crossfit_train_fakes(
                        train, attrs, horizon=horizon, outer_block=outer_block,
                        epochs=args.final_epochs, seed=seed, batch_size=args.batch_size,
                        device=device, cache=cache / "final")
                    outer_ckpt = _checkpoint(cache / "final", horizon, outer_block, "outer",
                                             attrs, args.final_epochs, seed)
                    generator = train_gan(train, attrs, epochs=args.final_epochs, seed=seed,
                                          batch_size=args.batch_size, device=device,
                                          checkpoint=outer_ckpt)
                    fake_test = generate(generator, test, attrs, device)
                    train_bundle = feature_bundle(train, fake_train)
                    test_bundle = feature_bundle(test, fake_test)

                    for record in test:
                        _, real_target = load_record(record)
                        fake = fake_test[record.fid]
                        real_ndvi = ((real_target[..., 2] - real_target[..., 0]) /
                                     (real_target[..., 2] + real_target[..., 0] + 1e-8))
                        fake_ndvi = ((fake[..., 2] - fake[..., 0]) /
                                     (fake[..., 2] + fake[..., 0] + 1e-8))
                        image_rows.append({
                            "horizon": horizon, "target": target, "outer_block": outer_block,
                            "fid": record.fid, "seed": seed, "attrs": ",".join(attrs),
                            "l1": float(np.mean(np.abs(fake - real_target))),
                            "ndvi_error": float(abs(np.mean(fake_ndvi) - np.mean(real_ndvi))),
                            "glcm_feature_mae": _glcm_feature_error(real_target, fake),
                        })

                    for rs_mode in args.rs_modes:
                        include_dose = rs_mode == "rs_dose"
                        for scenario in SCENARIOS:
                            matrices = scenario_matrices(
                                scenario, train, test, train_bundle, test_bundle,
                                target, include_dose)
                            pred, model = fit_predict(*matrices, seed=seed + outer_block,
                                                      n_jobs=args.n_jobs)
                            for record, value in zip(test, pred):
                                prediction_rows.append({
                                    "horizon": horizon, "target": target,
                                    "rs_mode": rs_mode, "scenario": scenario,
                                    "outer_block": outer_block, "fid": record.fid,
                                    "seed": seed, "attrs": ",".join(attrs), "model": model,
                                    "y": record.target(target), "prediction": float(value),
                                })
                    _write_incremental(prediction_rows, out / "oof_predictions.csv")
                    _write_incremental(image_rows, out / "gan_image_metrics.csv")

    summarize(prediction_rows, out, args.bootstrap)


def summarize(prediction_rows: list[dict], out: Path, bootstrap: int) -> None:
    predictions = pd.DataFrame(prediction_rows)
    identity = ["horizon", "target", "rs_mode", "scenario", "seed", "fid"]
    if predictions.duplicated(identity).any():
        raise AssertionError("mais de uma predição OOF para a mesma parcela/cenário/semente")
    counts = predictions.groupby(identity[:-1]).fid.nunique()
    if not (counts == 24).all():
        raise AssertionError(f"predições OOF incompletas; contagens observadas: {sorted(counts.unique())}")
    per_seed_rows = []
    for group_key, frame in predictions.groupby(
            ["horizon", "target", "rs_mode", "scenario", "seed"]):
        y, pred = frame.y.to_numpy(), frame.prediction.to_numpy()
        per_seed_rows.append({
            "horizon": group_key[0], "target": group_key[1], "rs_mode": group_key[2],
            "scenario": group_key[3], "seed": group_key[4], "n": len(frame),
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
        })
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    per_seed.groupby(["horizon", "target", "rs_mode", "scenario"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        seeds=("seed", "nunique"),
    ).to_csv(out / "metrics_seed_summary.csv", index=False)

    key = ["horizon", "target", "rs_mode", "scenario", "outer_block", "fid", "y"]
    ensemble = predictions.groupby(key, as_index=False).agg(
        prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
        seeds=("seed", "nunique"))
    ensemble.to_csv(out / "oof_predictions_ensemble.csv", index=False)

    metric_rows = []
    for group_key, frame in ensemble.groupby(["horizon", "target", "rs_mode", "scenario"]):
        y, pred = frame.y.to_numpy(), frame.prediction.to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        metric_rows.append({
            "horizon": group_key[0], "target": group_key[1], "rs_mode": group_key[2],
            "scenario": group_key[3], "n": len(frame), "r2": float(r2_score(y, pred)),
            "rmse": rmse, "mae": float(mean_absolute_error(y, pred)),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "metrics.csv", index=False)

    contrasts = (
        ("augment_synthetic_vs_real", "augment_synthetic", "augment_real"),
        ("augment_hybrid_vs_real", "augment_hybrid", "augment_real"),
        ("forecast_synthetic_vs_history", "forecast_synthetic", "forecast_history"),
        ("forecast_fusion_vs_history", "forecast_fusion", "forecast_history"),
        ("combined_vs_history", "combined", "forecast_history"),
        ("combined_vs_fusion", "combined", "forecast_fusion"),
    )
    contrast_rows = []
    for (horizon, target, rs_mode), frame in ensemble.groupby(["horizon", "target", "rs_mode"]):
        wide = frame.pivot(index=["outer_block", "fid", "y"], columns="scenario",
                           values="prediction").reset_index()
        for contrast, candidate, baseline in contrasts:
            if candidate not in wide or baseline not in wide:
                continue
            y = wide.y.to_numpy()
            base = wide[baseline].to_numpy()
            cand = wide[candidate].to_numpy()
            base_r2 = float(r2_score(y, base))
            candidate_r2 = float(r2_score(y, cand))
            lo, hi = paired_stratified_bootstrap_delta(
                y, base, cand, wide.outer_block.to_numpy(), n_bootstrap=bootstrap)
            delta = candidate_r2 - base_r2
            contrast_rows.append({
                "horizon": horizon, "target": target, "rs_mode": rs_mode,
                "contrast": contrast,
                "candidate": candidate, "baseline": baseline,
                "baseline_r2": base_r2, "candidate_r2": candidate_r2,
                "delta_r2": delta, "delta_r2_lo": lo, "delta_r2_hi": hi,
                "improvement_supported": bool(delta > 0 and lo > 0),
                "absolute_useful": bool(candidate_r2 > 0),
            })
    pd.DataFrame(contrast_rows).to_csv(out / "contrasts.csv", index=False)


def prepare(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        records = build_records(args.stage4, args.table, horizon)
        rows = [{"horizon": horizon, "fid": r.fid, "block": r.block,
                 "dose_n": r.dose_n, "history_stages": ",".join(r.history_stages),
                 "target_stage": r.target_stage, "biomass": r.biomass,
                 "productivity": r.productivity,
                 "input_nc_base": expected_input_channels(r.history_stages, ())}
                for r in records]
        pd.DataFrame(rows).to_csv(out / f"records_{horizon}.csv", index=False)
        print(f"{horizon}: {len(records)} parcelas completas")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--stage4", required=True)
        cmd.add_argument("--table", required=True)
        cmd.add_argument("--out", required=True)
        cmd.add_argument("--horizons", nargs="+", choices=tuple(HORIZONS), default=["V13", "R5"])
        if name == "run":
            cmd.add_argument("--targets", nargs="+", choices=["Biomassa", "Produtividade"],
                             default=["Biomassa", "Produtividade"])
            cmd.add_argument("--rs-modes", nargs="+", choices=["rs_puro", "rs_dose"],
                             default=["rs_puro", "rs_dose"])
            cmd.add_argument("--screen-epochs", type=int, default=50)
            cmd.add_argument("--final-epochs", type=int, default=200)
            cmd.add_argument("--screen-seed", type=int, default=7)
            cmd.add_argument("--final-seeds", type=int, nargs="+", default=[7, 11, 23])
            cmd.add_argument("--max-aux", type=int, default=4)
            cmd.add_argument("--batch-size", type=int, default=4)
            cmd.add_argument("--bootstrap", type=int, default=10_000)
            cmd.add_argument("--device", default="auto")
            cmd.add_argument("--n-jobs", type=int, default=1)
            cmd.add_argument("--fixed-attrs", nargs="*", default=None,
                             help="pula a triagem; útil para smoke tests")
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
