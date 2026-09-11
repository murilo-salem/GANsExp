#!/usr/bin/env python3
"""Fusao tardia aninhada de R2 real, R5 sintetico e dose para biomassa R5.

``run`` seleciona todos os hiperparametros e os pesos da fusao dentro de um
leave-one-block-out interno e mede o resultado em um leave-one-block-out
externo de 2022/23. ``replicate`` so aceita uma decisao confirmatoria produzida
por ``run`` e aplica o pipeline congelado em 2023/24.

Os geradores sao treinados e usados integralmente em memoria. Nenhum arquivo
``.pt`` ou ``.pth`` e produzido.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code" / "pipeline")]

from milho_experiment.temporal import paired_stratified_bootstrap_delta  # noqa: E402
from stage14_architecture_search import (  # noqa: E402
    SearchRecord,
    _representation_features,
    assert_no_weights,
    evaluation_units,
    generate,
    load_source_records,
    load_target_records,
    train_generator,
)


TARGET = "Biomassa"
MODEL = "attention_unet"
TRACK = "A"
HISTORICAL_SYNTHETIC_R2 = 0.825468


@dataclass(frozen=True, order=True)
class TreeConfig:
    select_k: int | str
    min_samples_leaf: int
    max_features: float

    @property
    def name(self) -> str:
        return (f"k{self.select_k}_leaf{self.min_samples_leaf}_"
                f"mf{self.max_features:g}")


TREE_CONFIGS = tuple(
    TreeConfig(k, leaf, max_features)
    for k in (4, 8, 12, "all")
    for leaf in (1, 2, 3)
    for max_features in (0.5, 1.0)
)


def make_tree(config: TreeConfig, seed: int, n_features: int) -> Pipeline:
    k = config.select_k
    if isinstance(k, int):
        k = min(k, n_features)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_regression, k=k)),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def make_dose_tree(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def simplex_weights(step: float) -> list[tuple[float, float, float]]:
    inverse = round(1.0 / step)
    if step <= 0 or not np.isclose(inverse * step, 1.0):
        raise ValueError("blend-step deve dividir 1 exatamente")
    return [(real / inverse, synthetic / inverse, dose / inverse)
            for real in range(inverse + 1)
            for synthetic in range(inverse - real + 1)
            for dose in [inverse - real - synthetic]]


def nested_partitions(records: Sequence[SearchRecord], outer_block: int):
    outer_train = [record for record in records if record.block != outer_block]
    outer_test = [record for record in records if record.block == outer_block]
    if not outer_train or not outer_test:
        raise ValueError(f"bloco externo ausente: {outer_block}")
    for inner_block in sorted({record.block for record in outer_train}):
        inner_train = [record for record in outer_train if record.block != inner_block]
        inner_test = [record for record in outer_train if record.block == inner_block]
        train_ids = {record.fid for record in inner_train}
        forbidden = {record.fid for record in inner_test + outer_test}
        if train_ids & forbidden:
            raise AssertionError("vazamento entre treino interno e validacao/teste")
        yield inner_block, inner_train, inner_test


def _matrices(records: Sequence[SearchRecord], synthetic: dict[int, np.ndarray] | None,
              branch: str):
    if branch == "dose":
        return evaluation_units(TRACK, records, None, TARGET, "dose_only")
    cache = {
        record.fid: _representation_features(
            TRACK, record, synthetic if branch == "synthetic" else None)
        for record in records
    }
    return evaluation_units(TRACK, records, synthetic, TARGET, "rs_only",
                            feature_cache=cache)


def _predict_grid(train: Sequence[SearchRecord], test: Sequence[SearchRecord], *,
                  train_fake: dict[int, np.ndarray], test_fake: dict[int, np.ndarray],
                  seed: int) -> list[dict]:
    rows: list[dict] = []
    for branch, train_synthetic, test_synthetic in (
        ("real", None, None),
        ("synthetic", train_fake, test_fake),
    ):
        train_x, train_y, _ = _matrices(train, train_synthetic, branch)
        test_x, test_y, units = _matrices(test, test_synthetic, branch)
        for config in TREE_CONFIGS:
            model = make_tree(config, seed, train_x.shape[1])
            model.fit(train_x, train_y)
            prediction = model.predict(test_x)
            rows.extend({"branch": branch, "config": config.name, "seed": seed,
                         **unit, "y": float(truth), "prediction": float(value)}
                        for unit, truth, value in zip(units, test_y, prediction))
    train_x, train_y, _ = _matrices(train, None, "dose")
    test_x, test_y, units = _matrices(test, None, "dose")
    dose_model = make_dose_tree(seed)
    dose_model.fit(train_x, train_y)
    prediction = dose_model.predict(test_x)
    rows.extend({"branch": "dose", "config": "fixed", "seed": seed,
                 **unit, "y": float(truth), "prediction": float(value)}
                for unit, truth, value in zip(units, test_y, prediction))
    return rows


def _ensemble_grid(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    keys = ["branch", "config", "unit_id", "block", "dose_n", "n_fids", "y"]
    return frame.groupby(keys, as_index=False, sort=True).prediction.mean()


def _config_complexity(name: str) -> tuple[int, int, float]:
    config = next(config for config in TREE_CONFIGS if config.name == name)
    k = 10_000 if config.select_k == "all" else int(config.select_k)
    return k, -config.min_samples_leaf, -config.max_features


def select_branch_config(grid: pd.DataFrame, branch: str) -> tuple[str, float]:
    candidates = []
    for config, part in grid[grid.branch == branch].groupby("config", sort=True):
        candidates.append((float(r2_score(part.y, part.prediction)), str(config)))
    if not candidates:
        raise ValueError(f"previsoes ausentes para ramo {branch}")
    score, name = min(candidates, key=lambda item: (-item[0], _config_complexity(item[1])))
    return name, score


def select_fusion(grid: pd.DataFrame, step: float) -> dict:
    real_config, real_r2 = select_branch_config(grid, "real")
    synthetic_config, synthetic_r2 = select_branch_config(grid, "synthetic")
    selected = pd.concat([
        grid[(grid.branch == "real") & (grid.config == real_config)],
        grid[(grid.branch == "synthetic") & (grid.config == synthetic_config)],
        grid[grid.branch == "dose"],
    ])
    wide = selected.pivot(index=["unit_id", "block", "dose_n", "n_fids", "y"],
                          columns="branch", values="prediction").reset_index()
    if wide[["real", "synthetic", "dose"]].isna().any().any():
        raise ValueError("ramos desalinhados durante a fusao")
    y = wide.y.to_numpy(float)
    choices = []
    for real, synthetic, dose in simplex_weights(step):
        prediction = real * wide.real + synthetic * wide.synthetic + dose * wide.dose
        sse = float(np.sum((y - prediction.to_numpy(float)) ** 2))
        choices.append((sse, -real, synthetic, real, synthetic, dose))
    _, _, _, real, synthetic, dose = min(choices)
    prediction = real * wide.real + synthetic * wide.synthetic + dose * wide.dose
    return {
        "real_config": real_config,
        "synthetic_config": synthetic_config,
        "real_branch_r2": real_r2,
        "synthetic_branch_r2": synthetic_r2,
        "weight_real": real,
        "weight_synthetic": synthetic,
        "weight_dose": dose,
        "fusion_r2": float(r2_score(y, prediction)),
    }


def _selected_predictions(grid: pd.DataFrame, selection: dict) -> pd.DataFrame:
    selected = pd.concat([
        grid[(grid.branch == "real") & (grid.config == selection["real_config"])],
        grid[(grid.branch == "synthetic") &
             (grid.config == selection["synthetic_config"])],
        grid[grid.branch == "dose"],
    ])
    wide = selected.pivot(index=["unit_id", "block", "dose_n", "n_fids", "y"],
                          columns="branch", values="prediction").reset_index()
    wide = wide.rename(columns={"real": "prediction_real",
                                "synthetic": "prediction_synthetic",
                                "dose": "prediction_dose"})
    wide["prediction_fused"] = (
        selection["weight_real"] * wide.prediction_real +
        selection["weight_synthetic"] * wide.prediction_synthetic +
        selection["weight_dose"] * wide.prediction_dose
    )
    return wide


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _replace_fold(path: Path, rows: pd.DataFrame, outer_block: int) -> None:
    previous = _read(path)
    if len(previous) and "outer_block" in previous:
        previous = previous[previous.outer_block != outer_block]
    result = pd.concat([previous, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


def _metrics(y, prediction) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def _summarize(oof: pd.DataFrame, bootstrap: int) -> tuple[pd.DataFrame, dict]:
    y = oof.y.to_numpy(float)
    blocks = oof.block.to_numpy(int)
    rows = []
    mapping = {
        "real": "prediction_real",
        "synthetic": "prediction_synthetic",
        "dose": "prediction_dose",
        "fused": "prediction_fused",
    }
    for model, column in mapping.items():
        rows.append({"model": model, **_metrics(y, oof[column].to_numpy(float))})
    summary = pd.DataFrame(rows)
    synthetic = oof.prediction_synthetic.to_numpy(float)
    fused = oof.prediction_fused.to_numpy(float)
    low, high = paired_stratified_bootstrap_delta(
        y, synthetic, fused, blocks, n_bootstrap=bootstrap, seed=2026)
    fused_r2 = float(summary.loc[summary.model == "fused", "r2"].iloc[0])
    synthetic_r2 = float(summary.loc[summary.model == "synthetic", "r2"].iloc[0])
    decision = {
        "historical_synthetic_r2": HISTORICAL_SYNTHETIC_R2,
        "nested_synthetic_r2": synthetic_r2,
        "nested_fused_r2": fused_r2,
        "delta_vs_nested_synthetic": fused_r2 - synthetic_r2,
        "delta_ci95_low": float(low),
        "delta_ci95_high": float(high),
        "improves_point_estimate": bool(
            fused_r2 > max(HISTORICAL_SYNTHETIC_R2, synthetic_r2)),
        "confirmatory_supported": bool(
            fused_r2 > HISTORICAL_SYNTHETIC_R2 and low > 0),
    }
    return summary, decision


def _metadata(args) -> dict:
    return {
        "phase": "selection",
        "protocol_version": 1,
        "target": TARGET,
        "generator": MODEL,
        "track": TRACK,
        "outer_cv": "leave-one-block-out",
        "inner_cv": "leave-one-block-out",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "width": args.width,
        "blend_step": args.blend_step,
        "tree_configs": [asdict(config) for config in TREE_CONFIGS],
        "checkpoint_policy": "none",
        "experimental_unit": "block+dose",
    }


def _device(args):
    import torch
    if args.device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def _release(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args)
    metadata_path = out / "meta.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("diretorio de saida pertence a outra configuracao")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    records = load_source_records(args.dataset, args.source_stage4,
                                  args.source_table, TRACK)
    blocks = sorted({record.block for record in records})
    if blocks != [1, 2, 3, 4]:
        raise ValueError(f"quatro blocos 1..4 eram esperados; recebido {blocks}")
    device = _device(args)
    base_path = out / "base_predictions.csv"
    grid_path = out / "grid_predictions.csv"
    weights_path = out / "fusion_weights.csv"
    history_path = out / "training_history.csv"

    for outer_block in blocks:
        existing_base = _read(base_path)
        existing_grid = _read(grid_path)
        existing_weights = _read(weights_path)
        complete = (
            len(existing_base[existing_base.outer_block == outer_block]) == 6
            if len(existing_base) and "outer_block" in existing_base else False
        ) and (
            len(existing_grid[existing_grid.outer_block == outer_block]) ==
            6 * (2 * len(TREE_CONFIGS) + 1)
            if len(existing_grid) and "outer_block" in existing_grid else False
        ) and (
            len(existing_weights[existing_weights.outer_block == outer_block]) == 1
            if len(existing_weights) and "outer_block" in existing_weights else False
        )
        if complete:
            continue

        outer_train = [record for record in records if record.block != outer_block]
        outer_test = [record for record in records if record.block == outer_block]
        inner_rows: list[dict] = []
        histories: list[dict] = []
        for inner_block, inner_train, inner_test in nested_partitions(records, outer_block):
            for seed in args.seeds:
                generator, history = train_generator(
                    inner_train, MODEL, epochs=args.epochs, seed=seed,
                    batch_size=args.batch_size, crop_size=args.crop_size,
                    width=args.width, device=device)
                train_fake = generate(generator, inner_train, device)
                test_fake = generate(generator, inner_test, device)
                predicted = _predict_grid(
                    inner_train, inner_test, train_fake=train_fake,
                    test_fake=test_fake, seed=seed)
                inner_rows.extend(predicted)
                histories.extend({"outer_block": outer_block, "scope": "inner",
                                  "validation_block": inner_block, "seed": seed, **row}
                                 for row in history)
                del generator
                _release(device)
        inner_grid = _ensemble_grid(inner_rows)
        selection = select_fusion(inner_grid, args.blend_step)

        outer_rows: list[dict] = []
        for seed in args.seeds:
            generator, history = train_generator(
                outer_train, MODEL, epochs=args.epochs, seed=seed,
                batch_size=args.batch_size, crop_size=args.crop_size,
                width=args.width, device=device)
            train_fake = generate(generator, outer_train, device)
            test_fake = generate(generator, outer_test, device)
            outer_rows.extend(_predict_grid(
                outer_train, outer_test, train_fake=train_fake,
                test_fake=test_fake, seed=seed))
            histories.extend({"outer_block": outer_block, "scope": "outer",
                              "validation_block": outer_block, "seed": seed, **row}
                             for row in history)
            del generator
            _release(device)
        outer_grid = _ensemble_grid(outer_rows)
        base = _selected_predictions(outer_grid, selection)
        base.insert(0, "outer_block", outer_block)
        grid_saved = outer_grid.copy()
        grid_saved.insert(0, "outer_block", outer_block)
        weights = pd.DataFrame([{"outer_block": outer_block, **selection}])
        history_frame = pd.DataFrame(histories)
        _replace_fold(base_path, base, outer_block)
        _replace_fold(grid_path, grid_saved, outer_block)
        _replace_fold(weights_path, weights, outer_block)
        _replace_fold(history_path, history_frame, outer_block)
        assert_no_weights(out)

    oof = _read(base_path).sort_values(["block", "dose_n"]).reset_index(drop=True)
    if len(oof) != 24 or oof.unit_id.nunique() != 24:
        raise AssertionError("a avaliacao externa deve conter exatamente 24 unidades")
    oof.to_csv(out / "oof_predictions.csv", index=False)
    summary, decision = _summarize(oof, args.bootstrap)
    fold_weights = _read(weights_path)
    decision.update({
        "gan_positive_folds": int((fold_weights.weight_synthetic > 0).sum()),
        "gan_mean_weight": float(fold_weights.weight_synthetic.mean()),
        "weights_stable_in_at_least_three_folds": bool(
            (fold_weights.weight_synthetic > 0).sum() >= 3),
    })
    summary.to_csv(out / "summary.csv", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    deployment = select_fusion(_read(grid_path).drop(columns="outer_block"), args.blend_step)
    deployment.update({"target": TARGET, "generator": MODEL, "track": TRACK,
                       "seeds": args.seeds, "epochs": args.epochs,
                       "batch_size": args.batch_size, "crop_size": args.crop_size,
                       "width": args.width})
    (out / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    winner_path = out / "winner.json"
    if decision["confirmatory_supported"]:
        winner_path.write_text(json.dumps(deployment, indent=2) + "\n")
    elif winner_path.exists():
        winner_path.unlink()
    assert_no_weights(out)


def replicate(args) -> None:
    if not args.target_stage4 or not args.target_table or not args.winner:
        raise ValueError("replicate requer --target-stage4, --target-table e --winner")
    winner_path = Path(args.winner)
    if not winner_path.is_file():
        raise ValueError("replicacao bloqueada: a selecao nao produziu winner.json")
    selection = json.loads(winner_path.read_text())
    if selection.get("target") != TARGET or selection.get("generator") != MODEL:
        raise ValueError("winner incompativel com este experimento")
    real_config = next(c for c in TREE_CONFIGS if c.name == selection["real_config"])
    synthetic_config = next(c for c in TREE_CONFIGS
                            if c.name == selection["synthetic_config"])
    source = load_source_records(args.dataset, args.source_stage4,
                                 args.source_table, TRACK)
    target = load_target_records(args.target_stage4, args.target_table, TRACK)
    device = _device(args)
    rows: list[dict] = []
    histories: list[dict] = []
    for seed in selection["seeds"]:
        generator, history = train_generator(
            source, MODEL, epochs=int(selection["epochs"]), seed=int(seed),
            batch_size=int(selection["batch_size"]),
            crop_size=int(selection["crop_size"]),
            width=int(selection["width"]), device=device)
        source_fake = generate(generator, source, device)
        target_fake = generate(generator, target, device)
        for branch, config, source_synthetic, target_synthetic in (
            ("real", real_config, None, None),
            ("synthetic", synthetic_config, source_fake, target_fake),
        ):
            train_x, train_y, _ = _matrices(source, source_synthetic, branch)
            test_x, test_y, units = _matrices(target, target_synthetic, branch)
            model = make_tree(config, int(seed), train_x.shape[1])
            model.fit(train_x, train_y)
            rows.extend({"branch": branch, "seed": int(seed), **unit,
                         "y": float(truth), "prediction": float(value)}
                        for unit, truth, value in zip(units, test_y,
                                                     model.predict(test_x)))
        train_x, train_y, _ = _matrices(source, None, "dose")
        test_x, test_y, units = _matrices(target, None, "dose")
        dose_model = make_dose_tree(int(seed)).fit(train_x, train_y)
        rows.extend({"branch": "dose", "seed": int(seed), **unit,
                     "y": float(truth), "prediction": float(value)}
                    for unit, truth, value in zip(units, test_y,
                                                 dose_model.predict(test_x)))
        histories.extend({"seed": int(seed), **row} for row in history)
        del generator
        _release(device)
    frame = pd.DataFrame(rows)
    ensemble = frame.groupby(
        ["branch", "unit_id", "block", "dose_n", "n_fids", "y"],
        as_index=False).prediction.mean()
    wide = ensemble.pivot(index=["unit_id", "block", "dose_n", "n_fids", "y"],
                          columns="branch", values="prediction").reset_index()
    wide = wide.rename(columns={"real": "prediction_real",
                                "synthetic": "prediction_synthetic",
                                "dose": "prediction_dose"})
    wide["prediction_fused"] = (
        selection["weight_real"] * wide.prediction_real +
        selection["weight_synthetic"] * wide.prediction_synthetic +
        selection["weight_dose"] * wide.prediction_dose)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "replicate_predictions.csv", index=False)
    summary, decision = _summarize(wide, args.bootstrap)
    summary.to_csv(out / "replicate_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False)
    (out / "replicate_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "phase": "replicate", "source": "2022/23", "target_season": "2023/24",
        "selection": selection, "checkpoint_policy": "none",
        "interpretation": "secondary_replication_not_pristine_holdout",
    }, indent=2) + "\n")
    assert_no_weights(out)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", choices=("run", "replicate"))
    result.add_argument("--dataset", required=True)
    result.add_argument("--source-stage4", required=True)
    result.add_argument("--source-table", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--target-stage4")
    result.add_argument("--target-table")
    result.add_argument("--winner")
    result.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--crop-size", type=int, default=128)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--blend-step", type=float, default=0.05)
    result.add_argument("--bootstrap", type=int, default=10_000)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.width < 8 or args.crop_size < 16 or args.crop_size % 8:
        raise ValueError("epochs>=1, width>=8 e crop-size>=16 divisivel por 8 sao obrigatorios")
    simplex_weights(args.blend_step)
    if args.phase == "run":
        run(args)
    else:
        replicate(args)


if __name__ == "__main__":
    main()
