#!/usr/bin/env python3
"""Baseline PLSR 2022/23 com unidades experimentais dose×bloco.

O arquivo 5-bandas contém dois polígonos (``Tipo=R/B``) para cada combinação
``estágio, dose, bloco``. Este comando os agrega antes de juntar o alvo de campo,
para que o Y não seja duplicado. Avalia dois conjuntos de atributos:

* ``rrenir_indices_texture_nir``: NDVI, NDRE, CIrededge, SAVI e GLCM do NIR;
* ``fiveband_indices_texture_nir``: os oito índices disponíveis e o mesmo GLCM.

O número de componentes é escolhido somente no GroupKFold interno de cada fold
externo. As métricas reportadas são previsões out-of-fold por unidade dose×bloco.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RRENIR_INDICES = ("NDVI", "NDRE", "CIrededge", "SAVI")
FIVEBAND_INDICES = ("NDVI", "GNDVI", "NDRE", "CIrededge", "SAVI", "EVI", "VARI", "TGI")
SAMPLE_RE = re.compile(r"^(?P<stage>[A-Za-z]\d+)_d(?P<dose>-?\d+(?:\.\d+)?)_b(?P<block>\d+)$")
T_CRITICAL_95_N5 = 2.7764451051977987


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rpd": float(np.std(y_true, ddof=1) / rmse),
    }


def parse_and_aggregate(features: pd.DataFrame) -> pd.DataFrame:
    """Agrega os dois polígonos R/B em uma linha por estágio×dose×bloco."""
    parts = features["sample"].str.extract(SAMPLE_RE)
    if parts.isna().any().any():
        bad = features.loc[parts.isna().any(axis=1), "sample"].unique().tolist()
        raise ValueError(f"amostras fora do padrão stage_dose_bloco: {bad[:5]}")
    frame = features.copy()
    frame["stage"] = parts["stage"].str.upper()
    frame["dose"] = pd.to_numeric(parts["dose"], errors="raise")
    frame["block"] = pd.to_numeric(parts["block"], errors="raise").astype(int)
    value_cols = [c for c in frame if c not in {"sample", "parcela", "medida", "stage", "dose", "block"}]
    counts = frame.groupby(["stage", "dose", "block"], sort=True).size()
    if not counts.eq(2).all():
        raise ValueError("a baseline 22/23 exige exatamente dois polígonos R/B por unidade; "
                         f"encontrado: {counts.value_counts().to_dict()}")
    out = frame.groupby(["stage", "dose", "block"], as_index=False, sort=True)[value_cols].mean()
    out["unit"] = out["dose"].map(lambda x: f"d{x:g}") + "_b" + out["block"].astype(str)
    return out


def join_targets(aggregated: pd.DataFrame, table_path: Path, target: str) -> pd.DataFrame:
    table = pd.read_excel(table_path)
    required = {"Estagio", "Dose_N", "Bloco", target}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"colunas ausentes na tabela de campo: {sorted(missing)}")
    targets = table[["Estagio", "Dose_N", "Bloco", target]].copy()
    targets["stage"] = targets.pop("Estagio").astype(str).str.upper()
    targets["dose"] = pd.to_numeric(targets.pop("Dose_N"), errors="raise")
    targets["block"] = pd.to_numeric(targets["Bloco"], errors="raise").astype(int)
    targets = targets.drop(columns="Bloco").dropna(subset=[target])
    if targets.duplicated(["stage", "dose", "block"]).any():
        raise ValueError("a tabela de campo tem alvos duplicados para estágio×dose×bloco")
    return aggregated.merge(targets, on=["stage", "dose", "block"], how="inner", validate="one_to_one")


def choose_components(X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_components: int, folds: int) -> int:
    n_splits = min(folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    best_components, best_score = 1, -np.inf
    for components in range(1, min(max_components, X.shape[1], len(y) - 1) + 1):
        prediction = np.empty(len(y), dtype=float)
        for train, test in splitter.split(X, y, groups):
            model = make_pipeline(StandardScaler(), PLSRegression(n_components=components))
            model.fit(X[train], y[train])
            prediction[test] = model.predict(X[test]).ravel()
        score = r2_score(y, prediction)
        if score > best_score:
            best_components, best_score = components, score
    return best_components


def outer_splitter(protocol: str, groups: np.ndarray, folds: int, seed: int):
    """Retorna o protocolo externo, sempre sem dividir uma unidade dose×bloco."""
    if protocol == "holdout_70_30":
        return GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    if protocol == "holdout_80_20":
        return GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    if protocol == "groupkfold_4":
        return GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    if protocol == "logo":
        return LeaveOneGroupOut()
    raise ValueError(f"protocolo externo desconhecido: {protocol}")


def nested_predictions(frame: pd.DataFrame, features: list[str], target: str, protocol: str,
                       folds: int, max_components: int, seed: int) -> pd.DataFrame:
    X = frame[features].to_numpy(float)
    y = frame[target].to_numpy(float)
    groups = frame["unit"].to_numpy()
    splitter = outer_splitter(protocol, groups, folds, seed)
    records: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        components = choose_components(X[train], y[train], groups[train], max_components, folds)
        model = make_pipeline(StandardScaler(), PLSRegression(n_components=components))
        model.fit(X[train], y[train])
        fold_rows = frame.iloc[test][["stage", "dose", "block", "unit"]].copy()
        fold_rows["protocol"] = protocol
        fold_rows["fold"] = fold
        fold_rows["n_components"] = components
        fold_rows["y_true"] = y[test]
        fold_rows["y_pred"] = model.predict(X[test]).ravel()
        records.append(fold_rows)
    return pd.concat(records, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds a avaliar; substitui --seed (ex.: 42 43 44 45 46)")
    parser.add_argument("--protocols", nargs="+",
                        default=["holdout_70_30", "holdout_80_20", "groupkfold_4", "logo"],
                        choices=["holdout_70_30", "holdout_80_20", "groupkfold_4", "logo"])
    parser.add_argument("--by-stage", action="store_true",
                        help="avalia também cada estágio com alvo separadamente")
    args = parser.parse_args()
    if args.folds < 2 or args.max_components < 1:
        raise ValueError("--folds deve ser >=2 e --max-components deve ser >=1")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(seeds) < 1:
        raise ValueError("é necessária ao menos uma seed")

    aggregated = parse_and_aggregate(pd.read_csv(args.features))
    glcm = [c for c in aggregated if c.startswith("glcm")]
    sets = {
        "rrenir_indices_texture_nir": [*RRENIR_INDICES, *glcm],
        "fiveband_indices_texture_nir": [*FIVEBAND_INDICES, *glcm],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.out / "baseline_2223_aggregated_features.csv", index=False)

    all_predictions, runs = [], []
    for target in ("Biomassa", "Produtividade"):
        data = join_targets(aggregated, args.table, target)
        for set_name, columns in sets.items():
            missing = set(columns) - set(data.columns)
            if missing:
                raise ValueError(f"{set_name}: atributos ausentes: {sorted(missing)}")
            if data[columns].isna().any().any():
                raise ValueError(f"{set_name}: há atributos ausentes após agregação")
            scopes = [("ALL", data)]
            if args.by_stage:
                scopes.extend((stage, part.copy()) for stage, part in data.groupby("stage", sort=True))
            for evaluation_stage, scoped in scopes:
                for protocol in args.protocols:
                    for seed in seeds:
                        prediction = nested_predictions(scoped, columns, target, protocol, args.folds,
                                                        args.max_components, seed)
                        prediction.insert(0, "seed", seed)
                        prediction.insert(0, "evaluation_stage", evaluation_stage)
                        prediction.insert(0, "feature_set", set_name)
                        prediction.insert(0, "target", target)
                        values = metric_values(prediction["y_true"].to_numpy(), prediction["y_pred"].to_numpy())
                        runs.append({
                            "target": target, "feature_set": set_name, "evaluation_stage": evaluation_stage,
                            "protocol": protocol, "seed": seed, "n_rows": len(prediction),
                            "n_units": prediction["unit"].nunique(), "n_features": len(columns),
                            "inner_folds": args.folds,
                            "components_by_fold": ",".join(map(str, prediction.groupby("fold")["n_components"].first())),
                            **values,
                        })
                        all_predictions.append(prediction)

    runs_df = pd.DataFrame(runs).sort_values(["target", "feature_set", "protocol", "seed"])
    grouping = ["target", "feature_set", "evaluation_stage", "protocol", "n_rows", "n_units", "n_features", "inner_folds"]
    summary_rows = []
    for keys, part in runs_df.groupby(grouping, sort=True):
        row = dict(zip(grouping, keys))
        row["n_seeds"] = len(part)
        for metric in ("r2", "rmse", "mae", "rpd"):
            mean = float(part[metric].mean())
            std = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
            half_width = T_CRITICAL_95_N5 * std / np.sqrt(len(part)) if len(part) == 5 else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width if np.isfinite(half_width) else np.nan
            row[f"{metric}_ci95_high"] = mean + half_width if np.isfinite(half_width) else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "evaluation_stage", "feature_set", "protocol"])
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df.to_csv(args.out / "summary.csv", index=False)
    runs_df.to_csv(args.out / "per_seed_metrics.csv", index=False)
    predictions_df.to_csv(args.out / "oof_predictions.csv", index=False)
    lines = [
        "=== BASELINE 2022/23 — PLSR ANINHADO ===",
        "Unidade independente: dose×bloco; dois polígonos R/B agregados por estágio.",
        f"Seleção interna: GroupKFold({args.folds}); componentes candidatos: 1–{args.max_components}; seeds={','.join(map(str, seeds))}.",
        "IC 95% entre seeds: média ± t(0,975;4) × DP / √5.",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.target:14} | estágio={row.evaluation_stage:3} | {row.feature_set:33} | {row.protocol:15} | n={row.n_rows:2d}, unidades={row.n_units:2d}, "
            f"p={row.n_features:2d} | R²={row.r2_mean:.3f}±{row.r2_std:.3f} "
            f"IC95=[{row.r2_ci95_low:.3f}, {row.r2_ci95_high:.3f}] | "
            f"RMSE={row.rmse_mean:.2f}±{row.rmse_std:.2f} | "
            f"MAE={row.mae_mean:.2f}±{row.mae_std:.2f} | RPD={row.rpd_mean:.2f}±{row.rpd_std:.2f}"
        )
    report = "\n".join(lines) + "\n"
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
