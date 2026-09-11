#!/usr/bin/env python3
"""Relatório consolidado do valor preditivo da GAN a partir do stage5_attribute_search.

Lê os `summary.csv` de um ou mais experimentos (biomassa, produtividade, ...) e gera
`GAN_VALUE.md` com, para cada alvo:

  - Ranking de configurações pelo ganho ΔR² = R²(fusion_veg) − R²(veg), médio sobre seeds.
  - Tabela completa: por cenário (veg/real/fake/fusion_veg/fusion_real) × modelo, R² com
    IC95 bootstrap (média sobre seeds).
  - Veredito honesto por alvo: a GAN pura prediz sozinha? a GAN agrega ao vegetativo?

Uso:
    python3 code/pipeline/stage5_gan_value_report.py \
        --result-dir artifacts/runs/2324_attr_search_biomassa/results --alvo Biomassa \
        --result-dir artifacts/runs/2324_attr_search_produtividade/results --alvo Produtividade \
        --out artifacts/runs/gan_value/GAN_VALUE.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUCCESS_DELTA_R2 = 0.05   # critério do projeto para considerar ganho preditivo


def _load_summary(result_dir: Path) -> pd.DataFrame:
    s = pd.read_csv(result_dir / "summary.csv")
    required = {"cfg", "attr_names", "n_cond", "seed", "scenario", "model",
                "r2", "rmse", "r2_lo", "r2_hi", "l1", "ndvi_error"}
    missing = required - set(s.columns)
    if missing:
        raise ValueError(f"{result_dir / 'summary.csv'}: colunas ausentes {sorted(missing)}")
    return s


def _aggregate(s: pd.DataFrame) -> pd.DataFrame:
    """Média sobre seeds por (cfg, cenário, modelo)."""
    group = ["cfg", "attr_names", "n_cond", "scenario", "model"]
    return s.groupby(group, as_index=False).agg(
        r2=("r2", "mean"), r2_std=("r2", "std"),
        r2_lo=("r2_lo", "mean"), r2_hi=("r2_hi", "mean"),
        rmse=("rmse", "mean"), l1=("l1", "mean"), ndvi_error=("ndvi_error", "mean"),
        n_seeds=("seed", "count"))


def _best_by_scenario(a: pd.DataFrame) -> pd.DataFrame:
    """Melhor modelo por (cfg, cenário)."""
    best = a.loc[a.groupby(["cfg", "scenario"])["r2"].idxmax()].copy()
    return best[["cfg", "attr_names", "n_cond", "scenario", "model", "r2",
                 "r2_lo", "r2_hi", "rmse"]]


def _pivot_value(best: pd.DataFrame) -> pd.DataFrame:
    """Tabela por config: veg, fake, fusion_veg, ΔR²."""
    idx = ["cfg", "attr_names", "n_cond"]
    p = best.pivot_table(index=idx, columns="scenario", values="r2")
    for col in ("veg", "fake", "real", "fusion_veg", "fusion_real"):
        if col not in p.columns:
            p[col] = np.nan
    p = p[["veg", "fake", "real", "fusion_veg", "fusion_real"]].copy()
    p["delta_r2_fusion_veg"] = p["fusion_veg"] - p["veg"]
    p["delta_r2_fake"] = p["fake"] - p["veg"]
    return p.sort_values("delta_r2_fusion_veg", ascending=False)


def _verdict(target: str, pivot: pd.DataFrame) -> list[str]:
    lines = [f"### Veredito — {target}", ""]
    top = pivot.iloc[0]
    dv = top["delta_r2_fusion_veg"]
    dfake = top["delta_r2_fake"]
    lines.append(
        f"Melhor configuração: **{top.name[0]}** ({top['attr_names']}), "
        f"n_cond={top['n_cond']}.")
    lines.append(
        f"  - R² vegetativo real = {top['veg']:.3f}; R² fusão (veg+GAN) = {top['fusion_veg']:.3f}; "
        f"ΔR² = {dv:+.3f}.")
    lines.append(
        f"  - R² GAN pura (fake) = {top['fake']:.3f}; ΔR² vs vegetativo = {dfake:+.3f}.")
    if dv >= SUCCESS_DELTA_R2:
        lines.append(
            f"  - **GAN agrega valor preditivo ao vegetativo** (ΔR² = {dv:+.3f} ≥ +{SUCCESS_DELTA_R2}).")
    elif dv > 0:
        lines.append(
            f"  - A GAN agrega levemente ao vegetativo (ΔR² = {dv:+.3f}), abaixo do limiar "
            f"+{SUCCESS_DELTA_R2} do projeto; interpretar com cautela.")
    else:
        lines.append(
            f"  - A GAN **não** agrega ao vegetativo (ΔR² = {dv:+.3f}); a imagem sintética não "
            f"adiciona predição útil sobre o vegetativo real nesta configuração.")
    if dfake <= 0:
        lines.append(
            f"  - A GAN pura (sem o vegetativo) fica abaixo do baseline vegetativo "
            f"(ΔR² = {dfake:+.3f}): o valor vem da fusão, não da GAN isolada.")
    lines.append("")
    return lines


def _render(target: str, s: pd.DataFrame, a: pd.DataFrame, best: pd.DataFrame,
            pivot: pd.DataFrame) -> list[str]:
    lines = [f"# Valor preditivo da GAN — {target}", ""]
    lines.append(f"Unidade independente: **parcela** (24 parcelas, 4 folds). "
                 f"R² médio sobre {int(s.seed.nunique())} seeds; IC95% bootstrap por parcela. "
                 f"Cenários: `veg` (V8/V13 reais, baseline pré-colheita), `real` (R2/R5 reais), "
                 f"`fake` (R2/R5 sintéticos GAN), `fusion_veg` (veg+sintético), "
                 f"`fusion_real` (real+sintético).")
    lines.append("")

    lines += ["## Ranking por ganho de fusão (ΔR² = fusion_veg − veg)", ""]
    if pivot.empty:
        lines.append("Sem dados.")
    else:
        lines += [
            "| cfg | n_cond | veg | fake | real | fusion_veg | ΔR² fusion | ΔR² fake |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for (name, attrs, nc), row in pivot.iterrows():
            lines.append(
                f"| {name} | {nc} | {row['veg']:.3f} | {row['fake']:.3f} | {row['real']:.3f} "
                f"| {row['fusion_veg']:.3f} | {row['delta_r2_fusion_veg']:+.3f} "
                f"| {row['delta_r2_fake']:+.3f} |")
    lines.append("")

    lines += _verdict(target, pivot)

    lines += ["## Detalhe por cenário × modelo (R² médio sobre seeds, IC95)", ""]
    scen_order = ["veg", "real", "fake", "fusion_veg", "fusion_real"]
    for scenario in scen_order:
        sub = best[best.scenario == scenario]
        if sub.empty:
            continue
        lines += [f"### Cenário `{scenario}`", ""]
        lines += ["| cfg | n_cond | modelo | R² | IC95 | RMSE |", "|---|---|---|---:|---:|---:|"]
        for _, r in sub.sort_values("r2", ascending=False).iterrows():
            lines.append(f"| {r.cfg} | {r.n_cond} | {r.model} | {r.r2:.3f} | "
                         f"[{r.r2_lo:.3f}, {r.r2_hi:.3f}] | {r.rmse:.0f} |")
        lines.append("")

    lines += ["---", ""]
    lines += [f"Fonte: `summary.csv` de {len(a)} linhas agregadas "
              f"({int(s.seed.nunique())} seeds).", ""]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", action="append", required=True,
                    help="diretório com summary.csv de um experimento")
    ap.add_argument("--alvo", action="append", required=True,
                    help="rótulo do alvo; repita na mesma ordem de --result-dir")
    ap.add_argument("--out", required=True, help="caminho do GAN_VALUE.md")
    args = ap.parse_args()
    if len(args.result_dir) != len(args.alvo):
        raise ValueError("--result-dir e --alvo devem ter o mesmo número de entradas")

    blocks = []
    for d, target in zip(args.result_dir, args.alvo):
        s = _load_summary(Path(d))
        a = _aggregate(s)
        best = _best_by_scenario(a)
        pivot = _pivot_value(best)
        blocks += [f"\n\n---\n\n"] + _render(target, s, a, best, pivot)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Relatório -> {out}")


if __name__ == "__main__":
    main()