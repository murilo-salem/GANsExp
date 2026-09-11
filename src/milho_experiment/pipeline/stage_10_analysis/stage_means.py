#!/usr/bin/env python3
"""Valor de cada índice/textura por safra × estágio × parcela.

Cada valor é a média sobre os pixels vegetados da parcela, conforme extraído
por safra em:
  - artifacts/runs/2223_baseline_plsr/results/baseline_2223_aggregated_features.csv
  - artifacts/runs/2324_5band_features/results/features_5band_2324.csv

Saídas (--out):
  parcelas_wide.csv   uma linha por parcela: safra, estagio, parcela, dose, bloco + atributos
  parcelas_tidy.csv   formato longo: safra, estagio, parcela, atributo, valor
  MEANS.md            tabelas Markdown por safra × família, linhas = estágio·parcela
  heatmap_means.png   resumo agregado (média entre parcelas): atributo × safra·estágio
  evolucao_indices.png  evolução dos 8 índices ao longo dos estágios, linha por parcela
  evolucao_ndvi_parcelas.png  NDVI por estágio, um mini-painel por unidade dose·bloco

Uso:
  python3 stage_means.py [--out artifacts/analysis/means_safra_estagio]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]

SOURCES = [
    ("2022_2023", REPO / "artifacts/runs/2223_baseline_plsr/results/baseline_2223_aggregated_features.csv"),
    ("2023_2024", REPO / "artifacts/runs/2324_5band_features/results/features_5band_2324.csv"),
]

STAGE_ORDER = ["V6", "V8", "V10", "V11", "V13", "V18", "R1", "R2", "R5"]
STAGE_PLOT_ORDER = ["V6", "V8", "V11", "V13", "V18", "R2", "R5"]
SAFRA_COLORS = {"2022_2023": "tab:blue", "2023_2024": "tab:orange"}
SAFRA_LABELS = {"2022_2023": "22/23", "2023_2024": "23/24"}
INDEX_PANELS = ["NDVI", "GNDVI", "NDRE", "CIrededge", "SAVI", "EVI", "VARI", "TGI"]

FAMILIES = [
    ("Índices espectrais", lambda c: c in {"NDVI", "GNDVI", "NDRE", "CIrededge", "SAVI", "EVI", "VARI", "TGI"}),
    ("Estatísticas espectrais", lambda c: c.startswith("spec_")),
    ("GLCM global", lambda c: c.startswith("glcm_")),
    ("GLCM janela w3", lambda c: c.startswith("glcmw3_")),
    ("GLCM janela w5", lambda c: c.startswith("glcmw5_")),
    ("GLCM janela w7", lambda c: c.startswith("glcmw7_")),
    ("GLCM RGB global", lambda c: c.startswith("rgb_glcm_")),
    ("GLCM RGB janela w3", lambda c: c.startswith("rgb_glcmw3_")),
    ("GLCM RGB janela w5", lambda c: c.startswith("rgb_glcmw5_")),
    ("GLCM RGB janela w7", lambda c: c.startswith("rgb_glcmw7_")),
]

ID_COLS = ["safra", "estagio", "parcela", "dose", "bloco"]
META_DROP = {"stage", "medida", "sample", "unit", "n_veg_pixels"}


def load_sources():
    frames = []
    for safra, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"fonte ausente: {path}")
        df = pd.read_csv(path).rename(columns={"stage": "estagio", "medida": "estagio", "block": "bloco"})
        if "sample" in df.columns:
            parsed = df["sample"].str.extract(r"_d([^_]+)_b([^_]+)$")
            df["dose"] = pd.to_numeric(parsed[0], errors="coerce")
            df["bloco"] = pd.to_numeric(parsed[1], errors="coerce")
        else:
            df["dose"] = pd.to_numeric(df["dose"], errors="coerce")
            df["bloco"] = pd.to_numeric(df["bloco"], errors="coerce")
        df["parcela"] = df["parcela"].astype(str) if "parcela" in df.columns else df["unit"]
        df.insert(0, "safra", safra)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def feature_columns(df):
    skip = set(META_DROP) | set(ID_COLS) | {c for c in df.columns if c.startswith("phys_")}
    ordered = []
    for _, pred in FAMILIES:
        for c in df.columns:
            if c not in skip and pred(c) and c not in ordered:
                ordered.append(c)
    leftovers = [c for c in df.columns if c not in skip and not any(p(c) for _, p in FAMILIES)]
    return ordered + leftovers


def sort_key(safra, estagio, parcela, dose, bloco):
    st = STAGE_ORDER.index(estagio) if estagio in STAGE_ORDER else 99
    return (st, dose if pd.notna(dose) else 1e9, bloco if pd.notna(bloco) else 1e9, str(parcela))


def write_wide(df, feat_cols, out):
    cols = ID_COLS + feat_cols
    wide = df[cols].sort_values(
        ["safra", "estagio", "dose", "bloco", "parcela"],
        key=lambda s: s.map(lambda e: STAGE_ORDER.index(e) if s.name == "estagio" and e in STAGE_ORDER else 99)
        if s.name == "estagio" else s,
    )
    wide.to_csv(out / "parcelas_wide.csv", index=False, float_format="%.6g")
    return wide


def write_tidy(df, feat_cols, out):
    keys = {tuple(r) for r in df[ID_COLS].itertuples(index=False)}
    order = {k: i for i, k in enumerate(sorted(keys, key=lambda k: sort_key(*k)))}
    tidy = df.melt(id_vars=ID_COLS, value_vars=feat_cols,
                   var_name="atributo", value_name="valor")
    fam_rank = {c: i for i, c in enumerate(feat_cols)}
    tidy["_row"] = [order[t] for t in zip(tidy["safra"], tidy["estagio"], tidy["parcela"], tidy["dose"], tidy["bloco"])]
    tidy["_fam"] = tidy["atributo"].map(fam_rank)
    tidy = tidy.sort_values(["_row", "_fam"]).drop(columns=["_row", "_fam"])
    tidy.to_csv(out / "parcelas_tidy.csv", index=False, float_format="%.6g")
    return tidy


def fmt(x):
    return "—" if pd.isna(x) else f"{x:.3g}"


def write_markdown(df, feat_cols, out):
    lines = [
        "# Índices/texturas por safra × estágio × parcela",
        "",
        "Cada valor é a média sobre os pixels vegetados da parcela. «—» = atributo não extraído na safra.",
        "",
    ]
    for safra in sorted(df["safra"].unique()):
        sub = df[df["safra"] == safra]
        sub = sub.sort_values(
            ["estagio", "dose", "bloco", "parcela"],
            key=lambda s: s.map(lambda e: STAGE_ORDER.index(e) if s.name == "estagio" and e in STAGE_ORDER else 99)
            if s.name == "estagio" else s,
        )
        lines += [f"## Safra {safra.replace('_', '/')}", ""]
        for fam_name, pred in FAMILIES:
            attrs = [c for c in feat_cols if pred(c)]
            if not attrs:
                continue
            lines += [
                f"### {fam_name}",
                "",
                "| estágio | parcela | dose | bloco | " + " | ".join(attrs) + " |",
                "|---|---|---|---|" + "---|" * len(attrs),
            ]
            for _, r in sub.iterrows():
                cells = [fmt(r[a]) for a in attrs]
                dose = "—" if pd.isna(r["dose"]) else str(int(r["dose"]))
                bloco = "—" if pd.isna(r["bloco"]) else str(int(r["bloco"]))
                lines.append(f"| {r['estagio']} | {r['parcela']} | {dose} | {bloco} | " + " | ".join(cells) + " |")
            lines.append("")
    (out / "MEANS.md").write_text("\n".join(lines), encoding="utf-8")


def _xpos(estagio):
    return STAGE_PLOT_ORDER.index(estagio)


def plot_evolution(df, feat_cols, out):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex=True)
    for ax, attr in zip(axes.ravel(), INDEX_PANELS):
        if attr not in feat_cols:
            ax.set_visible(False)
            continue
        sub = df[["safra", "dose", "bloco", "estagio", attr]].dropna(subset=[attr])
        for safra, g in sub.groupby("safra"):
            color = SAFRA_COLORS.get(safra, "tab:green")
            piv = g.pivot_table(index=["dose", "bloco"], columns="estagio", values=attr)
            stages = [s for s in STAGE_PLOT_ORDER if s in piv.columns]
            xs = [_xpos(s) for s in stages]
            for _, row in piv[stages].iterrows():
                ax.plot(xs, row.values, color=color, lw=0.7, alpha=0.25)
            mu = piv[stages].mean()
            ax.plot(xs, mu.values, color=color, lw=2.6)
        lo, hi = sub[attr].quantile([0.02, 0.98])
        pad = 0.06 * (hi - lo if hi > lo else max(abs(hi), 1e-6))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(attr, fontsize=11)
        ax.set_xticks(range(len(STAGE_PLOT_ORDER)), STAGE_PLOT_ORDER, rotation=45, fontsize=7)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
    handles = []
    for safra in ["2022_2023", "2023_2024"]:
        color = SAFRA_COLORS[safra]
        handles += [
            Line2D([], [], color=color, lw=0.7, label=f"{SAFRA_LABELS[safra]} parcela"),
            Line2D([], [], color=color, lw=2.6, label=f"{SAFRA_LABELS[safra]} média"),
        ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Evolução dos índices por estágio fenológico (linha fina = parcela, grossa = média da safra)",
                 fontsize=13, y=0.955)
    fig.text(0.01, 0.005,
             "Em 23/24, GNDVI/EVI usam RGB em DN + RRENIR; VARI/TGI usam RGB em DN. "
             "Eixo y aparado nos quantis 2–98% de cada painel.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out / "evolucao_indices.png", dpi=150)
    plt.close(fig)


def plot_evolution_per_parcela(df, attr, out):
    units = df[["dose", "bloco"]].drop_duplicates().sort_values(["dose", "bloco"])
    fig, axes = plt.subplots(6, 4, figsize=(15, 15), sharex=True, sharey=True)
    vals = df[attr].dropna()
    lo, hi = vals.min(), vals.max()
    pad = 0.05 * (hi - lo)
    for ax, (_, u) in zip(axes.ravel(), units.iterrows()):
        for safra in sorted(df["safra"].unique()):
            g = df[(df["safra"] == safra) & (df["dose"] == u["dose"]) & (df["bloco"] == u["bloco"])]
            g = g.dropna(subset=[attr]).assign(_x=lambda d: d["estagio"].map(_xpos)).sort_values("_x")
            if g.empty:
                continue
            ax.plot(g["_x"], g[attr], color=SAFRA_COLORS.get(safra, "tab:green"), lw=1.8,
                    marker="o", ms=3.5)
        ax.set_title(f"d{int(u['dose'])}·b{int(u['bloco'])}", fontsize=9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(STAGE_PLOT_ORDER)))
        ax.set_xticklabels(STAGE_PLOT_ORDER, rotation=45, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    handles = [Line2D([], [], color=SAFRA_COLORS[s], lw=1.8, marker="o", ms=3.5,
                      label=SAFRA_LABELS[s]) for s in sorted(df["safra"].unique())]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{attr} por estágio fenológico — um painel por unidade dose·bloco",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "evolucao_ndvi_parcelas.png", dpi=150)
    plt.close(fig)


def write_heatmap(df, feat_cols, out):
    mean = df.groupby(["safra", "estagio"])[feat_cols].mean()
    cols = sorted(mean.index, key=lambda se: (se[0], STAGE_ORDER.index(se[1]) if se[1] in STAGE_ORDER else 99))
    labels = [f"{s.replace('_', '/')}·{e}" for (s, e) in cols]
    M = mean.loc[cols].T
    Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1, ddof=1), axis=0)

    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.5, 0.32 * len(M) + 2.5))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(Z.values, cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(M.index)), M.index, fontsize=6.5)
    for i in range(len(M.index)):
        for j in range(len(cols)):
            if np.isnan(Z.values[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="#555555")
    for edge in np.cumsum([sum(1 for (s, _) in cols if s == sf) for sf in sorted({s for (s, _) in cols})])[:-1]:
        ax.axvline(edge - 0.5, color="black", lw=1.2)
    fig.colorbar(im, ax=ax, pad=0.02, label="z-score (por atributo)")
    ax.set_title("Resumo (média entre parcelas) dos índices/texturas por safra × estágio")
    fig.tight_layout()
    fig.savefig(out / "heatmap_means.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/analysis/means_safra_estagio")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_sources()
    feat_cols = feature_columns(df)

    wide = write_wide(df, feat_cols, out := args.out)
    tidy = write_tidy(df, feat_cols, out)
    write_markdown(df, feat_cols, out)
    write_heatmap(df, feat_cols, out)
    plot_evolution(df, feat_cols, out)
    plot_evolution_per_parcela(df, "NDVI", out)

    counts = df.groupby(["safra", "estagio"]).size()
    print("Parcelas por safra × estágio:\n" + counts.to_string())
    print(f"\n{len(wide)} parcelas em {out/'parcelas_wide.csv'}; {len(tidy)} linhas em {out/'parcelas_tidy.csv'}")
    print(f"tabelas em {out/'MEANS.md'}; heatmap em {out/'heatmap_means.png'}")
    print(f"evolução em {out/'evolucao_indices.png'} e {out/'evolucao_ndvi_parcelas.png'}")


if __name__ == "__main__":
    main()
