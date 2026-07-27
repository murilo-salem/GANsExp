#!/usr/bin/env python3
"""EDA + QA da matriz de atributos hiperespectrais (features_12dez.csv).

Gera figuras e um relatório-resumo (EDA_REPORT.md):
  - espectros médios por parcela (+ dispersão M vs S)
  - distribuições dos índices de vegetação e das texturas GLCM
  - PCA (scree + PC1×PC2 por parcela/medida + loadings vs comprimento de onda)
  - clustering (KMeans k=2..6 com silhueta + dendrograma hierárquico)
  - correlações (índices+texturas) e correlação banda-a-banda
  - QA/outliers (Mahalanobis no espaço PCA, n_veg_pixels, faixa de reflectância)

Uso:
  python3 eda_report.py --features out/features_12dez.csv --out out/eda
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from plsr_kfold import INDEX_COLS, select_block

TEX_PREFIX = "glcm"


def spec_cols(df):
    cols = [c for c in df.columns if c.startswith("b") and c[1:4].isdigit()]
    waves = [int(re.search(r"_(\d+)nm", c).group(1)) for c in cols]
    return cols, np.array(waves)


def fig_spectra(df, out):
    cols, waves = spec_cols(df)
    order = np.argsort(waves)
    waves, X = waves[order], df[cols].values[:, order]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for _, r in df.iterrows():
        ax[0].plot(waves, r[cols].values[order], lw=0.6, alpha=0.5,
                   color=("tab:blue" if r["medida"] == "M" else "tab:orange"))
    ax[0].set(title="Espectro médio por amostra (azul=M, laranja=S)",
              xlabel="comprimento de onda (nm)", ylabel="reflectância")
    mu, sd = X.mean(0), X.std(0)
    ax[1].plot(waves, mu, color="k", label="média")
    ax[1].fill_between(waves, mu - sd, mu + sd, alpha=0.25, label="±1 dp")
    ax[1].set(title="Espectro médio ± desvio (todas as amostras)",
              xlabel="comprimento de onda (nm)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "spectra.png", dpi=120); plt.close(fig)


def fig_distributions(df, out):
    idx = [c for c in INDEX_COLS if c in df.columns]
    tex = [c for c in df.columns if c.startswith(TEX_PREFIX)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].boxplot([df[c].dropna() for c in idx], tick_labels=idx, showmeans=True)
    ax[0].set(title="Índices de vegetação"); ax[0].tick_params(axis="x", rotation=45)
    ax[1].boxplot([df[c].dropna() for c in tex], tick_labels=tex, showmeans=True)
    ax[1].set(title="Texturas GLCM (global + janelas)")
    ax[1].tick_params(axis="x", rotation=90)
    fig.tight_layout(); fig.savefig(out / "distributions.png", dpi=120); plt.close(fig)


def fig_pca(df, out, block="spectrum"):
    X = StandardScaler().fit_transform(select_block(df, block).values)
    p = PCA(n_components=min(10, X.shape[0], X.shape[1])).fit(X)
    scores = p.transform(X)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ev = p.explained_variance_ratio_ * 100
    ax[0].bar(range(1, len(ev) + 1), ev); ax[0].plot(range(1, len(ev) + 1), np.cumsum(ev), "r-o", ms=3)
    ax[0].set(title=f"Scree ({block})", xlabel="componente", ylabel="% variância")
    parc = df["parcela"].values
    sc = ax[1].scatter(scores[:, 0], scores[:, 1], c=parc, cmap="tab20", s=30)
    ax[1].set(title="PC1×PC2 por parcela", xlabel=f"PC1 ({ev[0]:.0f}%)", ylabel=f"PC2 ({ev[1]:.0f}%)")
    fig.colorbar(sc, ax=ax[1], fraction=0.04, label="parcela")
    for med, col in (("M", "tab:blue"), ("S", "tab:orange")):
        m = df["medida"].values == med
        ax[2].scatter(scores[m, 0], scores[m, 1], c=col, s=30, label=med)
    ax[2].set(title="PC1×PC2 por medida (M/S)", xlabel="PC1", ylabel="PC2"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / f"pca_{block}.png", dpi=120); plt.close(fig)
    # loadings vs wavelength (só faz sentido no espectro/derivada)
    cols = select_block(df, block).columns
    waves = [int(m.group(1)) if (m := re.search(r"_(\d+)nm", c)) else i for i, c in enumerate(cols)]
    o = np.argsort(waves)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k in range(3):
        ax.plot(np.array(waves)[o], p.components_[k][o], lw=1, label=f"PC{k+1}")
    ax.set(title=f"Loadings PC1-3 ({block})", xlabel="nm"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"pca_loadings_{block}.png", dpi=120); plt.close(fig)
    return scores, ev


def fig_clustering(df, scores, out):
    Z = scores[:, :5]
    ks, sils = list(range(2, 7)), []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
        sils.append(silhouette_score(Z, lab))
    best_k = ks[int(np.argmax(sils))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(ks, sils, "o-"); ax[0].set(title="Silhueta × k (KMeans, espaço PCA)",
                                          xlabel="k", ylabel="silhueta")
    ax[0].axvline(best_k, color="r", ls="--", label=f"melhor k={best_k}"); ax[0].legend()
    dendrogram(linkage(Z, method="ward"), labels=df["sample"].values, ax=ax[1],
               leaf_font_size=6)
    ax[1].set(title="Dendrograma (Ward)")
    fig.tight_layout(); fig.savefig(out / "clustering.png", dpi=120); plt.close(fig)
    return best_k, max(sils)


def fig_correlations(df, out):
    cols = [c for c in INDEX_COLS if c in df.columns] + [c for c in df.columns if c.startswith(TEX_PREFIX)]
    corr = df[cols].corr()
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set(xticks=range(len(cols)), yticks=range(len(cols)),
              title="Correlação: índices + texturas")
    ax[0].set_xticklabels(cols, rotation=90, fontsize=6); ax[0].set_yticklabels(cols, fontsize=6)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    scols, waves = spec_cols(df); o = np.argsort(waves)
    bc = np.corrcoef(df[scols].values[:, o].T)
    im2 = ax[1].imshow(bc, cmap="coolwarm", vmin=-1, vmax=1, extent=[waves[o][0], waves[o][-1]] * 2)
    ax[1].set(title="Correlação banda-a-banda (redundância)", xlabel="nm", ylabel="nm")
    fig.colorbar(im2, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(out / "correlations.png", dpi=120); plt.close(fig)


def qa_outliers(df, scores):
    Z = scores[:, :5]
    mu = Z.mean(0)
    cov = np.cov(Z.T)
    inv = np.linalg.pinv(cov)
    d2 = np.array([(z - mu) @ inv @ (z - mu) for z in Z])
    thr = np.percentile(d2, 95) * 1.5
    df = df.copy()
    df["mahal2"] = d2
    flags = []
    for _, r in df.iterrows():
        why = []
        if r["mahal2"] > thr:
            why.append(f"Mahalanobis²={r['mahal2']:.1f}")
        if r["n_veg_pixels"] < 0.5 * df["n_veg_pixels"].median():
            why.append(f"pouca vegetação ({int(r['n_veg_pixels'])})")
        if why:
            flags.append((r["sample"], "; ".join(why)))
    return flags, df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="out/eda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)

    fig_spectra(df, out)
    fig_distributions(df, out)
    scores, ev = fig_pca(df, out, "spectrum")
    fig_pca(df, out, "derivative")
    best_k, best_sil = fig_clustering(df, scores, out)
    fig_correlations(df, out)
    flags, _ = qa_outliers(df, scores)

    # concordância do agrupamento k=2 com a medida M/S + separação no PC1
    lab2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(scores[:, :5])
    ms = (df["medida"].values == "M").astype(int)
    agree = max((lab2 == ms).mean(), (lab2 != ms).mean()) * 100
    pc1_M, pc1_S = scores[ms == 1, 0].mean(), scores[ms == 0, 0].mean()

    lines = [
        "# EDA + QA — atributos hiperespectrais (voo 12.12)", "",
        f"Amostras: **{len(df)}** ({df['parcela'].nunique()} parcelas, "
        f"M={int((df.medida=='M').sum())} / S={int((df.medida=='S').sum())}) · "
        f"{df.shape[1]} atributos.", "",
        "## Estrutura (PCA, bloco espectro)",
        f"- PC1 explica {ev[0]:.0f}% e PC1+PC2 {ev[0]+ev[1]:.0f}% da variância "
        f"(→ forte redundância entre as 300 bandas). Ver `pca_spectrum.png`, `pca_loadings_spectrum.png`.",
        "", "## Agrupamento",
        f"- Melhor k (KMeans, silhueta): **k={best_k}** (silhueta={best_sil:.2f}). "
        f"{'Estrutura de grupos fraca' if best_sil < 0.25 else 'Há agrupamento perceptível'} "
        "— ver `clustering.png` (checar contra tratamentos quando o croqui for recuperado).",
        f"- **A medida M/S é a principal fonte de variação:** o agrupamento k=2 coincide com "
        f"M/S em **{agree:.0f}%** dos casos e há separação no PC1 (média PC1: M={pc1_M:+.1f} vs "
        f"S={pc1_S:+.1f}). Ver painel direito de `pca_spectrum.png`. Confirmar o que M/S representa "
        "(ex.: manhã/tarde, folha superior/inferior, sol/sombra) antes de usar como amostras independentes.",
        "", "## QA / outliers",
    ]
    if flags:
        lines.append(f"- **{len(flags)} amostra(s) sinalizada(s):**")
        lines += [f"  - `{s}`: {w}" for s, w in flags]
    else:
        lines.append("- Nenhum outlier forte (Mahalanobis/vegetação).")
    lines += [
        "- Cubos já perdidos na truncagem do dataset.zip (fora do CSV): `20M`, `5S`.",
        "", "## Figuras", "",
        "| arquivo | conteúdo |", "|---|---|",
        "| `spectra.png` | espectros por amostra + média±dp |",
        "| `distributions.png` | boxplots de índices e texturas |",
        "| `pca_spectrum.png` / `pca_derivative.png` | scree + PC1×PC2 (parcela/medida) |",
        "| `pca_loadings_*.png` | loadings PC1-3 vs comprimento de onda |",
        "| `clustering.png` | silhueta × k + dendrograma |",
        "| `correlations.png` | correlação índices+texturas e banda-a-banda |",
        "", "> Modelagem supervisionada (PLSR) permanece bloqueada até o Y de campo "
        "(`../../MISSING_DATA.md`).",
    ]
    (out / "EDA_REPORT.md").write_text("\n".join(lines) + "\n")
    print("Figuras + EDA_REPORT.md em", out)
    print("Outliers sinalizados:", [s for s, _ in flags] or "nenhum")


if __name__ == "__main__":
    main()
