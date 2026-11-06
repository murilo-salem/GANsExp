#!/usr/bin/env python3
"""Compara o perfil espectral RRENIR real e sintético no mesmo estágio.

O eixo espectral usa os três canais disponíveis no projeto, na ordem
Vermelho -> Red edge -> NIR. Como os metadados dos ortomosaicos não registram
os centros das bandas, o gráfico não atribui comprimentos de onda nominais.

Exemplo:
    python3 -m milho_experiment.pipeline.stage_10_analysis.spectral_real_vs_synthetic \
        --real-manifest artifacts/runs/2324_stage4_phenology/results/manifest.csv \
        --synthetic-dir artifacts/runs/2324_gan_example_images/results \
        --season 2023_2024 --target-stage R2 --source-stage V13 \
        --out artifacts/analysis/spectral_real_vs_synthetic_2324_r2_v13
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


BANDS = ("Vermelho", "Red edge", "NIR")
SYNTHETIC_RE = re.compile(
    r"^p(?P<fid>\d+)_(?P<source>[VR]\d+)to(?P<target>[VR]\d+).*_fake\.npy$",
    re.IGNORECASE,
)


def discover_synthetic(directory: str | Path, target_stage: str,
                       source_stage: str | None = None) -> pd.DataFrame:
    """Descobre tensores sintéticos e extrai parcela/estágios do nome."""
    target_stage = target_stage.upper()
    source_stage = source_stage.upper() if source_stage else None
    rows = []
    for path in sorted(Path(directory).glob("*_fake.npy")):
        match = SYNTHETIC_RE.match(path.name)
        if not match:
            continue
        parsed = {key: value.upper() if key != "fid" else int(value)
                  for key, value in match.groupdict().items()}
        if parsed["target"] != target_stage:
            continue
        if source_stage and parsed["source"] != source_stage:
            continue
        rows.append({**parsed, "synthetic_path": str(path.resolve())})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(
            f"nenhum *_fake.npy encontrado para {source_stage or '*'}->{target_stage} "
            f"em {Path(directory)}"
        )
    if frame["fid"].duplicated().any():
        sources = sorted(frame["source"].unique())
        raise ValueError(
            "mais de um sintético por parcela; informe --source-stage "
            f"(disponíveis: {sources})"
        )
    return frame


def _load_rrenir(path: str | Path) -> np.ndarray:
    image = np.load(path).astype(np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"tensor RRENIR inválido em {path}: shape={image.shape}")
    return image


def paired_profiles(real_manifest: str | Path, synthetic_dir: str | Path,
                    target_stage: str, source_stage: str | None = None,
                    season: str | None = None) -> pd.DataFrame:
    """Retorna uma linha por parcela, origem e banda para pares real/sintético."""
    manifest_path = Path(real_manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"fid", "stage", "npy"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifesto sem colunas obrigatórias: {sorted(missing)}")

    manifest["stage"] = manifest["stage"].astype(str).str.upper()
    real = manifest[manifest["stage"] == target_stage.upper()].copy()
    if season is not None:
        if "safra" not in real:
            raise ValueError("--season exige a coluna 'safra' no manifesto")
        real = real[real["safra"].astype(str) == str(season)]
    if real.empty:
        raise ValueError(f"nenhum real encontrado para safra={season!r}, estágio={target_stage}")
    if real["fid"].duplicated().any():
        raise ValueError("mais de uma imagem real por parcela no estágio selecionado")
    real["real_path"] = [str((manifest_path.parent / path).resolve()) for path in real["npy"]]

    synthetic = discover_synthetic(synthetic_dir, target_stage, source_stage)
    pairs = real.merge(synthetic, on="fid", how="inner", validate="one_to_one")
    if pairs.empty:
        raise ValueError("não há parcelas comuns entre o real e o sintético")

    rows = []
    for pair in pairs.itertuples(index=False):
        real_image = _load_rrenir(pair.real_path)
        synthetic_image = _load_rrenir(pair.synthetic_path)
        if real_image.shape != synthetic_image.shape:
            raise ValueError(
                f"formas diferentes na parcela {pair.fid}: "
                f"real={real_image.shape}, sintético={synthetic_image.shape}"
            )
        # Usa o suporte espacial do alvo real. Nos produtos com máscara, isso
        # remove o preenchimento zero externo à parcela; nos legados, mantém o crop todo.
        support = np.isfinite(real_image).all(axis=-1) & (np.abs(real_image).sum(axis=-1) > 1e-8)
        support &= np.isfinite(synthetic_image).all(axis=-1)
        if not support.any():
            raise ValueError(f"parcela {pair.fid} sem pixels válidos")
        for origin, image in (("Real", real_image), ("Sintético", synthetic_image)):
            means = image[support].mean(axis=0)
            for band_index, (band, value) in enumerate(zip(BANDS, means)):
                rows.append({
                    "safra": season or getattr(pair, "safra", ""),
                    "target_stage": target_stage.upper(),
                    "source_stage": pair.source,
                    "fid": int(pair.fid),
                    "origem": origin,
                    "band_index": band_index,
                    "banda": band,
                    "reflectancia_media": float(value),
                })
    return pd.DataFrame(rows)


def summarize(profiles: pd.DataFrame) -> pd.DataFrame:
    """Resume parcelas e calcula IC95% t da média por origem/banda."""
    grouped = profiles.groupby(
        ["safra", "target_stage", "source_stage", "origem", "band_index", "banda"],
        sort=False,
    )["reflectancia_media"]
    summary = grouped.agg(media="mean", desvio_padrao="std", n="count").reset_index()
    summary["erro_ic95"] = np.where(
        summary["n"] > 1,
        t.ppf(0.975, summary["n"] - 1) * summary["desvio_padrao"] / np.sqrt(summary["n"]),
        np.nan,
    )
    summary["ic95_inferior"] = summary["media"] - summary["erro_ic95"]
    summary["ic95_superior"] = summary["media"] + summary["erro_ic95"]
    return summary


def paired_metrics(profiles: pd.DataFrame) -> pd.DataFrame:
    """Calcula erro pareado por banda e ângulo espectral por parcela."""
    wide = profiles.pivot(index=["fid", "band_index", "banda"], columns="origem",
                          values="reflectancia_media").reset_index()
    wide["diferenca_sintetico_menos_real"] = wide["Sintético"] - wide["Real"]
    metrics = []
    for (band_index, band), values in wide.groupby(["band_index", "banda"], sort=True):
        delta = values["diferenca_sintetico_menos_real"].to_numpy()
        metrics.append({
            "escala": "banda", "banda": band, "band_index": band_index,
            "n": len(values), "vies": float(delta.mean()),
            "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        })

    cube = profiles.pivot(index="fid", columns=["origem", "band_index"],
                          values="reflectancia_media")
    real = cube["Real"].to_numpy()
    synthetic = cube["Sintético"].to_numpy()
    cosine = np.sum(real * synthetic, axis=1) / (
        np.linalg.norm(real, axis=1) * np.linalg.norm(synthetic, axis=1) + 1e-12
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    metrics.append({
        "escala": "perfil", "banda": "todas", "band_index": np.nan,
        "n": len(angles), "vies": np.nan, "mae": np.nan, "rmse": np.nan,
        "angulo_espectral_medio_graus": float(angles.mean()),
    })
    return pd.DataFrame(metrics)


def plot_profiles(profiles: pd.DataFrame, summary: pd.DataFrame, output: str | Path) -> None:
    colors = {"Real": "#1769aa", "Sintético": "#d95f02"}
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for origin in ("Real", "Sintético"):
        parcel = profiles[profiles["origem"] == origin]
        for _, values in parcel.groupby("fid"):
            values = values.sort_values("band_index")
            ax.plot(values["band_index"], values["reflectancia_media"],
                    color=colors[origin], lw=0.8, alpha=0.18)
        values = summary[summary["origem"] == origin].sort_values("band_index")
        x = values["band_index"].to_numpy()
        y = values["media"].to_numpy()
        error = values["erro_ic95"].to_numpy()
        ax.plot(x, y, color=colors[origin], marker="o", ms=6, lw=2.6, label=origin)
        if np.isfinite(error).any():
            ax.fill_between(x, y - error, y + error, color=colors[origin], alpha=0.16)

    first = profiles.iloc[0]
    n = profiles["fid"].nunique()
    season = str(first["safra"]).replace("_", "/")
    ax.set_title(
        f"Perfil espectral real × sintético — safra {season}, estágio {first['target_stage']}\n"
        f"entrada {first['source_stage']} · n={n} parcelas"
    )
    ax.set_xticks(range(len(BANDS)), BANDS)
    ax.set_ylabel("Reflectância normalizada média")
    ax.set_xlabel("Canal RRENIR (ordem espectral)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.text(0.01, -0.18,
            "Linhas finas: parcelas; linha grossa: média; faixa: IC95% t entre parcelas. "
            "Centros espectrais não informados nos metadados.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--season")
    parser.add_argument("--target-stage", required=True)
    parser.add_argument("--source-stage")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = paired_profiles(args.real_manifest, args.synthetic_dir, args.target_stage,
                               args.source_stage, args.season)
    summary = summarize(profiles)
    metrics = paired_metrics(profiles)
    profiles.to_csv(out / "perfis_por_parcela.csv", index=False, float_format="%.8g")
    summary.to_csv(out / "resumo_curvas.csv", index=False, float_format="%.8g")
    metrics.to_csv(out / "metricas_pareadas.csv", index=False, float_format="%.8g")
    plot_profiles(profiles, summary, out / "curva_espectral_real_vs_sintetico.png")
    print(f"{profiles['fid'].nunique()} pares real/sintético -> {out}")


if __name__ == "__main__":
    main()
