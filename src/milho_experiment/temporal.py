"""Primitivas do experimento temporal de valor preditivo das GANs.

O módulo não depende de PyTorch. Ele concentra a montagem auditável das
parcelas, a ordem dos canais temporais e as métricas pareadas usadas pelo
orquestrador da etapa 08 de cenários temporais.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .indices import attribute_channels


HORIZONS: dict[str, tuple[tuple[str, ...], str]] = {
    "V13": (("V6", "V8"), "V13"),
    "R5": (("V6", "V8", "V13", "R2"), "R5"),
}

AUX_CANDIDATES = (
    "chlorophyll",
    "NDVI",
    "NDRE",
    "SAVI",
    "EVI2",
    "glcm_asm",
    "glcm_contrast",
    "glcm_entropy",
    "glcm_homogeneity",
)


def _dose_key(value: object) -> str:
    """Normaliza doses numéricas sem perder suporte a rótulos textuais."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else format(number, ".12g")


@dataclass(frozen=True)
class TemporalRecord:
    fid: int
    block: int
    dose_n: object
    history_stages: tuple[str, ...]
    target_stage: str
    history_paths: tuple[Path, ...]
    target_path: Path
    biomass: float
    productivity: float

    def target(self, name: str) -> float:
        key = name.strip().lower()
        if key == "biomassa":
            return self.biomass
        if key == "produtividade":
            return self.productivity
        raise ValueError(f"alvo desconhecido: {name!r}")


def build_records(stage4: str | Path, table: str | Path,
                  target_stage: str) -> list[TemporalRecord]:
    """Monta uma amostra por parcela e exige Y do estágio correto.

    A tabela de campo é indexada por ``(Estagio, Bloco, Dose_N)``. Isso evita
    o comportamento legado em que uma biomassa de outro estágio podia
    sobrescrever a biomassa do alvo.
    """
    stage4 = Path(stage4)
    target_stage = target_stage.upper()
    if target_stage not in HORIZONS:
        raise ValueError(f"horizonte inválido: {target_stage!r}; válidos: {tuple(HORIZONS)}")
    history_stages, expected_target = HORIZONS[target_stage]
    assert target_stage == expected_target

    manifest = pd.read_csv(stage4 / "manifest.csv")
    required = {"fid", "stage", "bloco", "dose_n", "npy"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifesto sem colunas obrigatórias: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["stage"] = manifest["stage"].astype(str).str.upper()
    manifest["dose_key"] = manifest["dose_n"].map(_dose_key)
    if manifest.duplicated(["fid", "stage"]).any():
        dup = manifest.loc[manifest.duplicated(["fid", "stage"], keep=False),
                           ["fid", "stage"]]
        raise ValueError(f"mais de uma imagem por parcela/estágio: {dup.to_dict('records')[:5]}")

    field = pd.read_excel(table).copy()
    field_required = {"Estagio", "Bloco", "Dose_N", "Biomassa", "Produtividade"}
    missing = field_required - set(field.columns)
    if missing:
        raise ValueError(f"tabela de campo sem colunas obrigatórias: {sorted(missing)}")
    field["Estagio"] = field["Estagio"].astype(str).str.upper()
    field["dose_key"] = field["Dose_N"].map(_dose_key)
    field = field[field["Estagio"] == target_stage]
    if field.duplicated(["Estagio", "Bloco", "dose_key"]).any():
        raise ValueError("alvo duplicado para estágio/bloco/dose")
    y_by_key = {
        (str(r.Estagio), int(r.Bloco), str(r.dose_key)): r
        for r in field.itertuples()
    }

    by_image = {(int(r.fid), str(r.stage)): r for r in manifest.itertuples()}
    records: list[TemporalRecord] = []
    needed = (*history_stages, target_stage)
    for fid in sorted(int(v) for v in manifest.fid.unique()):
        rows = [by_image.get((fid, stage)) for stage in needed]
        if any(row is None for row in rows):
            continue
        first = rows[0]
        assert first is not None
        block = int(first.bloco)
        dose_key = _dose_key(first.dose_n)
        if any(int(row.bloco) != block or _dose_key(row.dose_n) != dose_key
               for row in rows if row is not None):
            raise ValueError(f"metadados inconsistentes entre estágios da parcela {fid}")
        yrow = y_by_key.get((target_stage, block, dose_key))
        if yrow is None:
            continue
        biomass = float(yrow.Biomassa)
        productivity = float(yrow.Produtividade)
        if not np.isfinite(biomass) or not np.isfinite(productivity):
            continue
        paths = tuple((stage4 / str(row.npy)).resolve() for row in rows if row is not None)
        if not all(path.is_file() for path in paths):
            absent = [str(path) for path in paths if not path.is_file()]
            raise FileNotFoundError(f"recortes ausentes da parcela {fid}: {absent}")
        records.append(TemporalRecord(
            fid=fid,
            block=block,
            dose_n=first.dose_n,
            history_stages=history_stages,
            target_stage=target_stage,
            history_paths=paths[:-1],
            target_path=paths[-1],
            biomass=biomass,
            productivity=productivity,
        ))
    if not records:
        raise RuntimeError(f"nenhuma parcela completa para o horizonte {target_stage}")
    return records


def load_record(record: TemporalRecord) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    history = tuple(np.load(path).astype(np.float32) for path in record.history_paths)
    target = np.load(record.target_path).astype(np.float32)
    shapes = {array.shape for array in (*history, target)}
    if len(shapes) != 1 or next(iter(shapes))[-1] != 3:
        raise ValueError(f"formas RRENIR incompatíveis na parcela {record.fid}: {sorted(shapes)}")
    return history, target


def temporal_input(history: Sequence[np.ndarray], aux_names: Sequence[str] = (),
                   normalize: str = "image") -> np.ndarray:
    """Concatena blocos cronológicos ``[Red, RE, NIR, auxiliares]``."""
    if not history:
        raise ValueError("histórico temporal vazio")
    names = tuple(aux_names)
    blocks = []
    for refl in history:
        if refl.ndim != 3 or refl.shape[-1] != 3:
            raise ValueError(f"imagem RRENIR deve ter shape (H,W,3); recebeu {refl.shape}")
        if names:
            aux = attribute_channels(refl, names=names, normalize=normalize)
            blocks.append(np.concatenate([refl, aux], axis=-1))
        else:
            blocks.append(refl)
    return np.concatenate(blocks, axis=-1).astype(np.float32, copy=False)


def expected_input_channels(history_stages: Sequence[str], aux_names: Sequence[str]) -> int:
    return len(tuple(history_stages)) * (3 + len(tuple(aux_names)))


def assert_disjoint(train: Iterable[TemporalRecord], test: Iterable[TemporalRecord]) -> None:
    train_ids = {record.fid for record in train}
    test_ids = {record.fid for record in test}
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(f"vazamento de parcelas: {sorted(overlap)}")


def paired_stratified_bootstrap_delta(
    y: Sequence[float],
    baseline: Sequence[float],
    candidate: Sequence[float],
    blocks: Sequence[int],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float]:
    """IC95% pareado de ΔR², reamostrando parcelas dentro de cada bloco."""
    from sklearn.metrics import r2_score

    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    blocks = np.asarray(blocks)
    if not (len(y) == len(baseline) == len(candidate) == len(blocks)):
        raise ValueError("y, predições e blocos devem ter o mesmo comprimento")
    rng = np.random.default_rng(seed)
    positions = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    deltas = []
    for _ in range(n_bootstrap):
        take = np.concatenate([rng.choice(pos, size=len(pos), replace=True) for pos in positions])
        if np.unique(y[take]).size < 2:
            continue
        deltas.append(float(r2_score(y[take], candidate[take]) -
                            r2_score(y[take], baseline[take])))
    if not deltas:
        return np.nan, np.nan
    return tuple(float(v) for v in np.quantile(deltas, [0.025, 0.975]))
