"""Leitura canônica dos dados de campo da safra de milho 2025/26.

A planilha recebida usa cabeçalhos em múltiplas linhas. Este módulo concentra essa
interpretação para que as etapas de imagem não dependam de posições de colunas
espalhadas pelo código.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _block_number(value) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Bloco inválido: {value!r}")
    return int(match.group(1))


def _experiment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma única linha para cada uma das 40 parcelas do experimento."""
    data = frame.iloc[5:].copy()
    ids = pd.to_numeric(data[3], errors="coerce")
    data = data[ids.between(1, 40)].copy()
    data["parcela"] = ids.loc[data.index].astype(int)
    return data.drop_duplicates("parcela", keep="first").sort_values("parcela")


def normalize_workbook(path: str | Path) -> pd.DataFrame:
    """Retorna uma tabela por parcela com alvos e metadados temporalmente explícitos.

    ``chl_total_v10`` e ``chl_total_r1`` são as médias dos terços médio e
    superior da folha. As leituras por terço também são mantidas para a trilha
    hiperespectral, em que M/S são observações distintas da mesma parcela.
    """
    path = Path(path)
    field = _experiment_rows(pd.read_excel(path, sheet_name="Dados Campo Milho", header=None))
    lab = _experiment_rows(pd.read_excel(path, sheet_name="Dados Laboratório Milho", header=None))

    meta = pd.DataFrame({
        "parcela": field["parcela"].to_numpy(),
        "dose": field[0].ffill().to_numpy(),
        "tratamento": field[1].ffill().to_numpy(),
        "bloco": field[2].map(_block_number).to_numpy(),
        "ndvi_g_25nov": pd.to_numeric(field[4], errors="coerce").to_numpy(),
        "ndvi_g_v10": pd.to_numeric(field[5], errors="coerce").to_numpy(),
        "ndvi_g_v13": pd.to_numeric(field[6], errors="coerce").to_numpy(),
        "chl_field_v13": pd.to_numeric(field[7], errors="coerce").to_numpy(),
        "chl_field_r1": pd.to_numeric(field[8], errors="coerce").to_numpy(),
        "biomassa_g_v10": pd.to_numeric(field[9], errors="coerce").to_numpy(),
        "biomassa_kg_ha_v10": pd.to_numeric(field[10], errors="coerce").to_numpy(),
    })
    lab_values = pd.DataFrame({
        "parcela": lab["parcela"].to_numpy(),
        "chl_v10_medio": pd.to_numeric(lab[6], errors="coerce").to_numpy(),
        "chl_v10_superior": pd.to_numeric(lab[9], errors="coerce").to_numpy(),
        "chl_r1_medio": pd.to_numeric(lab[12], errors="coerce").to_numpy(),
        "chl_r1_superior": pd.to_numeric(lab[15], errors="coerce").to_numpy(),
    })
    out = meta.merge(lab_values, on="parcela", validate="one_to_one")
    out["chl_total_v10"] = out[["chl_v10_medio", "chl_v10_superior"]].mean(axis=1)
    out["chl_total_r1"] = out[["chl_r1_medio", "chl_r1_superior"]].mean(axis=1)
    if len(out) != 40 or out.parcela.nunique() != 40:
        raise ValueError("A planilha deve produzir exatamente 40 parcelas únicas")
    return out.sort_values("parcela").reset_index(drop=True)

