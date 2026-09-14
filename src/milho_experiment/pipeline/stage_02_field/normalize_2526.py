#!/usr/bin/env python3
"""Normaliza a planilha de campo/laboratório da safra 2025/26 em um CSV canônico."""
from __future__ import annotations

import argparse
from pathlib import Path

from .season2526 import normalize_workbook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_workbook(args.workbook)
    df.to_csv(out, index=False)
    print(f"{len(df)} parcelas x {df.shape[1]} colunas -> {out}")
    print("cobertura:")
    for c in ("biomassa_kg_ha_v10", "chl_total_v10", "chl_total_r1"):
        print(f"  {c}: {df[c].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
