#!/usr/bin/env python3
"""Cobertura de índices/texturas por safra × estágio fenológico.

Verifica, para cada atributo extraído nas safras de:
  - artifacts/runs/2223_baseline_plsr/results/baseline_2223_aggregated_features.csv
  - artifacts/runs/2324_5band_features/results/features_5band_2324.csv
se ele foi calculado para todas as parcelas de cada estágio. Estágios do
STAGE_ORDER sem linhas na safra são marcados explicitamente como «sem voo».

Legenda da matriz:
  ✓        calculado para todas as parcelas do estágio
  k/n      calculado parcialmente (k de n parcelas)
  —        atributo não extraído na safra (coluna ausente na fonte)
  sem voo  estágio sem linhas na fonte da safra

Saídas (--out):
  COVERAGE.md          matriz atributo × safra·estágio por família + resumo das lacunas
  coverage_matrix.csv  formato longo: safra, estagio, familia, atributo, n_parcelas,
                       n_calculados, pct

Uso:
  python3 stage_coverage.py [--out artifacts/analysis/stage_coverage]
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

SOURCES = [
    ("2022_2023", REPO / "artifacts/runs/2223_baseline_plsr/results/baseline_2223_aggregated_features.csv"),
    ("2023_2024", REPO / "artifacts/runs/2324_5band_features/results/features_5band_2324.csv"),
]

STAGE_ORDER = ["V6", "V8", "V10", "V11", "V13", "V18", "R1", "R2", "R5"]

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

META_DROP = {"stage", "medida", "sample", "unit", "n_veg_pixels",
             "phys_CHL_total", "phys_N_acumulado", "phys_N_percent"}

SAFRA_LABELS = {"2022_2023": "22/23", "2023_2024": "23/24"}


def load_seasons():
    seasons = {}
    for safra, path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"fonte ausente: {path}")
        df = pd.read_csv(path).rename(columns={"stage": "estagio", "medida": "estagio"})
        seasons[safra] = df
    return seasons


def attrs_by_family(seasons):
    fam_attrs = {}
    seen = set()
    for name, pred in FAMILIES:
        attrs = []
        for df in seasons.values():
            for c in df.columns:
                if c not in seen and c not in META_DROP and pred(c):
                    seen.add(c)
                    attrs.append(c)
        fam_attrs[name] = attrs
    return fam_attrs


def coverage_records(seasons, fam_attrs):
    records = []
    for safra, df in seasons.items():
        for estagio, sub in df.groupby("estagio"):
            for fam_name, attrs in fam_attrs.items():
                for a in attrs:
                    if a not in df.columns:
                        continue
                    n, k = len(sub), int(sub[a].notna().sum())
                    records.append({"safra": safra, "estagio": estagio,
                                    "familia": fam_name, "atributo": a,
                                    "n_parcelas": n, "n_calculados": k})
    rec = pd.DataFrame(records)
    st_rank = {e: i for i, e in enumerate(STAGE_ORDER)}
    fam_rank = {name: i for i, (name, _) in enumerate(FAMILIES)}
    att_rank = {a: i for i, a in enumerate([c for attrs in fam_attrs.values() for c in attrs])}
    return rec.assign(_s=rec["estagio"].map(st_rank), _f=rec["familia"].map(fam_rank),
                      _a=rec["atributo"].map(att_rank)) \
              .sort_values(["safra", "_s", "_f", "_a"]).drop(columns=["_s", "_f", "_a"])


def write_csv(rec, out):
    rec_out = rec.assign(pct=(100 * rec["n_calculados"] / rec["n_parcelas"]).round(1))
    rec_out.to_csv(out / "coverage_matrix.csv", index=False, float_format="%.6g")


def cell(n, has_col, k):
    if n == 0:
        return "sem voo"
    if not has_col:
        return "—"
    return "✓" if k == n else f"{k}/{n}"


def write_markdown(seasons, fam_attrs, rec, out):
    lines = [
        "# Cobertura de índices/texturas por safra × estágio",
        "",
        "«✓» calculado para todas as parcelas · «k/n» parcial · "
        "«—» atributo não extraído na safra · «sem voo» estágio sem linhas na safra.",
        "",
    ]
    lookups = {}
    for r in rec.itertuples(index=False):
        lookups[(r.safra, r.estagio, r.atributo)] = (r.n_parcelas, r.n_calculados)
    counts = {safra: df["estagio"].value_counts() for safra, df in seasons.items()}
    cols_present = {safra: set(df.columns) for safra, df in seasons.items()}

    for safra, _ in SOURCES:
        lines.append(f"## Safra {SAFRA_LABELS[safra]} ({counts[safra].sum()} parcelas)")
        lines.append("")
        for fam_name, attrs in fam_attrs.items():
            if not attrs:
                continue
            head = "| atributo | " + " | ".join(STAGE_ORDER) + " |"
            sep = "|---|" + "---|" * len(STAGE_ORDER)
            lines += [f"### {fam_name}", "", head, sep]
            for a in attrs:
                cells = []
                for e in STAGE_ORDER:
                    n = int(counts[safra].get(e, 0))
                    has_col = a in cols_present[safra]
                    k = lookups.get((safra, e, a), (0, 0))[1] if (n > 0 and has_col) else 0
                    cells.append(cell(n, has_col, k))
                lines.append(f"| {a} | " + " | ".join(cells) + " |")
            lines.append("")

    lines += ["## Resumo das lacunas", ""]
    all_attrs = [a for attrs in fam_attrs.values() for a in attrs]
    for safra, df in seasons.items():
        not_extracted = [a for a in all_attrs if a not in cols_present[safra]]
        no_flight = [e for e in STAGE_ORDER if int(counts[safra].get(e, 0)) == 0]
        odd = sorted(set(df["estagio"].dropna()) - set(STAGE_ORDER))
        lines.append(f"- **Safra {SAFRA_LABELS[safra]}**: "
                     f"sem voo em `{', '.join(no_flight) or 'nenhum'}`; "
                     f"atributos não extraídos: `{', '.join(not_extracted) or 'nenhum'}`"
                     + (f"; estágios fora do padrão: `{', '.join(odd)}`" if odd else ""))
    gaps = rec[(rec["n_parcelas"] > 0) & (rec["n_calculados"] < rec["n_parcelas"])]
    if gaps.empty:
        lines.append("- Nenhum cálculo parcial ou falho dentro dos estágios com voo.")
    else:
        lines.append("- Cálculos parciais/falhos:")
        for r in gaps.itertuples(index=False):
            lines.append(f"  - `{r.atributo}` em {SAFRA_LABELS[r.safra]}·{r.estagio}: "
                         f"{r.n_calculados}/{r.n_parcelas} parcelas")
    lines.append("")
    lines.append("Fontes: " + "; ".join(str(p.relative_to(REPO)) for _, p in SOURCES))
    (out / "COVERAGE.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(seasons, fam_attrs, rec):
    all_attrs = [a for attrs in fam_attrs.values() for a in attrs]
    print("Parcelas por safra × estágio:")
    for safra, df in seasons.items():
        counts = df["estagio"].value_counts()
        row = ", ".join(f"{e}:{counts.get(e, 0)}" for e in STAGE_ORDER if e in counts.index)
        print(f"  {SAFRA_LABELS[safra]}: {row}")
        missing = [a for a in all_attrs if a not in df.columns]
        no_flight = [e for e in STAGE_ORDER if e not in counts.index]
        if no_flight:
            print(f"    sem voo: {', '.join(no_flight)}")
        if missing:
            print(f"    não extraídos nesta safra: {', '.join(missing)}")
    gaps = rec[(rec["n_parcelas"] > 0) & (rec["n_calculados"] < rec["n_parcelas"])]
    print(f"\n{len(rec)} combinações avaliadas; cálculos parciais/falhos: {len(gaps)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/analysis/stage_coverage")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seasons = load_seasons()
    fam_attrs = attrs_by_family(seasons)
    rec = coverage_records(seasons, fam_attrs)

    write_csv(rec, args.out)
    write_markdown(seasons, fam_attrs, rec, args.out)
    print_summary(seasons, fam_attrs, rec)
    print(f"\nmatriz em {args.out/'COVERAGE.md'}; tabela em {args.out/'coverage_matrix.csv'}")


if __name__ == "__main__":
    main()
