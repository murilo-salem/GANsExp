# Experimento de milho — pipeline reproduzível

Este repositório reúne ortomosaicos, atributos espectrais/texturais, modelos agronômicos e uma
GAN fenológica. A fonte de verdade científica está em `code/pipeline/RESULTS_STATUS.md`; o novo
fluxo reproduzível está organizado em `configs/`, `src/`, `scripts/`, `docs/` e `artifacts/`.

## Início rápido

```bash
python3 -m pip install -e ".[analysis]"
make test
make dry-run
```

Defina `ABC_DATA_DIR` quando os dados não estiverem em `./data`, `ABC_ARTIFACTS_DIR` para gravar
resultados fora da cópia de trabalho e `ABC_ARCHIVE_DIR` para backups externos (padrão:
`../ABC-archive`). Os dados ficam em `raw/` e os itens reprovados em `quarantine/`. Novas execuções usam:

```bash
python3 scripts/abc_run.py --config configs/safras/2324_multisafra.toml --check-inputs
python3 scripts/abc_run.py --config configs/safras/2324_multisafra.toml
```

## Mapa

- `src/milho_experiment/`: caminhos, validação anti-vazamento e manifestos.
- `code/pipeline/` e `code/analysis/`: CLIs existentes, mantidos durante a migração.
- `configs/`: receitas TOML reproduzíveis por experimento.
- `artifacts/`: resultados novos e arquivo histórico, fora do Git.
- `dataset_registry/`: catálogo versionado, checksums e quarentena dos binários externos.
- `docs/`: arquitetura, validação, catálogo e migração.
- `legacy/preprocessing/`: scripts preservados de pré-processamento não portáveis.
- `pytorch-CycleGAN-and-pix2pix/`: framework GAN vendorizado; veja `vendor/PIX2PIX_VENDOR.md`.
