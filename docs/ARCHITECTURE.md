# Arquitetura do experimento

O projeto possui três camadas:

1. **Dados externos** (`ABC_DATA_DIR`, padrão `data/`): ativos utilizáveis em `raw/` e itens
   reprovados preservados em `quarantine/`; o catálogo versionado está em `dataset_registry/`.
2. **Código**: módulos compartilhados em `src/milho_experiment`, CLIs existentes em `code/` e o
   framework GAN vendorizado.
3. **Artefatos** (`ABC_ARTIFACTS_DIR`, padrão `artifacts/`): resultados de execução acompanhados
   por manifesto; resultados históricos ficam em `artifacts/archive/legacy/`.

O comando oficial novo é `python scripts/abc_run.py --config <arquivo.toml>`. Ele preserva a
compatibilidade com os entrypoints atuais e registra configuração, hashes de entradas e revisão Git.
