# Arquitetura do experimento

O projeto possui três camadas:

1. **Dados externos** (`ABC_DATA_DIR`, padrão `data/`): ativos utilizáveis em `raw/` e itens
   reprovados preservados em `quarantine/`; o catálogo versionado está em `dataset_registry/`.
2. **Código**: módulos compartilhados e etapas da pipeline em `src/milho_experiment`; o framework
   Pix2Pix permanece como checkout externo, acessado pelo adaptador `milho_experiment.pix2pix`.
3. **Artefatos** (`ABC_ARTIFACTS_DIR`, padrão `artifacts/`): resultados de execução acompanhados
   por manifesto; resultados históricos ficam em `artifacts/archive/legacy/`.

O comando oficial é `python scripts/abc_run.py --config <arquivo.toml>`. As receitas executam
módulos Python (`execution.module`) e registram configuração, hashes, revisão Git e etapa da pipeline.
