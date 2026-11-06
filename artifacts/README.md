# Artefatos de execução

- `runs/<run-id>/`: saídas de novas execuções e o `manifest.json` correspondente.
- `archive/`: resultados históricos preservados durante a migração gradual.

Os dados brutos, checkpoints e artefatos grandes não são versionados. Use `ABC_DATA_DIR` e
`ABC_ARTIFACTS_DIR` para armazená-los fora da cópia de trabalho quando necessário.

