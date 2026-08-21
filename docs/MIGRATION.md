# Migração gradual

O código atual não é removido nem tem caminhos alterados nesta fase. Os entrypoints em
`code/pipeline/` e `code/analysis/` continuam sendo a camada de compatibilidade. O novo fluxo
adiciona configurações TOML e `scripts/abc_run.py` por cima deles.

## Regras para novas mudanças

1. Implementar lógica compartilhada em `src/milho_experiment/`, não em novos scripts soltos.
2. Criar uma receita TOML e executá-la pelo runner para todo experimento reprodutível.
3. Gravar saídas em `artifacts/runs/<run-id>/`; o arquivo histórico é somente leitura em
   `artifacts/archive/legacy/`.
4. Atualizar `docs/EXPERIMENT_CATALOG.md` e declarar a unidade de validação.
5. Depois que uma receita substituir um CLI sem regressões, transformar o CLI antigo em wrapper
   deprecatado e mover sua implementação para `src/`.

## Legado preservado

`legacy/preprocessing/` foi movido sem edição funcional. Seus caminhos absolutos e dependências
não declaradas impedem execução portátil; ele é referência de histórico, não parte do pipeline
oficial.
