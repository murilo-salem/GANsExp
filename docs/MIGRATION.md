# Organização direta pela pipeline

O código próprio está em `src/milho_experiment/pipeline/stage_01_*` até
`stage_11_*`. Os módulos são executados por `python -m` e as receitas ficam em
`configs/stages/<etapa>/`. Não há camada de compatibilidade em `code/`.

## Regras para novas mudanças

1. Implementar lógica compartilhada em `src/milho_experiment/`, na etapa correspondente.
2. Criar uma receita TOML e executá-la pelo runner para todo experimento reprodutível.
3. Gravar saídas em `artifacts/runs/<run-id>/`; o arquivo histórico é somente leitura em
   `artifacts/archive/legacy/`.
4. Atualizar `docs/EXPERIMENT_CATALOG.md` e declarar a unidade de validação.
5. Declarar `execution.module`, nunca um caminho de script, em receitas novas.

## Legado preservado

`legacy/preprocessing/` foi movido sem edição funcional. Seus caminhos absolutos e dependências
não declaradas impedem execução portátil; ele é referência de histórico, não parte do pipeline
oficial.
