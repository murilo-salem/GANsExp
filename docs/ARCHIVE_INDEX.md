# Índice de resultados históricos

Os resultados já existentes foram preservados no arquivo abaixo. Eles não devem ser sobrescritos
por novas execuções.

| Área histórica | Conteúdo | Classificação |
|---|---|---|
| `artifacts/archive/legacy/pipeline/multisafra/` | PLSR intra/inter-safra | referência oficial atual |
| `artifacts/archive/legacy/pipeline/plsr_2324/` | features e PLSR 23/24 | referência de features |
| `artifacts/archive/legacy/pipeline/stage4_*` | recortes fenológicos | intermediário reprodutível |
| `artifacts/archive/legacy/pipeline/stage5_*` | inferência GAN | diagnóstico GAN |
| `artifacts/archive/legacy/pipeline/stage8_*` | cenários e CV GAN | proof-of-mechanism/diagnóstico |
| `artifacts/archive/legacy/analysis/` | EDA e seleção VIP | exploratório |
| `artifacts/archive/legacy/gan/` | datasets, checkpoints e resultados GAN | histórico gerado |

Ao reexecutar um desses fluxos, use `artifacts/runs/<run-id>/` e acrescente o manifesto ao
catálogo de experimentos. Só então o resultado pode ser promovido para o arquivo histórico.
