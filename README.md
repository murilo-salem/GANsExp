# Experimento de milho — sensoriamento remoto, fenologia e previsão agronômica

Este repositório reúne dados, código e artefatos de um experimento de estimativa de **biomassa** e **produtividade** de milho a partir de ortomosaicos, atributos espectrais e de textura e modelos de projeção fenológica. O objetivo é medir o que pode ser estimado por sensoriamento remoto sob validação por parcela e testar se imagens sintéticas geradas por GAN acrescentam valor à previsão agronômica.

O resultado central é consistente: textura extraída dos ortomosaicos prediz biomassa em duas safras. As GANs reproduzem parte da aparência e dos índices de estágios futuros, mas **ainda não demonstraram ganho preditivo agronômico**. Resultados negativos e calibrações que exigem biomassa local são reportados explicitamente, sem equivalê-los a previsão remota.

## Estado científico atual

A unidade independente de validação é a **parcela**. Estágios repetidos, pares fenológicos e amostras da mesma parcela ficam juntos em treino ou teste. O protocolo padrão é `GroupKFold` por parcela; preditores físicos co-medidos não são apresentados como sensoriamento remoto puro.

| Linha de evidência | Resultado-chave | Interpretação |
|---|---:|---|
| Biomassa por textura, 22/23 | R² = 0,69; n = 192 | Replica o sinal em uma safra independente. |
| Biomassa por textura, 23/24 | R² = 0,80; n = 120 | Resultado sólido dentro da safra. |
| Biomassa pooled 22/23 + 23/24 | R² = 0,57; n = 312 | Modelo único mais conservador e generalizável. |
| Produtividade pooled | R² = 0,44 | Estimativa estável recomendada; resultados isolados são sensíveis. |
| GAN por fold, produtividade 23/24 | vegetativo R² OOF = 0,791; fusão GAN = 0,724 | A imagem sintética preserva sinal, mas não supera o vegetativo real. |
| GAN V10/V13 → R1, 25/26 | L1 mínimo = 0,1200 | Aproximação de imagem; clorofila R1 não teve predição útil. |
| GAN por fold, busca de condicionamento 23/24 | biomassa R² OOF = 0,716 (2 canais); produtividade R² OOF = 0,735 (5 canais) | O melhor conjunto de canais condicionais depende do alvo; as GANs superam o baseline "real" na fusão. |

As métricas, fontes e ressalvas completas estão em [Status honesto dos resultados](docs/results/RESULTS_STATUS.md).

## Pergunta e delineamento

O experimento responde a quatro perguntas:

1. Atributos de ortomosaicos explicam biomassa e produtividade sem reutilizar uma parcela entre treino e avaliação?
2. Esse sinal se mantém entre as safras 2022/23 e 2023/24?
3. Uma GAN consegue projetar um estágio reprodutivo a partir de estágio anterior de modo útil para uma tarefa agronômica?
4. A safra 2025/26, com novos voos, cubos hiperespectrais e alvos de biomassa/clorofila, amplia a evidência disponível?

| Safra | Dados principais | Alvos disponíveis | Papel |
|---|---|---|---|
| 2022/23 | RGB e RRENIR em V8, V11, V18, R2 e R5 | biomassa e produtividade | baseline, textura e origem da transferência R2→R5 |
| 2023/24 | multiespectral em V6, V8, V13, R2 e R5 | biomassa e produtividade | replicação, GAN fenológica e alvo de transferência |
| 2025/26 | ortomosaicos V10, V13 e R1; cubos Pika L | biomassa V10 e clorofila V10/R1 | extensão fenológica e hiperespectral |

Há 38 medidas de biomassa V10 em 2025/26; as parcelas 29 e 36 não têm pesagem. Altura e produtividade de colheita ainda não estão disponíveis nessa safra. Os cubos 12/12 `5S` e `20M` não têm header ENVI; os de 14/01 estão completos.

## Pipeline

```text
dados brutos + geometria de parcelas
        │
        ├── recortes/máscaras por estágio → índices + textura GLCM → PLSR/árvores → biomassa/produtividade
        │
        └── pares fenológicos (V→R ou R2→R5) → modelos de imagem/GAN → atributos sintéticos
                                                                  │
                                                                  └── comparação com baseline real
```

- **Preparação espacial:** recortes por parcela e estágio, normalização e registro de dados de campo.
- **Atributos:** bandas, índices de vegetação, estatísticas e descritores Haralick/GLCM; há também tensores espaciais RGB + índices + textura.
- **Modelos agronômicos:** PLSR, Extra Trees e Elastic Net, com agrupamento por parcela e seleção restrita aos folds de treino quando aplicável.
- **Modelos fenológicos:** Pix2Pix vegetativo→reprodutivo e variantes R2→R5; imagens geradas são avaliadas por L1, índices, textura e tarefa agronômica a jusante.

Consulte a [arquitetura](docs/ARCHITECTURE.md), o [protocolo de validação](docs/VALIDATION_PROTOCOL.md) e o [guia da pipeline](docs/pipeline/README.md) para interfaces e comandos de cada etapa.

## Experimentos concluídos e em andamento

| Experimento | Evidência e conclusão | Estado |
|---|---|---|
| `multisafra` | Textura prediz biomassa nas duas safras; pooled sustenta biomassa R²=0,57 e produtividade R²=0,44. Transferência direta mantém R² negativo, mesmo com adaptação de domínio. | referência oficial |
| `gan_2324_cv` | GAN V8/V13→R2/R5 reproduz imagem/índices, mas cenários agronômicos históricos tiveram R² absoluto negativo. | proof-of-mechanism |
| `gan_fusion_2324` | Retreino da GAN por fold eliminou vazamento externo; a fusão sintética não superou vegetativo+dose. | diagnóstico robusto |
| `gan_2526` | V10/V13→R1 selecionou checkpoint 150 por L1; R1 real e sintético não predizem clorofila no holdout de 10 parcelas. | proof-of-mechanism |
| `rfinal_2324` | `stats_aug`, DANN e CycleGAN falharam ao transferir R2→R5: R² sintético entre −3,42 e −5,22. | resultado negativo |
| `residual_gan_2324` | GAN residual R2→R5 obteve R²=0,624, abaixo do baseline R2 (0,724; ΔR²=−0,099). | resultado negativo; reavaliação OOF pendente |
| `gan_architecture_search` | Pares registrados, pesos e protocolo de seleção preparados; arquiteturas ainda não foram treinadas. | preparado |
| `rgb_texture_tensors` | Geração reprodutível de tensores espaciais RGB, índices e GLCM em mosaicos elegíveis. | reprodutível |
| `gan_attr_search` | GAN retreinada por fold para cada conjunto de canais condicionais: biomassa ótima com 2 canais (R²=0,716), produtividade com 5 (R²=0,735). Atributos agora configuráveis e calculados por módulo canônico. | concluído (n=24) |
| `2324_temporal_gan_value` | CV aninhada para V6+V8→V13 e V6+V8+V13+R2→R5, com busca progressiva de índices/GLCM e contrastes real, sintético e híbrido. | implementado; treino confirmatório pendente |
| `2324_temporal_gan_loss_ablation` | Ablação aninhada de losses para V6+V8+V13+R2→R5, sem canais auxiliares e com seleção agronômica interna dos pesos. | implementado; treino pendente |

Na trilha R2→R5, a calibração local do `stats_aug` alcançou R² OOF=0,644, mas utiliza biomassa local em 18 parcelas para calibrar cada fold. É calibração local de R5, não transferência remota sem rótulos. O relatório agregado da GAN residual antecede os quatro retreinos por fold mais recentes; a avaliação OOF deve ser regenerada antes de atualizar essa conclusão.

O catálogo com fontes de verdade e estados está em [docs/EXPERIMENT_CATALOG.md](docs/EXPERIMENT_CATALOG.md).

## Limites que orientam a interpretação

- Transferência inter-safra é o principal limite: sensor, GSD, iluminação e escala não foram resolvidos apenas por modelagem.
- GAN e cenários têm poucas parcelas independentes (6 em 2023/24 e 10 em 2025/26); métricas por par não aumentam o tamanho amostral independente.
- Clorofila, N e outras co-medidas de campo são tetos híbridos de referência, não preditores de sensoriamento remoto puro.
- O histórico contém resultados antigos mais otimistas. Os números a serem citados são os de `RESULTS_STATUS.md`, não métricas legadas isoladas.

## Reprodução rápida

Requer Python 3 e os extras de análise:

```bash
python3 -m pip install -e ".[analysis]"
make test
make dry-run
```

Os caminhos podem ser externos à cópia de trabalho:

```bash
export ABC_DATA_DIR=/caminho/para/dados
export ABC_ARTIFACTS_DIR=/caminho/para/artefatos
export ABC_ARCHIVE_DIR=/caminho/para/arquivo-historico
```

Cada execução reprodutível é descrita por uma receita TOML e disparada pelo runner:

```bash
python3 scripts/abc_run.py --config configs/stages/07_modeling/2324_multisafra.toml --check-inputs
python3 scripts/abc_run.py --config configs/stages/07_modeling/2324_multisafra.toml
```

O runner registra configuração, hashes de insumos e revisão Git. Resultados novos ficam em `artifacts/runs/<run-id>/`; resultados históricos ficam em `artifacts/archive/legacy/`. O fluxo de migração está em [docs/MIGRATION.md](docs/MIGRATION.md).

## Organização e referências

- `src/milho_experiment/` — caminhos, validação, manifestos e utilitários compartilhados.
- `configs/` — receitas TOML para execuções reproduzíveis.
- `scripts/abc_run.py` — ponto de entrada do fluxo novo.
- `src/milho_experiment/pipeline/` — implementações organizadas pelas 11 etapas.
- `artifacts/` — saídas recentes e índice do arquivo histórico; binários grandes permanecem externos.
- `dataset_registry/` — inventário, checksums e quarentena de ativos externos.
- `legacy/preprocessing/` — pré-processamento preservado para referência, não parte do fluxo oficial.

- [Status honesto dos resultados](docs/results/RESULTS_STATUS.md)
- [Catálogo de experimentos](docs/EXPERIMENT_CATALOG.md)
- [Protocolo de validação](docs/VALIDATION_PROTOCOL.md)
- [Arquitetura](docs/ARCHITECTURE.md)
- [Índice de resultados históricos](docs/ARCHIVE_INDEX.md)
- [Artefatos de execução](artifacts/README.md)
