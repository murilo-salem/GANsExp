# GAN temporal — protocolo de valor preditivo

O experimento `2324_temporal_gan_value` testa dois horizontes da safra 2023/24:

- `V6 + V8 → V13`;
- `V6 + V8 + V13 + R2 → R5`.

Cada entrada concatena, em ordem cronológica, `Red + RedEdge + NIR` e os canais auxiliares
selecionados. A busca completa (`--candidate-set all`, padrão do CLI) considera **38 candidatos**:

- Cinco índices: NDVI, NDRE, CIrededge (clorofila), SAVI e EVI2.
- Dezoito descritores globais do NIR, com prefixo `glcm_global_`: ASM, contraste,
  correlação, variância, IDM, sum_average, sum_variance, sum_entropy, entropia,
  diff_entropy, IMC1, IMC2, MCC, dissimilaridade, inércia, cluster_shade,
  cluster_prominence e homogeneidade.
- Quinze mapas locais do NIR: ASM, contraste, entropia, correlação e homogeneidade
  nas janelas 3, 5 e 7 (`glcmw3_*`, `glcmw5_*`, `glcmw7_*`).

Todos os extratores já existentes são reutilizados em `milho_experiment.gan_attributes`.
CIrededge não é duplicado como `chlorophyll`, mas o alias antigo continua aceito.
Inércia é numericamente igual a contraste; ambos são mantidos para contemplar o catálogo
completo. O limite de **quatro auxiliares selecionados** permanece; todos os 38 são
considerados pela seleção, que pode parar antes caso não haja melhora interna.

Descritores globais usam GLCM de 16 níveis e a máscara do extrator agronômico (NDVI > 0,3,
com fallback para a imagem inteira abaixo de 200 pixels). São escalares codificados como
planos constantes com `0.5 + atan(sign(x) * log1p(abs(x))) / pi`; aplicar normalização por
percentis em um plano constante apagaria seu valor. Essa transformação não ajusta parâmetros
usando parcelas de teste. Valores globais indefinidos recebem zero bruto (0,5 codificado).
Os mapas locais usam 8 níveis e normalização por imagem. Os aliases espaciais antigos
`glcm_*` mantêm sua semântica de janela 5, permitindo restaurar pesos anteriores.

GNDVI, EVI, VARI e TGI já existem no extrator de cinco bandas, mas exigem verde/azul,
ausentes no tensor RRENIR. Texturas de luminância RGB também exigem essas bandas.
Não se fabricam esses canais nesta pipeline. A inclusão deles exige uma versão dos
dados temporais com as bandas reais correspondentes.

### Referência anterior e busca ampliada

`--candidate-set legacy` preserva os nove candidatos originais (cinco índices e quatro
mapas GLCM de janela 5). O TOML original declara esse modo explicitamente para preservar
a reprodução da referência. O R² de produtividade **0,683219 → 0,723257** pertence a essa
busca anterior; **não é um resultado da busca ampliada**.

A configuração `2324_temporal_gan_value_all_features.toml` executa somente R5,
Produtividade e `rs_puro`, com todos os 38 candidatos e saída separada. Nenhum novo
resultado agronômico é atribuído a ela antes de seu treino e avaliação.

## Validação

A parcela é a unidade independente. Cada bloco de seis parcelas é um fold externo; os três
blocos restantes formam a CV interna. A triagem usa 50 épocas e semente 7. O subconjunto
selecionado é retreinado com 200 épocas e sementes 7, 11 e 23. Imagens sintéticas usadas no
treino downstream são produzidas por cross-fitting entre os blocos internos.

A seleção progressiva compara internamente o ΔR² de `combined` contra `forecast_history`.
O contraste de previsão antecipada escolhido é `forecast_fusion` contra `forecast_history`;
essa diferença entre o escore de triagem e o contraste final é mantida explicitamente.
O conjunto de atributos tabulares e o `SelectKBest` do regressor não foram alterados
pela ampliação dos candidatos da GAN.

São reportadas separadamente:

- composição do treino: real, sintético e real+sintético;
- previsão: histórico real, alvo sintético e histórico+alvo sintético;
- cenário combinado: exemplos `[histórico, alvo real]` e `[histórico, alvo sintético]` no treino,
  com somente `[histórico, alvo sintético]` no teste.

Biomassa é ligada explicitamente ao estágio-alvo; produtividade é a produtividade final. O
protocolo principal é `rs_puro`, acompanhado pela sensibilidade `rs_dose`.

## Critério

O resultado principal é o R² OOF da média das três sementes. Há evidência de melhoria somente
quando ΔR² é positivo e o limite inferior do IC95% pareado, estratificado por bloco, também é
positivo. Um ganho entre R² negativos é registrado sem ser chamado de predição útil.

## Execução

Validação rápida dos pares, sem treinar GANs:

```bash
python3 code/pipeline/stage12_temporal_gan_value.py prepare \
  --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
  --table data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx \
  --out artifacts/runs/2324_temporal_gan_value/prepare
```

Execução da busca ampliada, longa:

```bash
python3 scripts/abc_run.py --config configs/stages/08_scenarios/2324_temporal_gan_value_all_features.toml
```

Reprodução da seleção anterior, com nove candidatos:

```bash
python3 scripts/abc_run.py --config configs/stages/08_scenarios/2324_temporal_gan_value.toml
```

As saídas principais são `channel_selection.csv`, `oof_predictions.csv`,
`oof_predictions_ensemble.csv`, `metrics.csv`, `contrasts.csv` e `gan_image_metrics.csv`.
