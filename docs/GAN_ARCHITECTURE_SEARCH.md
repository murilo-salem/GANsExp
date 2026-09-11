# Busca de arquiteturas R2/V8+R2 → R5

O estágio 14 compara duas trilhas sem misturar seus rankings:

- **A:** R2 → R5: identidade, transformação afim, U-Net L1 e quatro GANs.
- **B:** V8+R2 → R5: ExtraTrees direto, ConvLSTM-lite e SimVP-lite.

A seleção é leave-one-block-out na safra 2022/23. Biomassa é o alvo
confirmatório e produtividade é secundário. Todas as representações são
avaliadas pelo mesmo ExtraTrees fixo. A imagem R5 real aparece somente como
oráculo diagnóstico e nunca pode vencer a seleção.

Na versão corrigida, toda saída recebe novamente a máscara espacial antes da
extração de atributos. Os dois recortes de uma mesma combinação bloco+dose
são promediados, formando 24 unidades de campo. O relatório separa `rs_only`,
`dose_only` e `rs_plus_dose`.

## Execução da seleção

```bash
python3 scripts/abc_run.py \
  --config configs/safras/2223_gan_architecture_search.toml \
  --check-inputs
```

Para um smoke test, chame o entrypoint diretamente com uma trilha, um modelo,
uma semente e `--epochs 1 --sample-images 0 --bootstrap 100`.

A rerun corrigida e reduzida usa:

```bash
python3 scripts/abc_run.py \
  --config configs/safras/2223_gan_architecture_search_masked.toml \
  --check-inputs
```

## Avaliação final bloqueada

Somente depois da criação de `winners.json`, execute:

```bash
python3 code/pipeline/stage14_architecture_search.py final \
  --dataset artifacts/runs/gan_architecture_search/dataset \
  --source-stage4 artifacts/runs/2223_stage4_residual/results \
  --source-table data/raw/safra_2022_2023/field/parametros_2223_normalizado.xlsx \
  --target-stage4 artifacts/runs/2324_stage4_residual/results \
  --target-table data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx \
  --winners artifacts/runs/2223_gan_architecture_search/results/winners.json \
  --out artifacts/runs/2324_gan_architecture_search_final/results
```

O comando `selection` não aceita caminhos da safra alvo. O comando `final`
exige vencedores explícitos. Nenhum dos dois grava `.pt` ou `.pth`; uma queda
durante o treino implica repetir apenas a combinação ainda não registrada.

`wgangp` não integra a matriz padrão. Ele pode ser solicitado explicitamente
com `--models wgangp` somente se perdas não finitas ou colapso ocorrerem em ao
menos duas sementes dos GANs convencionais.

## Saídas

- `oof_predictions.csv`: uma predição por parcela, modelo e semente.
- `image_metrics.csv`: MAE, PSNR, SSIM aproximado e ângulo espectral.
- `training_history.csv`: perdas escalares por época.
- `summary.csv`: ΔR², IC bootstrap, Holm, estabilidade e métricas visuais.
- `winners.json`: no máximo um vencedor confirmatório por trilha.
- `samples/`: pequenos painéis R2 | R5 sintético | R5 real em PNG.
