# Ablação de losses da GAN temporal — R5

O experimento `2324_temporal_gan_loss_ablation` mantém a arquitetura e o
protocolo de validação do experimento temporal anterior, mas usa somente o
histórico RRENIR `V6 + V8 + V13 + R2` para gerar `R5`. Índices e texturas são
usados como objetivos diferenciáveis, nunca como canais de entrada.

## Braços

- `l1`: L1 mascarado;
- `l1_gan`: L1 mascarado e adversarial vanilla;
- `l1_indices_texture`: L1, NDVI/NDRE e textura local multiescala;
- `l1_gan_feature_matching`: L1, adversarial e feature matching do PatchGAN;
- `full`: todas as perdas anteriores e gradientes Sobel.

A máscara exclui somente o fundo zerado fora da parcela. O PatchGAN permanece
em uma escala para que a arquitetura não seja um fator da ablação. GLCM
discreto é usado somente como métrica externa, pois sua quantização não é
diferenciável.

## Validação

Os quatro blocos são os folds externos. A busca de pesos ocorre somente nos
três blocos de treino, com 50 épocas e seed 7. `alpha` é escolhido em
`{0.1, 0.3, 1.0}` e o peso de feature matching em `{1, 5, 10}`. Os modelos
finais usam 200 épocas e seeds 7, 11 e 23.

Para biomassa, a seleção interna maximiza `augment_hybrid - augment_real`.
Para produtividade, maximiza `forecast_fusion - forecast_history`. A
configuração completa é confirmatória; seus dois testes recebem correção de
Holm.

A execução oficial usa retenção `rotating`: durante a unidade ativa há no
máximo um `.resume.pth`, sobrescrito a cada 10 épocas e removido assim que a
unidade termina. Ele guarda G, D, otimizadores, época e estados aleatórios.
Não são preservados checkpoints finais por fold; os resultados científicos
persistentes são as tabelas e métricas produzidas após a avaliação em memória.

## Execução

```bash
python3 scripts/abc_run.py \
  --config configs/safras/2324_temporal_gan_loss_ablation.toml
```

Para conferir os dados sem treinar:

```bash
python3 code/pipeline/stage13_temporal_gan_loss_ablation.py prepare \
  --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
  --table data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx
```

Os resultados incluem seleção de pesos, histórico de treino, métricas de
imagem, predições OOF, métricas downstream e contrastes confirmatórios.
