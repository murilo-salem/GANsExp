# GAN temporal — protocolo de valor preditivo

O experimento `2324_temporal_gan_value` testa dois horizontes da safra 2023/24:

- `V6 + V8 → V13`;
- `V6 + V8 + V13 + R2 → R5`.

Cada entrada concatena, em ordem cronológica, `Red + RedEdge + NIR` e os mapas auxiliares
selecionados. A seleção progressiva considera CIrededge/clorofila, NDVI, NDRE, SAVI, EVI2 e
quatro mapas GLCM, com no máximo quatro auxiliares. GNDVI não participa porque exige a banda
verde, ausente no tensor RRENIR.

## Validação

A parcela é a unidade independente. Cada bloco de seis parcelas é um fold externo; os três
blocos restantes formam a CV interna. A triagem usa 50 épocas e semente 7. O subconjunto
selecionado é retreinado com 200 épocas e sementes 7, 11 e 23. Imagens sintéticas usadas no
treino downstream são produzidas por cross-fitting entre os blocos internos.

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

Execução oficial, longa:

```bash
python3 scripts/abc_run.py --config configs/safras/2324_temporal_gan_value.toml
```

As saídas principais são `channel_selection.csv`, `oof_predictions.csv`,
`oof_predictions_ensemble.csv`, `metrics.csv`, `contrasts.csv` e `gan_image_metrics.csv`.
