# Status honesto dos resultados — projeto milho

Registro consolidado dos resultados **reais** (relidos dos `.txt` de saída), incluindo as
melhorias puramente de código feitas com os dados atuais e as correções de números que estavam
otimistas no `README.md`/`PIPELINE_FLOWCHART.md`. Sem coletar dado novo.

Validação padrão: **GroupKFold por parcela** (anti-vazamento) sobre o bloco de **textura** (GLCM
Haralick). `n` = amostras (parcela × estágio).

## 1. O que funciona (resultado sólido)

| Alvo | Cenário | R² | n | Fonte |
|------|---------|----|---|-------|
| **Biomassa** | within 23/24 (texture) | **0.80** | 120 | `multisafra_Biomassa.txt` |
| **Biomassa** | within 22/23 (texture) | **0.69** | 192 | idem — **replica em 2 safras** |
| **Biomassa** | **pooled 22/23+23/24** (texture+dummy safra) | **0.57** | 312 | modelo único, generalização honesta |
| Biomassa | texture+physical 23/24 | 0.95 | 72 | ceiling com clorofila/N de campo (não é RS puro) |

O vínculo **textura → biomassa** é o resultado central e defensável: se sustenta em duas safras
independentes e num modelo único pooled.

## 2. Melhorias feitas por código (dados atuais)

### 2.1 Modelo pooled multi-safra — `stage_multisafra.py` (`run_pooled`)
Junta as duas safras (colunas de textura comuns + **dummy de safra**) num só GroupKFold.
**Ganho real para Produtividade**, que era instável por safra isolada:

| Alvo | within 22/23 | within 23/24 | **pooled (n=312)** |
|------|--------------|--------------|--------------------|
| Biomassa | 0.69 | 0.80 | **0.57** (mais robusto/generalizável) |
| **Produtividade** | 0.44 | 0.30* | **0.44 estável** |

\* *Produtividade 23/24 é sensível ao nº de componentes: cai a **0.065** com nc alto
(`plsrG_Produtividade_texture.txt`, nc=10) e sobe a 0.30 com nc=3. O pooled a 0.44 é a
estimativa estável — recomendado como número oficial de Produtividade.*

### 2.2 Adaptação de domínio inter-safra — `stage_multisafra.py` (`cross_safra(domain_adapt=True)`)
Padronização z-score por-domínio (média/desvio de cada safra) antes de transferir — alinha os
momentos das features entre safras (CORAL-lite, não-supervisionado, sem rótulos da safra-alvo).

| Direção | Alvo | baseline R² | +DA R² | baseline corr | +DA corr |
|---------|------|-------------|--------|---------------|----------|
| 22/23→23/24 | Biomassa | −12.98 | **−2.85** | −0.48 | −0.55 |
| 22/23→23/24 | Produtividade | −11.30 | **−1.08** | 0.23 | **0.50** |
| 23/24→22/23 | Produtividade | 0.13 | −0.18 | 0.51 | 0.44 |

**Conclusão honesta:** a DA **resgata o caso catastrófico** (erro de escala explosivo cai ~4–10×)
e melhora o ranking (corr) na direção 22/23→23/24. Mas o **R² absoluto cross-safra continua
negativo** — a calibração entre safras **não transfere só com código**. É limite de
domínio/dado (GSD/iluminação/sensor + poucas parcelas), não de modelagem.

## 3. Correções aos documentos anteriores (eram otimistas)

- **Produtividade**: `README`/`FLOWCHART` reportavam R²=0.30 como se fosse estável. O honesto é
  **0.065–0.44 dependendo de nc/agrupamento**; o número robusto é o **pooled 0.44**.
- **Etapa 8 (cenários)** — `stage8_2324/report_Biomassa.txt`: R² é **negativo tanto no real
  (−1.14) quanto no sintético (−0.49)** nas 24 parcelas de validação. A frase "corr 0.48=0.48"
  descreve só o ranking; **em R² absoluto não prediz**. É **proof-of-mechanism**, não resultado
  preditivo. (O sintético ficou até menos ruim que o real — coerente com a tese, mas com n minúsculo.)
- **GAN — épocas**: teto já atingido. Doce spot **~2000 épocas** (L1 0.103, ΔNDVI 0.046);
  2000→4000 **não melhora** e o discriminador colapsa. Mais épocas/gerações **não ajudam**.

## 4. Situação dos bloqueios após a safra 2025/26

Resolvidos por dados novos:

- **Safra adicional:** 2025/26 tem 40 parcelas e três ortomosaicos fenológicos (V10, V13 e R1).
- **Trilha hiperespectral:** há cubos Pika L em 12/12 (76 cubos BIL+HDR válidos) e 14/01
  (79 válidos), além de clorofila laboratorial nas 40 parcelas nas duas datas.
- **Y real para V10:** biomassa está preenchida para 38 das 40 parcelas em 12/12
  (as parcelas 29 e 36 estão sem pesagem).

Continuam bloqueados:

- **Altura**: alvo nunca medido em nenhuma tabela.
- **Produtividade de colheita:** não consta na planilha 2025/26; biomassa só está disponível em V10,
  não em R1.
- **GAN/etapa-8 publicáveis**: a nova validação ainda tem apenas 10 parcelas independentes.

Os cubos inválidos de 12/12 (`5S` e `20M`) não têm header ENVI; 14/01 está completo.

## 5. Produtividade com GAN re-treinada por fold (23/24)

Foi executada a ressalva metodológica: para cada bloco externo (quatro blocos de seis parcelas),
a GAN foi treinada sem aquele bloco, gerou R2/R5 sintético a partir de V8/V13 e somente então o
modelo de produtividade foi ajustado nos outros três blocos. A seleção de atributos e
hiperparâmetros é interna aos três blocos de treino. Logo, as 24 predições são OOF por parcela,
sem reutilizar o bloco de teste.

| Entrada + dose N | Melhor modelo | R² OOF | RMSE |
|---|---|---:|---:|
| Vegetativo real | Extra Trees | **0.791** | 914.9 |
| Reprodutivo sintético GAN | Elastic Net | 0.362 | 1597.1 |
| Vegetativo + sintético GAN | Extra Trees | 0.724 | 1051.4 |

Conclusão: a GAN preserva algum sinal de produtividade, mas a fusão perde 0.067 R² frente ao
vegetativo real. Portanto ela **não** é evidência de melhoria preditiva ainda. A saída auditável
está em `out/stage8_2324_ganfusion_cv/` (`summary.csv` e `oof_predictions.csv`).

## 6. Primeiro treino 2025/26 — V10/V13 → R1

Execução inicial concluída em 200 épocas (150 com taxa fixa + 50 de decaimento), usando
30 parcelas no treino e o bloco 4 inteiro como validação (10 parcelas; 20 pares, que foram
agregados por parcela). Checkpoints 50/100/150/200 foram comparados no mesmo holdout:

| Checkpoint | L1 médio por parcela | \|ΔNDVI\| médio |
|---|---:|---:|
| 50 | 0.1215 | 0.0122 |
| 100 | 0.1223 | **0.0117** |
| **150** | **0.1200** | 0.0154 |
| 200 | 0.1213 | 0.0134 |

O checkpoint 150 é o selecionado por L1. Por transição, V10→R1 teve L1=0.1214 e
V13→R1 L1=0.1185. É uma boa aproximação de imagem/índice, mas **não demonstra predição
agronômica**: no cenário de clorofila R1, o PLSR com R1 real teve R²=−0.212 e com R1
sintético R²=−0.382 (10 parcelas; IC bootstrap amplo). O resultado correto é
proof-of-mechanism da geração, não estimativa de clorofila publicável.

## 7. Projeção antecipada R2 → R5 entre safras (22/23 → 23/24)

O experimento `stage9_rfinal_forecast.py` treinou a projeção R2→R5 somente com 22/23 e manteve R5 de 23/24 bloqueado até a avaliação. Nenhuma das três adaptações transferiu uma estimativa de biomassa útil para 23/24:

| Método | R² biomassa sintética | RMSE | L1 da imagem |
|---|---:|---:|---:|
| `stats_aug` | −3.424 | 10482.4 | **0.1062** |
| DANN | −5.225 | 12434.2 | 0.1062 |
| CycleGAN | −4.530 | 11719.4 | 0.1123 |

Mesmo o R5 real passado pelo PLSR treinado em 22/23 teve R²=−0.014 em 23/24; portanto o problema dominante é a transferência de domínio, não apenas a geração de imagem. A calibração local OOF do `stats_aug` elevou o R² de −3.424 para **0.644** (RMSE=2973.6), mas usa biomassa de 18 parcelas para calibrar e prediz as 6 parcelas restantes por fold. É resultado de **calibração local R5**, não previsão remota em safra sem rótulos.

## 8. GAN residual R2 → R5 orientada à biomassa

Uma variante residual, treinada em 22/23 para prever a alteração R2→R5 com perda auxiliar de Δbiomassa, foi comparada em 23/24 ao baseline que usa R2 diretamente:

| Cenário | R² | r | RMSE |
|---|---:|---:|---:|
| Baseline R2 | **0.724** | 0.851 | 2618.8 |
| GAN residual | 0.624 | 0.793 | 3054.1 |

O ΔR² é **−0.099**: a variante não atingiu o critério pré-definido de ganho (+0.05) e não deve substituir o baseline. Os quatro retreinos por fold foram concluídos depois deste relatório agregado; a avaliação OOF deve ser regenerada antes de qualquer atualização desta conclusão.

## 9. Busca de arquiteturas GAN R2 → R5

O estágio `stage11_gan_search.py` preparou pares R2→R5 registrados, pesos e as arquiteturas `identity`, `unet_light`, `resnet9`, `attention_unet` e `multiscale`. A seleção será por ganho de biomassa em GroupKFold de 22/23 (sucesso: ΔR² ≥ 0.05), com 23/24 bloqueada até a avaliação final. **Ainda não há treinos ou resultados comparativos** nessa busca.

## 10. Busca do número/composição de canais condicionais da GAN (23/24)

Para responder qual o melhor conjunto de atributos que entram na GAN, cada configuração
foi avaliada com a GAN retreinada por bloco externo (4 folds de 6 parcelas) e R² OOF por
parcela na tarefa agronômica downstream. Os atributos condicionais agora são configuráveis
(`--gan_attr_names`) e calculados pelo módulo canônico `milho_experiment.indices`,
consistente com a pipeline agronômica. Resultados em `artifacts/runs/2324_attr_search_*/`.

| Alvo | Melhor nº de canais | Composição | R² OOF | Baseline "real" |
|---|---:|---|---:|---:|
| **Biomassa** | **2** | chlorophyll + NDVI | **0.716** | 0.658 (extratrees) |
| **Produtividade** | **5** | chl+NDVI+NDRE+SAVI+EVI2 | **0.735** | 0.590 (extratrees) |

**Leitura:** o "número de objetos" ótimo depende do alvo (2 para biomassa, 5 para
produtividade). As GANs com condicionamento adequado preservam ou superam o sinal do
reprodutivo real na fusão, mas com n pequeno (24 parcelas) as diferenças são indicativas.
Métricas de imagem (L1/ΔNDVI) variam pouco entre as configurações (~0,11 / ~0,05).

## Como reproduzir
```bash
python3 code/pipeline/stage_multisafra.py \
  --feat2223 artifacts/archive/legacy/pipeline/band5_2223/features_5band_2223.csv \
  --table2223 "data/raw/safra_2022_2023/field/parametros_2223_normalizado.xlsx" \
  --feat2324 artifacts/archive/legacy/pipeline/plsr_2324/features_ortho_2324.csv \
  --tgt2324  artifacts/archive/legacy/pipeline/plsr_2324/targets_ortho_2324.csv \
  --target Produtividade --out artifacts/runs/2324_multisafra_produtividade/results
```
Saída: `artifacts/runs/<run-id>/results/multisafra_<alvo>.txt` (within, cross baseline/+DA, pooled).
