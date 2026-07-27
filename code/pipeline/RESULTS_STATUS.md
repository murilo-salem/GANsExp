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

## 4. O que continua bloqueado (precisa de dado novo, não de código)

- **Altura**: alvo nunca medido em nenhuma tabela.
- **Safra 2024/25**: não adquirida.
- **Trilha hiperespectral (300 bandas)**: cubos brutos ausentes de `data/`; `features_12dez.csv`
  **sem coluna de Y** → PLSR só rodou com alvo sintético.
- **GAN/etapa-8 publicáveis**: exigem mais parcelas (hoje 24–48).

## Como reproduzir
```bash
python3 code/pipeline/stage_multisafra.py \
  --feat2223 code/pipeline/out/band5_2223/features_5band_2223.csv \
  --table2223 "data/Safra2022a2023/tabela de dados/parametros_2223_normalizado.xlsx" \
  --feat2324 code/pipeline/out/plsr_2324/features_ortho_2324.csv \
  --tgt2324  code/pipeline/out/plsr_2324/targets_ortho_2324.csv \
  --target Produtividade --out code/pipeline/out/multisafra
```
Saída: `code/pipeline/out/multisafra/multisafra_<alvo>.txt` (within, cross baseline/+DA, pooled).
