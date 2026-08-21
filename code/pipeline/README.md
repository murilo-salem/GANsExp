# Pipeline milho — etapas 4–8 do diagrama (safras 2023/24 e 2022/23)

> Migração em curso: novos experimentos devem usar `configs/` + `scripts/abc_run.py` na raiz.
> Este diretório mantém os CLIs compatíveis e os resultados históricos; veja `docs/MIGRATION.md`.

> ⚠️ **Números oficiais/honestos em [`RESULTS_STATUS.md`](RESULTS_STATUS.md).** Alguns resultados
> abaixo estão otimistas: Produtividade estável é ~**0.44** (pooled), não 0.30; a etapa 8 tem
> **R² negativo** (proof-of-mechanism); mais épocas de GAN não melhoram (teto em ~2000).
> Melhorias novas por código: modelo **pooled** multi-safra e **adaptação de domínio** inter-safra.

Implementa as etapas do experimento do diagrama, de forma auto-consistente sobre a **safra 23/24**
(orto RRENIR + tabela de campo + shapefile, SIRGAS 2000 / UTM 22S) e generalizada à **22/23**.

> **Fechamento do roteiro (passos 1–4)** — ver seção "Roadmap concluído" ao final:
> cubo 5-bandas (22/23), GAN completa condicionada por atributos (7 canais), etapa 8 (cenários) e
> multi-safra. Bloqueios de dados: **altura** (não medida) e **safra 24/25** (ausente) ficam fora.

> Dependências instaladas no user-site: `tifffile`, `pyshp`, `openpyxl` (leitura de GeoTIFF/
> shapefile/xlsx sem GDAL) e, para a GAN, `dominate`/`wandb` (visdom não é necessário).

## Base — `geo.py`
Lê o ortomosaico GeoTIFF via `tifffile` + geotransform (ModelPixelScale/Tiepoint), converte
UTM→pixel e recorta parcelas do shapefile. Índices RRENIR (NDVI/NDRE/CIrededge/SAVI) e mapa de
clorofila (proxy CIrededge). CRS: EPSG:31982.

## Tensores RGB + textura por mosaico — `build_rgb_texture_tensors.py`
Gera um Zarr espacial por mosaico elegível, com 22 canais: `R,G,B`, `VARI,ExG,GLI,TGI` e os
mapas GLCM de luminância (`asm`, `contrast`, `entropy`, `correlation`, `homogeneity`) nas janelas
3/5/7. A execução é em blocos, preserva a geometria no manifesto e não carrega o mosaico inteiro.

```bash
python3 scripts/abc_run.py --config configs/analysis/rgb_texture_tensors.toml --check-inputs
```

São incluídos V8/V11/V18/R2/R5 de 22/23 e V13/R1 de 25/26. A safra 23/24 e V10 de 25/26 são
registrados como excluídos porque não têm RGB físico completo.

## Etapa 4 — Divisão por estágio fenológico — `stage4_phenology.py`
Recorta cada parcela de cada orto e classifica: `V6/V8/V13`→**vegetativo**, `R2/R5`→**reprodutivo**.
```bash
python3 code/pipeline/stage4_phenology.py \
  --ortho-dir "data/raw/safra_2023_2024/orthomosaics" \
  --shapefile "data/raw/safra_2023_2024/geometry/Shape_parcelas23_24.shp" \
  --out artifacts/runs/2324_stage4_phenology/results --size 256
```
Saída: `manifest.csv` (120 recortes = 24 parcelas × 5 estágios), `npy/`, `vegetativo/`, `reprodutivo/`.
Sinal fenológico validado (NDVI médio): V6 0.37 → V8 0.62 → V13 0.69 → R2 0.71 → R5 0.61.

## Etapa 5 — Pix2Pix condicional vegetativo→reprodutivo
Realiza a GAN do diagrama: **entrada = estágio vegetativo + mapa de clorofila (4 canais)**,
**saída = estágio reprodutivo sintético (3 canais)**, perda `L_GAN + λ·L1` (padrão pix2pix).

1. Montar pares alinhados (split por parcela, sem vazamento):
```bash
python3 code/pipeline/stage5_make_pairs.py \
  --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
  --out artifacts/archive/legacy/gan/datasets/pheno_2324 \
  --veg V8 V13 --rep R2 R5 --val-blocos 4      # 96 pares: 72 treino / 24 val
```
2. Treinar (dataset `phenology`, `input_nc=7/output_nc=3` definidos pelo próprio dataset):
```bash
cd pytorch-CycleGAN-and-pix2pix
WANDB_MODE=disabled python3 train.py --dataroot ../artifacts/archive/legacy/gan/datasets/pheno_2324 --name pheno_2324 \
  --model pix2pix --dataset_mode phenology --direction AtoB \
  --load_size 286 --crop_size 256 --batch_size 4 \
  --n_epochs 150 --n_epochs_decay 50 --no_html \
  --checkpoints_dir ../artifacts/runs/2324_pheno_gan/checkpoints
```
3. Inferir + validar quantitativamente (painel `veg | sintético | real`, L1 e ΔNDVI):
```bash
python3 code/pipeline/stage5_infer.py \
  --dataroot artifacts/archive/legacy/gan/datasets/pheno_2324 --phase val \
  --ckpt artifacts/archive/legacy/gan/checkpoints/pheno_2324/latest_net_G.pth \
  --out artifacts/runs/2324_stage5_infer/results
```
Arquivos: `data/phenology_dataset.py` (dataset condicional). Smoke-test de 2 épocas já validou a
fiação (G_L1 41.5→19.8). **Nota:** com só 24 parcelas o volume é pequeno — os pares V×R e o
tiling/augmentation ajudam, mas é um proof-of-mechanism; para resultado publicável, mais dados.

### Run longo 4000 épocas — `pheno_2324_4k` (experimento de convergência)
Para testar se mais épocas melhoram a geração, treinos 2000+2000 (lr cheio 1–2000, decaimento
linear 2001–4000), `save_epoch_freq=500`, ~0.28 s/época em RTX 5090 (~20 min total):
```bash
WANDB_MODE=disabled python3 train.py --dataroot ../artifacts/archive/legacy/gan/datasets/pheno_2324 --name pheno_2324_4k \
  --model pix2pix --dataset_mode phenology --direction AtoB \
  --load_size 286 --crop_size 256 --batch_size 4 \
  --n_epochs 2000 --n_epochs_decay 2000 --no_html --save_epoch_freq 500 \
  --checkpoints_dir ../artifacts/runs/2324_pheno_gan_4k/checkpoints
```
Inferência por checkpoint: `--ckpt artifacts/runs/2324_pheno_gan_4k/checkpoints/pheno_2324_4k/{2000,4000}_net_G.pth`.

**Resultado — platô em ~2000 épocas (validação, 24 pares):**
| Checkpoint | L1 | \|ΔNDVI\| | vs baseline 200-ép |
|---|---|---|---|
| 200 (baseline `pheno_2324`) | 0.1115 | 0.0546 | — |
| **2000** (`pheno_2324_4k`) | **0.1026** | **0.0461** | **-7.9% L1 / -15.6% ΔNDVI** |
| 4000 (`pheno_2324_4k`) | 0.1029 | 0.0470 | -7.6% L1 / -14.1% ΔNDVI (≈ igual a 2000) |

**Conclusão:** 200→2000 traz melhora real; 2000→4000 **não melhora** (leve regressão). A perda de
treino `G_L1` platô em ~15.5 desde a época ~1500. O discriminador colapsa progressivamente
(`D_real`/`D_fake` 0.49→0.19, `G_GAN` 1.4→2.6) após época 2000 — instabilidade adversarial tardia
sem benefício de qualidade (a geração é carregada pelo termo L1, `λ=100`). **Doce spot ≈ 2000
épocas.** Checkpoints `2000`/`4000` mantidos em `checkpoints/pheno_2324_4k/`.

## Plug do Y (etapas 6-7) — `stage6_features_ortho.py` + `plsr_kfold.py`
Extrai por parcela/estágio o mesmo schema de atributos (índices, estatísticas, GLCM global+3/5/7)
do orto RRENIR, junta o Y real (`Biomassa/Produtividade/CHL_total`) da tabela normalizada **e
também os atributos físicos de campo como X** (`phys_CHL_total`, `phys_N_percent`,
`phys_N_acumulado`) — realiza o "clorofila (físico)" do diagrama etapa 6.
```bash
python3 code/pipeline/stage6_features_ortho.py \
  --ortho-dir "data/raw/safra_2023_2024/orthomosaics" \
  --shapefile "data/raw/safra_2023_2024/geometry/Shape_parcelas23_24.shp" \
  --table "data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx" \
  --out artifacts/runs/2324_features_ortho/results
cd code/analysis && python3 plsr_kfold.py \
  --features ../../artifacts/runs/2324_features_ortho/results/features_ortho_2324.csv \
  --target-csv ../../artifacts/runs/2324_features_ortho/results/targets_ortho_2324.csv \
  --join sample --target-col Biomassa --block texture+physical --group parcela
```
`--group parcela` usa **GroupKFold** (novo): sem ele, o alvo repetido entre estágios da mesma
parcela vaza entre treino/teste e infla o R². `--block` aceita combinações com `+`
(`texture+physical`, `indices+texture+physical`, ...); bloco `physical` = cols `phys_*`.

### Anti-vazamento (fix)
O merge do alvo adiciona a coluna `Biomassa` ao df; o bloco `all` antigo a incluía em X →
R²≈1.0 espúrio. Agora `plsr_kfold.py` remove de X o alvo e sua contraparte `phys_<alvo>`
(reportado no relatório como "Anti-vazamento: removido de X -> [...]").

### Descritores de textura — Haralick completo (`glcm_features` em `code/analysis/hyperspectral_features.py`)
GLCM simétrica (offsets (0,1)+(1,0), 16 níveis). 17 descritores globais + `homogeneity` (alias de
IDM): Angular Second Moment, Contrast, Correlation, Variance, Inverse Difference Moment, Sum Average,
Sum Variance, Sum Entropy, Entropy, Difference Entropy, Information Measure of Correlation 1 e 2,
Maximal Correlation Coefficient, Dissimilarity, Inertia, Cluster Shade, Cluster Prominence
(prefixo `glcm_*`). Mantém-se a versão em janela 3/5/7 (`glcmwN_*`, 5 descritores). Total texture: 33.

### Resultados honestos (GroupKFold por parcela, 24 parcelas × 5 estágios = 120, bloco texture)
| Alvo | R² | RPD | Comentário |
|------|----|-----|-----------|
| **Biomassa** | **0.80** (0.84 c/ VIP) | 2.3–2.5 | sinal real; VIP destaca variance, cluster_prominence/shade, sum_average, correlation |
| Biomassa | **0.95** (texture+physical, n=72) | 4.6 | + clorofila/N de campo (diagrama etapa 6); phys são preditores co-medidos, exige dado de campo |
| Produtividade | **0.30** | 1.19 | com Haralick completo (era ~0.06 só c/ 5 descritores básicos) |
| CHL_total | 0.31 | 1.20 | |

⚠️ **Não confiar** no bloco `all` com muitos componentes: o anti-vazamento remove o alvo de X, mas
`all` soma colinearidade + preditores físicos co-medidos → R² inflado. Use bloco físico `texture`
e cap de componentes (`--max-comp 8`), ou seleção VIP (`band_selection.py`). Científico: `physical`
não é sensoriamento remoto puro (CHL/N vêm de campo) — útil como ceiling de referência.

### Produtividade com GAN por fold — `stage8_gan_fusion_cv.py`
O experimento pré-colheita robusto usa uma GAN treinada novamente para cada bloco externo e compara
vegetativo real, reprodutivo sintético e a fusão dos dois, sempre com a dose de N. O modelo de
produtividade e a seleção de atributos ficam restritos aos outros três blocos.
```bash
python3 code/pipeline/stage8_gan_fusion_cv.py \
  --stage4 artifacts/archive/legacy/pipeline/stage4_2324 \
  --table "data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx" \
  --ckpt-dir artifacts/archive/legacy/gan/checkpoints \
  --out artifacts/runs/2324_ganfusion_cv/results
```
Na execução atual (24 parcelas; 4 folds), Extra Trees com **vegetativo+dose** obteve R² OOF 0.791;
a fusão GAN obteve 0.724 e GAN+dose isolada 0.362. Assim, a GAN foi usada e validada, mas ainda não
agrega valor preditivo à produtividade nesta safra. Consulte o `REPORT.md` do diretório de saída.

---

## Roadmap concluído (passos 1–4 — fechamento até a etapa 8)

### Passo 1 — Cubo 5-bandas por pixel — `build_5band.py` (só 22/23)
Funde RGB (Azul/Verde) + RRENIR (Vermelho/RedEdge/NIR) em B1–B5 por parcela/estágio; habilita
GNDVI/EVI/VARI/TGI (etapa 3 de fato 5-bandas). Viável **só na 22/23** (no 23/24 o único RGB é
`zero` e está corrompido). ⚠️ Azul/Verde vêm de câmera RGB (DN, não calibrada) → VARI/EVI ficam
instáveis; NDVI/NDRE/CIrededge (do RRENIR calibrado) são sólidos. `geo.py` ganhou `bands5_indices`.
Saída: `out/band5_2223/features_5band_2223.csv` (240 amostras).

### Passo 2 — GAN completa + condicionamento por atributos — `data/phenology_dataset.py`
Entrada da GAN passou de 4→**7 canais**: 3 bandas + **clorofila + NDVI + NDRE + SAVI** como mapas
(`attribute_channels`, `N_COND=4`) — realiza os "Atributos usados na GAN" do diagrama. Treino
completo (200 épocas) → `checkpoints/pheno_2324`. Inferência final: **L1=0.111, ΔNDVI=0.055**
(vs smoke 0.123/0.131 — ΔNDVI ~2.4× melhor).

### Passo 3 — Etapa 8: cenários futuros — `stage8_scenarios.py`
Veg real → GAN → reprodutivo sintético → atributos → **PLSR treinado no reprodutivo real** →
prediz Y. Compara predição do sintético vs real vs Y medido. Resultado (Biomassa, val):
**corr(pred_synth, medido) = corr(pred_real, medido) = 0.48** — o dado sintético preserva a mesma
capacidade preditiva do real (ponto central do diagrama). R² absoluto negativo por dados escassos
(~6 parcelas de validação). Painéis `veg|sintético|real` em `out/stage8_2324/paineis/`.

### Passo 4 — Multi-safra + validação cruzada — `stage_multisafra.py`
| | Biomassa | Produtividade |
|--|--|--|
| within 22/23 (GroupKFold, texture) | R²=0.69 (n=192) | R²=0.44 |
| within 23/24 (GroupKFold, texture) | R²=0.80 (n=120) | R²=0.30 |
| cross 23/24→22/23 | R²<0 (corr −0.06) | R²=0.14 (corr 0.51) |
| cross 22/23→23/24 | R²<0 | R²<0 |

**Achado positivo:** o vínculo textura→biomassa **replica em duas safras independentes** (0.69 e
0.80). **Limitação:** a calibração absoluta **não transfere entre safras** (domain shift de
GSD/iluminação/sensor) — exigiria adaptação de domínio/normalização inter-safra.

## Estado final vs diagrama
Etapas 1–8 implementadas e rodando ponta-a-ponta em 22/23+23/24. A lacuna da **etapa 6**
("Entradas X: espectrais + textura + **clorofila (físico)**") foi fechada: `stage6` emite
`phys_CHL_total/phys_N_percent/phys_N_acumulado` e `plsr_kfold` aceita bloco `physical`
(combinável, ex.: `texture+physical`). Pendências externas (fora do código): **altura**
(coleta de campo) e **safra 24/25** (aquisição). Ressalva científica: dados pequenos
(24–48 parcelas) ⇒ GAN e etapa 8 são proof-of-mechanism; bloco `all`/muitos componentes no
PLSR inflacionam R² (use `texture`/VIP + `--group parcela`; o anti-vazamento já remove o alvo
de X). `physical` usa preditores de campo co-medidos → ceiling de referência, não RS puro.

## Projeção causal R2 → R5 em 23/24 — `stage9_rfinal_forecast.py`

### Variante residual orientada a biomassa — `stage10_residual_gan.py`

Para testar ganho incremental sobre R2, recrie os recortes com máscara da parcela e
escala única por voo; a GAN prevê o delta R2→R5 e recebe uma perda auxiliar de
Δbiomassa apenas em 22/23.

```bash
python3 code/pipeline/stage4_phenology.py --ortho-dir data/raw/safra_2022_2023/orthomosaics \
  --shapefile data/raw/safra_2022_2023/geometry/Shape_parcelas_2223.shp \
  --out artifacts/runs/2223_stage4_residual/results --reflectance-scale mosaic
python3 code/pipeline/stage10_residual_gan.py train \
  --source-stage4 artifacts/runs/2223_stage4_residual/results \
  --source-table data/raw/safra_2022_2023/field/parametros_2223_normalizado.xlsx \
  --out artifacts/runs/2324_residual_gan/checkpoint
```

O experimento de previsão antecipada usa pares R2→R5 e biomassa R5 apenas da safra
22/23. Em 23/24, somente R2 é disponibilizado às três adaptações (``stats_aug``,
``dann`` e ``cyclegan``); R5 real é lido exclusivamente no comando de avaliação.

```bash
python3 scripts/abc_run.py --config configs/safras/2223_stage4.toml
python3 scripts/abc_run.py --config configs/safras/2324_stage4.toml
python3 code/pipeline/stage9_rfinal_forecast.py prepare \
  --source-stage4 artifacts/runs/2223_stage4_phenology/results \
  --target-stage4 artifacts/runs/2324_stage4_phenology/results \
  --out artifacts/runs/2324_rfinal_forecast/dataset

for method in stats_aug dann cyclegan; do
  python3 code/pipeline/stage9_rfinal_forecast.py train --dataset artifacts/runs/2324_rfinal_forecast/dataset \
    --method "$method" --out "artifacts/runs/2324_rfinal_forecast/$method/checkpoint"
  python3 code/pipeline/stage9_rfinal_forecast.py predict --dataset artifacts/runs/2324_rfinal_forecast/dataset \
    --method "$method" --checkpoint "artifacts/runs/2324_rfinal_forecast/$method/checkpoint" \
    --out "artifacts/runs/2324_rfinal_forecast/$method/predictions"
  python3 code/pipeline/stage9_rfinal_forecast.py evaluate --dataset artifacts/runs/2324_rfinal_forecast/dataset \
    --predictions "artifacts/runs/2324_rfinal_forecast/$method/predictions" \
    --source-table data/raw/safra_2022_2023/field/parametros_2223_normalizado.xlsx \
    --target-table data/raw/safra_2023_2024/field/parametros_2324_normalizado.xlsx \
    --out "artifacts/runs/2324_rfinal_forecast/$method/evaluation"
done
```

Para a discussão como **calibração local R5**, fixe `stats_aug` antes de abrir as
predições e calibre a escala do PLSR em três blocos, avaliando o quarto:

```bash
python3 code/pipeline/stage9_rfinal_forecast.py calibrate \
  --predictions artifacts/runs/2324_rfinal_forecast/stats_aug/evaluation/predicoes_parcela.csv \
  --method stats_aug \
  --out artifacts/runs/2324_rfinal_forecast/stats_aug/calibration_local_r5
```

O resultado é OOF por bloco (18 parcelas de calibração, 6 de teste); não deve ser
apresentado como transferência cross-safra sem biomassa local.

## Safra 2025/26 — extensão multiespectral/GAN

A safra 2025/26 adiciona 40 parcelas, ortomosaicos em **V10** (12/12, 4 bandas
`B,R,RE,NIR`), **V13** (23/12, 5 bandas `B,G,R,RE,NIR`) e **R1** (14/01, 5 bandas).
O pipeline novo reduz os voos ao núcleo comum `[Red, RedEdge, NIR]`, mantendo a GAN
compatível com a arquitetura de três bandas. A planilha contém biomassa em V10 e
clorofila laboratorial em V10/R1; altura e produtividade de colheita permanecem ausentes.

```bash
python3 code/pipeline/normalize_2526.py \
  --workbook "data/raw/safra_2025_2026/field/PLANILHA_VOOS REGULARES_COM SENSOR.xlsx" \
  --out artifacts/runs/2526_field_targets/results/field_targets.csv

python3 code/pipeline/stage4_2526.py \
  --v10 "data/raw/safra_2025_2026/orthomosaics/12.12.2025/Milho_12.12.2025_comsensor_mosaico.tif" \
  --v13 "data/raw/safra_2025_2026/orthomosaics/23.12.2025_voo regular (30m)/Milho_23.12.2025_comsensor_vooregular_mosaico_modificado.tif" \
  --r1 "data/raw/safra_2025_2026/orthomosaics/14.01.2026/milho_14_01_2026_com_sensor_mosaico_modificado.tif" \
  --shapefile "data/raw/safra_2025_2026/geometry/parecelas_milho.shp" \
  --workbook "data/raw/safra_2025_2026/field/PLANILHA_VOOS REGULARES_COM SENSOR.xlsx" \
  --out artifacts/runs/2526_stage4_phenology/results

python3 code/pipeline/stage5_make_pairs.py \
  --stage4 artifacts/runs/2526_stage4_phenology/results \
  --out artifacts/runs/2526_pheno_gan/dataset \
  --veg V10 V13 --rep R1 --val-blocos 4
```

Após o treino, passe `--pairs .../pairs.csv` a `stage5_infer.py` para obter métricas
por par **e por parcela**. `stage8_2526.py` treina em R1 real e avalia CHL_total em
10 parcelas de validação, agregando os pares V10 e V13 antes das métricas.
