# Análise Hiperespectral — Milho 2025-26

Código novo (não faz parte do Repo original). Trabalha sobre os cubos hiperespectrais
**Resonon Pika L (300 bandas, 392–1033 nm)** do voo 12.12.

## `hyperspectral_features.py`
Leitor ENVI-BIL puro em numpy (sem rasterio/spectral/skimage) + extração de atributos por amostra.
Para cada cubo `<parcela><M|S>.bil` gera: máscara de vegetação (NDVI), 6 índices
(NDVI/GNDVI/NDRE/CIrededge/SAVI/EVI2), 5 estatísticas espectrais, 5 texturas GLCM (banda NIR)
e o espectro médio das 300 bandas.

```bash
python3 hyperspectral_features.py \
  --input "../../data/2025-2026/hyperspectral_12dez" \
  --out   out/features_12dez.csv
```

Blocos de atributos: 6 índices veg · 5 estatísticas espectrais · 5 GLCM global (NIR) ·
15 **GLCM por janela** `glcmwN_*` (janelas 3/5/7 — opt-in, `--glcm-windows 3,5,7`) ·
300 bandas (espectro médio) · **300 derivadas espectrais** `d1_*` (Savitzky-Golay, jan 7, ordem 2).

GLCM por janela é vetorizada (scipy `uniform_filter`, sem skimage): contraste/homogeneidade/
correlação vêm de somas-janela de funções de (a,b); só ASM/entropia iteram os pares (levels²).
~1,6 s/cubo com 8 níveis.

## Saída atual — `out/features_12dez.csv`
**60 amostras × 635 atributos** (32 parcelas, medidas M/S). 2 cubos pulados (`20M`, `5S`) por
`.hdr` ausente (truncagem do dataset.zip). Coluna `parcela` permite juntar ao Y agronômico
quando as tabelas forem recuperadas (ver `../../MISSING_DATA.md`).

## `plsr_kfold.py`
PLSR + K-Fold genérico. Escolhe `n_components` por CV (maior R² médio) e reporta
**R²/RMSE/MAE/RPD** por fold e agregado. Blocos de X selecionáveis
(`spectrum|derivative|indices|texture|all`).

```bash
# smoke-test (alvo sintético — só valida a mecânica)
python3 plsr_kfold.py --features out/features_12dez.csv --synthetic --block derivative

# uso real, quando houver a tabela de campo (colunas: parcela,<alvo>)
python3 plsr_kfold.py --features out/features_12dez.csv \
    --target-csv campo.csv --target-col produtividade --join parcela --block all
```
Saída: `out/plsr_report.txt` + `out/plsr_report.pred.csv`.

## `band_selection.py`
Seleção de bandas/atributos por **VIP** (Variable Importance in Projection) + **forward K-Fold**
guiado pelo ranking VIP (escalável a 300 bandas, ao contrário da força bruta). Reporta o melhor
subconjunto (R²/RMSE/MAE/RPD), o subconjunto `VIP>thr` e o top-VIP com comprimento de onda.

```bash
python3 band_selection.py --features out/features_12dez.csv --synthetic --block derivative
python3 band_selection.py --features out/features_12dez.csv \
    --target-csv campo.csv --target-col biomassa --block spectrum --max-features 20
```
Saída: `out/band_selection.txt` + `out/vip_scores.csv`. No smoke-test o VIP apontou 674–681 nm
(clorofila/vermelho), 455–464 nm (azul) e 550–556 nm (verde) — regiões fisicamente coerentes.

## `image_visuals.py` — imagens dos cubos + texturas
Renderiza, por cubo: **RGB verdadeira**, **falsa-cor IV**, **mapa de NDVI** e **painel de textura
GLCM** (ASM/Contraste/Entropia/Correlação/Homogeneidade) em mapas por pixel nas janelas 3/5/7.
Monta ainda **galerias** (todas as amostras) e comparações **M vs S** por parcela. Reusa
`glcm_window_maps()` (mapas por pixel) de `hyperspectral_features.py`.

```bash
python3 image_visuals.py --input ../../data/2025-2026/hyperspectral_12dez --out out/visuals
python3 image_visuals.py ... --samples 6   # renderiza individualmente só 6 cubos (textura é cara)
```
Saída em `out/visuals/`: `cubes/*_{rgb,falsecolor,ndvi}.png` · `texture/*_glcm.png` ·
`pair_MvsS/parcela_*.png` · `gallery_rgb.png` · `gallery_ndvi.png`.

## `eda_report.py` — EDA + QA da matriz de atributos
Espectros por parcela, distribuições de índices/texturas, **PCA** (scree + PC1×PC2 + loadings),
**clustering** (silhueta × k + dendrograma), **correlações** (índices+texturas e banda-a-banda),
**QA/outliers** (Mahalanobis + `n_veg_pixels`), e um resumo `EDA_REPORT.md`.

```bash
python3 eda_report.py --features out/features_12dez.csv --out out/eda
```
Achados (12.12): PC1 explica ~90% (redundância espectral forte); **a medida M/S é a maior fonte de
variação** (k=2 coincide ~68% com M/S, separação clara no PC1) — confirmar o que M/S representa
antes de tratar como amostras independentes; nenhum outlier forte.

## TODO
- [x] Derivada espectral (`d1_*`, Savitzky-Golay).
- [x] Esqueleto PLSR + K-Fold (R²/RMSE/MAE/RPD, seleção de componentes).
- [x] Seleção de bandas por VIP + forward (`band_selection.py`).
- [x] GLCM com janelas deslizantes 3×3/5×5/7×7 (`--glcm-windows 3,5,7`).
- [x] Visualização de cubos + texturas (`image_visuals.py`) e EDA/QA (`eda_report.py`).
- [ ] **Plugar o Y real** (altura/biomassa/produtividade) via `--target-csv` após re-download das tabelas.
