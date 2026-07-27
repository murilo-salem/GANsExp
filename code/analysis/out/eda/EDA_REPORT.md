# EDA + QA — atributos hiperespectrais (voo 12.12)

Amostras: **60** (32 parcelas, M=31 / S=29) · 635 atributos.

## Estrutura (PCA, bloco espectro)
- PC1 explica 90% e PC1+PC2 96% da variância (→ forte redundância entre as 300 bandas). Ver `pca_spectrum.png`, `pca_loadings_spectrum.png`.

## Agrupamento
- Melhor k (KMeans, silhueta): **k=2** (silhueta=0.50). Há agrupamento perceptível — ver `clustering.png` (checar contra tratamentos quando o croqui for recuperado).
- **A medida M/S é a principal fonte de variação:** o agrupamento k=2 coincide com M/S em **68%** dos casos e há separação no PC1 (média PC1: M=+5.4 vs S=-5.8). Ver painel direito de `pca_spectrum.png`. Confirmar o que M/S representa (ex.: manhã/tarde, folha superior/inferior, sol/sombra) antes de usar como amostras independentes.

## QA / outliers
- Nenhum outlier forte (Mahalanobis/vegetação).
- Cubos já perdidos na truncagem do dataset.zip (fora do CSV): `20M`, `5S`.

## Figuras

| arquivo | conteúdo |
|---|---|
| `spectra.png` | espectros por amostra + média±dp |
| `distributions.png` | boxplots de índices e texturas |
| `pca_spectrum.png` / `pca_derivative.png` | scree + PC1×PC2 (parcela/medida) |
| `pca_loadings_*.png` | loadings PC1-3 vs comprimento de onda |
| `clustering.png` | silhueta × k + dendrograma |
| `correlations.png` | correlação índices+texturas e banda-a-banda |

> Modelagem supervisionada (PLSR) permanece bloqueada até o Y de campo (`../../MISSING_DATA.md`).
