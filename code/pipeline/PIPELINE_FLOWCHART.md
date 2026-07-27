# Fluxograma completo do pipeline milho (etapas 1–8 + roteiro 1–4)

> ⚠️ Resultados honestos e correções em [`RESULTS_STATUS.md`](RESULTS_STATUS.md)
> (Produtividade, etapa 8, teto da GAN; + pooled e adaptação de domínio inter-safra).

Diagrama técnico de tudo que foi implementado, com especificações e resultados.
CRS de todos os produtos geoespaciais: **SIRGAS 2000 / UTM 22S (EPSG:31982)**.

```mermaid
flowchart TD
    %% ================= AQUISIÇÃO =================
    subgraph ACQ["1 · AQUISIÇÃO DE DADOS"]
        direction TB
        A2223["Safra 2022/23<br/>RRENIR + RGB por estágio<br/>V8/V11/V18/R2/R5<br/>uint16, GSD ~1.36 cm"]
        A2324["Safra 2023/24<br/>RRENIR por estágio V6/V8/V13/R2/R5<br/>uint16, GSD ~1.63 cm, tiled GeoTIFF<br/>RGB: só 'zero' (corrompido)"]
        HYP["Cubos hiperespectrais (trilha à parte)<br/>Resonon Pika L · 300 bandas · ENVI-BIL<br/>voo 12.dez · 62 cubos"]
        SHP["Shapefiles de parcelas<br/>23/24: 24 parcelas · 22/23: 48 shapes<br/>chave (Dose_N, Bloco)"]
        TBL["Tabelas de campo (Y)<br/>parametros_normalizado.xlsx<br/>Biomassa · Produtividade · CHL_total · %N<br/>por parcela × estágio"]
    end

    %% ================= BASE GEO =================
    GEO["geo.py — base geoespacial<br/>tifffile + ModelPixelScale/Tiepoint (sem GDAL)<br/>UTM→pixel · crop_bbox por parcela (leitura cacheada)<br/>read_parcels (pyshp) · to_reflectance (norm. p99.5)<br/>rrenir_indices · bands5_indices · chlorophyll_map"]
    A2223 --> GEO
    A2324 --> GEO
    SHP --> GEO

    %% ================= ETAPA 3 FEATURES =================
    subgraph FEAT["3 · EXTRAÇÃO DE ATRIBUTOS"]
        direction TB
        F_SPEC["Espectrais<br/>NDVI · GNDVI · NDRE · CIrededge · SAVI · EVI2<br/>(5-bandas: + EVI/VARI/TGI)<br/>estatísticas: mean/std/p10/p50/p90"]
        F_TEX["Textura — GLCM Haralick COMPLETO (17)<br/>ASM · Contrast · Correlation · Variance · IDM<br/>Sum Avg/Var/Entropy · Entropy · Diff Entropy<br/>IMC1 · IMC2 · MCC · Dissimilarity · Inertia<br/>Cluster Shade · Cluster Prominence<br/>global (16 níveis, simétrica) + janelas 3/5/7 (8 níveis)"]
    end
    GEO --> FEAT
    HYP -.->|"hyperspectral_features.py<br/>(320 atributos/amostra)"| FEAT

    %% ================= ETAPA 4 FENOLOGIA =================
    subgraph PHENO["4 · DIVISÃO FENOLÓGICA — stage4_phenology.py"]
        direction TB
        P_VEG["VEGETATIVO (real)<br/>V6/V8/V13"]
        P_REP["REPRODUTIVO (real)<br/>R2/R5"]
        P_OUT["120 recortes 256×256 (24 parc × 5 est)<br/>npy float32 + PNG falsa-cor + manifest.csv"]
    end
    GEO --> PHENO
    P_VEG --> P_OUT
    P_REP --> P_OUT
    PVAL["Validação fenológica (NDVI médio)<br/>V6 0.37 → V8 0.62 → V13 0.69 → R2 0.71 → R5 0.61"]:::result
    P_OUT --> PVAL

    %% ================= ETAPA 5 GAN =================
    subgraph GAN["5 · PIX2PIX CONDICIONAL veg→reprodutivo"]
        direction TB
        MK["stage5_make_pairs.py<br/>pares V8/V13 × R2/R5 alinhados<br/>96 pares · split por parcela (72 treino / 24 val)"]
        DS["phenology_dataset.py — ENTRADA 7 CANAIS<br/>3 bandas + attribute_channels:<br/>Clorofila(CIrededge) · NDVI · NDRE · SAVI<br/>normalizados [-1,1]"]
        TRN["Treino pix2pix<br/>G = U-Net 256 (54.4 M) · D = PatchGAN (2.77 M)<br/>L = L_GAN + λ·L1 (λ=100) · Adam lr 2e-4<br/>batch 4 · load 286 / crop 256 · 200 épocas<br/>→ checkpoints/pheno_2324"]
        INF["stage5_infer.py<br/>painel veg | sintético | real + métricas"]
    end
    P_OUT --> MK --> DS --> TRN --> INF
    GRES["Inferência final: L1 = 0.111 · |ΔNDVI| = 0.055<br/>(vs smoke 2ép: 0.123 / 0.131 — ΔNDVI ~2.4× melhor)"]:::result
    INF --> GRES

    %% ================= ETAPA 6/7 PLSR =================
    subgraph PLSR["6·7 · MODELAGEM PLSR + VALIDAÇÃO"]
        direction TB
        S6["stage6_features_ortho.py<br/>features por parcela/estágio (23/24)<br/>+ atributos físicos phys_CHL_total/phys_N_percent/phys_N_acumulado<br/>+ junção Y por (Estagio, Dose_N, Bloco)"]
        MODEL["plsr_kfold.py<br/>StandardScaler + PLSRegression<br/>otimiza nº componentes por CV<br/>GroupKFold por parcela (anti-vazamento)<br/>bloco combinável texture+physical (diagrama)<br/>métricas: R² · RMSE · MAE · RPD<br/>guarda: remove alvo e phys_<alvo> de X"]
        VIP["band_selection.py<br/>VIP + forward selection"]
    end
    FEAT --> S6 --> MODEL
    TBL --> S6
    MODEL --> VIP
    PRES["RESULTADOS (23/24, GroupKFold por parcela)<br/>Biomassa R²=0.80 (texture, n=120) · 0.95 (texture+physical, n=72)<br/>Produtividade R²=0.30 · CHL_total R²=0.31<br/>⚠ bloco 'all' agora sem alvo em X (fix de vazamento)"]:::result
    MODEL --> PRES
    VIP --> PRES

    %% ================= ETAPA 8 CENÁRIOS =================
    subgraph SCEN["8 · CENÁRIOS FUTUROS — stage8_scenarios.py"]
        direction TB
        S8A["Veg REAL → G treinado → Reprodutivo SINTÉTICO"]
        S8B["extrai atributos do sintético"]
        S8C["PLSR treinado no reprodutivo REAL → prediz Y"]
        S8D["compara: sintético vs real vs Y medido"]
    end
    TRN --> S8A --> S8B --> S8C --> S8D
    MODEL --> S8C
    SRES["corr(pred_sintético, medido) = corr(pred_real, medido) = 0.48<br/>→ dado sintético preserva o poder preditivo (tese do diagrama)<br/>R² absoluto negativo (só ~6 parcelas de validação)"]:::result
    S8D --> SRES

    %% ================= PASSO 1 · 5-BANDAS =================
    subgraph B5["PASSO 1 · CUBO 5-BANDAS (só 22/23) — build_5band.py"]
        direction TB
        B5A["funde RGB(Azul,Verde) + RRENIR(Verm,RedEdge,NIR)<br/>B1–B5 por parcela/estágio · reamostra 256²<br/>habilita GNDVI/EVI/VARI/TGI"]
        B5B["⚠ Azul/Verde = DN de câmera RGB (não calibrado)<br/>VARI/EVI instáveis; NDVI/NDRE sólidos<br/>240 amostras → features_5band_2223.csv"]
    end
    A2223 --> B5A --> B5B
    B5B --> FEAT

    %% ================= PASSO 4 · MULTI-SAFRA =================
    subgraph MS["PASSO 4 · MULTI-SAFRA — stage_multisafra.py"]
        direction TB
        MSA["within-safra GroupKFold (texture)"]
        MSB["cross-safra: treina numa, testa na outra"]
    end
    B5B --> MSA
    S6 --> MSA
    MSA --> MSB
    MRES["within: Biomassa R²=0.69 (22/23, n=192) + 0.80 (23/24) → REPLICA<br/>cross-safra: R²<0 (não transfere · domain shift GSD/luz/sensor)"]:::result
    MSB --> MRES

    %% ================= BLOQUEIOS =================
    BLOCK["BLOQUEIOS EXTERNOS (fora do código)<br/>• Altura: não medida em nenhuma tabela<br/>• Safra 2024/25: ausente (aquisição pendente)"]:::blocker
    TBL -.-> BLOCK

    classDef result fill:#d5f5e3,stroke:#1e8449,color:#145a32;
    classDef blocker fill:#fdedec,stroke:#c0392b,color:#922b21;
```

## Especificações técnicas resumidas

| Componente | Especificação |
|---|---|
| **CRS / geo** | SIRGAS 2000 UTM 22S (EPSG:31982); leitura `tifffile` + tags GeoTIFF; sem GDAL |
| **RRENIR** | 3 bandas uint16 [Red, RedEdge, NIR]; tiled 256; GSD 1.63 cm (23/24) / 1.36 cm (22/23) |
| **RGB** | 3 bandas uint16 [R, G, B]; só 22/23 por estágio |
| **Índices** | NDVI, GNDVI, NDRE, CIrededge, SAVI, EVI2 (+ EVI, VARI, TGI no 5-bandas) |
| **GLCM** | Haralick 17 descritores; matriz simétrica; global 16 níveis + janelas 3/5/7 (8 níveis) |
| **GAN** | pix2pix; G U-Net256 (54.4 M), D PatchGAN (2.77 M); input_nc=7, output_nc=3; L_GAN+λL1 (λ=100); Adam 2e-4; 200 épocas |
| **Condicionamento** | 4 canais de atributos: Clorofila(CIrededge), NDVI, NDRE, SAVI (`attribute_channels`) |
| **PLSR** | StandardScaler + PLSRegression; GroupKFold por parcela; R²/RMSE/MAE/RPD; VIP forward; bloco combinável texture+physical; guarda anti-vazamento (alvo e phys_<alvo> fora de X) |
| **Deps (user-site)** | tifffile, pyshp, openpyxl, dominate, wandb (torch 2.11 + CUDA) |

## Resultados consolidados

| Etapa | Métrica | Valor |
|---|---|---|
| 4 Fenologia | NDVI V6→R2→R5 | 0.37 → 0.71 → 0.61 (coerente) |
| 5 GAN | L1 / ΔNDVI (val) — 200 ép | 0.1115 / 0.0546 |
| 5 GAN | L1 / ΔNDVI (val) — 2000 ép (doce spot) | 0.1026 / 0.0461 (-7.9% / -15.6%) |
| 5 GAN | L1 / ΔNDVI (val) — 4000 ép | 0.1029 / 0.0470 (platô vs 2000; D colapsa) |
| 6-7 PLSR 23/24 | Biomassa R² (texture / texture+physical) | 0.80 (n=120) / 0.95 (n=72) |
| 6-7 PLSR 23/24 | Produtividade / CHL R² | 0.30 / 0.31 |
| 8 Cenários | corr(sintético,medido) vs corr(real,medido) | 0.48 = 0.48 |
| 4-passo Multi-safra | Biomassa within (22/23 / 23/24) | 0.69 / 0.80 |
| 4-passo Multi-safra | cross-safra | R²<0 (não transfere) |

**Distância ao diagrama: ~90%.** Etapas 1–8 rodando em 22/23+23/24; falta apenas dado externo
(altura, safra 24/25). Ressalva: 24–48 parcelas ⇒ GAN e etapa 8 são proof-of-mechanism.
