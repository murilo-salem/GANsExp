# Catálogo de experimentos

| ID | Safra | Objetivo | Estado | Fonte de verdade |
|---|---|---|---|---|
| `multisafra` | 22/23 + 23/24 | Biomassa/Produtividade RS | oficial | `artifacts/archive/legacy/pipeline/multisafra/` |
| `gan_2324_cv` | 23/24 | Projeção V→R por GAN | proof-of-mechanism | `artifacts/archive/legacy/pipeline/stage8_2324_gancv_produtividade/` |
| `gan_fusion_2324` | 23/24 | Produtividade com GAN por fold | diagnóstico robusto | `artifacts/archive/legacy/pipeline/stage8_2324_ganfusion_cv/` |
| `gan_2526` | 25/26 | V10/V13→R1 e CHL | proof-of-mechanism | `artifacts/archive/legacy/pipeline/stage8_2526/` |
| `rgb_texture_tensors` | 22/23 + 25/26 | Tensor RGB + índices + GLCM por estágio | reprodutível | `artifacts/runs/rgb_texture_tensors/` |

Para um novo experimento, criar um TOML em `configs/`, executar via `scripts/abc_run.py` e registrar
o novo `run-id` nesta tabela com a unidade de validação e cenário de preditores.
