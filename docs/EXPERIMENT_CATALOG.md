# Catálogo de experimentos

| ID | Safra | Objetivo | Estado | Fonte de verdade |
|---|---|---|---|---|
| `multisafra` | 22/23 + 23/24 | Biomassa/Produtividade RS | oficial | `artifacts/archive/legacy/pipeline/multisafra/` |
| `gan_2324_cv` | 23/24 | Projeção V→R por GAN | proof-of-mechanism | `artifacts/archive/legacy/pipeline/stage8_2324_gancv_produtividade/` |
| `gan_fusion_2324` | 23/24 | Produtividade com GAN por fold | diagnóstico robusto | `artifacts/archive/legacy/pipeline/stage8_2324_ganfusion_cv/` |
| `gan_2526` | 25/26 | V10/V13→R1 e CHL | proof-of-mechanism | `artifacts/archive/legacy/pipeline/stage8_2526/` |
| `rgb_texture_tensors` | 22/23 + 25/26 | Tensor RGB + índices + GLCM por estágio | reprodutível | `artifacts/runs/rgb_texture_tensors/` |
| `rfinal_2324` | 22/23→23/24 | Transferência R2→R5 para biomassa | resultado negativo | `artifacts/runs/2324_rfinal_forecast/` |
| `residual_gan_2324` | 22/23→23/24 | GAN residual R2→R5 orientada à biomassa | resultado negativo; reavaliação OOF pendente | `artifacts/runs/2324_residual_gan/` |
| `gan_architecture_search` | 22/23→23/24 | Busca registrada de arquiteturas R2→R5 | preparado; sem treinos | `artifacts/runs/gan_architecture_search/` |
| `gan_attr_search_biomassa` | 23/24 | Número/composição de canais condicionais da GAN por R² biomassa | busca concluída; melhor = 2 canais (chl+NDVI), R²=0.716 | `artifacts/runs/2324_attr_search_biomassa/` |
| `gan_attr_search_produtividade` | 23/24 | Número/composição de canais condicionais da GAN por R² produtividade | busca concluída; melhor = 5 canais, R²=0.735 | `artifacts/runs/2324_attr_search_produtividade/` |
| `2324_temporal_gan_value` | 23/24 | GAN temporal V6+V8→V13 e V6+V8+V13+R2→R5; aumento, fusão e cenário combinado | implementado; execução confirmatória pendente | `artifacts/runs/2324_temporal_gan_value/` |

Para um novo experimento, criar um TOML em `configs/`, executar via `scripts/abc_run.py` e registrar
o novo `run-id` nesta tabela com a unidade de validação e cenário de preditores.
