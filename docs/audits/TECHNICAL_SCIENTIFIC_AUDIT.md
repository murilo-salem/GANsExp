# Auditoria técnica e científica — ABC

Data da auditoria: 2026-08-06. Escopo: repositório principal, customizações do clone
`pytorch-CycleGAN-and-pix2pix`, dados locais e artefatos publicados. Nenhum dado, modelo
ou arquivo previamente existente foi alterado; as reproduções foram escritas em
`/tmp/abc-audit-20260806`.

## Resultado executivo

O pipeline principal (fenologia, pares, inferência, atributos e cenários 2023/24 e
2025/26) está operacional e as inferências a partir dos **checkpoints existentes** são
reproduzíveis no ambiente auditado. Os resultados devem continuar a ser tratados como
*proof of mechanism*, e não como estimativas agronômicas generalizáveis.

Os riscos prioritários são a normalização independente de cada recorte, a ausência de
máscara de polígono no cálculo das features, validação/model selection não aninhadas e
a falta de ambiente e protocolo determinístico versionados. O subdiretório
`preprocessing/` legado não é executável nem portável no estado atual.

## Evidências de reprodução

| Fluxo | Resultado observado | Situação |
|---|---:|---|
| 2023/24, stage 4 | 120 recortes; 96 pares (72 treino / 24 validação) | igual ao documentado |
| 2023/24, GAN 2.000 épocas | L1 0,1026; erro NDVI 0,0461 | checkpoint reproduzido |
| 2023/24, GAN 4.000 épocas | L1 0,1029; erro NDVI 0,0470 | checkpoint reproduzido |
| 2023/24, PLSR textura/biomassa | R² OOF 0,7987; RMSE 2564,26 | reproduzido |
| 2025/26, stage 4 | 120 recortes; 80 pares (60 / 20) | igual ao documentado |
| 2025/26, GAN épocas 50/100/150/200 | L1 0,1215 / 0,1223 / 0,1200 / 0,1213 | checkpoints reproduzidos |
| 2025/26, cenário CHL R1 | real R² -0,212; sintético R² -0,382 | reproduzido |

Os pares de validação correspondem a somente seis parcelas independentes em 2023/24
e dez em 2025/26. Há múltiplos pares por parcela, mas a agregação por parcela foi
confirmada somente na etapa 8 de 2025/26; métricas de imagem por par não são novas
unidades experimentais.

Foi iniciado um retreino isolado do experimento longo 2023/24. Ele chegou à época 212
sem falha de arquitetura/dados e foi interrompido. O comando não estabelece semente,
determinismo CUDA ou versões bloqueadas; um retreino completo não pode confirmar os
mesmos pesos publicados e não é uma reprodução científica independente.

## Achados prioritários

### Crítico — quantificação espectral não é preservada

`geo.to_reflectance()` normaliza cada recorte por seu percentil 99,5. Assim, o valor de
um pixel depende da própria parcela e data, em vez de uma calibração comum do sensor.
Isso remove variação de intensidade entre estágios/safras e impede interpretar as
entradas como reflectância comparável, sobretudo em GAN temporal e transferência.

**Ação:** conservar DN calibrado ou aplicar fator de reflectância por voo/banda; congelar
estatísticas de normalização no treino e aplicá-las à validação/inferência.

### Alto — extração usa `bbox`, não o polígono da parcela

`Ortho.crop_bbox()` devolve todo o retângulo envolvente e `stage4`, `stage6` e
`build_5band` não rasterizam a geometria do shapefile. Bordas, solo, ruas e parcelas
vizinhas podem entrar nas médias, GLCM e alvo visual da GAN. Também não há checagem de
CRS nem cálculo de cobertura válida.

**Ação:** rasterizar cada polígono no grid do ortomosaico, propagar a máscara a índices,
textura e redimensionamento, e salvar a fração válida por amostra.

### Alto — estimativa PLSR é otimista

`plsr_kfold.py` escolhe `n_components` no mesmo `GroupKFold` usado para reportar R².
Seleção de blocos/VIP também não é encapsulada num CV externo. O agrupamento por parcela
corrige vazamento entre estágios, mas não esse viés de seleção em amostras pequenas.

**Ação:** usar CV aninhada agrupada; selecionar componentes, VIP e bloco somente no treino
interno de cada fold externo, com IC por bootstrap de parcelas.

### Alto — split de validação é fixo

`stage5_make_pairs.py` reserva `Bloco >= 4`: todo bloco 4 é teste (seis parcelas em
2023/24; dez em 2025/26). É um holdout espacial válido se essa for a pergunta, mas não
mede a variabilidade entre splits e pode capturar gradiente espacial/desenho experimental.

**Ação:** declarar esse holdout explicitamente e executar leave-one-block-out ou CV por
bloco, preservando todas as transições de uma parcela juntas.

### Alto — documentação contraditória e interface desatualizada

O README descreve entrada GAN de “4 canais”, mas o dataset implementa 7 (três bandas +
quatro atributos). O fluxograma divulga produtividade 0,30, enquanto
`RESULTS_STATUS.md` corrige para faixa 0,065–0,44 e recomenda pooled 0,44. O fluxograma
também mantém bloqueios que o status já atualizou.

**Ação:** fazer de `RESULTS_STATUS.md` a única fonte de métricas e sincronizar README e
fluxograma; testar a compatibilidade canais/dataset/checkpoint.

### Médio — reprodução operacional incompleta

Não há `requirements.txt`, lockfile, CI ou testes automatizados. No ambiente auditado
faltam `msal`, `rasterio` e `cv2`; onze utilitários de pré-processamento falharam antes
de executar por dependências ausentes ou caminhos inexistentes.

**Ação:** publicar ambiente bloqueado (incluindo Torch/CUDA), comandos únicos, testes de
CLI e testes unitários para geo, joins e splits.

### Médio — pré-processamento legado não é portável

Onze scripts em `preprocessing/` não têm guarda `if __name__ == "__main__"`; vários
executam ao importar e contêm caminhos de `/home/lucas-fontoura/...`. Isso torna
importação/teste inseguro e os scripts não executáveis neste repositório.

**Ação:** converter em CLIs parametrizadas, mover lógica para funções puras, retirar
caminhos pessoais e arquivar fluxos obsoletos.

### Médio — gestão de artefatos e Git

Há 1.067 arquivos gerados versionados em `code/**/out/**`; resultados não trazem manifesto
de insumos/checkpoint/ambiente. Existe um ambiente `code/pipeline/.venv-sharepoint/` não
ignorado. O clone upstream e o repositório principal contêm alterações locais não
commitadas.

**Ação:** ignorar ambientes, versionar somente relatórios/manifests pequenos, registrar
hashes de dados/checkpoints/commit upstream e fixar o clone por submódulo ou patch.

### Médio — integridade de dados sem validação automática

`RGB_zero_2023_2024.tif` é inválido. Nos hiper de 12/12 há 78 BIL e 77 HDR: `5S` e
`20M` não têm HDR e `19S` é HDR órfão; 14/01 está completo (79/79). O pipeline hiper
segue após exceções, sem manifesto de falhas ou política de cobertura mínima.

**Ação:** adicionar pré-validação com pares, checksum e política explícita para ausências.

### Baixo — segurança e robustez geo

Não foram encontrados segredos privados hard-coded. O fluxo SharePoint usa autenticação
por dispositivo, mas `msal` não está declarado. `GeoTransform` aceita apenas
`ModelPixelScale`/`ModelTiepoint`, sem validar EPSG ou transformações afins/rotação.

## Pontos positivos confirmados

- O split GAN não mistura a mesma parcela entre treino e validação.
- A guarda que remove o alvo e `phys_<alvo>` do X em PLSR está ativa.
- A etapa 8 de 2025/26 agrega V10→R1 e V13→R1 antes das métricas agronômicas.
- O status é cauteloso sobre R² negativo de cenários, teto GAN e transferência inter-safra.
- Scripts da trilha principal passam compilação e CLI, exceto o downloader sem `msal`.

## Ordem recomendada de correção

1. Congelar ambiente, seeds, dados/checkpoints e manifests de execução.
2. Corrigir máscara poligonal e estratégia de normalização/calibração antes de novos experimentos.
3. Implementar CV aninhada por parcela/bloco e intervalos de incerteza.
4. Unificar documentação e tornar o status a fonte única de resultados.
5. Refatorar ou arquivar `preprocessing/`; depois adicionar CI e testes.
