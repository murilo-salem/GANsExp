# Figura 5 — Ganho preditivo pareado por parcela

Distribuição das contribuições pareadas de 24 parcelas para o ganho global
ΔR² = R²(Real + Sintético) − R²(Real), em V13 e R5, para biomassa aérea e
produtividade. Cada ponto representa uma parcela, após a média das predições
OOF das sementes 7, 11 e 23. As cores identificam os quatro blocos externos
de seis parcelas. O losango representa o ΔR² global; a barra, o IC95%
percentil de 10.000 reamostragens pareadas de parcelas dentro de cada bloco.
O teste utiliza atributos do alvo real nos dois cenários de treinamento.

Como cada parcela tem apenas uma observação, não se calcula R² individual.
Os pontos representam `n × (erro²_real − erro²_híbrido) / SST`;
a média dos pontos é exatamente o ΔR² global. Os violinos são descritivos,
com apenas seis parcelas por bloco. O CSV também fornece as contribuições
sem escala, cuja soma é o ΔR², e os valores por fold com denominador próprio.

| Alvo | Horizonte | ΔR² | IC95% | Parcelas com menor erro |
|---|---|---:|---|---:|
| Biomassa | V13 | −0,0227 | [−0,1853; +0,0858] | 13/24 |
| Biomassa | R5 | +0,1054 | [+0,0248; +0,2492] | 15/24 |
| Produtividade | V13 | +0,0517 | [−0,0333; +0,1413] | 14/24 |
| Produtividade | R5 | +0,0185 | [−0,1077; +0,1415] | 12/24 |

Somente biomassa em R5 atende ao critério registrado de melhoria
(ΔR² positivo e limite inferior do IC95% positivo). Mesmo nesse contraste,
há perdas em nove parcelas e no bloco 3. Portanto, não há melhoria uniforme
entre parcelas nem evidência de ganho em todos os alvos/horizontes. São
quatro contrastes descritivos, sem correção por multiplicidade. O intervalo
condiciona-se às predições ajustadas e aos quatro blocos observados, sem
retreino nem estimativa da incerteza de generalização a novos blocos.

O código-fonte separa os blocos externos, faz seleção de canais dentro do
treino externo e usa GroupKFold e Pipeline para selecionar os regressores.
Porém, o cross-fitting das imagens sintéticas precede a busca interna do
regressor: a cadeia completa GAN–regressor não é refeita em cada split
interno. A figura e as verificações de identidade OOF não certificam ausência
de vazamento em todas as etapas. Essa afirmação exigiria auditoria completa
da execução e isolamento integral da cadeia em cada divisão relevante.

Origem: `artifacts/runs/2324_temporal_gan_value/results/oof_predictions.csv`,
modo `rs_puro`, busca histórica de nove candidatos. A proveniência JSON
registra o SHA256 do arquivo de entrada e os parâmetros da execução.

Reprodução a partir da raiz do repositório:

```bash
MPLBACKEND=Agg .venv/bin/python -m milho_experiment.pipeline.stage_10_analysis.paired_predictive_gain
```
