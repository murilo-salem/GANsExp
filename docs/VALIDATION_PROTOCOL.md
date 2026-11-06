# Protocolo de validação

A unidade independente é a **parcela**. Estágios repetidos da mesma parcela devem ficar no mesmo
lado da validação; o protocolo padrão é `GroupKFold` por parcela, com seleção de hiperparâmetros
somente no treino interno. Métricas de pares V×R devem ser agregadas por parcela antes do relatório.

Resultados devem declarar um dos cenários: `rs_puro`, `rs_dose`, `hibrido_campo` ou `gan`.
Co-medidas físicas (CHL, N) não podem ser apresentadas como predição exclusivamente remota. KFold
por observação é apenas diagnóstico e deve identificar explicitamente a sobreposição de parcelas.

