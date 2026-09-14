# Pix2Pix/CycleGAN vendorizado

O framework vendorizado permanece temporariamente em `pytorch-CycleGAN-and-pix2pix/` para não
quebrar checkpoints, datasets e comandos históricos. A migração física para `vendor/` só ocorrerá
quando os scripts compatíveis usarem a resolução central de caminhos.

Estado a registrar antes de qualquer atualização upstream:

- commit upstream atual e `git diff` local;
- datasets adicionais (`phenology_dataset.py` e `multispectral_dataset.py`);
- alterações em `train.py`, opções e utilitários;
- versão de PyTorch/CUDA e nomes de checkpoints usados por cada run.

