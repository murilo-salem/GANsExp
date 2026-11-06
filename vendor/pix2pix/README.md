# Dependência externa Pix2Pix

O checkout em `pytorch-CycleGAN-and-pix2pix/` é uma dependência externa e não faz
parte do pacote `milho_experiment`. Configure outro local com `ABC_PIX2PIX_DIR`.

Antes de atualizar o checkout, registre o commit upstream, o diff local e os datasets
adicionados. O adaptador `milho_experiment.pix2pix` valida a presença de
`models/networks.py` antes de expor o checkout aos módulos GAN.
