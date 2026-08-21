# Catálogo do dataset

`assets.csv` é o inventário canônico dos arquivos sob `ABC_DATA_DIR` (por padrão,
`data/`). Cada linha é um arquivo e usa caminhos relativos a essa raiz; portanto o
catálogo funciona quando os binários ficam fora da cópia de trabalho.

`archives.csv` registra backups externos; o diretório padrão é controlado por
`ABC_ARCHIVE_DIR` e resolve para `../ABC-archive`.

Campos principais:

- `asset_id`: identificador estável derivado do caminho canônico;
- `season`, `domain`, `acquisition_date`, `stage`, `sample_id`: contexto extraído
  da árvore, quando disponível;
- `asset_group`: agrupa componentes do mesmo ativo, como `.bil`/`.bil.hdr` e
  os cinco arquivos de um shapefile;
- `status`: `usable` ou `quarantined`;
- `sha256` e `size_bytes`: evidência de integridade.

## Árvore de dados

```text
data/
  raw/safra_YYYY_YYYY/{orthomosaics,hyperspectral,satellite,field,geometry,design,external_results,metadata}/
  quarantine/safra_YYYY_YYYY/{orthomosaics,hyperspectral}/
```

Os nomes recebidos dos instrumentos e fornecedores são preservados. A identidade
canônica vem do caminho e do `asset_id`, não de renomear arquivos brutos.

## Operação

```bash
python3 scripts/build_dataset_registry.py --hash
python3 scripts/validate_dataset_layout.py --hash --read-images
```

Atualize `quarantine.csv` antes de regenerar o catálogo quando um ativo for
reclassificado. Itens em quarentena são preservados para recuperação, mas não
podem ser usados por receitas de execução.
