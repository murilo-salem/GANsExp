import rasterio
import numpy as np
from pathlib import Path

def vegetation_indices(blue, green, red, rededge, nir, L=0.5):
    """
    Calcula diversos índices de vegetação para sensores multiespectrais.
    Parâmetros:
        blue, green, red, rededge, nir : arrays NumPy das bandas espectrais
        L : parâmetro de ajuste do SAVI (padrão = 0.5)
    Retorna:
        dicionário com todos os índices calculados
    """
 
    # Evita divisão por zero
    eps = 1e-10
 
    # Índices clássicos
    NDVI = (nir - red) / (nir + red + eps)
    GNDVI = (nir - green) / (nir + green + eps)
    SAVI = ((nir - red) * (1 + L)) / (nir + red + L + eps)
    MSAVI = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    EVI = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
    VARI = (green - red) / (green + red - blue + eps)
 
    # Índices com RedEdge
    NDRE = (nir - rededge) / (nir + rededge + eps)
    CIred_edge = (nir / (rededge + eps)) - 1
 
    # Índices avançados
    MCARI = (rededge - red) - 0.2 * (rededge - green)
    TCARI = 3 * ((rededge - red) - 0.2 * (rededge - green) * (rededge / (red + eps)))
 
    # OSAVI (para combinar com TCARI)
    OSAVI = (nir - red) / (nir + red + 0.16 + eps)
 
    # CCCI (Canopy Chlorophyll Content Index)
    NDVI_min, NDVI_max = np.nanmin(NDVI), np.nanmax(NDVI)
    NDRE_min, NDRE_max = np.nanmin(NDRE), np.nanmax(NDRE)
 
    CCCI = ((NDRE - NDRE_min) / (NDRE_max - NDRE_min + eps)) * \
           ((NDVI - NDVI_min) / (NDVI_max - NDVI_min + eps))
 
    return {
        "NDVI": NDVI,
        "GNDVI": GNDVI,
        "SAVI": SAVI,
        "MSAVI": MSAVI,
        "EVI": EVI,
        "VARI": VARI,
        "NDRE": NDRE,
        "CIred_edge": CIred_edge,
        "MCARI": MCARI,
        "TCARI": TCARI,
        "OSAVI": OSAVI,
        "CCCI": CCCI
    }


def vegetation_indices_3bands(red, rededge, nir, L=0.5):
    eps = 1e-10

    NDVI = (nir - red) / (nir + red + eps)
    SAVI = ((nir - red) * (1 + L)) / (nir + red + L + eps)
    MSAVI = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2

    NDRE = (nir - rededge) / (nir + rededge + eps)
    CIred_edge = (nir / (rededge + eps)) - 1

    OSAVI = (nir - red) / (nir + red + 0.16 + eps)

    # CCCI
    NDVI_min, NDVI_max = np.nanmin(NDVI), np.nanmax(NDVI)
    NDRE_min, NDRE_max = np.nanmin(NDRE), np.nanmax(NDRE)

    CCCI = ((NDRE - NDRE_min) / (NDRE_max - NDRE_min + eps)) * \
           ((NDVI - NDVI_min) / (NDVI_max - NDVI_min + eps))

    return {
        "NDVI": NDVI,
        "SAVI": SAVI,
        "MSAVI": MSAVI,
        "NDRE": NDRE,
        "CIred_edge": CIred_edge,
        "OSAVI": OSAVI,
        "CCCI": CCCI
    }


file = "/home/lucas-fontoura/Documents/Pix2Pix/pytorch-CycleGAN-and-pix2pix/results/test_multiespectral2/test_latest/images/RRENIR_v8_real.tif"

with rasterio.open(file) as src:
    red = src.read(1).astype(np.float32)
    rededge = src.read(2).astype(np.float32)
    nir = src.read(3).astype(np.float32)

    profile = src.profile  # para salvar depois

indices = vegetation_indices_3bands(red, rededge, nir)

# ndvi = indices["NDVI"]
# savi = indices["SAVI"]
# msavi = indices["MSAVI"]
ndre = indices["NDRE"]
# ciedge = indices["CIred_edge"]
# osavi = indices["OSAVI"]
# ccci = indices["CCCI"]


profile.update(dtype=rasterio.float32, count=1)

# with rasterio.open(file.replace(".tif", "_ndvi.tif"), "w", **profile) as dst:
#     dst.write(ndvi.astype(rasterio.float32), 1)

# with rasterio.open(file.replace(".tif", "_savi.tif"), "w", **profile) as dst:
#     dst.write(savi.astype(rasterio.float32), 1)

# with rasterio.open(file.replace(".tif", "_msavi.tif"), "w", **profile) as dst:
#     dst.write(msavi.astype(rasterio.float32), 1)

with rasterio.open(file.replace(".tif", "_ndre.tif"), "w", **profile) as dst:
    dst.write(ndre.astype(rasterio.float32), 1)

# with rasterio.open(file.replace(".tif", "_ciedge.tif"), "w", **profile) as dst:
#     dst.write(ciedge.astype(rasterio.float32), 1)

# with rasterio.open(file.replace(".tif", "_osavi.tif"), "w", **profile) as dst:
#     dst.write(osavi.astype(rasterio.float32), 1)

# with rasterio.open(file.replace(".tif", "_ccci.tif"), "w", **profile) as dst:
#     dst.write(ccci.astype(rasterio.float32), 1)