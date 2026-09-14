import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt

def compute_ndvi_safe(image_chw, red_idx=0, nir_idx=2):
    # Desnormaliza para [0, 1] primeiro
    img = (image_chw + 1) / 2.0
    
    red = img[red_idx].clip(0, 1)
    nir = img[nir_idx].clip(0, 1)
    
    denom = nir + red
    # Só calcula onde o denominador é significativo
    ndvi = np.where(denom > 0.02, (nir - red) / denom, 0.0)
    return ndvi.clip(-1, 1)

folderA = '/home/lucas-fontoura/Documents/Pix2Pix/Data/pix2pix_full_cornfield_256/train/input'
folderB = '/home/lucas-fontoura/Documents/Pix2Pix/Data/pix2pix_full_cornfield_256/train/target'

filesA = sorted(os.listdir(folderA))[20:25]
filesB = sorted(os.listdir(folderB))[20:25]

fig, axes = plt.subplots(len(filesA), 3, figsize=(12, 4*len(filesA)))

for i, f in enumerate(filesA):
    with rasterio.open(os.path.join(folderA, f)) as src:
        A = src.read([1,2,3]).astype(np.float32)  # (3, H, W) em [-1, 1]
    with rasterio.open(os.path.join(folderB, f)) as src:
        B = src.read([1,2,3]).astype(np.float32)

    # Desnormaliza de [-1,1] para [0,1]
    A_vis = (A + 1) / 2.0
    B_vis = (B + 1) / 2.0

    # NDVI com os valores reais (ainda em [-1,1])
    # canal 0=RED, canal 2=NIR (índices no array C,H,W)
    ndvi_A = compute_ndvi_safe(A)
    ndvi_B = compute_ndvi_safe(B)

    # Imprime estatísticas para diagnóstico
    print(f"\n{f}")
    print(f"  A raw range: [{A.min():.3f}, {A.max():.3f}]")
    print(f"  B raw range: [{B.min():.3f}, {B.max():.3f}]")
    print(f"  NDVI_A: [{ndvi_A.min():.3f}, {ndvi_A.max():.3f}]  mean={ndvi_A.mean():.3f}")
    print(f"  NDVI_B: [{ndvi_B.min():.3f}, {ndvi_B.max():.3f}]  mean={ndvi_B.mean():.3f}")
    print(f"  Delta NDVI mean={( ndvi_B - ndvi_A).mean():.3f}  std={(ndvi_B - ndvi_A).std():.3f}")

    axes[i,0].imshow(A_vis.transpose(1,2,0).clip(0,1))
    axes[i,0].set_title(f'V8 - {f[:20]}')
    axes[i,1].imshow(B_vis.transpose(1,2,0).clip(0,1))
    axes[i,1].set_title('R5 real')
    im = axes[i,2].imshow(ndvi_B - ndvi_A, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes[i,2].set_title('Delta NDVI (R5 - V8)')
    plt.colorbar(im, ax=axes[i,2])

plt.tight_layout()
plt.savefig('diagnostico_pares.png', dpi=150)
print("\nSalvo em diagnostico_pares.png")

