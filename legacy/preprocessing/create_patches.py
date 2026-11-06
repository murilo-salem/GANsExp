import os
import rasterio
import numpy as np

input_path = "/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens/safra_milho_crop.tif"
output_dir = "/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens/safra_milho_crop_patches"
os.makedirs(output_dir, exist_ok=True)

patch_size = 256
stride = 256

with rasterio.open(input_path) as src:
    
    img = src.read()  # (bands, height, width)
    bands, height, width = img.shape
    
    patch_id = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            
            patch = img[:, y:y+patch_size, x:x+patch_size]

            patch_meta = src.meta.copy()
            patch_meta.update({
                "height": patch_size,
                "width": patch_size,
                "transform": rasterio.windows.transform(
                    rasterio.windows.Window(x, y, patch_size, patch_size),
                    src.transform
                )
            })

            # if np.mean(patch) < -0.90: # Não mantem o alinhamento entre patches
            #     continue

            out_path = os.path.join(output_dir, f"patch_{patch_id}.tif")

            with rasterio.open(out_path, "w", **patch_meta) as dst:
                dst.write(patch)

            patch_id += 1

print(f"{patch_id} patches gerados")