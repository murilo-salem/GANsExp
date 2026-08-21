import os
import rasterio
import numpy as np

stages = ["v8_patches_256", "v11_patches_256", "v18_patches_256", "R2_patches_256", "R5_patches_256"]
base_dir = "/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/normalized_per_image"

patch_ids = sorted(os.listdir(os.path.join(base_dir, stages[0])))

valid_patches = []

# for patch_name in patch_ids:
    
#     means = []

#     for s in stages:
#         path = os.path.join(base_dir, s, patch_name)

#         with rasterio.open(path) as src:
#             img = src.read()
#             means.append(np.mean(img))

#     if not all(m < -0.9 for m in means):
#         valid_patches.append(patch_name)

for patch_name in patch_ids:
    valid = True
    for s in stages:
        path = os.path.join(base_dir, s, patch_name)
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        
        # Pixel é nodata se a média dos canais é menor que -0.95
        nodata_ratio = (img.mean(axis=0) < -0.95).mean()
        
        if nodata_ratio > 0.15:  # remove se mais de 15% é nodata
            valid = False
            break
    
    if valid:
        valid_patches.append(patch_name)

# print((valid_patches, len(valid_patches)))
valid_patches = set(valid_patches)

for s in stages:
    path = os.path.join(base_dir, s)

    for patch_name in os.listdir(path):
        if patch_name not in valid_patches:
            os.remove(os.path.join(path, patch_name))

# Check if all stages have the same patches
for s in stages:
    print(s, len(os.listdir(os.path.join(base_dir, s))))