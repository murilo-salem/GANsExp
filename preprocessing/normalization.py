import glob
import rasterio
import numpy as np

def normalize_dataset(files, output_suffix="_normalized"):
    # Passo 1: calcula stats globais só desse domínio
    mins, maxs = [], []
    for f in files:
        with rasterio.open(f) as src:
            img = src.read()
        mins.append(img.min(axis=(1,2)))
        maxs.append(img.max(axis=(1,2)))
    
    global_min = np.min(mins, axis=0)
    global_max = np.max(maxs, axis=0)
    print(f"  Global min por banda: {global_min}")
    print(f"  Global max por banda: {global_max}")

    # Passo 2: normaliza e salva
    for f in files:
        with rasterio.open(f) as src:
            img = src.read().astype(np.float32)
            profile = src.profile

        img_norm = np.zeros_like(img, dtype=np.float32)
        for b in range(img.shape[0]):
            img_norm[b] = 2 * (img[b] - global_min[b]) / (global_max[b] - global_min[b] + 1e-6) - 1

        profile.update(dtype=rasterio.float32)
        out_path = f.replace(".tif", f"{output_suffix}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(img_norm)

    return global_min, global_max

# V8 e R5 separados
# files_v8 = glob.glob("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/RRENIR_v8.tif")
files_v11 = glob.glob("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/RRENIR_v11.tif")
files_v18 = glob.glob("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/RRENIR_v18.tif")
files_r2 = glob.glob("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/RRENIR_R2.tif")
# files_r5 = glob.glob("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/RRENIR_R5.tif")


# print("Normalizando V8:")
# min_v8, max_v8 = normalize_dataset(files_v8)

print("Normalizando V11:")
min_v11, max_v11 = normalize_dataset(files_v11)

print("Normalizando V18:")
min_v18, max_v18 = normalize_dataset(files_v18)

print("Normalizando R2:")
min_r2, max_r2 = normalize_dataset(files_r2)

# print("Normalizando R5:")
# min_r5, max_r5 = normalize_dataset(files_r5)

# Salva os stats para usar no teste/inferência
# np.save("stats_v8.npy", {"min": min_v8, "max": max_v8})
# np.save("stats_r5.npy", {"min": min_r5, "max": max_r5})