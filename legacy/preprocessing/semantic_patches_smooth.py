from PIL import Image, ImageFilter
from pathlib import Path
import numpy as np

# Cores exatas das classes
CLASS_COLORS = {
    (128, 128, 128): 0,
    (144, 238, 144): 1,
    (139,  69,  19): 2,
    (  0, 191, 255): 3,
    ( 34, 139,  34): 4,
}
COLORS_ARRAY = np.array(list(CLASS_COLORS.keys()))

def smooth_semantic(sem_path, blur_radius=3):
    """
    Suaviza um mapa semântico:
    1. Blur gaussiano em cada canal
    2. Reatribui cada pixel à cor mais próxima
    → resultado: regiões mais suaves mas ainda com cores exatas das classes
    """
    img = np.array(Image.open(sem_path).convert("RGB")).astype(np.float32)
    
    # Blur por canal
    from scipy.ndimage import gaussian_filter
    blurred = np.stack([
        gaussian_filter(img[:,:,c], sigma=blur_radius)
        for c in range(3)
    ], axis=2)
    
    # Reatribui à cor de classe mais próxima (evita cores intermediárias)
    H, W, _ = blurred.shape
    pixels = blurred.reshape(-1, 3)
    dists = np.linalg.norm(pixels[:, None, :] - COLORS_ARRAY[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    result = COLORS_ARRAY[nearest].reshape(H, W, 3).astype(np.uint8)
    
    return Image.fromarray(result)

# Processa todos os semânticos
input_dir  = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/normalized_per_image/semantic_R5_patches_256")
output_dir = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens_mesma_grid/crop_full_cornfield/normalized_per_image/semantic_R5_patches_256_smooth")
output_dir.mkdir(parents=True, exist_ok=True)

for p in sorted(input_dir.glob("*.png")):
    smooth = smooth_semantic(p, blur_radius=8)  # ajuste o blur
    smooth.save(output_dir / p.name)
    print(f"✓ {p.name}")