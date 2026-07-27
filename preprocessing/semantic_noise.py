import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import random
from scipy.ndimage import gaussian_filter

# Mesmas cores do treino
CLASS_COLORS = np.array([
    [128, 128, 128],  # 0: outro
    [144, 238, 144],  # 1: vegetação rala
    [139,  69,  19],  # 2: solo
    [  0, 191, 255],  # 3: água
    [ 34, 139,  34],  # 4: vegetação densa
], dtype=np.uint8)

def generate_semantic_map(size=256, n_blobs=6):
    """
    Gera um mapa semântico sintético com regiões suaves (tipo campo real).
    """
    # Ruído base com blur → regiões orgânicas
    noise = np.random.rand(size, size)
    noise = gaussian_filter(noise, sigma=random.uniform(20, 50))

    # Divide o range do ruído em N classes por threshold
    thresholds = np.percentile(noise, [15, 35, 55, 75])
    labels = np.zeros((size, size), dtype=np.uint8)
    labels[noise > thresholds[0]] = 1
    labels[noise > thresholds[1]] = 4  # veg densa por cima
    labels[noise < thresholds[2]] = 2  # solo nas regiões baixas
    labels[noise < thresholds[3] - 0.6] = 3  # água (raro)

    return CLASS_COLORS[labels]


# ── Gera N mapas e roda inferência ────────────────────────────────────────────
N = 10
synthetic_dir = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/smooth_pix2pix_dataset_semantic_256/semantic_noise")
synthetic_dir.mkdir(exist_ok=True)

for i in range(N):
    sem = generate_semantic_map()
    Image.fromarray(sem).save(synthetic_dir / f"synthetic_{i:04d}_sem.png")

print(f"✓ {N} mapas semânticos gerados em {synthetic_dir}")