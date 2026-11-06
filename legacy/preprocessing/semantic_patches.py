import numpy as np
from sklearn.cluster import MiniBatchKMeans
import rasterio
from pathlib import Path
from PIL import Image

# ── Configurações ─────────────────────────────────────────────────────────────
N_CLASSES        = 5
SAMPLE_PER_PATCH = 500

patches_dir  = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens/safra_milho_crop_patches")
semantic_dir = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/Multispectral_satelital/Imagens/safra_milho_crop_semantic_patches")
semantic_dir.mkdir(exist_ok=True)

# Cores por classe — mesmas nos dois métodos
CLASS_COLORS = np.array([
    [128, 128, 128],  # 0: outro / ambíguo
    [144, 238, 144],  # 1: vegetação rala
    [139,  69,  19],  # 2: solo exposto
    [  0, 191, 255],  # 3: água / sombra
    [ 34, 139,  34],  # 4: vegetação densa
], dtype=np.uint8)

# Thresholds dos índices — ajuste conforme sua cena
THRESH = dict(
    veg_densa = 0.3,
    veg_rala  = 0.15,
    solo_ndvi = 0.1,
    solo_r    = 0.3,
    agua_ndvi = 0.05,
    agua_nir  = 0.15,
)

# ── Funções ───────────────────────────────────────────────────────────────────
def compute_labels_indices(img):
    """
    Retorna mapa de labels (H, W) usando NDVI/NDRE.
    Pixels ambíguos ficam com label 0 ('outro').
    img: (3, H, W) normalizado [-1, 1], bandas: Red, RedEdge, NIR
    """
    r   = (img[0] + 1) / 2
    re  = (img[1] + 1) / 2
    nir = (img[2] + 1) / 2
    eps = 1e-8

    ndvi = (nir - r)  / (nir + r  + eps)
    # ndre disponível se quiser refinar: (nir - re) / (nir + re + eps)

    labels = np.zeros(img.shape[1:], dtype=np.uint8)  # 0 = outro
    labels[(ndvi > THRESH["veg_rala"])  & (ndvi <= THRESH["veg_densa"])] = 1
    labels[(ndvi < THRESH["solo_ndvi"]) & (r    >  THRESH["solo_r"])]    = 2
    labels[(ndvi < THRESH["agua_ndvi"]) & (nir  <  THRESH["agua_nir"])]  = 3
    labels[ndvi > THRESH["veg_densa"]]                                    = 4  # prioridade máxima
    return labels


def combine_indices_kmeans(img, kmeans, kmeans_to_class):
    """
    1. Classifica com índices
    2. Pixels ainda como 'outro' (label 0) → usa K-Means para decidir
    """
    labels = compute_labels_indices(img)

    ambiguous_mask = (labels == 0)
    if ambiguous_mask.any():
        C, H, W = img.shape
        pixels_all = img.reshape(C, -1).T                    # (H*W, 3)
        amb_idx    = ambiguous_mask.flatten()
        pixels_amb = pixels_all[amb_idx]                     # só os ambíguos

        km_labels  = kmeans.predict(pixels_amb)              # cluster id
        cls_labels = np.vectorize(kmeans_to_class.get)(km_labels)  # → classe 0-4

        labels_flat          = labels.flatten()
        labels_flat[amb_idx] = cls_labels.astype(np.uint8)
        labels               = labels_flat.reshape(H, W)

    return CLASS_COLORS[labels]


# ── 1. Coleta amostras para treinar K-Means ───────────────────────────────────
patch_list  = sorted(patches_dir.glob("*.tif"))
all_samples = []

for patch in patch_list:
    with rasterio.open(patch) as src:
        img = src.read().astype(np.float32)
    pixels = img.reshape(img.shape[0], -1).T
    idx    = np.random.choice(len(pixels), SAMPLE_PER_PATCH, replace=False)
    all_samples.append(pixels[idx])

all_samples = np.vstack(all_samples)
print(f"Treinando K-Means com {len(all_samples)} amostras...")

# ── 2. Treina K-Means global ──────────────────────────────────────────────────
kmeans    = MiniBatchKMeans(n_clusters=N_CLASSES, random_state=42, n_init=10)
kmeans.fit(all_samples)

# Mapeia cada cluster para uma classe pela ordem de NIR dos centroides
nir_order      = np.argsort(kmeans.cluster_centers_[:, 2])[::-1]  # NIR alto → veg densa
kmeans_to_class = {int(old): int(new) for new, old in enumerate(nir_order)}
# resultado: cluster com maior NIR → classe 4 (veg densa), ..., menor NIR → classe 0

# ── 3. Gera semânticos ────────────────────────────────────────────────────────
for patch in patch_list:
    with rasterio.open(patch) as src:
        img = src.read().astype(np.float32)

    semantic_rgb = combine_indices_kmeans(img, kmeans, kmeans_to_class)

    out = semantic_dir / (patch.stem + "_sem.png")
    Image.fromarray(semantic_rgb).save(out)
    print(f"✓ {patch.name}")

print("Concluído!")