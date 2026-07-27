# gera_inputs_teste.py
import numpy as np
from PIL import Image
from pathlib import Path

CLASS_COLORS = np.array([
    [128, 128, 128],
    [144, 238, 144],
    [139,  69,  19],
    [  0, 191, 255],
    [ 34, 139,  34],
], dtype=np.float32)

def snap_to_classes(img_float):
    H, W, _ = img_float.shape
    pixels = img_float.reshape(-1, 3)
    dists = np.linalg.norm(pixels[:, None, :] - CLASS_COLORS[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return CLASS_COLORS[nearest].reshape(H, W, 3).astype(np.uint8)

sem_dir = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/smooth_pix2pix_dataset_semantic_256/train/input")
out_dir = Path("/home/lucas-fontoura/Documents/Pix2Pix-seman-real/data/smooth_pix2pix_dataset_semantic_256/interpolated_train_images")
out_dir.mkdir(exist_ok=True)

patches = sorted(sem_dir.glob("*.png"))

count = 0
for i, pa in enumerate(patches):
    for pb in patches[i+1:]:
        # Só entre patches do mesmo estado
        state_a = next((s for s in ['R2','R5','V8','V11','V18'] if s in pa.stem), None)
        state_b = next((s for s in ['R2','R5','V8','V11','V18'] if s in pb.stem), None)
        if state_a != state_b:
            continue

        a = np.array(Image.open(pa).convert("RGB")).astype(np.float32)
        b = np.array(Image.open(pb).convert("RGB")).astype(np.float32)

        for _ in range(3):  # 3 variações por par
            sy = np.random.randint(-40, 40)
            sx = np.random.randint(-40, 40)
            b_shifted = np.roll(np.roll(b, sy, axis=0), sx, axis=1)
            alpha = np.random.uniform(0.2, 0.8)
            mixed = ((1 - alpha) * a + alpha * b_shifted).astype(np.float32)
            result = snap_to_classes(mixed)
            name = f"interp_{pa.stem}_{pb.stem}_{count:04d}.png"
            Image.fromarray(result).save(out_dir / name)
            count += 1

    if count >= 99:
        break

print(f"✓ {count} semânticos gerados em {out_dir}")