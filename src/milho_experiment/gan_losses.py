"""Perdas diferenciáveis para a ablação da GAN temporal RRENIR.

As funções trabalham com tensores de reflectância em ``[0, 1]``. A máscara
representa a área da parcela e nunca é inferida a partir da imagem gerada.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


EPS = 1e-6
TEXTURE_WINDOWS = (5, 11, 21)


def parcel_mask(target: torch.Tensor, threshold: float = EPS) -> torch.Tensor:
    """Reconstrói a máscara da parcela a partir do alvo RRENIR zerado fora dela."""
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError(f"alvo deve ter shape (N,3,H,W); recebeu {tuple(target.shape)}")
    return (target.abs().amax(dim=1, keepdim=True) > threshold).to(target.dtype)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    denominator = expanded.sum().clamp_min(1.0)
    return (value * expanded).sum() / denominator


def masked_l1(fake: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((fake - target).abs(), mask)


def _normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denominator = a + b
    sign = torch.where(denominator >= 0, 1.0, -1.0)
    denominator = torch.where(denominator.abs() < EPS, sign * EPS, denominator)
    return ((a - b) / denominator).clamp(-1.0, 1.0)


def vegetation_index_loss(fake: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """MAE médio de NDVI e NDRE."""
    fake_ndvi = _normalized_difference(fake[:, 2:3], fake[:, 0:1])
    real_ndvi = _normalized_difference(target[:, 2:3], target[:, 0:1])
    fake_ndre = _normalized_difference(fake[:, 2:3], fake[:, 1:2])
    real_ndre = _normalized_difference(target[:, 2:3], target[:, 1:2])
    return 0.5 * (
        _masked_mean((fake_ndvi - real_ndvi).abs(), mask)
        + _masked_mean((fake_ndre - real_ndre).abs(), mask)
    )


def _local_stats(image: torch.Tensor, mask: torch.Tensor, window: int):
    padding = window // 2
    density = F.avg_pool2d(mask, window, stride=1, padding=padding)
    safe = density.clamp_min(EPS)
    mean = F.avg_pool2d(image * mask, window, stride=1, padding=padding) / safe
    second = F.avg_pool2d(image.square() * mask, window, stride=1, padding=padding) / safe
    std = (second - mean.square()).clamp_min(0.0).add(EPS).sqrt()

    pair_h_mask = mask[..., :, 1:] * mask[..., :, :-1]
    pair_v_mask = mask[..., 1:, :] * mask[..., :-1, :]
    pair_h = image[..., :, 1:] * image[..., :, :-1]
    pair_v = image[..., 1:, :] * image[..., :-1, :]
    pair_h = F.pad(pair_h * pair_h_mask, (0, 1, 0, 0))
    pair_v = F.pad(pair_v * pair_v_mask, (0, 0, 0, 1))
    pair_h_mask = F.pad(pair_h_mask, (0, 1, 0, 0))
    pair_v_mask = F.pad(pair_v_mask, (0, 0, 0, 1))
    corr_h = F.avg_pool2d(pair_h, window, stride=1, padding=padding) / \
        F.avg_pool2d(pair_h_mask, window, stride=1, padding=padding).clamp_min(EPS)
    corr_v = F.avg_pool2d(pair_v, window, stride=1, padding=padding) / \
        F.avg_pool2d(pair_v_mask, window, stride=1, padding=padding).clamp_min(EPS)
    valid = (density >= 0.8).to(image.dtype)
    return (mean, std, corr_h, corr_v), valid


def multiscale_texture_loss(fake: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor,
                            windows: Iterable[int] = TEXTURE_WINDOWS) -> torch.Tensor:
    """Compara estatísticas locais diferenciáveis do NIR em várias escalas."""
    losses = []
    for window in windows:
        if window <= 0 or window % 2 == 0:
            raise ValueError("janelas de textura devem ser ímpares e positivas")
        fake_stats, valid = _local_stats(fake[:, 2:3], mask, window)
        target_stats, _ = _local_stats(target[:, 2:3], mask, window)
        losses.extend(_masked_mean((a - b).abs(), valid)
                      for a, b in zip(fake_stats, target_stats))
    if not losses:
        raise ValueError("ao menos uma janela de textura é necessária")
    return torch.stack(losses).mean()


def gradient_loss(fake: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """MAE dos gradientes Sobel, ignorando a borda artificial da parcela."""
    dtype, device, channels = fake.dtype, fake.device, fake.shape[1]
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=dtype, device=device) / 8.0
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    fake_x = F.conv2d(fake, kernel_x, padding=1, groups=channels)
    fake_y = F.conv2d(fake, kernel_y, padding=1, groups=channels)
    real_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
    real_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
    interior = (F.avg_pool2d(mask, 3, stride=1, padding=1) >= 1.0 - EPS).to(mask.dtype)
    return 0.5 * (
        _masked_mean((fake_x - real_x).abs(), interior)
        + _masked_mean((fake_y - real_y).abs(), interior)
    )


def discriminator_features(discriminator, value: torch.Tensor):
    """Executa o PatchGAN e devolve ativações após cada LeakyReLU."""
    if not hasattr(discriminator, "model"):
        raise TypeError("feature matching requer PatchGAN com atributo 'model'")
    features = []
    current = value
    for layer in discriminator.model:
        current = layer(current)
        if isinstance(layer, torch.nn.LeakyReLU):
            features.append(current)
    return current, features


def feature_matching_loss(fake_features: Iterable[torch.Tensor],
                          real_features: Iterable[torch.Tensor]) -> torch.Tensor:
    pairs = list(zip(fake_features, real_features))
    if not pairs:
        raise ValueError("feature matching requer ao menos uma ativação")
    return torch.stack([(fake - real.detach()).abs().mean() for fake, real in pairs]).mean()


def calibrate_scales(fake: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     components: Iterable[str]) -> dict[str, float]:
    """Iguala a magnitude inicial de cada termo auxiliar à do L1 mascarado."""
    with torch.no_grad():
        reference = float(masked_l1(fake, target, mask).detach())
        functions = {
            "indices": vegetation_index_loss,
            "texture": multiscale_texture_loss,
            "gradient": gradient_loss,
        }
        scales = {}
        for name in components:
            if name not in functions:
                raise ValueError(f"componente desconhecido: {name}")
            magnitude = float(functions[name](fake, target, mask).detach())
            scales[name] = reference / max(magnitude, EPS)
    return scales
