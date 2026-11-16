"""JAI-specific evaluation metrics for Bayer-space and NIR-space supervision.

These metrics operate on the same representations used during training:
- Bayer-space visible: predicted RGB is Bayer-selected via the CFA mask, then
  compared against the normalized Source0 raw target.
- NIR-space: predicted single-channel NIR is compared directly against the
  normalized Source1 raw target.

All metrics assume targets are normalized using **effective bit depth** (12-bit
for JAI FS-3200D), NOT container bit depth (16-bit).

SSIM and LPIPS are intentionally NOT included:
- SSIM requires 2D spatial patches; training batches are flattened ray samples.
  Computing SSIM would require full-image rendering, which depends on valid
  camera poses (not yet available).
- LPIPS requires 3-channel RGB images; Bayer-space supervision is 1-channel.
  NIR is also 1-channel.  Expanding to 3 channels for LPIPS would be misleading.

These metrics will be added when full-image rendering with real poses is available.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from nerfstudio.model_components.jai_losses import bayer_select


def bayer_space_metrics(
    pred_rgb: Tensor,
    target_source0: Tensor,
    bayer_mask: Tensor,
) -> Dict[str, Tensor]:
    """Compute Bayer-space metrics for the visible branch.

    Args:
        pred_rgb: Predicted RGB, shape ``(*, 3)`` float32 in [0, 1].
        target_source0: Normalized Source0 target, shape ``(*,)`` or ``(*, 1)``
            float32 in [0, 1].
        bayer_mask: One-hot BayerRG mask, shape ``(*, 3)`` float32.

    Returns:
        Dict with keys ``bayer_mae``, ``bayer_mse``, ``bayer_psnr``.
    """
    if target_source0.dim() >= 1 and target_source0.shape[-1] == 1:
        target_source0 = target_source0.squeeze(-1)

    pred_bayer = bayer_select(pred_rgb, bayer_mask)
    diff = pred_bayer - target_source0

    mae = diff.abs().mean()
    mse = (diff ** 2).mean()
    # PSNR: 10 * log10(1 / MSE), clamped for numerical stability
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))

    return {
        "bayer_mae": mae,
        "bayer_mse": mse,
        "bayer_psnr": psnr,
    }


def nir_space_metrics(
    pred_nir: Tensor,
    target_source1: Tensor,
) -> Dict[str, Tensor]:
    """Compute NIR-space metrics for the infrared branch.

    Args:
        pred_nir: Predicted NIR, shape ``(*, 1)`` or ``(*,)`` float32 in [0, 1].
        target_source1: Normalized Source1 target, same shape, float32 in [0, 1].

    Returns:
        Dict with keys ``nir_mae``, ``nir_mse``, ``nir_psnr``.
    """
    if pred_nir.dim() >= 1 and pred_nir.shape[-1] == 1:
        pred_nir = pred_nir.squeeze(-1)
    if target_source1.dim() >= 1 and target_source1.shape[-1] == 1:
        target_source1 = target_source1.squeeze(-1)

    diff = pred_nir - target_source1

    mae = diff.abs().mean()
    mse = (diff ** 2).mean()
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))

    return {
        "nir_mae": mae,
        "nir_mse": mse,
        "nir_psnr": psnr,
    }
