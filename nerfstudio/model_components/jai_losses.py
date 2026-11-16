"""JAI FS-3200D loss utilities: Bayer-select loss and NIR reconstruction loss.

These losses are designed for the JAI dual-source capture setup where:
- Source0 (visible) is a BayerRG raw mosaic — NOT a demosaiced RGB image
- Source1 (NIR) is a single-channel near-infrared image

Bayer-select loss
-----------------
The model predicts a full 3-channel RGB image for the visible branch.
At each pixel the Bayer CFA physically measures only ONE of R, G, or B.
The Bayer-select loss picks the predicted channel that matches what the
sensor measured and compares it against the raw Source0 target::

    pred_bayer = (pred_rgb * bayer_mask).sum(dim=-1)   # select measured channel
    loss = criterion(pred_bayer, source0_target)

This avoids demosaicing entirely — the model learns to predict full RGB,
but supervision comes only from the channel the sensor actually observed.

NIR reconstruction loss
-----------------------
The model predicts a single-channel NIR output for the infrared branch.
This is a straightforward reconstruction loss against normalized Source1::

    loss = criterion(pred_nir, source1_target)

Both losses default to L1 but accept any element-wise criterion.

Relationship to existing thermal losses
---------------------------------------
The thermal pipeline in ``thermal_nerfacto.py`` uses:
- ``is_thermal`` masks to separate RGB / thermal pixels in a mixed batch
- ``tv_pixel_loss`` for thermal pixel smoothness (2x2 patch TV)
- ``cross_channel_loss`` to align thermal gradients with RGB gradients
- ``density_loss`` (L1) to regularize RGB vs thermal density fields

None of these apply to JAI:
- JAI has no ``is_thermal`` mask — Source0 and Source1 are always paired
- JAI Source0 is Bayer raw (not demosaiced RGB), so TV/cross-channel losses
  designed for thermal-vs-RGB do not transfer
- JAI density regularization (if needed) will be defined in a future task

The existing thermal pipeline is NOT modified by this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Bayer-select operation (pure, no loss reduction)
# ---------------------------------------------------------------------------

def bayer_select(
    pred_rgb: Tensor,
    bayer_mask: Tensor,
) -> Tensor:
    """Select the predicted channel at each pixel according to the Bayer mask.

    This is the core operation for Bayer-select supervision: at each pixel,
    exactly one of R, G, or B was measured by the sensor.  The Bayer mask is
    a one-hot ``(*, 3)`` tensor indicating which channel was measured.

    Args:
        pred_rgb: Predicted RGB image, shape ``(*, 3)``.
        bayer_mask: One-hot Bayer CFA mask, shape ``(*, 3)``.  At each spatial
            position exactly one channel is 1.0, the others are 0.0.

    Returns:
        Scalar-per-pixel tensor of shape ``(*,)`` — the predicted value at the
        channel the sensor measured.
    """
    return (pred_rgb * bayer_mask).sum(dim=-1)


# ---------------------------------------------------------------------------
# Bayer-select loss
# ---------------------------------------------------------------------------

def bayer_select_loss(
    pred_rgb: Tensor,
    target_source0: Tensor,
    bayer_mask: Tensor,
    criterion: Literal["l1", "mse"] = "l1",
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Bayer-select loss for Source0 visible-branch supervision.

    Computes the reconstruction error between the predicted RGB image and the
    raw Bayer mosaic target, using only the channel the sensor measured at each
    pixel.

    Args:
        pred_rgb: Predicted RGB image, shape ``(*, 3)`` float32 in [0, 1].
        target_source0: Normalized Source0 Bayer target, shape ``(*,)`` or
            ``(*, 1)`` float32 in [0, 1].  Normalized using effective bit depth
            (12-bit for JAI FS-3200D), NOT container bit depth (16-bit).
        bayer_mask: One-hot BayerRG mask, shape ``(*, 3)`` float32.
        criterion: ``"l1"`` (default) or ``"mse"``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (if ``reduction != "none"``) or per-pixel loss tensor.
    """
    # Flatten target to (*, ) if it has a trailing dim of 1
    if target_source0.dim() >= 1 and target_source0.shape[-1] == 1:
        target_source0 = target_source0.squeeze(-1)

    pred_bayer = bayer_select(pred_rgb, bayer_mask)

    if criterion == "l1":
        per_pixel = (pred_bayer - target_source0).abs()
    elif criterion == "mse":
        per_pixel = (pred_bayer - target_source0) ** 2
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    if reduction == "mean":
        return per_pixel.mean()
    elif reduction == "sum":
        return per_pixel.sum()
    else:
        return per_pixel


# ---------------------------------------------------------------------------
# NIR reconstruction loss
# ---------------------------------------------------------------------------

def nir_reconstruction_loss(
    pred_nir: Tensor,
    target_source1: Tensor,
    criterion: Literal["l1", "mse"] = "l1",
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Single-channel NIR reconstruction loss for Source1.

    Straightforward reconstruction loss between predicted NIR output and the
    normalized Source1 target.

    Args:
        pred_nir: Predicted NIR output, shape ``(*, 1)`` or ``(*,)`` float32
            in [0, 1].
        target_source1: Normalized Source1 NIR target, shape ``(*, 1)`` or
            ``(*,)`` float32 in [0, 1].  Normalized using effective bit depth.
        criterion: ``"l1"`` (default) or ``"mse"``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss (if ``reduction != "none"``) or per-pixel loss tensor.
    """
    # Align shapes: squeeze trailing 1 if present
    if pred_nir.dim() >= 1 and pred_nir.shape[-1] == 1:
        pred_nir = pred_nir.squeeze(-1)
    if target_source1.dim() >= 1 and target_source1.shape[-1] == 1:
        target_source1 = target_source1.squeeze(-1)

    if criterion == "l1":
        per_pixel = (pred_nir - target_source1).abs()
    elif criterion == "mse":
        per_pixel = (pred_nir - target_source1) ** 2
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    if reduction == "mean":
        return per_pixel.mean()
    elif reduction == "sum":
        return per_pixel.sum()
    else:
        return per_pixel
