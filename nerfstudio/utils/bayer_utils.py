"""Bayer CFA (Color Filter Array) mask utilities for raw image supervision.

This module generates per-pixel channel masks for a BayerRG pattern, which is
the Bayer layout used by the JAI FS-3200D Source0 sensor.

BayerRG layout (RGGB)
---------------------
The 2x2 repeating tile is:

    col 0   col 1
    ┌───┬───┐
    │ R │ G │   row 0  (even)
    ├───┼───┤
    │ G │ B │   row 1  (odd)
    └───┴───┘

- R  is at even row, even col   → (r % 2 == 0) and (c % 2 == 0)
- G1 is at even row, odd  col   → (r % 2 == 0) and (c % 2 == 1)
- G2 is at odd  row, even col   → (r % 2 == 1) and (c % 2 == 0)
- B  is at odd  row, odd  col   → (r % 2 == 1) and (c % 2 == 1)

For supervision purposes G1 and G2 are combined into a single green mask
because the NeRF field predicts a single green channel.

Coordinate convention
---------------------
Row index = y (top to bottom), column index = x (left to right).
Masks are boolean arrays of shape (H, W).

Usage in a future Bayer-select loss
------------------------------------
Given a predicted RGB image ``pred_rgb`` of shape (H, W, 3) and a raw Bayer
measurement ``gt_raw`` of shape (H, W):

    r_mask, g_mask, b_mask = bayer_masks_rggb(H, W)
    pred_bayer = (
        pred_rgb[..., 0] * r_mask
        + pred_rgb[..., 1] * g_mask
        + pred_rgb[..., 2] * b_mask
    )
    loss = L1(pred_bayer, gt_raw)

This selects, at each pixel, only the channel the sensor actually measured.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


def bayer_masks_rggb(
    height: int,
    width: int,
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Generate R, G, B boolean masks for a BayerRG (RGGB) pattern.

    Args:
        height: Image height in pixels (should be even for a complete tiling).
        width:  Image width in pixels (should be even for a complete tiling).

    Returns:
        Tuple of three boolean arrays (r_mask, g_mask, b_mask), each of
        shape (height, width).
        - r_mask: True where the sensor measures Red.
        - g_mask: True where the sensor measures Green (G1 or G2 combined).
        - b_mask: True where the sensor measures Blue.

        Exactly one of the three masks is True at each pixel.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Height and width must be positive, got ({height}, {width})."
        )

    # Row and column parity arrays
    row_even = np.zeros(height, dtype=np.bool_)
    row_even[0::2] = True
    col_even = np.zeros(width, dtype=np.bool_)
    col_even[0::2] = True

    row_odd = ~row_even
    col_odd = ~col_even

    # Outer products to build 2-D masks
    r_mask = np.outer(row_even, col_even)       # even row, even col
    g1_mask = np.outer(row_even, col_odd)       # even row, odd  col
    g2_mask = np.outer(row_odd, col_even)       # odd  row, even col
    b_mask = np.outer(row_odd, col_odd)         # odd  row, odd  col

    g_mask = g1_mask | g2_mask

    return r_mask, g_mask, b_mask


def bayer_mask_3ch(
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    """Generate a (H, W, 3) float32 mask for BayerRG (RGGB).

    Channel order is [R, G, B].  At each pixel exactly one channel is 1.0
    and the other two are 0.0.  This format is convenient for element-wise
    multiplication with a predicted RGB image.

    Args:
        height: Image height in pixels.
        width:  Image width in pixels.

    Returns:
        float32 array of shape (height, width, 3).
    """
    r, g, b = bayer_masks_rggb(height, width)
    mask = np.zeros((height, width, 3), dtype=np.float32)
    mask[..., 0] = r.astype(np.float32)
    mask[..., 1] = g.astype(np.float32)
    mask[..., 2] = b.astype(np.float32)
    return mask


def bayer_masks_rggb4(
    height: int,
    width: int,
) -> Tuple[
    npt.NDArray[np.bool_], npt.NDArray[np.bool_],
    npt.NDArray[np.bool_], npt.NDArray[np.bool_],
]:
    """Generate 4 boolean masks (R, G1, G2, B) for a BayerRG (RGGB) pattern.

    Unlike :func:`bayer_masks_rggb`, this does NOT merge G1/G2.  This is what
    you want when handling physically distinct per-site quantities such as
    black level, panel DN, or Bayer-channel-wise reflectance.

    Returns:
        (r_mask, g1_mask, g2_mask, b_mask) — each of shape (H, W) bool.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Height and width must be positive, got ({height}, {width}).")
    row_even = np.zeros(height, dtype=np.bool_); row_even[0::2] = True
    col_even = np.zeros(width, dtype=np.bool_);  col_even[0::2] = True
    row_odd = ~row_even
    col_odd = ~col_even
    r_mask = np.outer(row_even, col_even)
    g1_mask = np.outer(row_even, col_odd)
    g2_mask = np.outer(row_odd, col_even)
    b_mask = np.outer(row_odd, col_odd)
    return r_mask, g1_mask, g2_mask, b_mask


def apply_bayer_black_level_rggb(
    raw: npt.NDArray,
    bl_rggb: Tuple[float, float, float, float],
) -> npt.NDArray[np.float32]:
    """Subtract a per-Bayer-channel black level from a raw mosaic.

    Returns a float32 array (never clipped — clipping should happen after
    normalization by white level so that negative pixels land at 0 cleanly).

    Args:
        raw: (H, W) raw mosaic, any numeric dtype.
        bl_rggb: (R, G1, G2, B) black level DN values.
    """
    h, w = raw.shape
    r_m, g1_m, g2_m, b_m = bayer_masks_rggb4(h, w)
    out = raw.astype(np.float32, copy=True)
    r_bl, g1_bl, g2_bl, b_bl = (float(x) for x in bl_rggb)
    out[r_m]  -= r_bl
    out[g1_m] -= g1_bl
    out[g2_m] -= g2_bl
    out[b_m]  -= b_bl
    return out


def verify_bayer_masks(
    r_mask: npt.NDArray[np.bool_],
    g_mask: npt.NDArray[np.bool_],
    b_mask: npt.NDArray[np.bool_],
) -> bool:
    """Sanity-check that Bayer masks are mutually exclusive and exhaustive.

    Args:
        r_mask, g_mask, b_mask: Boolean masks of identical shape.

    Returns:
        True if the masks pass all checks.

    Raises:
        AssertionError: With a descriptive message on failure.
    """
    assert r_mask.shape == g_mask.shape == b_mask.shape, "Mask shapes differ."
    h, w = r_mask.shape

    # Mutual exclusivity: no pixel belongs to two channels
    assert not np.any(r_mask & g_mask), "R and G masks overlap."
    assert not np.any(r_mask & b_mask), "R and B masks overlap."
    assert not np.any(g_mask & b_mask), "G and B masks overlap."

    # Exhaustive: every pixel belongs to exactly one channel
    assert np.all(r_mask | g_mask | b_mask), "Some pixels have no mask."

    # Expected counts for RGGB
    total = h * w
    # For dimensions that may not be even, compute expected counts carefully
    r_expected = ((h + 1) // 2) * ((w + 1) // 2)
    b_expected = (h // 2) * (w // 2)
    g_expected = total - r_expected - b_expected

    assert r_mask.sum() == r_expected, (
        f"R count {r_mask.sum()} != expected {r_expected}"
    )
    assert b_mask.sum() == b_expected, (
        f"B count {b_mask.sum()} != expected {b_expected}"
    )
    assert g_mask.sum() == g_expected, (
        f"G count {g_mask.sum()} != expected {g_expected}"
    )

    return True
