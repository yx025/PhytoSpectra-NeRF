#!/usr/bin/env python3
"""Task 5 — Standalone validation of JAI Bayer-select and NIR losses.

Loads real dataset samples from a scene directory and verifies that the loss
functions produce correct, non-degenerate scalar values.

Tests performed:
1. Bayer-select loss on Source0 with a simulated "perfect" prediction
   (pred_rgb channels filled from the raw target via the Bayer mask) —
   loss should be near zero.
2. Bayer-select loss on Source0 with a random prediction — loss should be
   a plausible non-zero value.
3. NIR reconstruction loss on Source1 with target-as-prediction — near zero.
4. NIR reconstruction loss on Source1 with random prediction — non-zero.
5. Shape/dtype sanity checks on all intermediate tensors.
6. Verification that the Bayer-select operation correctly matches the mask.

Usage
-----
::
    python scripts/4.test_jai_losses.py data/JAI/3.18.2026/cucumber1
    python scripts/4.test_jai_losses.py data/JAI/3.18.2026/cucumber1 --num-samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ImportError:
    print("ERROR: torch is not available.")
    sys.exit(1)

from nerfstudio.data.dataparsers.jai_dataparser import JaiDataParserConfig, JaiDataParser
from nerfstudio.data.datasets.jai_dataset import JaiDataset
from nerfstudio.model_components.jai_losses import (
    bayer_select,
    bayer_select_loss,
    nir_reconstruction_loss,
)
from nerfstudio.process_data.jai_utils import DEFAULT_EFFECTIVE_BITS


def _fmt(t: torch.Tensor) -> str:
    return f"shape={list(t.shape)} dtype={t.dtype}"


def test_on_sample(sample: dict, idx: int) -> tuple[int, int]:
    """Run loss tests on one dataset sample. Returns (passes, failures)."""
    passes = 0
    failures = 0

    def _check(name: str, ok: bool, detail: str = ""):
        nonlocal passes, failures
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        else:
            passes += 1
        msg = f"  {status}: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    h, w = sample["height"], sample["width"]
    fi = sample["frame_index"]

    print(f"\n{'='*72}")
    print(f"  Sample {idx}  (frame_index={fi}, {w}x{h})")
    print(f"{'='*72}")

    # Extract targets from sample
    s0_norm = sample["source0_normalized"]   # (H, W, 1) float32
    s1_norm = sample["source1_normalized"]   # (H, W, 1) float32
    bayer_mask = sample["source0_bayer_mask"]  # (H, W, 3) float32

    print(f"\n  Inputs:")
    print(f"    source0_normalized : {_fmt(s0_norm)}")
    print(f"    source1_normalized : {_fmt(s1_norm)}")
    print(f"    bayer_mask         : {_fmt(bayer_mask)}")

    # ------------------------------------------------------------------
    # Test 1: Bayer-select with "perfect" prediction
    # ------------------------------------------------------------------
    # Construct a pred_rgb where each channel is filled with the raw target
    # value at pixels where that channel is measured (and zero elsewhere).
    # bayer_select should reconstruct the target exactly.
    print(f"\n  --- Bayer-select loss (Source0) ---")

    s0_flat = s0_norm.squeeze(-1)  # (H, W)
    # "Perfect" prediction: broadcast target into all 3 channels, masked by Bayer
    # This means pred_rgb[..., c] = target wherever mask[..., c] == 1
    # For a perfect prediction, we set all channels to the target value.
    pred_rgb_perfect = s0_flat.unsqueeze(-1).expand(-1, -1, 3).clone()  # (H, W, 3)

    selected_perfect = bayer_select(pred_rgb_perfect, bayer_mask)
    _check("bayer_select output shape", list(selected_perfect.shape) == [h, w],
           f"got {list(selected_perfect.shape)}")
    _check("bayer_select output dtype", selected_perfect.dtype == torch.float32)

    loss_perfect = bayer_select_loss(pred_rgb_perfect, s0_norm, bayer_mask, criterion="l1")
    _check("perfect pred Bayer L1 ~ 0", loss_perfect.item() < 1e-6,
           f"loss={loss_perfect.item():.8f}")

    loss_perfect_mse = bayer_select_loss(pred_rgb_perfect, s0_norm, bayer_mask, criterion="mse")
    _check("perfect pred Bayer MSE ~ 0", loss_perfect_mse.item() < 1e-12,
           f"loss={loss_perfect_mse.item():.10e}")

    # ------------------------------------------------------------------
    # Test 2: Bayer-select with random prediction
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    pred_rgb_random = torch.rand(h, w, 3)
    loss_random = bayer_select_loss(pred_rgb_random, s0_norm, bayer_mask, criterion="l1")
    _check("random pred Bayer L1 > 0", loss_random.item() > 0.01,
           f"loss={loss_random.item():.6f}")
    _check("random pred Bayer L1 is scalar", loss_random.dim() == 0)

    loss_random_mse = bayer_select_loss(pred_rgb_random, s0_norm, bayer_mask, criterion="mse")
    _check("random pred Bayer MSE > 0", loss_random_mse.item() > 0.001,
           f"loss={loss_random_mse.item():.6f}")

    # Test "none" reduction
    loss_none = bayer_select_loss(pred_rgb_random, s0_norm, bayer_mask,
                                  criterion="l1", reduction="none")
    _check("Bayer L1 reduction='none' shape", list(loss_none.shape) == [h, w],
           f"got {list(loss_none.shape)}")

    print(f"\n    Summary: perfect_L1={loss_perfect.item():.8f}, "
          f"random_L1={loss_random.item():.6f}, random_MSE={loss_random_mse.item():.6f}")

    # ------------------------------------------------------------------
    # Test 3: Bayer-select correctness — verify channel selection
    # ------------------------------------------------------------------
    # At each pixel, bayer_select should pick exactly the channel where mask==1
    print(f"\n  --- Bayer-select correctness check ---")
    # Use a pred with distinct channel values to verify selection
    pred_check = torch.zeros(h, w, 3)
    pred_check[..., 0] = 0.1  # R channel
    pred_check[..., 1] = 0.5  # G channel
    pred_check[..., 2] = 0.9  # B channel

    selected_check = bayer_select(pred_check, bayer_mask)
    # At R pixels (mask[...,0]==1), selected should be 0.1
    r_mask = bayer_mask[..., 0].bool()
    g_mask = bayer_mask[..., 1].bool()
    b_mask = bayer_mask[..., 2].bool()

    r_ok = torch.allclose(selected_check[r_mask], torch.tensor(0.1))
    g_ok = torch.allclose(selected_check[g_mask], torch.tensor(0.5))
    b_ok = torch.allclose(selected_check[b_mask], torch.tensor(0.9))
    _check("bayer_select picks R=0.1 at R pixels", r_ok)
    _check("bayer_select picks G=0.5 at G pixels", g_ok)
    _check("bayer_select picks B=0.9 at B pixels", b_ok)

    # ------------------------------------------------------------------
    # Test 4: NIR loss with "perfect" prediction
    # ------------------------------------------------------------------
    print(f"\n  --- NIR reconstruction loss (Source1) ---")

    pred_nir_perfect = s1_norm.clone()  # (H, W, 1)
    loss_nir_perfect = nir_reconstruction_loss(pred_nir_perfect, s1_norm, criterion="l1")
    _check("perfect pred NIR L1 ~ 0", loss_nir_perfect.item() < 1e-6,
           f"loss={loss_nir_perfect.item():.8f}")

    # ------------------------------------------------------------------
    # Test 5: NIR loss with random prediction
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    pred_nir_random = torch.rand(h, w, 1)
    loss_nir_random = nir_reconstruction_loss(pred_nir_random, s1_norm, criterion="l1")
    _check("random pred NIR L1 > 0", loss_nir_random.item() > 0.01,
           f"loss={loss_nir_random.item():.6f}")
    _check("random pred NIR L1 is scalar", loss_nir_random.dim() == 0)

    loss_nir_random_mse = nir_reconstruction_loss(pred_nir_random, s1_norm, criterion="mse")
    _check("random pred NIR MSE > 0", loss_nir_random_mse.item() > 0.001,
           f"loss={loss_nir_random_mse.item():.6f}")

    # Test shape-squeezing: (H,W) target should also work
    loss_nir_squeezed = nir_reconstruction_loss(
        pred_nir_random.squeeze(-1), s1_norm.squeeze(-1), criterion="l1")
    _check("NIR loss works with (H,W) inputs",
           abs(loss_nir_squeezed.item() - loss_nir_random.item()) < 1e-6)

    print(f"\n    Summary: perfect_L1={loss_nir_perfect.item():.8f}, "
          f"random_L1={loss_nir_random.item():.6f}, random_MSE={loss_nir_random_mse.item():.6f}")

    return passes, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 5 — Validate JAI Bayer-select and NIR loss functions.",
    )
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS)
    parser.add_argument("--file-ext", type=str, default="raw", choices=["raw", "npy"])
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    print(f"Scene: {scene_dir}")

    config = JaiDataParserConfig(
        data=scene_dir,
        effective_bits=args.effective_bits,
        file_ext=args.file_ext,
    )
    dp = JaiDataParser(config)
    outputs = dp.get_dataparser_outputs("train")
    dataset = JaiDataset(dataparser_outputs=outputs, scale_factor=1.0)

    n = len(dataset)
    if n == 0:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    print(f"Dataset: {n} frames, {outputs.metadata['width']}x{outputs.metadata['height']}")
    print(f"Container bits: {outputs.metadata['container_bits']}, "
          f"Effective bits: {outputs.metadata['effective_bits']}")

    num_samples = min(args.num_samples, n)
    total_passes = 0
    total_failures = 0

    for i in range(num_samples):
        sample = dataset.get_full_sample(i)
        p, f = test_on_sample(sample, i)
        total_passes += p
        total_failures += f

    print(f"\n{'='*72}")
    print(f"  LOSS VALIDATION SUMMARY")
    print(f"{'='*72}")
    print(f"  Samples tested : {num_samples}")
    print(f"  Total checks   : {total_passes + total_failures}")
    print(f"  Passed         : {total_passes}")
    print(f"  Failed         : {total_failures}")
    print()

    if total_failures > 0:
        print("RESULT: SOME CHECKS FAILED.")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
