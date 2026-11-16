#!/usr/bin/env python3
"""Step 3B — JAI dataset inspection script.

Constructs the full JaiDataParser -> JaiDataset pipeline, randomly samples
frames, and validates every field in the per-sample dict.  Optionally saves
debug preview images to disk.

Usage
-----
::
    # Inspect 5 random samples:
    python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1

    # Inspect 10 samples with previews:
    python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1 \
        --num-samples 10 --output-dir ./debug_previews

    # Reproducible run, no previews:
    python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1 \
        --num-samples 3 --seed 42 --no-previews
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ImportError:
    print("ERROR: torch is not available. JaiDataset requires torch.")
    print("Install torch or activate the nerfstudio conda environment.")
    sys.exit(1)

import numpy as np

from nerfstudio.data.dataparsers.jai_dataparser import JaiDataParserConfig, JaiDataParser
from nerfstudio.data.datasets.jai_dataset import JaiDataset
from nerfstudio.process_data.jai_utils import DEFAULT_EFFECTIVE_BITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENSOR_KEYS = [
    "image",
    "source0_raw",
    "source1_raw",
    "source0_normalized",
    "source1_normalized",
    "source0_bayer_mask",
]

META_KEYS = [
    "image_idx",
    "source_id",
    "frame_index",
    "viewpoint_id",
    "shared_viewpoint",
    "width",
    "height",
    "container_bits",
    "effective_bits",
]

PATH_KEYS = [
    "source0_path",
    "source1_path",
]


def _fmt_tensor(t: torch.Tensor) -> str:
    """One-line summary of a tensor."""
    return (
        f"shape={str(list(t.shape)):18s}  dtype={str(t.dtype):16s}  "
        f"min={t.float().min().item():12.6f}  max={t.float().max().item():12.6f}"
    )


def _print_sample(idx: int, sample: Dict[str, Any]) -> None:
    """Print a structured summary of one dataset sample."""
    print(f"\n{'='*72}")
    print(f"  Sample index : {idx}")
    print(f"  frame_index  : {sample['frame_index']}")
    print(f"  source0_path : {sample['source0_path']}")
    print(f"  source1_path : {sample['source1_path']}")
    print(f"{'='*72}")

    print("\n  Tensor fields:")
    for key in TENSOR_KEYS:
        t = sample[key]
        print(f"    {key:30s}  {_fmt_tensor(t)}")

    print("\n  Metadata fields:")
    for key in META_KEYS:
        print(f"    {key:30s}  value={sample[key]}")


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def _validate_sample(
    idx: int,
    sample: Dict[str, Any],
    expected_effective_bits: int,
) -> Tuple[List[str], List[str]]:
    """Run all validation checks on a single sample.

    Returns:
        (passes, failures) — lists of human-readable check descriptions.
    """
    passes: List[str] = []
    failures: List[str] = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        msg = f"{name}" + (f" — {detail}" if detail else "")
        if ok:
            passes.append(msg)
        else:
            failures.append(msg)

    h = sample["height"]
    w = sample["width"]

    # --- Frame pairing consistency ---
    s0_name = Path(str(sample["source0_path"])).name
    s1_name = Path(str(sample["source1_path"])).name
    # Both should contain the same frame index substring
    fi = sample["frame_index"]
    fi_str = f"_{fi:06d}_"
    _check(
        "paired frame index consistent",
        fi_str in s0_name and fi_str in s1_name,
        f"frame_index={fi}, source0={s0_name}, source1={s1_name}",
    )

    # --- Shared-viewpoint metadata ---
    _check(
        "shared_viewpoint is True",
        sample["shared_viewpoint"] is True,
        f"got {sample['shared_viewpoint']}",
    )
    _check(
        "viewpoint_id == frame_index (shared viewpoint, no cross-camera extrinsic)",
        sample["viewpoint_id"] == fi,
        f"viewpoint_id={sample['viewpoint_id']}, frame_index={fi}",
    )

    # --- Shape checks ---
    _check("image shape == (H,W,1)",
           list(sample["image"].shape) == [h, w, 1],
           f"got {list(sample['image'].shape)}")
    _check("source0_raw shape == (H,W)",
           list(sample["source0_raw"].shape) == [h, w],
           f"got {list(sample['source0_raw'].shape)}")
    _check("source1_raw shape == (H,W)",
           list(sample["source1_raw"].shape) == [h, w],
           f"got {list(sample['source1_raw'].shape)}")
    _check("source0_normalized shape == (H,W,1)",
           list(sample["source0_normalized"].shape) == [h, w, 1],
           f"got {list(sample['source0_normalized'].shape)}")
    _check("source1_normalized shape == (H,W,1)",
           list(sample["source1_normalized"].shape) == [h, w, 1],
           f"got {list(sample['source1_normalized'].shape)}")

    # --- Source0 and Source1 dimension match ---
    s0_hw = list(sample["source0_raw"].shape)
    s1_hw = list(sample["source1_raw"].shape)
    _check("Source0 and Source1 dimensions match",
           s0_hw == s1_hw,
           f"Source0={s0_hw}, Source1={s1_hw}")

    # --- Bayer mask spatial alignment ---
    bm_shape = list(sample["source0_bayer_mask"].shape)
    _check("bayer_mask spatial dims == (H,W,3)",
           bm_shape == [h, w, 3],
           f"got {bm_shape}")
    # Bayer mask should sum to 1.0 at every pixel (one-hot)
    bm_sum = sample["source0_bayer_mask"].sum(dim=-1)
    bm_ok = torch.allclose(bm_sum, torch.ones(h, w))
    _check("bayer_mask sums to 1.0 at every pixel", bm_ok)

    # --- Dtype checks ---
    _check("image dtype == float32",
           sample["image"].dtype == torch.float32)
    _check("source0_raw dtype == int32",
           sample["source0_raw"].dtype == torch.int32)
    _check("source1_raw dtype == int32",
           sample["source1_raw"].dtype == torch.int32)
    _check("source0_normalized dtype == float32",
           sample["source0_normalized"].dtype == torch.float32)
    _check("source1_normalized dtype == float32",
           sample["source1_normalized"].dtype == torch.float32)
    _check("bayer_mask dtype == float32",
           sample["source0_bayer_mask"].dtype == torch.float32)

    # --- container_bits and effective_bits ---
    _check("container_bits present and == 16",
           sample["container_bits"] == 16,
           f"got {sample['container_bits']}")
    _check("effective_bits present and correct",
           sample["effective_bits"] == expected_effective_bits,
           f"got {sample['effective_bits']}, expected {expected_effective_bits}")

    # --- Normalization uses effective bits, not container bits ---
    # If effective_bits == 12, max normalized value should be raw_max / 4095.
    # With container 16-bit normalization, max would be raw_max / 65535 (much smaller).
    # Check: source0_normalized max should be close to source0_raw max / (2^effective - 1),
    # not source0_raw max / (2^container - 1).
    eff = sample["effective_bits"]
    cont = sample["container_bits"]

    for src_name, raw_key, norm_key in [
        ("Source0", "source0_raw", "source0_normalized"),
        ("Source1", "source1_raw", "source1_normalized"),
    ]:
        raw_max = sample[raw_key].float().max().item()
        norm_max = sample[norm_key].max().item()
        if raw_max > 0:
            expected_norm_eff = raw_max / (2**eff - 1)
            expected_norm_cont = raw_max / (2**cont - 1)
            # The actual normalized max should be close to effective-based normalization
            eff_delta = abs(norm_max - expected_norm_eff)
            cont_delta = abs(norm_max - expected_norm_cont)
            uses_effective = eff_delta < cont_delta or eff == cont
            _check(
                f"{src_name} normalization uses effective bits ({eff}), not container ({cont})",
                uses_effective,
                f"norm_max={norm_max:.6f}, expected_eff={expected_norm_eff:.6f}, "
                f"expected_cont={expected_norm_cont:.6f}",
            )
        else:
            _check(f"{src_name} normalization check (raw_max==0, skipped)", True)

    # --- Value range checks ---
    _check("source0_normalized in [0, 1]",
           sample["source0_normalized"].min() >= 0 and sample["source0_normalized"].max() <= 1.0)
    _check("source1_normalized in [0, 1]",
           sample["source1_normalized"].min() >= 0 and sample["source1_normalized"].max() <= 1.0)

    # Raw values should not exceed 2^effective_bits - 1 for well-formed data
    s0_raw_max = sample["source0_raw"].max().item()
    s1_raw_max = sample["source1_raw"].max().item()
    eff_max = 2**eff - 1
    _check(f"source0_raw max <= {eff_max} (12-bit range)",
           s0_raw_max <= eff_max,
           f"got {s0_raw_max}")
    _check(f"source1_raw max <= {eff_max} (12-bit range)",
           s1_raw_max <= eff_max,
           f"got {s1_raw_max}")

    return passes, failures


# ---------------------------------------------------------------------------
# Preview saving
# ---------------------------------------------------------------------------

def _save_previews(
    idx: int,
    sample: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    """Save debug preview PNGs for one sample. Returns list of saved paths."""
    try:
        from PIL import Image
    except ImportError:
        return ["(skipped: Pillow not available)"]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    fi = sample["frame_index"]

    # Source0 normalized grayscale
    s0 = sample["source0_normalized"].squeeze(-1).numpy()  # (H, W)
    img_s0 = Image.fromarray((s0 * 255).clip(0, 255).astype(np.uint8), mode="L")
    p = output_dir / f"debug_frame{fi:06d}_source0_normalized.png"
    img_s0.save(p)
    saved.append(str(p))

    # Source1 normalized grayscale
    s1 = sample["source1_normalized"].squeeze(-1).numpy()
    img_s1 = Image.fromarray((s1 * 255).clip(0, 255).astype(np.uint8), mode="L")
    p = output_dir / f"debug_frame{fi:06d}_source1_normalized.png"
    img_s1.save(p)
    saved.append(str(p))

    # Bayer mask channel visualizations (R, G, B as separate grayscale images)
    bm = sample["source0_bayer_mask"].numpy()  # (H, W, 3)
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        ch = bm[..., ch_idx]
        img_ch = Image.fromarray((ch * 255).astype(np.uint8), mode="L")
        p = output_dir / f"debug_frame{fi:06d}_bayer_{ch_name}.png"
        img_ch.save(p)
        saved.append(str(p))

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3B — JAI dataset inspection and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1\n"
            "  python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1 "
            "--num-samples 10 --output-dir ./debug_previews\n"
            "  python scripts/3.test_jai_dataset.py data/JAI/3.18.2026/cucumber1 "
            "--seed 42 --no-previews\n"
        ),
    )
    parser.add_argument("scene_dir", type=Path,
                        help="Path to scene directory containing Source0/Source1 raw files.")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to inspect (default: 5).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible sampling (default: 0).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for debug preview images (default: <scene_dir>/debug_previews).")
    parser.add_argument("--no-previews", action="store_true",
                        help="Disable saving preview images.")
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS,
                        help=f"Effective sensor bit depth (default: {DEFAULT_EFFECTIVE_BITS}).")
    parser.add_argument("--file-ext", type=str, default="raw", choices=["raw", "npy"],
                        help="File extension to load (default: raw).")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir():
        print(f"ERROR: Scene directory does not exist: {scene_dir}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------------
    print(f"Scene directory : {scene_dir}")
    print(f"File extension  : {args.file_ext}")
    print(f"Effective bits  : {args.effective_bits}")
    print(f"Random seed     : {args.seed}")
    print()

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
        print("ERROR: Dataset is empty — no paired frames found.")
        sys.exit(1)

    print(f"Dataset size    : {n} train frames")
    print(f"Image dimensions: {outputs.metadata['width']} x {outputs.metadata['height']}")
    print(f"Container bits  : {outputs.metadata['container_bits']}")
    print(f"Effective bits  : {outputs.metadata['effective_bits']}")
    print(f"Shared viewpoint: {outputs.metadata.get('shared_viewpoint', '(missing)')}")

    # Per-source intrinsics
    s0_intr = outputs.metadata.get("source0_intrinsics")
    s1_intr = outputs.metadata.get("source1_intrinsics")
    if s0_intr is not None:
        print(f"Source0 intrins.: fx={s0_intr.fx} fy={s0_intr.fy} "
              f"cx={s0_intr.cx} cy={s0_intr.cy} ({s0_intr.width}x{s0_intr.height})")
    else:
        print(f"Source0 intrins.: None (no calibration data yet)")
    if s1_intr is not None:
        print(f"Source1 intrins.: fx={s1_intr.fx} fy={s1_intr.fy} "
              f"cx={s1_intr.cx} cy={s1_intr.cy} ({s1_intr.width}x{s1_intr.height})")
    else:
        print(f"Source1 intrins.: None (no calibration data yet)")

    # Report any scanner warnings
    warnings = outputs.metadata.get("jai_warnings", [])
    if warnings:
        print(f"\nScanner warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # -----------------------------------------------------------------------
    # Select samples
    # -----------------------------------------------------------------------
    num_samples = min(args.num_samples, n)
    random.seed(args.seed)
    sample_indices = sorted(random.sample(range(n), num_samples))
    print(f"\nInspecting {num_samples} samples: {sample_indices}")

    # -----------------------------------------------------------------------
    # Inspect each sample
    # -----------------------------------------------------------------------
    output_dir = args.output_dir or (scene_dir / "debug_previews")
    save_previews = not args.no_previews

    total_passes = 0
    total_failures = 0
    all_failures: List[Tuple[int, str]] = []

    for idx in sample_indices:
        sample = dataset.get_full_sample(idx)
        _print_sample(idx, sample)

        passes, failures = _validate_sample(idx, sample, args.effective_bits)
        total_passes += len(passes)
        total_failures += len(failures)

        print(f"\n  Validation ({len(passes)} passed, {len(failures)} failed):")
        for p in passes:
            print(f"    PASS: {p}")
        for f in failures:
            print(f"    FAIL: {f}")
            all_failures.append((idx, f))

        # Previews
        if save_previews:
            saved = _save_previews(idx, sample, output_dir)
            print(f"\n  Debug previews saved ({len(saved)} files):")
            for s in saved:
                print(f"    {s}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  INSPECTION SUMMARY")
    print(f"{'='*72}")
    print(f"  Dataset size         : {n}")
    print(f"  Samples inspected    : {num_samples}")
    print(f"  Sample indices       : {sample_indices}")
    print(f"  Total checks         : {total_passes + total_failures}")
    print(f"  Passed               : {total_passes}")
    print(f"  Failed               : {total_failures}")
    if all_failures:
        print(f"\n  Failed checks:")
        for fidx, fmsg in all_failures:
            print(f"    sample {fidx}: {fmsg}")
    if save_previews:
        print(f"\n  Previews saved to    : {output_dir}")
    print()

    if total_failures > 0:
        print("RESULT: SOME CHECKS FAILED.")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
