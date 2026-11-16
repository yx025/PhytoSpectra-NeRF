#!/usr/bin/env python3
"""Standalone CLI test for JAI raw loading and Bayer mask utilities.

This script does NOT depend on the Nerfstudio training pipeline.  It exercises
the raw loader and Bayer mask code, prints diagnostics, and optionally writes
debug preview images to disk.

Usage
-----
With real files (filenames must follow the naming convention)::

    python scripts/test_jai_raw.py \\
        --source0 /path/to/Source0_000001_1920x1200_16b.raw \\
        --source1 /path/to/Source1_000001_1920x1200_16b.raw \\
        --output-dir ./debug_previews

With synthetic test data (no real files needed)::

    python scripts/test_jai_raw.py --synthetic --output-dir ./debug_previews

Options::

    --effective-bits  Effective sensor bit depth (default: 12)
    --black-level     Sensor black level in DN (default: 0)
    --white-level     Sensor white level in DN (default: auto from effective bits)
    --no-preview      Skip writing preview images
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure the repo root is on sys.path so imports work without pip install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nerfstudio.process_data.jai_utils import (
    RawFileMetadata,
    load_jai_raw,
    normalize_raw,
    parse_filename,
)
from nerfstudio.utils.bayer_utils import (
    bayer_mask_3ch,
    bayer_masks_rggb,
    verify_bayer_masks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_array_stats(name: str, arr: np.ndarray) -> None:
    print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


def _print_metadata(meta: RawFileMetadata) -> None:
    print(f"  source          : {meta.source}")
    print(f"  frame_idx       : {meta.frame_idx}")
    print(f"  width           : {meta.width}")
    print(f"  height          : {meta.height}")
    print(f"  container_bits  : {meta.container_bits}")
    print(f"  effective_bits  : {meta.effective_bits}")
    print(f"  ext             : {meta.ext}")
    print(f"  is_bayer        : {meta.is_bayer}")
    print(f"  is_nir          : {meta.is_nir}")
    print(f"  path            : {meta.path}")


def _save_grayscale_preview(arr_float: np.ndarray, path: Path, label: str) -> None:
    """Save a [0,1] float32 array as an 8-bit grayscale PNG."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [skip] PIL not available, cannot save {label}")
        return
    img_u8 = (np.clip(arr_float, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(path)
    print(f"  saved {label}: {path}")


def _save_mask_preview(mask: np.ndarray, path: Path, label: str, color: tuple) -> None:
    """Save a boolean mask as a colored PNG (mask pixels get the color, rest black)."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  [skip] PIL not available, cannot save {label}")
        return
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[mask] = color
    Image.fromarray(img, mode="RGB").save(path)
    print(f"  saved {label}: {path}")


def _save_debug_demosaic_preview(
    raw_float: np.ndarray,
    r_mask: np.ndarray,
    g_mask: np.ndarray,
    b_mask: np.ndarray,
    path: Path,
) -> None:
    """Save a naive nearest-neighbor demosaiced preview.

    WARNING: This is a debug visualization ONLY. It is NOT suitable for
    training and is NOT part of the core pipeline. The demosaicing here is
    the simplest possible (no interpolation) and produces visible artifacts.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [skip] PIL not available, cannot save demosaic preview")
        return

    h, w = raw_float.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Assign raw values to their respective channels
    rgb[..., 0][r_mask] = raw_float[r_mask]
    rgb[..., 1][g_mask] = raw_float[g_mask]
    rgb[..., 2][b_mask] = raw_float[b_mask]

    # Naive nearest-neighbor fill: replicate within each 2x2 block
    # R channel: fill from (even_r, even_c)
    rgb[0::2, 1::2, 0] = rgb[0::2, 0::2, 0]  # right neighbor
    rgb[1::2, 0::2, 0] = rgb[0::2, 0::2, 0]  # below neighbor
    rgb[1::2, 1::2, 0] = rgb[0::2, 0::2, 0]  # diagonal neighbor
    # B channel: fill from (odd_r, odd_c)
    rgb[0::2, 0::2, 2] = rgb[1::2, 1::2, 2]  # diagonal neighbor
    rgb[0::2, 1::2, 2] = rgb[1::2, 1::2, 2]  # above neighbor
    rgb[1::2, 0::2, 2] = rgb[1::2, 1::2, 2]  # left neighbor
    # G channel: fill non-green positions with average of neighbors
    rgb[0::2, 0::2, 1] = rgb[0::2, 1::2, 1]  # from G1 (right)
    rgb[1::2, 1::2, 1] = rgb[1::2, 0::2, 1]  # from G2 (left)

    img_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="RGB").save(path)
    print(f"  saved debug demosaic preview (NOT for training): {path}")


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def _create_synthetic_data(tmp_dir: Path) -> tuple:
    """Create small synthetic .raw files for testing without real data.

    Mimics real JAI data: 12-bit effective values (0–4095) stored in a 16-bit
    container, with '16b' in the filename.
    """
    width, height = 64, 48
    container_bits = 16  # filename label — storage container
    effective_bits = 12  # actual sensor range
    max_val = 2**effective_bits - 1  # 4095

    # Source0: BayerRG pattern - random values in 12-bit range
    rng = np.random.default_rng(42)
    s0 = rng.integers(0, max_val, size=(height, width), dtype=np.uint16)
    s0_path = tmp_dir / f"Source0_000001_{width}x{height}_{container_bits}b.raw"
    s0.tofile(s0_path)

    # Source1: Mono NIR - smooth gradient in 12-bit range
    s1 = np.linspace(0, max_val, height * width, dtype=np.uint16).reshape(height, width)
    s1_path = tmp_dir / f"Source1_000001_{width}x{height}_{container_bits}b.raw"
    s1.tofile(s1_path)

    # Also save .npy versions
    s0_npy_path = tmp_dir / f"Source0_000001_{width}x{height}_{container_bits}b.npy"
    np.save(s0_npy_path, s0)
    s1_npy_path = tmp_dir / f"Source1_000001_{width}x{height}_{container_bits}b.npy"
    np.save(s1_npy_path, s1)

    return s0_path, s1_path, s0_npy_path, s1_npy_path


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_load_and_report(
    path: Path,
    effective_bits: int,
    black_level: float,
    white_level: float | None,
) -> tuple:
    """Load a single raw file, print diagnostics, return (raw_arr, norm_arr, metadata)."""
    print(f"\n--- Loading: {path.name} ---")
    raw, meta = load_jai_raw(path, effective_bits=effective_bits)

    print("Parsed metadata:")
    _print_metadata(meta)

    print("Raw array:")
    _print_array_stats("raw", raw)

    norm = normalize_raw(raw, bit_depth=meta.effective_bits,
                         black_level=black_level, white_level=white_level)
    print(f"Normalized (effective_bits={meta.effective_bits}, "
          f"black_level={black_level}, "
          f"white_level={white_level if white_level is not None else 2**meta.effective_bits - 1}):")
    _print_array_stats("normalized", norm)

    return raw, norm, meta


def test_bayer_masks(height: int, width: int) -> tuple:
    """Generate and verify Bayer masks, print results."""
    print(f"\n--- Bayer masks for ({height} x {width}) ---")

    r_mask, g_mask, b_mask = bayer_masks_rggb(height, width)
    total = height * width

    print(f"  R pixels: {r_mask.sum():>8d}  ({100 * r_mask.sum() / total:.1f}%)")
    print(f"  G pixels: {g_mask.sum():>8d}  ({100 * g_mask.sum() / total:.1f}%)")
    print(f"  B pixels: {b_mask.sum():>8d}  ({100 * b_mask.sum() / total:.1f}%)")
    print(f"  Total   : {total:>8d}")

    try:
        verify_bayer_masks(r_mask, g_mask, b_mask)
        print("  Verification: PASSED")
    except AssertionError as e:
        print(f"  Verification: FAILED - {e}")
        return None

    # Also test the 3ch mask
    mask_3ch = bayer_mask_3ch(height, width)
    assert mask_3ch.shape == (height, width, 3)
    assert np.allclose(mask_3ch.sum(axis=-1), 1.0), "3ch mask does not sum to 1 at every pixel"
    print("  3ch mask: shape OK, sums to 1.0 at every pixel")

    return r_mask, g_mask, b_mask


def test_npy_loading(npy_path: Path, expected_raw: np.ndarray) -> None:
    """Test .npy loading and verify it matches the .raw data."""
    print(f"\n--- Testing .npy loading: {npy_path.name} ---")
    raw_npy, meta_npy = load_jai_raw(npy_path)
    _print_array_stats("npy", raw_npy)
    if np.array_equal(raw_npy, expected_raw):
        print("  .npy matches .raw: PASSED")
    else:
        print("  .npy matches .raw: FAILED (arrays differ)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test JAI raw loading and Bayer mask utilities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source0", type=Path, default=None,
                        help="Path to a Source0 BayerRG12 .raw or .npy file")
    parser.add_argument("--source1", type=Path, default=None,
                        help="Path to a Source1 Mono12 .raw or .npy file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for debug preview images")
    parser.add_argument("--effective-bits", type=int, default=12,
                        help="Effective sensor bit depth for normalization (default: 12)")
    parser.add_argument("--black-level", type=float, default=0.0,
                        help="Sensor black level in DN (default: 0)")
    parser.add_argument("--white-level", type=float, default=None,
                        help="Sensor white level in DN (default: auto from effective bits)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip writing preview images")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic test data (no real files needed)")
    args = parser.parse_args()

    # Resolve data paths
    using_synthetic = False
    tmp_dir_obj = None

    if args.synthetic:
        using_synthetic = True
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="jai_test_")
        tmp_dir = Path(tmp_dir_obj.name)
        s0_path, s1_path, s0_npy, s1_npy = _create_synthetic_data(tmp_dir)
        print(f"Created synthetic test data in {tmp_dir}")
    else:
        if args.source0 is None or args.source1 is None:
            parser.error("Provide --source0 and --source1, or use --synthetic")
        s0_path = args.source0
        s1_path = args.source1
        s0_npy = None
        s1_npy = None

    # Output directory
    out_dir = args.output_dir
    write_previews = not args.no_preview
    if write_previews and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Test loading ----
    s0_raw, s0_norm, s0_meta = test_load_and_report(
        s0_path, args.effective_bits, args.black_level, args.white_level)
    s1_raw, s1_norm, s1_meta = test_load_and_report(
        s1_path, args.effective_bits, args.black_level, args.white_level)

    # ---- Test Bayer masks ----
    masks = test_bayer_masks(s0_meta.height, s0_meta.width)

    # ---- Test .npy loading (synthetic only) ----
    if using_synthetic:
        test_npy_loading(s0_npy, s0_raw)
        test_npy_loading(s1_npy, s1_raw)

    # ---- Write preview images ----
    if write_previews and out_dir is not None and masks is not None:
        r_mask, g_mask, b_mask = masks
        print(f"\n--- Writing debug previews to {out_dir} ---")

        _save_grayscale_preview(s0_norm, out_dir / "source0_gray.png",
                                "Source0 grayscale")
        _save_grayscale_preview(s1_norm, out_dir / "source1_gray.png",
                                "Source1 NIR grayscale")
        _save_mask_preview(r_mask, out_dir / "bayer_r.png",
                           "Bayer R mask", (255, 0, 0))
        _save_mask_preview(g_mask, out_dir / "bayer_g.png",
                           "Bayer G mask", (0, 255, 0))
        _save_mask_preview(b_mask, out_dir / "bayer_b.png",
                           "Bayer B mask", (0, 0, 255))

        # Debug-only demosaic preview (NOT for training)
        _save_debug_demosaic_preview(
            s0_norm, r_mask, g_mask, b_mask,
            out_dir / "source0_debug_demosaic.png",
        )
    elif write_previews and out_dir is None:
        print("\n[info] No --output-dir specified; skipping preview images.")

    # Cleanup
    if tmp_dir_obj is not None:
        tmp_dir_obj.cleanup()

    print("\n=== All tests completed. ===")


if __name__ == "__main__":
    main()


'''
python scripts/1.test_jai_raw.py     --source0 data/JAI/3.18.2026/cucumber1/Source0_000001_2048x1536_16b.raw     --source1 data/JAI/3.18.2026/cucumber1/Source1_000001_2048x1536_16b.raw     --output-dir debug_previews
'''