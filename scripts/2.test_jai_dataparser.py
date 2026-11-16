#!/usr/bin/env python3
"""Standalone CLI test for the JAI dataparser scene scanner and frame pairing.

Scans a scene directory, pairs Source0/Source1 by frame index, validates
consistency, and prints a detailed report.  Does NOT depend on training.

Usage
-----
::
    python scripts/2.test_jai_dataparser.py data/JAI/3.18.2026/cucumber1
    python scripts/test_jai_dataparser.py /path/to/scene_dir
    python scripts/test_jai_dataparser.py /path/to/scene_dir --file-ext npy
    python scripts/test_jai_dataparser.py /path/to/scene_dir --effective-bits 12 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nerfstudio.process_data.jai_utils import (
    DEFAULT_EFFECTIVE_BITS,
    JaiFramePair,
    load_transforms_json,
    scan_scene_directory,
)


def _print_pair(pair: JaiFramePair, verbose: bool = False) -> None:
    print(f"  frame {pair.frame_idx:06d}  "
          f"{pair.width}x{pair.height}  "
          f"container={pair.container_bits}b  effective={pair.effective_bits}b")
    if verbose:
        print(f"    Source0: {pair.source0.path.name}")
        print(f"    Source1: {pair.source1.path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect JAI scene directory: scan, pair, and validate frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scene_dir", type=Path,
                        help="Path to scene directory containing Source0/Source1 files")
    parser.add_argument("--file-ext", type=str, default="raw",
                        choices=["raw", "npy"],
                        help="File extension to scan (default: raw)")
    parser.add_argument("--effective-bits", type=int, default=DEFAULT_EFFECTIVE_BITS,
                        help=f"Effective sensor bit depth (default: {DEFAULT_EFFECTIVE_BITS})")
    parser.add_argument("--eval-fraction", type=float, default=0.1,
                        help="Fraction of frames for eval split (default: 0.1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-frame file paths")
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    print(f"Scene directory: {scene_dir}")
    print(f"File extension : .{args.file_ext}")
    print(f"Effective bits : {args.effective_bits}")
    print()

    # --- Raw scan ---
    print("=== Directory scan ===")
    try:
        pairs, warnings = scan_scene_directory(
            scene_dir,
            effective_bits=args.effective_bits,
            file_ext=args.file_ext,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Valid paired frames: {len(pairs)}")
    if pairs:
        ref = pairs[0]
        print(f"Resolution         : {ref.width} x {ref.height}")
        print(f"Container bits     : {ref.container_bits}")
        print(f"Effective bits     : {ref.effective_bits}")
        print(f"Frame index range  : {pairs[0].frame_idx} .. {pairs[-1].frame_idx}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # --- Pair listing ---
    print(f"\n=== Paired frames ({len(pairs)}) ===")
    for pair in pairs:
        _print_pair(pair, verbose=args.verbose)

    # --- Train/eval split preview (pure python, no torch) ---
    print(f"\n=== Train/eval split (eval_fraction={args.eval_fraction}) ===")
    n = len(pairs)
    n_eval = max(1, int(n * args.eval_fraction))
    if n_eval > 1:
        eval_indices = set(
            int(round(i * (n - 1) / (n_eval - 1))) for i in range(n_eval)
        )
    else:
        eval_indices = {0}

    train_pairs = [p for i, p in enumerate(pairs) if i not in eval_indices]
    eval_pairs = [p for i, p in enumerate(pairs) if i in eval_indices]
    print(f"Train frames: {len(train_pairs)}")
    print(f"Eval  frames: {len(eval_pairs)}")
    print(f"Total       : {len(train_pairs) + len(eval_pairs)}")

    if args.verbose:
        print("\nTrain frame indices:", [p.frame_idx for p in train_pairs])
        print("Eval  frame indices:", [p.frame_idx for p in eval_pairs])

    # --- Nerfstudio DataParser test (requires torch) ---
    try:
        from nerfstudio.data.dataparsers.jai_dataparser import (
            JaiDataParserConfig,
            JaiDataParser,
        )
        print(f"\n=== Nerfstudio JaiDataParser test ===")
        config = JaiDataParserConfig(
            data=scene_dir,
            effective_bits=args.effective_bits,
            file_ext=args.file_ext,
            eval_fraction=args.eval_fraction,
        )
        dp = JaiDataParser(config)
        train_out = dp.get_dataparser_outputs("train")
        eval_out = dp.get_dataparser_outputs("val")
        print(f"  DataparserOutputs created successfully")
        print(f"  Train image_filenames: {len(train_out.image_filenames)}")
        print(f"  Eval  image_filenames: {len(eval_out.image_filenames)}")
        print(f"  Metadata keys: {sorted(train_out.metadata.keys())}")
        cams = train_out.cameras
        print(f"  Cameras: n={len(cams)}, "
              f"fx={cams.fx[0].item():.1f}, fy={cams.fy[0].item():.1f}, "
              f"cx={cams.cx[0].item():.1f}, cy={cams.cy[0].item():.1f} (placeholders)")
    except ImportError:
        print(f"\n  (torch not available — skipping JaiDataParser test)")

    # --- transforms.json check ---
    transforms_path = scene_dir / "transforms.json"
    transforms = load_transforms_json(transforms_path)
    if transforms is not None:
        print(f"\n=== transforms.json found ===")
        print(f"  Keys: {list(transforms.keys())}")
        if "frames" in transforms:
            print(f"  Frames: {len(transforms['frames'])}")
    else:
        print(f"\n  (no transforms.json found — expected at this stage)")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
