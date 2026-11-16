"""Estimate black level from one or more covered-cap JAI raw captures.

Covered-cap = lens cap on, no light, so the DN response is pure sensor dark
current + read noise.  Use this to populate the ``black_level.json`` consumed
by ``jai_compute_reflectance.py`` and (optionally) the dataparser's
``bayer_black_level`` / ``nir_black_level`` knobs.

Output JSON::

    {
      "source0_rggb": [bl_R, bl_G1, bl_G2, bl_B],   // per-Bayer-site median DN
      "source1":      bl_nir,                         // scalar median DN
      "n_frames":     N
    }

Example::

    python scripts/jai_estimate_black_level.py \
        --scene data/JAI/4.9.2026/12.cucum1/black \
        --out   data/JAI/4.9.2026/12.cucum1/black/black_level.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerfstudio.utils.bayer_utils import bayer_masks_rggb4


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    import re
    m = re.match(r"^Source\d+_\d+_(\d+)x(\d+)_\d+b\.raw$", path.name)
    if not m:
        raise ValueError(f"Cannot parse filename dims: {path.name}")
    w, h = int(m.group(1)), int(m.group(2))
    return np.fromfile(path, dtype=np.dtype("<u2")).reshape(h, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--file-ext", choices=("raw", "npy"), default="raw")
    args = ap.parse_args()

    s0_files = sorted(args.scene.glob(f"Source0_*.{args.file_ext}"))
    s1_files = sorted(args.scene.glob(f"Source1_*.{args.file_ext}"))
    if not s0_files or not s1_files:
        raise SystemExit("Need both Source0_* and Source1_* dark frames.")

    # --- Source0 per-Bayer-channel ---
    s0_stack = np.stack([_load(p).astype(np.float32) for p in s0_files], axis=0)
    h, w = s0_stack.shape[1], s0_stack.shape[2]
    r_m, g1_m, g2_m, b_m = bayer_masks_rggb4(h, w)
    bl_r  = float(np.median(s0_stack[:, r_m]))
    bl_g1 = float(np.median(s0_stack[:, g1_m]))
    bl_g2 = float(np.median(s0_stack[:, g2_m]))
    bl_b  = float(np.median(s0_stack[:, b_m]))

    # --- Source1 scalar ---
    s1_stack = np.stack([_load(p).astype(np.float32) for p in s1_files], axis=0)
    bl_nir = float(np.median(s1_stack))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source0_rggb": [bl_r, bl_g1, bl_g2, bl_b],
        "source1": bl_nir,
        "n_frames": int(len(s0_files)),
    }, indent=2))
    print(f"Wrote {args.out}")
    print(f"  S0 (R,G1,G2,B) = ({bl_r:.1f}, {bl_g1:.1f}, {bl_g2:.1f}, {bl_b:.1f})")
    print(f"  S1 NIR         = {bl_nir:.1f}")


if __name__ == "__main__":
    main()
