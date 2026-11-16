"""Compute reflectance .npy files for a full JAI scene directory (remote-safe).

This is the headless counterpart to ``jai_pick_panel_roi_local.py``:

1. Read a ``panel_roi.json`` produced locally.
2. Optionally read a ``black_level.json`` (per-Bayer-channel + NIR).
3. For every Source0 / Source1 file in the scene directory, compute a
   float32 reflectance image in ``[0, 1]`` and save it as .npy with the
   SAME filename pattern the JaiDataParser already understands.

The filename convention is preserved so the reflectance files can be used
directly by training:

::

    ns-train jai-nerfacto \
        --pipeline.datamanager.dataparser.data          /scene/reflectance \
        --pipeline.datamanager.dataparser.file-ext      npy \
        --pipeline.datamanager.dataparser.intensity-mode reflectance \
        --pipeline.datamanager.dataparser.poses-source  /path/to/colmap_out

Reflectance formulas
--------------------
For NIR (single scalar panel value ``R_panel``)::

    reflectance(x) = ((DN(x) - DN_dark) / (DN_panel - DN_dark + eps)) * R_panel

For Bayer raw — each pixel uses the denominator of its own Bayer channel
(R, G1, G2, B).  We take the panel statistic CHANNEL-WISE on the raw mosaic::

    DN_panel[c] = median(DN[roi, mask_c])            # c ∈ {R,G1,G2,B}
    DN_dark[c]  = per-channel black level (or 0)
    reflectance(x, c) = ((DN(x) - DN_dark[c]) / (DN_panel[c] - DN_dark[c] + eps)) * R_panel[c]

The output is still a single 2-D mosaic — the model learns RGB via the
existing Bayer-select loss.  Green pixels (G1 & G2) use the same nominal
``R_panel`` for green but their own measured panel DN.

Coordinate mapping
------------------
The JSON ROI is in RAW pixel coordinates (see ``jai_pick_panel_roi_local.py``).
If the stored ``raw_width`` / ``raw_height`` match the current file, the ROI
is used directly; otherwise coordinates are scaled by the ratio, which
handles the "preview was resized" case.

Values are clipped to ``[0, clip_max]`` (default 1.2) to keep specular
highlights from exploding, then saved as float32.

python scripts/jai_compute_reflectance.py \
  --scene data/JAI/4.9.2026/12.cucum1 \
  --out data/JAI/4.9.2026/12.cucum1/reflectance \
  --panel-roi data/JAI/4.9.2026/12.cucum1/panel_roi.json \
  --black-level data/JAI/4.9.2026/12.cucum1/black/black_level.json \
  --file-ext raw



NIR:
reflectance(x) = ((DN(x) - DN_dark) / (DN_panel - DN_dark + eps)) * R_panel

Bayer:
reflectance(x,c) = ((DN(x) - DN_dark[c]) / (DN_panel[c] - DN_dark[c] + eps)) * R_panel[c]
"""



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# This script is intentionally self-contained: no torch, and bayer_utils is the
# only in-repo import so it can run on a minimal SSH Python env.
from nerfstudio.utils.bayer_utils import bayer_masks_rggb4


EPS = 1e-6


_WARNED_98 = False


def _warn_98_rule(img: np.ndarray) -> None:
    """WARNING only (behaviour unchanged, 2026-10-27): the panel filter keeps pixels < 0.98 * iinfo(dtype).max.
    JAI raws are 12-bit values stored low-aligned in uint16 (max 4095), so 0.98 * 65535 = 64224 can never be reached and
    the filter never removes a pixel; the panel statistic is the median of ALL panel ROI pixels, saturated ones included.
    See paper/second_experiment/01_nir_calib_check/RESULTS.md (section A)."""
    global _WARNED_98
    if _WARNED_98 or not np.issubdtype(img.dtype, np.integer):
        return
    if int(img.max()) < 0.98 * np.iinfo(img.dtype).max:
        import warnings
        warnings.warn(
            "jai_compute_reflectance: the 98 %% saturation filter (0.98 * %d) cannot trigger for this data "
            "(image max %d, 12-bit low-aligned); the panel median uses all ROI pixels including saturated ones (>= 4080)."
            % (np.iinfo(img.dtype).max, int(img.max())), RuntimeWarning, stacklevel=3)
        _WARNED_98 = True


def _scale_roi(roi: dict, cur_w: int, cur_h: int) -> Tuple[int, int, int, int]:
    raw_w = int(roi.get("raw_width", cur_w))
    raw_h = int(roi.get("raw_height", cur_h))
    sx = cur_w / raw_w
    sy = cur_h / raw_h
    x0 = int(round(roi["x0"] * sx))
    x1 = int(round(roi["x1"] * sx))
    y0 = int(round(roi["y0"] * sy))
    y1 = int(round(roi["y1"] * sy))
    x0, x1 = max(0, min(x0, x1)), min(cur_w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(cur_h, max(y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"Panel ROI too small after scaling: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _load_raw_or_npy(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    # Headerless uint16 .raw; dimensions come from filename.
    import re
    m = re.match(r"^Source\d+_\d+_(\d+)x(\d+)_\d+b\.raw$", path.name)
    if not m:
        raise ValueError(f"Cannot infer dimensions from filename: {path.name}")
    w, h = int(m.group(1)), int(m.group(2))
    arr = np.fromfile(path, dtype=np.dtype("<u2"))
    if arr.size != w * h:
        raise ValueError(f"Size mismatch for {path.name}: expected {w*h}, got {arr.size}")
    return arr.reshape((h, w))


def compute_nir_reflectance(
    img: np.ndarray,
    panel_roi: dict,
    panel_reflectance: float,
    dn_dark: float,
    clip_max: float = 1.2,
) -> np.ndarray:
    """NIR reflectance for a full image using a single panel ROI."""
    h, w = img.shape
    x0, y0, x1, y1 = _scale_roi(panel_roi, w, h)
    _warn_98_rule(img)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    # Drop near-saturated pixels if the image is integer.
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        roi = roi[roi < 0.98 * info.max]
    if roi.size == 0:
        raise ValueError("NIR panel ROI is empty after saturation filtering.")
    dn_panel = float(np.median(roi))
    denom = max(dn_panel - dn_dark, EPS)
    out = (img.astype(np.float32) - dn_dark) / denom * panel_reflectance
    return np.clip(out, 0.0, clip_max).astype(np.float32)


def compute_bayer_reflectance(
    img: np.ndarray,
    panel_roi: dict,
    panel_reflectance_rgb: Tuple[float, float, float],
    bl_rggb: Tuple[float, float, float, float],
    clip_max: float = 1.2,
) -> np.ndarray:
    """Per-Bayer-channel reflectance on a raw mosaic (R, G1, G2, B)."""
    h, w = img.shape
    r_m, g1_m, g2_m, b_m = bayer_masks_rggb4(h, w)
    x0, y0, x1, y1 = _scale_roi(panel_roi, w, h)

    _warn_98_rule(img)
    img_f = img.astype(np.float32)

    # Restrict masks to the panel ROI
    roi_mask = np.zeros_like(r_m)
    roi_mask[y0:y1, x0:x1] = True

    def _panel_stat(site_mask: np.ndarray) -> float:
        vals = img_f[site_mask & roi_mask]
        if np.issubdtype(img.dtype, np.integer):
            vals = vals[vals < 0.98 * np.iinfo(img.dtype).max]
        if vals.size == 0:
            raise ValueError("Bayer panel ROI empty for a channel; pick a larger ROI.")
        return float(np.median(vals))

    dn_r  = _panel_stat(r_m)
    dn_g1 = _panel_stat(g1_m)
    dn_g2 = _panel_stat(g2_m)
    dn_b  = _panel_stat(b_m)

    bl_r, bl_g1, bl_g2, bl_b = bl_rggb
    pr_r, pr_g, pr_b = panel_reflectance_rgb

    def _apply(site_mask, dn_panel, bl, r_panel):
        denom = max(dn_panel - bl, EPS)
        out = (img_f[site_mask] - bl) / denom * r_panel
        return out

    out = np.zeros_like(img_f)
    out[r_m]  = _apply(r_m,  dn_r,  bl_r,  pr_r)
    out[g1_m] = _apply(g1_m, dn_g1, bl_g1, pr_g)
    out[g2_m] = _apply(g2_m, dn_g2, bl_g2, pr_g)
    out[b_m]  = _apply(b_m,  dn_b,  bl_b,  pr_b)

    return np.clip(out, 0.0, clip_max).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute per-frame reflectance from a JAI scene.",
    )
    ap.add_argument("--scene", type=Path, required=True,
                    help="Scene directory with Source0_* / Source1_* files.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory to write reflectance .npy files.")
    ap.add_argument("--panel-roi", type=Path, required=True,
                    help="panel_roi.json from jai_pick_panel_roi_local.py.")
    ap.add_argument("--black-level", type=Path, default=None,
                    help="Optional black_level.json: "
                         '{"source0_rggb":[r,g1,g2,b], "source1":v}')
    ap.add_argument("--file-ext", choices=("raw", "npy"), default="raw",
                    help="Input file extension to scan for.")
    ap.add_argument("--clip-max", type=float, default=1.2)
    args = ap.parse_args()

    roi_spec = json.loads(args.panel_roi.read_text())
    s0_roi = roi_spec.get("source0")
    s1_roi = roi_spec.get("source1")
    if s0_roi is None or s1_roi is None:
        raise SystemExit(
            "panel_roi.json must contain both 'source0' and 'source1'. "
            "Run jai_pick_panel_roi_local.py once per kind."
        )
    panel_ref_nir = float(roi_spec.get("panel_reflectance_nir", 0.2))
    panel_ref_bayer = tuple(roi_spec.get("panel_reflectance_bayer", [0.2, 0.2, 0.2]))

    if args.black_level is not None and args.black_level.exists():
        bl = json.loads(args.black_level.read_text())
        bl_s0 = tuple(bl.get("source0_rggb", [0.0, 0.0, 0.0, 0.0]))
        bl_s1 = float(bl.get("source1", 0.0))
    else:
        bl_s0 = (0.0, 0.0, 0.0, 0.0)
        bl_s1 = 0.0

    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.scene.glob(f"Source*.{args.file_ext}"))
    if not files:
        raise SystemExit(f"No Source*.{args.file_ext} files found in {args.scene}")

    for path in files:
        img = _load_raw_or_npy(path)
        if path.name.startswith("Source0"):
            ref = compute_bayer_reflectance(img, s0_roi, panel_ref_bayer, bl_s0, args.clip_max)
        else:
            ref = compute_nir_reflectance(img, s1_roi, panel_ref_nir, bl_s1, args.clip_max)
        # Save with .npy extension, preserving the parseable filename pattern.
        out_name = path.stem + ".npy"
        np.save(args.out / out_name, ref)
        print(f"  {path.name} → {out_name}  "
              f"(min={ref.min():.3f}, max={ref.max():.3f}, med={np.median(ref):.3f})")

    print(f"\nDone.  Wrote {len(files)} reflectance files → {args.out}")
    print("Train with:")
    print("  ns-train jai-nerfacto \\")
    print(f"    --pipeline.datamanager.dataparser.data {args.out} \\")
    print("    --pipeline.datamanager.dataparser.file-ext npy \\")
    print("    --pipeline.datamanager.dataparser.intensity-mode reflectance")


if __name__ == "__main__":
    main()
