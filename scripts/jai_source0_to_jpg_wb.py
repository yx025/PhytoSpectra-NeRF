"""Batch convert JAI Source0 Bayer raw files to white-balanced JPGs.

Visualization / preprocessing utility ONLY. Does not modify the training
pipeline, dataparser, model, or reflectance processing.

Reads headerless ``Source0_<frame>_<W>x<H>_16b.raw`` (or ``.npy``) files,
normalizes raw DN, applies a per-Bayer-channel white balance, demosaics to
RGB, applies exposure + gamma, and writes 8-bit JPGs.

Pipeline (each frame)
---------------------
1.  Load raw uint16 mosaic via ``jai_utils.load_jai_raw``.
2.  Optional black-level subtraction (scalar DN).
3.  Normalize to float32 [0, 1] using the effective 12-bit range.
4.  Compute per-channel WB gains (``none``/``grayworld``/``panel``).
5.  Apply gains to the *mosaic* (each CFA site multiplied by its channel's gain).
6.  Demosaic (OpenCV ``COLOR_BayerBG2RGB`` for a RGGB CFA; falls back to a
    2x2 bin if OpenCV is unavailable).
7.  Exposure scale + gamma (default 2.2) + clip to [0, 1].
8.  Quantize to uint8 and save as JPG.

White-balance math
------------------
Gains are computed as ``gain_c = target_c / mean_c`` where ``mean_c`` is the
mean of raw-normalized pixels at CFA sites of channel ``c``.

- ``none``     → gains = (1, 1, 1)
- ``grayworld`` → target = mean(G); gain_r = meanG/meanR, gain_b = meanG/meanB,
                 gain_g = 1. Uses the full image.
- ``panel``    → as above but means are computed over the panel ROI from
                 ``panel_roi.json`` (key ``source0``). Gains are normalized so
                 the green gain equals 1 (neutral panel assumption; this is a
                 visual white balance, NOT a physically calibrated reflectance).

Example
-------
    python scripts/jai_source0_to_jpg_wb.py \
   --input-dir data/JAI/4.9.2026/12.cucum1 \
   --output-dir data/JAI/4.9.2026/12.cucum1/jpg \
   --input-ext raw \
   --wb-mode panel \
   --panel-roi data/JAI/4.9.2026/12.cucum1/panel_roi.json \
   --gamma 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path when script is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nerfstudio.process_data.jai_utils import (  # noqa: E402
    DEFAULT_EFFECTIVE_BITS,
    load_jai_raw,
    normalize_raw,
)
from nerfstudio.utils.bayer_utils import bayer_masks_rggb4  # noqa: E402

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------

def _channel_means_rggb(
    mosaic: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float, float]:
    """Return (R, G, B) means of a normalized Bayer mosaic.

    Green combines G1 and G2. ``roi`` is ``(x0, y0, x1, y1)`` inclusive-exclusive.
    """
    if roi is not None:
        x0, y0, x1, y1 = roi
        # Snap to even indices so the RGGB tile alignment is preserved.
        x0 &= ~1
        y0 &= ~1
        sub = mosaic[y0:y1, x0:x1]
    else:
        sub = mosaic
    h, w = sub.shape
    r_m, g1_m, g2_m, b_m = bayer_masks_rggb4(h, w)
    r_mean = float(sub[r_m].mean())
    g_mean = float(sub[g1_m | g2_m].mean())
    b_mean = float(sub[b_m].mean())
    return r_mean, g_mean, b_mean


def compute_wb_gains(
    mosaic_norm: np.ndarray,
    mode: str,
    panel_roi_json: Optional[Path],
) -> Tuple[float, float, float]:
    """Compute (gain_r, gain_g, gain_b) per the selected mode."""
    if mode == "none":
        return 1.0, 1.0, 1.0

    if mode == "grayworld":
        r, g, b = _channel_means_rggb(mosaic_norm)
        eps = 1e-6
        return g / max(r, eps), 1.0, g / max(b, eps)

    if mode == "panel":
        if panel_roi_json is None or not panel_roi_json.exists():
            raise FileNotFoundError(
                f"--wb-mode panel requires --panel-roi pointing to an existing "
                f"JSON file, got: {panel_roi_json}"
            )
        data = json.loads(panel_roi_json.read_text())
        s0 = data.get("source0")
        if not s0:
            raise ValueError(
                f"panel_roi.json missing 'source0' block: {panel_roi_json}"
            )
        rw = int(s0.get("raw_width", mosaic_norm.shape[1]))
        rh = int(s0.get("raw_height", mosaic_norm.shape[0]))
        if (rh, rw) != mosaic_norm.shape:
            raise ValueError(
                f"panel_roi.json source0 raw size ({rw}x{rh}) does not match "
                f"current frame ({mosaic_norm.shape[1]}x{mosaic_norm.shape[0]})."
            )
        roi = (int(s0["x0"]), int(s0["y0"]), int(s0["x1"]), int(s0["y1"]))
        r, g, b = _channel_means_rggb(mosaic_norm, roi=roi)
        target = data.get("panel_reflectance_bayer", [1.0, 1.0, 1.0])
        tr, tg, tb = float(target[0]), float(target[1]), float(target[2])
        eps = 1e-6
        gains = np.array([tr / max(r, eps), tg / max(g, eps), tb / max(b, eps)],
                         dtype=np.float64)
        # Normalize so green gain = 1 (visual WB, not absolute calibration).
        gains /= max(gains[1], eps)
        return float(gains[0]), float(gains[1]), float(gains[2])

    raise ValueError(f"Unknown wb-mode: {mode}")


def apply_wb_to_mosaic(
    mosaic_norm: np.ndarray,
    gains: Tuple[float, float, float],
) -> np.ndarray:
    """Scale each CFA site in the mosaic by its channel's gain."""
    h, w = mosaic_norm.shape
    r_m, g1_m, g2_m, b_m = bayer_masks_rggb4(h, w)
    out = mosaic_norm.astype(np.float32, copy=True)
    gr, gg, gb = gains
    out[r_m] *= gr
    out[g1_m] *= gg
    out[g2_m] *= gg
    out[b_m] *= gb
    return out


# ---------------------------------------------------------------------------
# Demosaic
# ---------------------------------------------------------------------------

def demosaic_rggb(mosaic_norm: np.ndarray) -> np.ndarray:
    """Demosaic a normalized RGGB Bayer mosaic to a float32 RGB image.

    Uses OpenCV's edge-aware demosaicing on a uint16 version of the mosaic.
    Falls back to a 2x2 bin (half resolution, nearest-upsampled back) if
    OpenCV is not importable.
    """
    m = np.clip(mosaic_norm, 0.0, None)
    peak = float(m.max()) if m.size else 1.0
    scale = 65535.0 / peak if peak > 0 else 1.0

    if _HAS_CV2:
        u16 = np.clip(m * scale, 0, 65535).astype(np.uint16)
        # RGGB Bayer in OpenCV's naming convention is COLOR_BayerBG2RGB.
        rgb_u16 = cv2.cvtColor(u16, cv2.COLOR_BayerBG2RGB)
        return rgb_u16.astype(np.float32) / scale

    # Fallback: coarse 2x2 binning then nearest upsample.
    h, w = m.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    m2 = m[:h2, :w2]
    r = m2[0::2, 0::2]
    g = 0.5 * (m2[0::2, 1::2] + m2[1::2, 0::2])
    b = m2[1::2, 1::2]
    small = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    if rgb.shape[:2] != (h, w):
        pad_h = h - rgb.shape[0]
        pad_w = w - rgb.shape[1]
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return rgb


# ---------------------------------------------------------------------------
# Display rendering
# ---------------------------------------------------------------------------

def render_to_uint8(
    rgb: np.ndarray,
    exposure: float,
    gamma: float,
) -> np.ndarray:
    """Exposure + gamma + clip + quantize to uint8."""
    out = rgb * float(exposure)
    np.clip(out, 0.0, 1.0, out=out)
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        out = np.power(out, 1.0 / gamma)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_clahe(rgb_u8: np.ndarray, clip: float, tiles: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on luminance.

    Operates on the L channel in LAB space so chroma is untouched.  This adds
    *local* contrast — pulling texture out of dark, low-contrast regions (e.g. a
    black tarp background) for COLMAP feature matching — without the foggy,
    washed-out look that a global exposure/gamma lift produces.  Requires
    OpenCV; if unavailable the input is returned unchanged.
    """
    if clip <= 0 or not _HAS_CV2:
        return rgb_u8
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def save_jpg(path: Path, rgb_u8: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CV2:
        bgr = rgb_u8[..., ::-1]
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return
    # Fallback using Pillow.
    from PIL import Image  # local import; only used if cv2 missing
    Image.fromarray(rgb_u8, mode="RGB").save(path, format="JPEG", quality=int(quality))


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _list_source0_files(input_dir: Path, ext_mode: str) -> list[Path]:
    if ext_mode == "auto":
        exts = ("raw", "npy")
    else:
        exts = (ext_mode,)
    files: list[Path] = []
    for e in exts:
        files.extend(sorted(input_dir.glob(f"Source0_*.{e}")))
    # De-dup while preserving order.
    seen: Dict[Path, None] = {}
    for f in files:
        seen.setdefault(f, None)
    return list(seen.keys())


def process_one(
    path: Path,
    output_dir: Path,
    wb_mode: str,
    panel_roi_json: Optional[Path],
    black_level: float,
    exposure: float,
    gamma: float,
    clahe_clip: float,
    clahe_tiles: int,
    overwrite: bool,
    jpg_quality: int,
) -> Optional[Path]:
    out_path = output_dir / (path.stem + ".jpg")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} (exists; use --overwrite)")
        return out_path

    raw, meta = load_jai_raw(path)
    if meta.source != "Source0":
        return None

    # Black level + normalize to [0, 1] (12-bit effective range by default).
    white_level = float(2 ** meta.effective_bits - 1)
    mosaic = normalize_raw(
        raw, bit_depth=meta.effective_bits,
        black_level=float(black_level), white_level=white_level,
    )

    gains = compute_wb_gains(mosaic, wb_mode, panel_roi_json)
    mosaic_wb = apply_wb_to_mosaic(mosaic, gains)
    rgb = demosaic_rggb(mosaic_wb)
    rgb_u8 = render_to_uint8(rgb, exposure=exposure, gamma=gamma)
    rgb_u8 = apply_clahe(rgb_u8, clip=clahe_clip, tiles=clahe_tiles)
    save_jpg(out_path, rgb_u8, quality=jpg_quality)

    clahe_str = f" clahe={clahe_clip:.1f}/{clahe_tiles}" if clahe_clip > 0 else ""
    print(
        f"[ok]   {out_path.name}  "
        f"wb={wb_mode} gains=({gains[0]:.3f},{gains[1]:.3f},{gains[2]:.3f}) "
        f"exp={exposure:.2f} gamma={gamma:.2f}{clahe_str}"
    )
    return out_path



# ---------------------------------------------------------------------------
# Built-in defaults for the common 12.cucum1 workflow.
#
# These reproduce exactly the command I run most often:
#
#     python scripts/jai_source0_to_jpg_wb.py \
#         --input-dir  data/JAI/4.9.2026/12.cucum1 \
#         --output-dir data/JAI/4.9.2026/12.cucum1/jpg \
#         --input-ext  raw \
#         --wb-mode    panel \
#         --panel-roi  data/JAI/4.9.2026/12.cucum1/panel_roi.json \
#         --gamma      1.5
#
# Any of these can still be overridden on the command line.  ``--output-dir``
# and ``--panel-roi`` default to ``<input-dir>/jpg`` and
# ``<input-dir>/panel_roi.json`` when not given, so moving to a new scene
# only requires passing ``--input-dir``.
# ---------------------------------------------------------------------------
_DEFAULT_INPUT_DIR = Path("data/JAI/4.9.2026/12.cucum1")
_DEFAULT_INPUT_EXT = "raw"
_DEFAULT_WB_MODE = "panel"
_DEFAULT_GAMMA = 1.5


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-convert JAI Source0 Bayer raws to white-balanced JPGs.  "
            "All flags have baked-in defaults for the 12.cucum1 workflow; "
            "pass any flag on the CLI to override.  Run with no args to use "
            "the defaults."
        ),
    )
    ap.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR,
                    help=f"Directory containing Source0_*.raw or .npy files "
                         f"(default: {_DEFAULT_INPUT_DIR}).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory (default: <input-dir>/jpg).")
    ap.add_argument("--input-ext", choices=("raw", "npy", "auto"),
                    default=_DEFAULT_INPUT_EXT,
                    help=f"Extension to scan for (default: {_DEFAULT_INPUT_EXT}).")
    ap.add_argument("--wb-mode", choices=("none", "grayworld", "panel"),
                    default=_DEFAULT_WB_MODE,
                    help=f"White balance mode (default: {_DEFAULT_WB_MODE}).")
    ap.add_argument("--panel-roi", type=Path, default=None,
                    help="panel_roi.json path.  Required when --wb-mode=panel. "
                         "Defaults to <input-dir>/panel_roi.json.")
    ap.add_argument("--black-level", type=float, default=0.0,
                    help="Scalar black-level DN to subtract (default 0). "
                         "For per-Bayer-channel black level use the reflectance "
                         "pipeline; this is a visualization tool.")
    ap.add_argument("--exposure", type=float, default=1.0,
                    help="Linear exposure multiplier applied after WB (default 1.0).")
    ap.add_argument("--gamma", type=float, default=_DEFAULT_GAMMA,
                    help=f"Display gamma applied before quantization "
                         f"(default {_DEFAULT_GAMMA}).")
    ap.add_argument("--clahe-clip", type=float, default=0.0,
                    help="CLAHE clip limit on luminance (0 = off, default). "
                         "Adds local contrast to reveal dark-region texture for "
                         "COLMAP without the foggy look of a global brightness "
                         "lift; try 3.0. Requires OpenCV.")
    ap.add_argument("--clahe-tiles", type=int, default=8,
                    help="CLAHE tile grid size (default 8 => 8x8 tiles).")
    ap.add_argument("--jpg-quality", type=int, default=95,
                    help="JPEG quality 1-100 (default 95).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing JPGs.")
    args = ap.parse_args()

    # Auto-derive output_dir and panel_roi from input_dir when not given.
    if args.output_dir is None:
        args.output_dir = args.input_dir / "jpg"
    if args.panel_roi is None:
        args.panel_roi = args.input_dir / "panel_roi.json"

    if not args.input_dir.is_dir():
        print(f"error: input dir does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    files = _list_source0_files(args.input_dir, args.input_ext)
    if not files:
        print(f"error: no Source0 files found under {args.input_dir} "
              f"(ext={args.input_ext})", file=sys.stderr)
        return 1

    if not _HAS_CV2:
        print("[warn] OpenCV not available; using fallback 2x2-bin demosaic "
              "(half-resolution quality). Install opencv-python for best results.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    n_err = 0
    for p in files:
        try:
            out = process_one(
                p,
                output_dir=args.output_dir,
                wb_mode=args.wb_mode,
                panel_roi_json=args.panel_roi,
                black_level=args.black_level,
                exposure=args.exposure,
                gamma=args.gamma,
                clahe_clip=args.clahe_clip,
                clahe_tiles=args.clahe_tiles,
                overwrite=args.overwrite,
                jpg_quality=args.jpg_quality,
            )
            if out is None:
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"[err]  {p.name}: {e}", file=sys.stderr)

    print(
        f"\nSummary: {n_ok} written, {n_skip} skipped, {n_err} failed. "
        f"Output: {args.output_dir}"
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
