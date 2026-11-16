"""Local panel-ROI picker for JAI reflectance preprocessing.

Run this script LOCALLY (not over SSH) on a machine with a GUI.  It loads ONE
reference raw image, opens a matplotlib window, and lets you click two corners
of the reflectance calibration panel.  The ROI is saved as JSON, which the
remote reflectance pipeline (``jai_compute_reflectance.py``) can then reuse on
every frame of the dataset.

Design notes
------------
- The ROI is saved in **raw pixel coordinates** (full-resolution Bayer mosaic /
  NIR image), NOT in the coordinates of any demosaiced/resized preview.
- If the preview has the same (H, W) as the raw image, picker coordinates are
  already raw coordinates, so no scaling is needed.
- If the preview is resized for display, we scale the picker coordinates back
  by ``(raw_w / preview_w, raw_h / preview_h)``.  This script always displays
  at raw resolution (matplotlib handles the zoom interactively), so the
  coordinates are raw out of the box.
- For Bayer raw, reflectance computation MUST be done on the mosaic, not on a
  demosaiced RGB image, so that per-Bayer-channel panel statistics preserve
  the original signal.  See ``jai_compute_reflectance.py`` for the math.

Usage
-----
::

    # NIR (Source1) panel
    python scripts/jai_pick_panel_roi_local.py \
        --npy /path/to/Source1_000044_2048x1536_16b.npy \
        --kind nir \
        --out /path/to/panel_roi.json

    # Bayer (Source0) panel — same coordinates if the panel is at the same
    # image location, but you can pick a separate ROI if needed.
    python scripts/jai_pick_panel_roi_local.py \
        --npy data/JAI/4.3.2026/cucumber1/Source0_000000_2048x1536_16b.npy \
        --kind bayer \
        --out data/JAI/4.3.2026/cucumber1   # merges into the same file

The output JSON schema looks like::

data    {
      "source0": {"x0": 120, "y0": 340, "x1": 210, "y1": 420,
                  "raw_width": 2048, "raw_height": 1536},
      "source1": {"x0": 118, "y0": 338, "x1": 208, "y1": 418,
                  "raw_width": 2048, "raw_height": 1536},
      "panel_reflectance_nir":   0.52,
      "panel_reflectance_bayer": [0.95, 0.95, 0.95]   # R, G, B
    }

You can edit the panel_reflectance_* values by hand if your calibration panel
has different nominal reflectances per channel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def pick_roi_interactive(img: np.ndarray, title: str) -> Dict[str, int]:
    import matplotlib.pyplot as plt  # local import — only needed in GUI mode
    fig, ax = plt.subplots(figsize=(9, 7))
    # Stretch display for visibility; picks are in data (raw) coordinates.
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title)
    plt.tight_layout()
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise SystemExit("Need exactly two clicks (top-left and bottom-right).")
    (x0, y0), (x1, y1) = pts
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya, yb = sorted((int(round(y0)), int(round(y1))))
    return {"x0": xa, "y0": ya, "x1": xb, "y1": yb}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick a reflectance-panel ROI.")
    ap.add_argument("--npy", type=Path, required=True,
                    help="Path to a 2-D .npy raw image (NIR mono or Bayer mosaic).")
    ap.add_argument("--kind", choices=("nir", "bayer"), required=True,
                    help="Which source this ROI belongs to.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path (merged if it already exists).")
    ap.add_argument("--panel-reflectance-nir", type=float, default=0.52,
                    help="Nominal reflectance of the panel in the NIR band.")
    ap.add_argument("--panel-reflectance-bayer", nargs=3, type=float,
                    default=[0.95, 0.95, 0.95],
                    metavar=("R", "G", "B"),
                    help="Nominal per-channel panel reflectance for visible Bayer.")
    args = ap.parse_args()

    img = np.load(args.npy)
    if img.ndim != 2:
        raise SystemExit(f"Expected 2-D image, got shape {img.shape}")
    h, w = img.shape

    title = (
        f"[{args.kind}] Click TWO points: top-left then bottom-right of PANEL.\n"
        f"Coordinates are saved in RAW pixel space ({w}x{h})."
    )
    roi = pick_roi_interactive(img, title)
    roi.update({"raw_width": int(w), "raw_height": int(h)})

    # Merge into existing JSON if present
    existing: Dict = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text())
    key = "source1" if args.kind == "nir" else "source0"
    existing[key] = roi
    existing.setdefault("panel_reflectance_nir", float(args.panel_reflectance_nir))
    existing.setdefault("panel_reflectance_bayer", list(args.panel_reflectance_bayer))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2))
    print(f"Saved ROI for {key} → {args.out}")
    print(f"  x0,y0,x1,y1 = {roi['x0']},{roi['y0']},{roi['x1']},{roi['y1']}  "
          f"(raw {w}x{h})")


if __name__ == "__main__":
    main()
