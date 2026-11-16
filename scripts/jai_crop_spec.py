"""Save, load, and convert the Nerfstudio viewer crop region.

The Nerfstudio viewer's built-in crop box is an Oriented Bounding Box (OBB)
defined by three parameters in the viewer's normalized world coordinates:

- **center** (T): ``[x, y, z]`` — center position of the box.
- **rotation** (RPY): ``[roll, pitch, yaw]`` in **radians** — Euler angles
  that build the 3×3 rotation matrix via ``SO3.from_rpy_radians()``.
- **scale** (S): ``[sx, sy, sz]`` — half-extents along each (rotated) axis.

The viewer's export panel (visible in the web UI under Export → Point Cloud)
already prints these three parameters as CLI flags::

    --obb_center 0.123 0.456 0.789 --obb_rotation 0.0 0.1 0.0 --obb_scale 1.2 1.2 1.0

This script provides a bridge between that representation and a persistent
JSON file, so you can:

1. Copy the OBB values from the viewer export panel.
2. Save them to ``crop.json`` using this script.
3. Reuse the same ``crop.json`` for exports on a DIFFERENT trained run
   (e.g. Path A = DN, Path B = reflectance) — both runs share the same
   ``poses_source``, so their normalized coordinate systems are identical,
   and the crop spec selects the exact same region in 3-D space.

JSON format
-----------
::

    {
      "crop_center": [0.123, 0.456, 0.789],
      "crop_rotation": [0.0, 0.1, 0.0],
      "crop_scale": [1.2, 1.2, 1.0],
      "comment": "Crop around the main plant canopy"
    }

``crop_rotation`` is ``[roll, pitch, yaw]`` in radians, matching the
viewer's internal representation and the ``--obb_rotation`` CLI flag on
``ns-export pointcloud`` / ``ns-export tsdf``.

Usage
-----
::

    # Save from viewer OBB values → JSON
    python scripts/jai_crop_spec.py save \
        --center   0.123  0.456  0.789 \
        --rotation 0.0    0.1    0.0   \
        --scale    1.2    1.2    1.0   \
        --comment  "Main plant canopy" \
        --out      crop.json

    # Print the ns-export CLI fragment from a saved JSON
    python scripts/jai_crop_spec.py to-cli --crop-json crop.json
    # → --obb_center 0.1230000000 0.4560000000 0.7890000000 \
    #   --obb_rotation 0.0000000000 0.1000000000 0.0000000000 \
    #   --obb_scale 1.2000000000 1.2000000000 1.0000000000

    # Dump a human-readable summary
    python scripts/jai_crop_spec.py show --crop-json crop.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def load_crop_spec(path: Path) -> Dict:
    """Read a crop-spec JSON and return the dict with validated fields."""
    data = json.loads(path.read_text())
    for key in ("crop_center", "crop_rotation", "crop_scale"):
        if key not in data:
            raise ValueError(f"crop-spec JSON is missing required key '{key}'.")
        val = data[key]
        if not isinstance(val, list) or len(val) != 3:
            raise ValueError(f"'{key}' must be a 3-element list, got: {val}")
    return data


def save_crop_spec(
    path: Path,
    center: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    comment: Optional[str] = None,
) -> None:
    d = {
        "crop_center": list(center),
        "crop_rotation": list(rotation),
        "crop_scale": list(scale),
    }
    if comment:
        d["comment"] = comment
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2) + "\n")


def crop_spec_to_cli(data: Dict) -> str:
    """Convert a loaded crop-spec dict to the CLI flag string for ns-export."""
    c = data["crop_center"]
    r = data["crop_rotation"]
    s = data["crop_scale"]
    c_str = " ".join(f"{v:.10f}" for v in c)
    r_str = " ".join(f"{v:.10f}" for v in r)
    s_str = " ".join(f"{v:.10f}" for v in s)
    return f"--obb_center {c_str} --obb_rotation {r_str} --obb_scale {s_str}"


def crop_spec_to_oriented_box(data: Dict):
    """Build a nerfstudio OrientedBox from a crop-spec dict."""
    from nerfstudio.data.scene_box import OrientedBox
    return OrientedBox.from_params(
        tuple(data["crop_center"]),
        tuple(data["crop_rotation"]),
        tuple(data["crop_scale"]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_save(args) -> int:
    save_crop_spec(
        args.out,
        center=tuple(args.center),
        rotation=tuple(args.rotation),
        scale=tuple(args.scale),
        comment=args.comment,
    )
    print(f"Saved crop spec → {args.out}")
    return 0


def _cmd_to_cli(args) -> int:
    data = load_crop_spec(args.crop_json)
    print(crop_spec_to_cli(data))
    return 0


def _cmd_show(args) -> int:
    data = load_crop_spec(args.crop_json)
    print(f"  center:   {data['crop_center']}")
    print(f"  rotation: {data['crop_rotation']}  (RPY radians)")
    print(f"  scale:    {data['crop_scale']}")
    if "comment" in data:
        print(f"  comment:  {data['comment']}")
    print()
    print("ns-export CLI fragment:")
    print(f"  {crop_spec_to_cli(data)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Save / load / convert the Nerfstudio viewer crop region.",
    )
    sub = ap.add_subparsers(dest="cmd")

    # save
    sp_save = sub.add_parser("save", help="Save OBB values to a crop-spec JSON.")
    sp_save.add_argument("--center",   type=float, nargs=3, required=True, metavar=("X","Y","Z"))
    sp_save.add_argument("--rotation", type=float, nargs=3, required=True, metavar=("R","P","Y"),
                         help="Roll, pitch, yaw in radians.")
    sp_save.add_argument("--scale",    type=float, nargs=3, required=True, metavar=("SX","SY","SZ"))
    sp_save.add_argument("--comment",  type=str, default=None)
    sp_save.add_argument("--out",      type=Path, required=True)

    # to-cli
    sp_cli = sub.add_parser("to-cli", help="Print ns-export CLI flags from a crop JSON.")
    sp_cli.add_argument("--crop-json", type=Path, required=True)

    # show
    sp_show = sub.add_parser("show", help="Print a human-readable summary.")
    sp_show.add_argument("--crop-json", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return 0
    if args.cmd == "save":
        return _cmd_save(args)
    elif args.cmd == "to-cli":
        return _cmd_to_cli(args)
    elif args.cmd == "show":
        return _cmd_show(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
