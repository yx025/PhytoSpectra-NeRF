#!/usr/bin/env python
"""Flag (and optionally exclude) bad frames of a JAI scene after COLMAP.

A frame is flagged when any of these hold:

* ``unposed``   — COLMAP did not register it (absent from transforms.json).
  These MUST go: the JAI dataparser gives unmatched raw frames an identity
  pose and still trains on them.
* ``few_pts``   — it observes < ``--min-pts-frac`` x the scene-median number of
  triangulated points (weak registration).
* ``reproj``    — mean reprojection error of its points is a robust outlier.
* ``radius``    — camera distance to the scene centre is a robust outlier
  (off-orbit / mis-registered position).
* ``aim``       — its optical axis misses the scene centre by an outlier amount
  (mis-registered rotation).
* ``blur``      — Laplacian sharpness < ``--blur-frac`` x the median of its
  temporal neighbours (motion blur).

Excluding a frame = moving its Source0 + Source1 raw files from the scene root
into ``<scene>/excluded_frames/``.  The dataparser only globs the scene root
and reads the frame index from the filename, so the frame simply disappears
from training.  ``jpg/`` and ``colmap_out/`` are NOT touched, so the
``frame_K.jpg -> raw idx K-1`` mapping used by the dataparser stays valid and
COLMAP never needs re-running.  ``--restore`` moves everything back.

Usage
-----
    python scripts/jai_flag_bad_frames.py --scene data/JAI/5.12/1.soybean3_up            # dry run
    python scripts/jai_flag_bad_frames.py --scene data/JAI/5.12/1.soybean3_up --apply
    python scripts/jai_flag_bad_frames.py --scene data/JAI/5.12/1.soybean3_up --restore
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, read_points3D_binary

EXCLUDE_DIR = "excluded_frames"
MANIFEST = "excluded.json"
_FRAME_RE = re.compile(r"frame_(\d+)\.\w+$")
_RAW_RE = re.compile(r"^Source0_(\d+)_")


def _robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); 0 where MAD is degenerate."""
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / mad if mad > 1e-12 else np.zeros_like(x)


def _sharpness(jpg: Path, max_side: int = 1024) -> float:
    img = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
    s = max_side / max(img.shape)
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def analyze(scene: Path, args) -> List[Dict]:
    out_dir = scene / "colmap_out"
    jpgs = sorted((scene / "jpg").glob("Source0_*.jpg"))
    raw_idx = [int(_RAW_RE.match(p.name).group(1)) for p in jpgs]
    if raw_idx != list(range(len(raw_idx))):
        raise SystemExit(f"{scene}: jpg frame indices are not contiguous from 0; "
                         "the dataparser's frame_K -> idx K-1 mapping would be wrong.")
    n = len(jpgs)
    rows = [{"idx": i, "reasons": []} for i in range(n)]

    # --- poses from transforms.json (what the dataparser actually uses) ---
    tj = json.loads((out_dir / "transforms.json").read_text())
    pose_by_idx = {int(_FRAME_RE.search(f["file_path"]).group(1)) - 1: f["transform_matrix"]
                   for f in tj["frames"]}
    posed = sorted(pose_by_idx)
    for r in rows:
        if r["idx"] not in pose_by_idx:
            r["reasons"].append("unposed")

    # --- COLMAP per-image support: #points and their mean track error ---
    sparse = out_dir / "colmap" / "sparse" / "0"
    images = read_images_binary(sparse / "images.bin")
    pts = read_points3D_binary(sparse / "points3D.bin")
    n_pts, err = {}, {}
    for im in images.values():
        k = int(_FRAME_RE.search(im.name).group(1)) - 1
        ids = im.point3D_ids[im.point3D_ids >= 0]
        n_pts[k] = len(ids)
        err[k] = float(np.mean([pts[i].error for i in ids])) if len(ids) else np.inf
    np_arr = np.array([n_pts.get(k, 0) for k in posed], dtype=float)
    er_arr = np.array([err.get(k, np.inf) for k in posed])
    med_pts, med_err = np.median(np_arr), np.median(er_arr[np.isfinite(er_arr)])
    er_z = _robust_z(np.where(np.isfinite(er_arr), er_arr, med_err * 10))

    # --- pose geometry, normalized exactly like the dataparser ---
    poses = torch.tensor([pose_by_idx[k] for k in posed], dtype=torch.float32)
    poses, _ = camera_utils.auto_orient_and_center_poses(poses, method="up", center_method="poses")
    c = poses[:, :3, 3].numpy()
    d = -poses[:, :3, 2].numpy()
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    proj = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    centre = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, c))
    radius = np.linalg.norm(c - centre, axis=1)
    v = centre - c
    miss = np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1) / np.median(radius)
    rad_z, miss_z = _robust_z(radius), _robust_z(miss)

    for j, k in enumerate(posed):
        r = rows[k]
        r.update(n_pts=int(np_arr[j]), reproj=float(er_arr[j]),
                 radius_rel=float(radius[j] / np.median(radius)), aim_miss=float(miss[j]))
        if np_arr[j] < args.min_pts_frac * med_pts:
            r["reasons"].append("few_pts")
        if er_z[j] > args.z and er_arr[j] > 1.5 * med_err:
            r["reasons"].append("reproj")
        if abs(rad_z[j]) > args.z and abs(radius[j] / np.median(radius) - 1) > 0.15:
            r["reasons"].append("radius")
        if miss_z[j] > args.z and miss[j] > args.min_aim_miss:
            r["reasons"].append("aim")

    # --- motion blur relative to temporal neighbours ---
    sharp = np.array([_sharpness(p) for p in jpgs])
    for i, r in enumerate(rows):
        nb = np.r_[sharp[max(0, i - 5):i], sharp[i + 1:i + 6]]
        r["sharp_rel"] = float(sharp[i] / np.median(nb))
        if r["sharp_rel"] < args.blur_frac:
            r["reasons"].append("blur")
    return rows


def _raw_files(scene: Path, idx: int) -> List[Path]:
    return sorted(scene.glob(f"Source[01]_{idx:06d}_*.raw")) + sorted(scene.glob(f"Source[01]_{idx:06d}_*.npy"))


def restore(scene: Path) -> None:
    ex = scene / EXCLUDE_DIR
    if not ex.is_dir():
        print(f"{scene}: nothing to restore")
        return
    moved = [p for p in ex.iterdir() if p.name != MANIFEST]
    for p in moved:
        shutil.move(str(p), str(scene / p.name))
    (ex / MANIFEST).unlink(missing_ok=True)
    ex.rmdir()
    print(f"{scene}: restored {len(moved)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="Move flagged frames into excluded_frames/.")
    ap.add_argument("--restore", action="store_true", help="Move excluded frames back.")
    ap.add_argument("--z", type=float, default=4.0, help="Robust z-score threshold.")
    ap.add_argument("--min-pts-frac", type=float, default=0.25)
    ap.add_argument("--min-aim-miss", type=float, default=0.05,
                    help="Absolute floor on axis miss distance / median cam radius.")
    ap.add_argument("--blur-frac", type=float, default=0.5)
    ap.add_argument("--max-exclude-frac", type=float, default=0.15,
                    help="Refuse --apply if more than this fraction is flagged (inspect instead).")
    args = ap.parse_args()

    if args.restore:
        restore(args.scene)
        return

    rows = analyze(args.scene, args)
    bad = [r for r in rows if r["reasons"]]
    print(f"{args.scene}: {len(bad)}/{len(rows)} frames flagged")
    for r in bad:
        extra = ", ".join(f"{k}={r[k]:.3g}" for k in ("n_pts", "reproj", "radius_rel", "aim_miss", "sharp_rel") if k in r)
        print(f"  {r['idx']:06d}  {'+'.join(r['reasons']):<20s} {extra}")

    if not args.apply:
        return
    if (args.scene / EXCLUDE_DIR / MANIFEST).exists():
        raise SystemExit("excluded_frames/ already applied; run --restore first to re-apply.")
    if len(bad) > args.max_exclude_frac * len(rows):
        raise SystemExit(f"{len(bad)} flagged > {args.max_exclude_frac:.0%} of frames — not applying; inspect this scene.")
    ex = args.scene / EXCLUDE_DIR
    ex.mkdir(exist_ok=True)
    for r in bad:
        r["files"] = [p.name for p in _raw_files(args.scene, r["idx"])]
        for name in r["files"]:
            shutil.move(str(args.scene / name), str(ex / name))
    (ex / MANIFEST).write_text(json.dumps({"thresholds": vars(args) | {"scene": str(args.scene)}, "frames": bad},
                                          indent=2, default=str))
    print(f"  moved {sum(len(r['files']) for r in bad)} raw files -> {ex}")


if __name__ == "__main__":
    main()
