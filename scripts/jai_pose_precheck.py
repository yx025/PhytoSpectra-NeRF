#!/usr/bin/env python
"""Pre-flight COLMAP pose-quality check for JAI scenes.

Scans every ``*/colmap_out/transforms.json`` under a root directory and, for
each scene, replicates the JAI dataparser's pose normalization
(``auto_orient_and_center_poses(method="up", center_method="poses")`` followed
by auto-scaling so the max camera translation magnitude is 1.0), then computes a
handful of geometry-consistency metrics.

The key metric is the **ray-convergence residual**: the mean perpendicular
distance from every camera's optical axis to the single least-squares point all
the axes "want" to meet at.  When COLMAP poses are mutually consistent, all
cameras genuinely point at one object centre and this residual is tiny.  When
the reconstruction has drift / mis-registered rotations (even if every frame
"registered"), the axes disagree and the residual blows up — those scenes
reliably collapse into color-block NeRF reconstructions.

Empirical anchors (data/JAI/6.5):
    cucumber2p  -> residual/cam_dist = 0.04  -> trains to a clean plant
    cucumber3p  -> residual/cam_dist = 0.18  -> color blocks

The reported ``risk`` uses the scale-free ratio ``residual / mean_cam_dist`` so
it is comparable across scenes captured at different COLMAP scales.

Usage
-----
    python scripts/jai_pose_precheck.py                 # scans data/JAI
    python scripts/jai_pose_precheck.py --root data/JAI/6.5
    python scripts/jai_pose_precheck.py --csv precheck.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from nerfstudio.cameras import camera_utils

# Scale-free risk thresholds on residual / mean_cam_dist (see anchors above).
OK_MAX = 0.07       # <= OK_MAX            -> LOW    (clean reconstruction expected)
MARGINAL_MAX = 0.12  # OK_MAX..MARGINAL_MAX -> MEDIUM (borderline, inspect)
#                      > MARGINAL_MAX       -> HIGH   (likely color blocks)
SURROUND_MAX = 1.1   # object offset / cam dist above this = one-sided capture;
#                      escalates a borderline (MEDIUM) scene to HIGH.


def _load_poses(tj_path: Path) -> Optional[torch.Tensor]:
    """Return (N, 4, 4) camera_to_world poses, ordered like ns-process-data."""
    with open(tj_path, "r") as f:
        tj = json.load(f)
    frames = tj.get("frames", [])
    if len(frames) < 3:
        return None
    frames = sorted(frames, key=lambda fr: fr["file_path"])
    poses = torch.tensor(
        [fr["transform_matrix"] for fr in frames], dtype=torch.float32
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return None
    return poses


def _convergence(centers: torch.Tensor, fwd: torch.Tensor) -> Optional[torch.Tensor]:
    """Least-squares point closest to all optical-axis rays  o + t * d."""
    eye = torch.eye(3)
    A = torch.zeros(3, 3)
    b = torch.zeros(3)
    for o, d in zip(centers, fwd):
        proj = eye - torch.outer(d, d)                # projector onto plane perp to d
        A += proj
        b += proj @ o
    try:
        return torch.linalg.solve(A, b)
    except RuntimeError:
        return None


def analyze_scene(tj_path: Path) -> Optional[Dict]:
    """Compute pose-consistency metrics for one scene; None if unusable.

    Two failure modes are detected, both of which produce color-block NeRFs:

    * ``incoherent`` — the cameras' optical axes do not meet at one point
      (high ``consistency`` = inlier ray-convergence residual / median camera
      distance).  COLMAP rotations/positions disagree on where the object is.
    * ``outlier``    — one or more cameras are grossly mis-registered far from
      the rest.  The dataparser scales by the *max* translation, so a far
      outlier collapses all real cameras into a microscopic region near the
      origin that the hash grid cannot resolve (``collapse`` = max/median
      camera distance).
    """
    poses = _load_poses(tj_path)
    if poses is None:
        return None
    n = poses.shape[0]

    # Replicate dataparser normalization: orient + center, then scale so the
    # largest camera translation magnitude is 1.0.  (Risk ratios below are
    # scale-free, so the exact scale only matters for the outlier diagnosis.)
    poses_n, _ = camera_utils.auto_orient_and_center_poses(
        poses.clone(), method="up", center_method="poses"
    )
    max_trans = float(poses_n[:, :3, 3].abs().max())
    scale = 1.0 / max_trans if max_trans > 0 else 1.0
    poses_n[:, :3, 3] *= scale

    centers = poses_n[:, :3, 3]                        # (N, 3) camera positions
    fwd = -poses_n[:, :3, 2]                            # (N, 3) forward (-z, OpenGL)
    fwd = fwd / fwd.norm(dim=1, keepdim=True).clamp_min(1e-8)

    dist = centers.norm(dim=1)
    med = float(dist.median())
    collapse = float(dist.max() / med) if med > 0 else float("inf")

    # Drop grossly mis-registered cameras (> 5x the median radius) so the
    # convergence metric reflects the bulk geometry, not one stray pose.
    inlier = dist <= 5.0 * med
    n_outliers = int((~inlier).sum())
    ci, fi = centers[inlier], fwd[inlier]
    if ci.shape[0] < 3:
        ci, fi = centers, fwd                          # degenerate: keep all

    obj = _convergence(ci, fi)
    if obj is None:
        return None
    perp = torch.stack([((obj - o) - ((obj - o) @ d) * d).norm() for o, d in zip(ci, fi)])
    residual = float(perp.mean())

    med_in = float(ci.norm(dim=1).median())
    consistency = residual / med_in if med_in > 0 else float("inf")
    obj_offset = float(obj.norm())
    surround = obj_offset / med_in if med_in > 0 else float("inf")  # ~0 surround, ~1 one-sided

    reasons = []
    if consistency > MARGINAL_MAX:
        reasons.append("incoherent")
    if n_outliers >= 1 or collapse > 5.0:
        reasons.append(f"outlier(x{n_outliers},collapse {collapse:.0f})")
    # One-sided capture (object essentially outside the camera hull) gives weak
    # multiview constraints; on its own it is survivable, but combined with
    # borderline pose coherence it reliably tips a scene into color blocks
    # (anchor: 6.5/soybean1p-250, consist 0.074 + surround 1.18 -> color blocks).
    one_sided = surround > SURROUND_MAX

    if reasons:
        risk = "HIGH"
    elif consistency > OK_MAX:
        risk = "HIGH" if one_sided else "MEDIUM"
        if one_sided:
            reasons.append(f"one-sided(surround {surround:.2f})")
    else:
        risk = "LOW"

    return {
        "n_frames": n,
        "dataparser_scale": scale,
        "residual": residual,
        "median_cam_dist": med_in,
        "consistency": consistency,
        "collapse": collapse,
        "n_outliers": n_outliers,
        "obj_offset": obj_offset,
        "surround": surround,
        "risk": risk,
        "reason": ",".join(reasons) if reasons else ("borderline" if risk == "MEDIUM" else ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/JAI"),
                    help="Directory to scan recursively (default: data/JAI).")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to write a CSV of all results.")
    args = ap.parse_args()

    tj_paths = sorted(args.root.rglob("colmap_out/transforms.json"))
    if not tj_paths:
        print(f"No */colmap_out/transforms.json found under {args.root}")
        return

    rows: List[Dict] = []
    skipped: List[str] = []
    for tj in tj_paths:
        # Scene name = path between root and /colmap_out
        scene = str(tj.parent.parent.relative_to(args.root))
        m = analyze_scene(tj)
        if m is None:
            skipped.append(scene)
            continue
        m["scene"] = scene
        rows.append(m)

    # Sort worst-first: HIGH before MEDIUM before LOW, then by consistency.
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows.sort(key=lambda r: (rank[r["risk"]], -r["consistency"]))

    print()
    print(f"{'scene':<34}{'N':>5}{'consist':>9}{'collapse':>10}{'surround':>10}"
          f"  {'risk':<7}reason")
    print("-" * 100)
    for r in rows:
        print(f"{r['scene']:<34}{r['n_frames']:>5}{r['consistency']:>9.3f}"
              f"{r['collapse']:>10.1f}{r['surround']:>10.2f}"
              f"  {r['risk']:<7}{r['reason']}")

    n_high = sum(r["risk"] == "HIGH" for r in rows)
    n_med = sum(r["risk"] == "MEDIUM" for r in rows)
    n_low = sum(r["risk"] == "LOW" for r in rows)
    print("-" * 88)
    print(f"Scanned {len(rows)} scenes:  "
          f"HIGH {n_high}   MEDIUM {n_med}   LOW {n_low}"
          + (f"   (skipped {len(skipped)})" if skipped else ""))
    print("\nconsist  = inlier ray-convergence residual / median cam distance "
          "(scale-free; pose coherence).")
    print(f"           LOW <= {OK_MAX}   MEDIUM <= {MARGINAL_MAX}   HIGH > {MARGINAL_MAX}")
    print("collapse = max/median cam distance (>5 => a mis-registered far camera "
          "wrecks the auto-scale).")
    print("surround = object offset / cam distance (~0 well-surrounded, ~1 one-sided arc).")
    print("reason   = why flagged: 'incoherent' (axes don't meet) and/or 'outlier' "
          "(stray camera).")
    if n_high:
        print("\nHIGH-risk scenes (re-run COLMAP with better surround / matching "
              "before trusting the NeRF):")
        for r in rows:
            if r["risk"] == "HIGH":
                print(f"  - {r['scene']}")
    if skipped:
        print("\nSkipped (too few / malformed frames): " + ", ".join(skipped))

    if args.csv is not None:
        import csv
        cols = ["scene", "risk", "reason", "n_frames", "consistency", "collapse",
                "n_outliers", "residual", "median_cam_dist", "obj_offset",
                "surround", "dataparser_scale"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
