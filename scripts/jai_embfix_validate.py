"""Validate old / A / A' point-cloud reflectance against measured pixels of held-out eval frames.

For every eval frame:
  * render the visible-branch median-depth map at the eval pose (full resolution);
  * project every point; keep it if |depth(pixel) - |p - c|| < --vis-tol * |p - c| (occlusion);
  * compare the point's value with the measured calibrated reflectance of that pixel:
    visible on the Bayer site (the channel the sensor measured there), NIR on every pixel;
  * plant pixels only, from the GT alone, per 2x2 Bayer block:
    NDVI = (NIR - R) / (NIR + R) > --ndvi-min, G > B (rejects the blue tarp), NIR > 0.1;
  * drop pixels whose raw DN >= 98 % of full scale (4095 for 12-bit) in that band.

Points falling into the same pixel are averaged (per method and group) before comparison,
so every sample is one pixel.  Groups: all, normal zenith 0-15 / 15-45 / >45 deg
(zenith = arccos|n_z| of the stored normal), A' n_views < 3 vs >= 3.

Writes validate.json next to the fixed exports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jai_embfix_common import (bayer_site_channel, branch_geometry, median_depth, project, read_ply,  # noqa: E402
                               training_frame_cameras)

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

BANDS = ("R", "G", "B", "NIR")
# The NIR head is a sigmoid (max 1.0) while calibrated GT NIR reaches 1.2 on bright leaves;
# "NIR<0.95" restricts to measured pixels below that ceiling (selection on GT, fair to all methods).
NIR_CEIL = 0.95
FIELDS = ("reflectance_r", "reflectance_g", "reflectance_b", "reflectance_nir")
ZEN_BINS = (("zen_0-15", 0, 15), ("zen_15-45", 15, 45), ("zen_gt45", 45, 90.01))


def load_raw(path_npy: Path, h: int, w: int) -> np.ndarray:
    """Raw DN mosaic for a reflectance .npy frame: <scene>/<stem>.raw next to reflectance/."""
    raw = path_npy.parent.parent / (path_npy.stem + ".raw")
    return np.fromfile(raw, dtype="<u2").reshape(h, w)


def plant_mask_gt(s0: torch.Tensor, s1: torch.Tensor, ndvi_min: float) -> torch.Tensor:
    R = s0[0::2, 0::2]
    G = 0.5 * (s0[0::2, 1::2] + s0[1::2, 0::2])
    B = s0[1::2, 1::2]
    N = 0.25 * (s1[0::2, 0::2] + s1[0::2, 1::2] + s1[1::2, 0::2] + s1[1::2, 1::2])
    ndvi = (N - R) / (N + R).clamp_min(1e-4)
    blk = (ndvi > ndvi_min) & (G > B) & (N > 0.1)
    return blk.repeat_interleave(2, 0).repeat_interleave(2, 1)


@torch.no_grad()
def depth_map(model, cams, j, chunk=262144):
    rb = cams.generate_rays(camera_indices=j, keep_shape=True)
    h, w = rb.shape
    rb = rb.flatten()
    out = torch.empty((h * w,), device=rb.origins.device)
    for s in range(0, h * w, chunk):
        out[s : s + chunk] = median_depth(model, branch_geometry(model, rb[s : s + chunk], "visible"))[:, 0]
    return out.view(h, w)


def metrics(pred: np.ndarray, gt: np.ndarray):
    if len(gt) == 0:
        return None
    ok = gt > 1e-3
    return {"n_pixels": int(len(gt)), "ratio_median": float(np.median(pred[ok] / gt[ok])),
            "mae": float(np.mean(np.abs(pred - gt))), "bias_mean": float(np.mean(pred - gt)),
            "gt_median": float(np.median(gt)), "pred_median": float(np.median(pred))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--old-ply", type=Path, required=True)
    ap.add_argument("--fix-dir", type=Path, required=True)
    ap.add_argument("--vis-tol", type=float, default=0.01)
    ap.add_argument("--ndvi-min", type=float, default=0.4)
    ap.add_argument("--sat-frac", type=float, default=0.98)
    ap.add_argument("--out-name", default="validate.json")
    ap.add_argument("--camera-frame", choices=("current", "auto"), default="current",
                    help="auto: eval cameras in the frame the checkpoint was trained in")
    args = ap.parse_args()

    _, pipe, _, _ = eval_setup(args.load_config, test_mode="test")
    model = pipe.model
    model.eval()
    dev = model.device
    ds = pipe.datamanager.eval_dataset
    cams = ds.cameras
    frame_info = {"frame": "current"}
    if args.camera_frame == "auto":
        _, cams, frame_info = training_frame_cameras(pipe, args.load_config)
    cams = cams.to(dev)

    clouds = {"old": read_ply(args.old_ply)[1]}
    for m in ("A", "Aprime", "Aprime_median", "Aprime_nograze"):  # whichever the fix dir holds
        if (args.fix_dir / f"refl_4ch_{m}.ply").exists():
            clouds[m] = read_ply(args.fix_dir / f"refl_4ch_{m}.ply")[1]
    base = clouds["old"]
    pts = torch.from_numpy(np.stack([base["x"], base["y"], base["z"]], 1)).to(dev)
    vals = {m: torch.from_numpy(np.stack([c[f] for f in FIELDS], 1)).to(dev) for m, c in clouds.items()}
    n = np.stack([base["nx"], base["ny"], base["nz"]], 1)
    zen = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-9), 0, 1)))
    zen = torch.from_numpy(zen).to(dev)
    groups = {"all": torch.ones_like(zen, dtype=torch.bool)}
    for name, lo, hi in ZEN_BINS:
        groups[name] = (zen >= lo) & (zen < hi)
    n_views = None
    if (args.fix_dir / "Aprime_extra.npz").exists():
        n_views = torch.from_numpy(np.load(args.fix_dir / "Aprime_extra.npz")["n_views"].astype(np.int64)).to(dev)
        groups["nviews_lt3"] = n_views < 3
        groups["nviews_ge3"] = n_views >= 3

    full = 2 ** 12 - 1
    acc = defaultdict(lambda: {"pred": defaultdict(list), "gt": []})  # (group, band) -> lists
    per_frame = []
    for j in range(len(ds)):
        data = ds[j]
        s0 = data["source0_normalized"][..., 0].to(dev)
        s1 = data["source1_normalized"][..., 0].to(dev)
        h, w = s0.shape
        pair = ds._pairs[j]
        sat0 = torch.from_numpy(load_raw(pair.source0.path, h, w).astype(np.int32) >= args.sat_frac * full).to(dev)
        sat1 = torch.from_numpy(load_raw(pair.source1.path, h, w).astype(np.int32) >= args.sat_frac * full).to(dev)
        plant = plant_mask_gt(s0, s1, args.ndvi_min)
        site = bayer_site_channel(h, w, dev)
        dmap = depth_map(model, cams, j)

        u, v, dist, ok = project(cams, j, pts)
        ui, vi = u.floor().long().clamp(0, w - 1), v.floor().long().clamp(0, h - 1)
        vis = ok & ((dmap[vi, ui].double() - dist).abs() < args.vis_tol * dist)
        pix = vi * w + ui
        per_frame.append({"frame": pair.source0.path.stem, "n_points_visible": int(vis.sum()),
                          "frac_plant_pixels": float(plant.float().mean()),
                          "frac_sat_visible": float(sat0.float().mean()), "frac_sat_nir": float(sat1.float().mean())})
        for g, gm in groups.items():
            sel = vis & gm & plant.view(-1)[pix]
            if sel.sum() == 0:
                continue
            p_idx, q = torch.where(sel)[0], pix[sel]
            uq, inv = torch.unique(q, return_inverse=True)
            cnt = torch.zeros(len(uq), device=dev).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
            means = {}
            for m, val in vals.items():
                s = torch.zeros((len(uq), 4), device=dev).index_add_(0, inv, val[p_idx])
                means[m] = s / cnt[:, None]
            ch = site.view(-1)[uq]
            vis_ok = ~sat0.view(-1)[uq]
            nir_ok = ~sat1.view(-1)[uq]
            for b, band in enumerate(BANDS[:3]):
                mk = vis_ok & (ch == b)
                acc[(g, band)]["gt"].append(s0.view(-1)[uq][mk].cpu().numpy())
                for m in vals:
                    acc[(g, band)]["pred"][m].append(means[m][mk, b].cpu().numpy())
            acc[(g, "NIR")]["gt"].append(s1.view(-1)[uq][nir_ok].cpu().numpy())
            for m in vals:
                acc[(g, "NIR")]["pred"][m].append(means[m][nir_ok, 3].cpu().numpy())
            unclip = nir_ok & (s1.view(-1)[uq] < NIR_CEIL)
            acc[(g, "NIR<0.95")]["gt"].append(s1.view(-1)[uq][unclip].cpu().numpy())
            for m in vals:
                acc[(g, "NIR<0.95")]["pred"][m].append(means[m][unclip, 3].cpu().numpy())
        print(f"  eval frame {j + 1}/{len(ds)}: {per_frame[-1]['n_points_visible']} visible pts, "
              f"plant {per_frame[-1]['frac_plant_pixels']:.1%}")

    result = {"config": str(args.load_config), "camera_frame": frame_info, "vis_tol": args.vis_tol, "ndvi_min": args.ndvi_min,
              "sat_frac": args.sat_frac, "per_frame": per_frame, "groups": {}}
    for (g, band), d in acc.items():
        gt = np.concatenate(d["gt"]) if d["gt"] else np.array([])
        result["groups"].setdefault(g, {})[band] = {
            m: metrics(np.concatenate(d["pred"][m]), gt) for m in vals}
    if n_views is not None:
        nv = n_views.cpu().numpy()
        result["n_views_all_points"] = {"median": float(np.median(nv)), "frac_lt3": float((nv < 3).mean()),
                                        "pctl_5_25_50_75_95": np.percentile(nv, [5, 25, 50, 75, 95]).tolist()}
    (args.fix_dir / args.out_name).write_text(json.dumps(result, indent=1))
    for band in BANDS + ("NIR<0.95",):
        r = result["groups"]["all"][band]
        print(f"{band:3s} n={r['old']['n_pixels']:7d} | " + " | ".join(
            f"{m} x{r[m]['ratio_median']:.3f} MAE {r[m]['mae']:.4f}" for m in vals))


if __name__ == "__main__":
    main()
