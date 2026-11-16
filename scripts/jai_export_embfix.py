"""Re-colour an existing JAI 4-ch reflectance PLY with embedding-fixed estimates A and A'.

Geometry is taken VERBATIM from the old PLY (xyz, normals); only reflectance_r/g/b/nir
(and the uint8 viewport colour derived from them) are replaced.  No training, no
parameter change: the checkpoint is only read.

  A   mean over all training embeddings e_i of render(generating ray, e_i)
  A'  mean over training frames k that see the point of render(ray c_k -> point, e_k);
      visibility: |median depth of that ray - |p - c_k|| < --vis-tol * |p - c_k|
      (visible-branch depth, used for both branches).  Median and view count go to
      Aprime_extra.npz.

The generating ray of each stored point (the old exporter's RNG was unseeded) is
recovered by projection (see jai_embfix_common).  As a proof, the same rays are also
rendered with the MEAN embedding -- the old export condition -- and must reproduce the
old PLY's reflectance and xyz; the mismatch counts are written to meta.json.

Outputs in --out-dir:
  refl_4ch_A.ply, refl_4ch_Aprime.ply   same layout / vertex order as the old PLY
  Aprime_extra.npz                      median_rgbn (P,4) float32, n_views (P,) int16
  gen_rays.npz                          cam, px, py, residual
  meta.json                             timings, reproduction check, geometry diff

Usage:
  python scripts/jai_export_embfix.py --load-config <run>/config.yml \
      --old-ply <seed>/refl_4ch.ply --out-dir embedding_fix_study/<scene>/seed0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jai_embfix_common import (  # noqa: E402
    branch_geometry, branch_shade, emb_table, median_depth, project, rays_at,
    read_ply, recover_generating_rays, training_frame_cameras, write_ply_like,
)

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

REFL_CLIP_MAX = 1.5  # same clamp as the old exporter


def sync():
    torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def estimate_A(model, cams, cam, px, py, chunk):
    """A plus the mean-embedding re-render of the same rays (reproduction check)."""
    P = cam.shape[0]
    tab_v, tab_n = emb_table(model, "visible"), emb_table(model, "nir")
    A = torch.zeros((P, 4), device=cam.device)
    ref = torch.zeros((P, 4), device=cam.device)
    xyz = torch.zeros((P, 3), device=cam.device)
    for s in range(0, P, chunk):
        sl = slice(s, s + chunk)
        rb = rays_at(cams, cam[sl], torch.stack([px[sl] + 0.5, py[sl] + 0.5], -1).double())
        gv, gn = branch_geometry(model, rb, "visible"), branch_geometry(model, rb, "nir")
        ref[sl, :3] = branch_shade(model, gv, "visible", tab_v.mean(0))
        ref[sl, 3:] = branch_shade(model, gn, "nir", tab_n.mean(0))
        xyz[sl] = rb.origins + rb.directions * median_depth(model, gv)
        acc = torch.zeros((rb.shape[0], 4), device=cam.device)
        for i in range(tab_v.shape[0]):
            acc[:, :3] += branch_shade(model, gv, "visible", tab_v[i])
            acc[:, 3:] += branch_shade(model, gn, "nir", tab_n[i])
        A[sl] = acc / tab_v.shape[0]
    return A, ref, xyz


@torch.no_grad()
def estimate_Aprime(model, cams, pts, gen, vis_tol, chunk, normals=None, max_angle_deg=None, need_median=True):
    """A' mean / median / view count.  With ``normals`` and ``max_angle_deg`` it also returns
    ``extra``: the mean over views whose angle to the point normal is <= max_angle_deg
    (arccos|d.n|, normal sign is ambiguous), that view count, and the per-point sum of
    squares over all views (for the across-view spread)."""
    P, K = pts.shape[0], len(cams)
    tab_v, tab_n = emb_table(model, "visible"), emb_table(model, "nir")
    dev = pts.device
    vals = torch.full((P, K, 4), float("nan"), dtype=torch.float16, device=dev) if need_median else None
    sums = torch.zeros((P, 4), device=dev)
    cnt = torch.zeros((P,), dtype=torch.int32, device=dev)
    want_extra = normals is not None and max_angle_deg is not None
    if want_extra:
        nrm = torch.nn.functional.normalize(normals.float(), dim=-1)
        cos_min = float(np.cos(np.radians(max_angle_deg)))
        sums_ng = torch.zeros((P, 4), device=dev)
        cnt_ng = torch.zeros((P,), dtype=torch.int32, device=dev)
        sumsq = torch.zeros((P, 4), device=dev)
    n_rays = 0
    for k in range(K):
        for s in range(0, P, chunk):
            idx = torch.arange(s, min(s + chunk, P), device=dev)
            u, v, dist, ok = project(cams, k, pts[idx])
            if ok.sum() == 0:
                continue
            idx, u, v, dist = idx[ok], u[ok], v[ok], dist[ok]
            # For the generating camera use the exact recovered pixel centre (= the old
            # export's ray): median depth is quantised, so a 1e-4 px offset can flip it.
            own = gen[0][idx] == k
            u = torch.where(own, gen[1][idx].double() + 0.5, u)
            v = torch.where(own, gen[2][idx].double() + 0.5, v)
            rb = rays_at(cams, torch.full((idx.shape[0],), k, device=dev), torch.stack([u, v], -1))
            gv = branch_geometry(model, rb, "visible")
            n_rays += idx.shape[0]
            vis = ((median_depth(model, gv)[:, 0].double() - dist).abs() < vis_tol * dist)
            if vis.sum() == 0:
                continue
            rgb = branch_shade(model, gv, "visible", tab_v[k])[vis]
            gn = branch_geometry(model, rb[vis], "nir")
            nir = branch_shade(model, gn, "nir", tab_n[k])
            out = torch.cat([rgb, nir], -1)
            iv = idx[vis]
            if need_median:
                vals[iv, k] = out.half()
            sums[iv] += out
            cnt[iv] += 1
            if want_extra:
                keep = (rb.directions[vis].float() * nrm[iv]).sum(-1).abs() >= cos_min
                sums_ng[iv[keep]] += out[keep]
                cnt_ng[iv[keep]] += 1
                sumsq[iv] += out * out
    mean = sums / cnt.clamp_min(1)[:, None]
    med = None
    if need_median:
        med = torch.empty((P, 4), device=dev)
        for s in range(0, P, 200_000):
            med[s : s + 200_000] = torch.nanmedian(vals[s : s + 200_000].float(), dim=1).values
    if want_extra:
        extra = {"mean_nograze": sums_ng / cnt_ng.clamp_min(1)[:, None], "n_views_nograze": cnt_ng,
                 "std_views": (sumsq / cnt.clamp_min(1)[:, None] - mean * mean).clamp_min(0).sqrt()}
        return mean, med, cnt, n_rays, extra
    return mean, med, cnt, n_rays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--old-ply", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vis-tol", type=float, default=0.01, help="relative depth tolerance for A' visibility")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--outputs", default="A,Aprime",
                    help="comma list of A, Aprime, Aprime_nograze (A' mean without views > --max-angle "
                         "from the point normal; one A' pass serves both A' outputs)")
    ap.add_argument("--max-angle", type=float, default=75.0)
    ap.add_argument("--camera-frame", choices=("current", "auto"), default="current",
                    help="auto: use the world frame the checkpoint was trained in (differs for runs "
                         "trained before the 2026-06-11 dataparser change); current: eval_setup cameras")
    args = ap.parse_args()
    outputs = [o.strip() for o in args.outputs.split(",") if o.strip()]
    bad = set(outputs) - {"A", "Aprime", "Aprime_nograze"}
    if bad:
        raise SystemExit(f"unknown --outputs {sorted(bad)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_dir.resolve() == args.old_ply.parent.resolve():
        raise SystemExit("refusing to write next to the old export; choose a new --out-dir")

    _, pipe, _, _ = eval_setup(args.load_config, test_mode="test")
    model = pipe.model
    model.eval()
    dev = model.device
    cams = pipe.datamanager.train_dataset.cameras
    frame_info = {"frame": "current"}
    if args.camera_frame == "auto":
        cams, _, frame_info = training_frame_cameras(pipe, args.load_config)
    cams = cams.to(dev)
    header, old = read_ply(args.old_ply)
    pts = torch.from_numpy(np.stack([old["x"], old["y"], old["z"]], 1)).to(dev)
    P = pts.shape[0]
    print(f"[embfix] {P} points, {len(cams)} train cams, {emb_table(model, 'visible').shape[0]} embeddings")

    t0 = sync()
    cam, px, py, res, res2 = recover_generating_rays(cams, pts)
    t1 = sync()
    A, ref, xyz = estimate_A(model, cams, cam, px, py, args.chunk)
    t2 = sync()
    ng = extra = None
    if "Aprime_nograze" in outputs:
        nrm = torch.from_numpy(np.stack([old["nx"], old["ny"], old["nz"]], 1)).to(dev)
        Ap, Ap_med, n_views, n_rays, extra = estimate_Aprime(
            model, cams, pts, (cam, px, py), args.vis_tol, args.chunk,
            normals=nrm, max_angle_deg=args.max_angle, need_median="Aprime" in outputs)
        ng = extra["mean_nograze"]
        no_ng = extra["n_views_nograze"] == 0
        ng[no_ng] = Ap[no_ng]  # every view grazing -> plain A' mean
    else:
        Ap, Ap_med, n_views, n_rays = estimate_Aprime(model, cams, pts, (cam, px, py), args.vis_tol, args.chunk)
    t3 = sync()

    # ---- reproduction check: mean-embedding render of the recovered ray == old PLY ----
    old_rgbn = torch.from_numpy(np.stack([old[f] for f in ("reflectance_r", "reflectance_g",
                                                            "reflectance_b", "reflectance_nir")], 1)).to(dev)
    rep_err = (ref.clamp(0, REFL_CLIP_MAX) - old_rgbn).abs().max(dim=1).values
    xyz_err = (xyz - pts).norm(dim=-1)

    # The few points with no visible view (median-depth quantisation, ~1e-5 of points) take A.
    zero = n_views == 0
    Ap[zero] = A[zero]
    if Ap_med is not None:
        Ap_med[zero] = A[zero]
    if ng is not None:
        ng[zero] = A[zero]
    to_np = lambda t: t.clamp(0, REFL_CLIP_MAX).cpu().numpy().astype(np.float32)  # noqa: E731
    est = {"A": A, "Aprime": Ap, "Aprime_nograze": ng}
    for o in outputs:
        v = to_np(est[o])
        write_ply_like(args.out_dir / f"refl_4ch_{o}.ply", header, old, v[:, :3], v[:, 3])
    extra_np = {"n_views": n_views.cpu().numpy().astype(np.int16)}
    if Ap_med is not None:
        extra_np["median_rgbn"] = to_np(Ap_med)
    if extra is not None:
        extra_np["n_views_nograze"] = extra["n_views_nograze"].cpu().numpy().astype(np.int16)
    np.savez_compressed(args.out_dir / "Aprime_extra.npz", **extra_np)
    np.savez_compressed(args.out_dir / "gen_rays.npz", cam=cam.cpu().numpy().astype(np.int16),
                        px=px.cpu().numpy().astype(np.int16), py=py.cpu().numpy().astype(np.int16),
                        residual=res.cpu().numpy().astype(np.float32))

    # ---- geometry diff (re-read from disk) ----
    geom = ["x", "y", "z"] + [f for f in ("nx", "ny", "nz") if f in old.dtype.names]
    diff = {}
    for name in (f"refl_4ch_{o}.ply" for o in outputs):
        h2, new = read_ply(args.out_dir / name)
        diff[name] = {"header_identical": h2 == header, "n_vertices_identical": len(new) == len(old),
                      **{f: bool(np.array_equal(new[f].view(np.uint32), old[f].view(np.uint32))) for f in geom}}

    nv = n_views.cpu().numpy()
    meta = {
        "config": str(args.load_config), "old_ply": str(args.old_ply), "n_points": P,
        "n_train_cams": len(cams), "vis_tol": args.vis_tol, "outputs": outputs, "camera_frame": frame_info,
        **({"nograze": {"max_angle_deg": args.max_angle,
                        "frac_points_fallback_all_grazing": float((extra["n_views_nograze"] == 0).float().mean()),
                        "frac_views_dropped": float(1 - extra["n_views_nograze"].sum() / n_views.sum().clamp_min(1))}}
           if extra is not None else {}),
        "ray_recovery": {"residual_px_p50": float(res.median()), "residual_px_max": float(res.max()),
                         "frac_second_candidate_within_0.01px": float((res2 < 0.01).float().mean())},
        # tcnn fp16 results shift ~1e-4 with batch composition, so tiers rather than bit equality
        "reproduction_check": {"mean_abs_refl_err": float(rep_err.mean()),
                               "max_abs_refl_err": float(rep_err.max()),
                               **{f"frac_refl_err_gt_{t:g}": float((rep_err > t).float().mean())
                                  for t in (1e-4, 1e-3, 1e-2)},
                               "max_xyz_err": float(xyz_err.max()),
                               **{f"frac_xyz_err_gt_{t:g}": float((xyz_err > t).float().mean())
                                  for t in (1e-5, 1e-3)}},
        "geometry_diff_bitwise": diff,
        "n_views": {"min": int(nv.min()), "p5": float(np.percentile(nv, 5)), "median": float(np.median(nv)),
                    "p95": float(np.percentile(nv, 95)), "max": int(nv.max()),
                    "frac_lt3": float((nv < 3).mean()), "hist": np.bincount(nv, minlength=len(cams) + 1).tolist()},
        "timing_s": {"ray_recovery": t1 - t0, "A_render": t2 - t1, "Aprime_render": t3 - t2,
                     "A_total": t2 - t0, "Aprime_extra_over_A": t3 - t2, "Aprime_rays": n_rays},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("ray_recovery", "reproduction_check", "timing_s")}, indent=1))
    print("geometry identical:", all(all(v for v in d.values()) for d in diff.values()))
    print("n_views:", {k: meta["n_views"][k] for k in ("min", "median", "max", "frac_lt3")})


if __name__ == "__main__":
    main()
