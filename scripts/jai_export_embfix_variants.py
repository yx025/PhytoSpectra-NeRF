"""Two more A' variants for an embedding-fix run directory (made by jai_export_embfix.py).

  Aprime_median   median over the visible views (from Aprime_extra.npz, same per-view values as A')
  Aprime_nograze  A' mean, but views whose angle to the point normal exceeds --max-angle
                  (arccos|d.n|, stored normal, sign ambiguous) are dropped; points left with
                  no view fall back to the plain A' mean (fraction reported)

Geometry and vertex order come verbatim from the old PLY; only reflectance (+ derived uint8
viewport colour) is replaced.  The A' pass is re-rendered to get per-view angles; its plain
mean is compared against the existing refl_4ch_Aprime.ply as a determinism check.

Outputs in --fix-dir: refl_4ch_Aprime_median.ply, refl_4ch_Aprime_nograze.ply,
variants_extra.npz (n_views_nograze, std_views = across-view std of A' per band),
meta_variants.json.
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
from jai_embfix_common import PLY_FLOAT_FIELDS, read_ply, write_ply_like  # noqa: E402
from jai_export_embfix import REFL_CLIP_MAX, estimate_Aprime  # noqa: E402

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--old-ply", type=Path, required=True)
    ap.add_argument("--fix-dir", type=Path, required=True)
    ap.add_argument("--max-angle", type=float, default=75.0)
    ap.add_argument("--chunk", type=int, default=131072)
    args = ap.parse_args()

    meta0 = json.loads((args.fix_dir / "meta.json").read_text())
    _, pipe, _, _ = eval_setup(args.load_config, test_mode="test")
    model = pipe.model
    model.eval()
    dev = model.device
    cams = pipe.datamanager.train_dataset.cameras.to(dev)
    header, old = read_ply(args.old_ply)
    pts = torch.from_numpy(np.stack([old["x"], old["y"], old["z"]], 1)).to(dev)
    nrm = torch.from_numpy(np.stack([old["nx"], old["ny"], old["nz"]], 1)).to(dev)
    g = np.load(args.fix_dir / "gen_rays.npz")
    gen = tuple(torch.from_numpy(g[k].astype(np.int64)).to(dev) for k in ("cam", "px", "py"))

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        mean, _, cnt, _, extra = estimate_Aprime(model, cams, pts, gen, meta0["vis_tol"], args.chunk,
                                                 normals=nrm, max_angle_deg=args.max_angle)
    torch.cuda.synchronize()
    t1 = time.time()

    clip = lambda t: np.clip(t, 0, REFL_CLIP_MAX).astype(np.float32)  # noqa: E731
    mean_np = clip(mean.cpu().numpy())
    cnt_ng = extra["n_views_nograze"].cpu().numpy()
    ng = clip(extra["mean_nograze"].cpu().numpy())
    ng[cnt_ng == 0] = mean_np[cnt_ng == 0]
    med = np.load(args.fix_dir / "Aprime_extra.npz")["median_rgbn"]
    # points with no visible view at all (~1e-5) take A, as in jai_export_embfix.py
    zero = cnt.cpu().numpy() == 0
    _, a_ply = read_ply(args.fix_dir / "refl_4ch_A.ply")
    a_np = np.stack([a_ply[f] for f in PLY_FLOAT_FIELDS], 1)
    med = med.copy()
    med[zero], ng[zero], mean_np[zero] = a_np[zero], a_np[zero], a_np[zero]

    write_ply_like(args.fix_dir / "refl_4ch_Aprime_median.ply", header, old, med[:, :3], med[:, 3])
    write_ply_like(args.fix_dir / "refl_4ch_Aprime_nograze.ply", header, old, ng[:, :3], ng[:, 3])
    np.savez_compressed(args.fix_dir / "variants_extra.npz", n_views_nograze=cnt_ng.astype(np.int16),
                        std_views=extra["std_views"].cpu().numpy().astype(np.float16))

    _, ap_old = read_ply(args.fix_dir / "refl_4ch_Aprime.ply")
    ap_prev = np.stack([ap_old[f] for f in PLY_FLOAT_FIELDS], 1)
    det = np.abs(ap_prev - mean_np).max(1)

    geom = ["x", "y", "z", "nx", "ny", "nz"]
    diff = {}
    for name in ("refl_4ch_Aprime_median.ply", "refl_4ch_Aprime_nograze.ply"):
        h2, new = read_ply(args.fix_dir / name)
        diff[name] = {"header_identical": h2 == header, "n_vertices_identical": len(new) == len(old),
                      **{f: bool(np.array_equal(new[f].view(np.uint32), old[f].view(np.uint32))) for f in geom}}
    n_all = cnt.cpu().numpy()
    meta = {
        "max_angle_deg": args.max_angle,
        "frac_points_all_views_grazing_fallback": float((cnt_ng == 0).mean()),
        "frac_views_dropped": float(1 - cnt_ng.sum() / max(n_all.sum(), 1)),
        "n_views_nograze_median": float(np.median(cnt_ng)),
        "aprime_rerender_vs_existing": {"max_abs": float(det.max()), "frac_gt_1e-3": float((det > 1e-3).mean())},
        "geometry_diff_bitwise": diff,
        "timing_s": {"Aprime_pass_with_angles": t1 - t0},
    }
    (args.fix_dir / "meta_variants.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
