"""Export a JAI 4-channel reflectance point cloud DIRECTLY from a checkpoint,
without eval_setup — i.e. WITHOUT needing the training data on disk.

Why this exists
---------------
`jai_export_4ch_reflectance_pointcloud.py` rebuilds the full pipeline via
nerfstudio's ``eval_setup``: the dataparser re-reads the config's ``data``
(reflectance .npy frames) and ``poses_source`` (colmap_out), and the
datamanager caches all images.  If those directories were deleted, that
exporter dies even though the trained model is intact.  This script instead:

1. loads the run's ``config.yml`` (yaml round-trip of the dataclass tree),
2. builds the JaiNerfactoModel with ``num_train_data`` recovered from the
   checkpoint's ``camera_optimizer.pose_adjustment`` shape,
3. loads the ``_model.*`` weights from the newest checkpoint,
4. shoots SYNTHETIC rays (origins on a spherical cap around the crop box,
   targets uniform inside the box) and back-projects depth exactly like the
   original exporter (acc > 0.5, OBB crop, clamp, optional outlier removal),
5. writes the identical PLY layout by reusing the original module's writer.

Caveat: reflectance is view-dependent (SH direction encoding).  The original
exporter uses training-camera ray directions; this one uses synthetic
directions, so per-point reflectance can differ slightly.  Geometry (density
surface) is view-independent and matches the original exporter.

Usage
-----
::

    python scripts/jai_export_4ch_from_ckpt.py \
        --load-config outputs/unnamed/jai-nerfacto/<ts>/config.yml \
        --output exports/train_log_pcd/foo_refl_4ch.ply \
        --num-points 3000000 --remove-outliers \
        --obb-center 0 0 -1 --obb-rotation 0 0 0 --obb-scale 2 2 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from jai_export_4ch_reflectance_pointcloud import _remove_outliers, _write_ply  # noqa: E402

from nerfstudio.cameras.rays import RayBundle  # noqa: E402
from nerfstudio.data.scene_box import OrientedBox, SceneBox  # noqa: E402


def load_model(config_path: Path, device: str):
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    ckpt_dir = config_path.parent / "nerfstudio_models"
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k[len("_model."):]: v for k, v in ckpt["pipeline"].items() if k.startswith("_model.")}

    num_train_data = state["camera_optimizer.pose_adjustment"].shape[0]
    scene_scale = getattr(config.pipeline.datamanager.dataparser, "scene_scale", 1.0)
    scene_box = SceneBox(
        aabb=torch.tensor(
            [[-scene_scale] * 3, [scene_scale] * 3], dtype=torch.float32
        )
    )

    model = config.pipeline.model.setup(
        scene_box=scene_box, num_train_data=num_train_data, metadata={},
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    print(f"[ckpt-export] loaded {ckpt_path.name} (step {ckpt['step']}, "
          f"{num_train_data} train cams) on {device}")
    return model


def synthetic_ray_batch(n, center, box_half, radius, device, gen):
    """Origins on a spherical cap above/around `center`; targets uniform in the box."""
    u = torch.randn(n, 3, generator=gen, device=device)
    u = u / u.norm(dim=-1, keepdim=True)
    u[:, 2] = 0.05 + 0.9 * u[:, 2].abs()          # elevation: sideways to steep
    u = u / u.norm(dim=-1, keepdim=True)
    origins = center + radius * u

    t = (torch.rand(n, 3, generator=gen, device=device) * 2.0 - 1.0) * box_half * 0.85
    targets = center + t
    dirs = targets - origins
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    return RayBundle(
        origins=origins,
        directions=dirs,
        pixel_area=torch.full((n, 1), 1e-6, device=device),
        camera_indices=torch.zeros((n, 1), dtype=torch.long, device=device),
    )


def sample_points(model, num_points, rays_per_batch, crop_obb, obb_center, obb_half,
                  radius, refl_clip_max, want_normals, device, seed=0):
    center = torch.tensor(obb_center, dtype=torch.float32, device=device)
    half = torch.tensor(obb_half, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    pts_list, rgb_list, nir_list, nrm_list = [], [], [], []
    collected = 0
    with torch.no_grad():
        while collected < num_points:
            rb = synthetic_ray_batch(rays_per_batch, center, half, radius, device, gen)
            outputs = model(rb)
            rgb = outputs["rgb"]
            nir = outputs["nir"].squeeze(-1)
            depth = outputs["depth"]
            acc = outputs["accumulation"]
            normals = None
            if want_normals:
                normals = outputs["normals"] * 2.0 - 1.0

            points = rb.origins + rb.directions * depth
            mask = acc[..., 0] > 0.5
            points, rgb, nir = points[mask], rgb[mask], nir[mask]
            if normals is not None:
                normals = normals[mask]

            if crop_obb is not None:
                m2 = crop_obb.within(points)
                points, rgb, nir = points[m2], rgb[m2], nir[m2]
                if normals is not None:
                    normals = normals[m2]

            if points.shape[0] == 0:
                continue
            pts_list.append(points.float().cpu())
            rgb_list.append(rgb.float().clamp(0.0, refl_clip_max).cpu())
            nir_list.append(nir.float().clamp(0.0, refl_clip_max).cpu())
            if normals is not None:
                nrm_list.append(normals.float().cpu())
            collected += int(points.shape[0])
            print(f"  sampled {collected}/{num_points}", end="\r", flush=True)
    print()
    pts = torch.cat(pts_list)[:num_points].numpy().astype(np.float32)
    rgb = torch.cat(rgb_list)[:num_points].numpy().astype(np.float32)
    nir = torch.cat(nir_list)[:num_points].numpy().astype(np.float32)
    nrm = torch.cat(nrm_list)[:num_points].numpy().astype(np.float32) if nrm_list else None
    return pts, rgb, nir, nrm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-points", type=int, default=3_000_000)
    ap.add_argument("--num-rays-per-batch", type=int, default=65536)
    ap.add_argument("--remove-outliers", action="store_true")
    ap.add_argument("--std-ratio", type=float, default=10.0)
    ap.add_argument("--no-normals", action="store_true")
    ap.add_argument("--refl-clip-max", type=float, default=1.5)
    ap.add_argument("--obb-center", type=float, nargs=3, default=[0.0, 0.0, -1.0])
    ap.add_argument("--obb-rotation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--obb-scale", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--cam-radius", type=float, default=1.1,
                    help="radius of the synthetic camera cap around the OBB center")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.load_config, device)

    want_normals = (not args.no_normals) and bool(getattr(model.config, "predict_normals", False))
    print(f"[ckpt-export] predict_normals={want_normals}")

    crop_obb = OrientedBox.from_params(
        tuple(args.obb_center), tuple(args.obb_rotation), tuple(args.obb_scale)
    )
    obb_half = [s / 2.0 for s in args.obb_scale]

    print(f"[ckpt-export] sampling {args.num_points} points with synthetic rays")
    points, rgb, nir, normals = sample_points(
        model, args.num_points, args.num_rays_per_batch, crop_obb,
        args.obb_center, obb_half, args.cam_radius,
        args.refl_clip_max, want_normals, device, seed=args.seed,
    )

    if args.remove_outliers:
        points, rgb, nir, normals = _remove_outliers(points, rgb, nir, normals,
                                                     std_ratio=args.std_ratio)
    _write_ply(args.output, points, rgb, nir, normals)
    print("[ckpt-export] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
