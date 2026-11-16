"""Export a JAI point cloud with **per-vertex 4-channel float reflectance** (R, G, B, NIR).

When to use this exporter
-------------------------
Use this script when the model was trained with ``--intensity-mode reflectance``.
It writes the visible-branch RGB and the NIR-branch output as **float32 reflectance
values** in [0, 1+], without any 256-level quantization.  This preserves precision
for downstream NDVI / chlorophyll / NDRE computations.

Compared to ``jai_export_reflectance_pointcloud.py`` (which writes uint8
red/green/blue and only ``reflectance_nir`` as float), this exporter writes
**four float channels**: ``reflectance_r``, ``reflectance_g``, ``reflectance_b``,
``reflectance_nir``.  uint8 RGB is also still written so the cloud renders in
Blender / MeshLab viewport with reasonable color, but the analytical values
live in the float channels.

PLY layout
----------
::

    x y z                          (float)
    red green blue                 (uint8, sRGB-style viewport color = clip(refl,0,1)*255)
    nx ny nz                       (float, iff predict_normals=True)
    reflectance_r                  (float, [0, ~1.2] before clamp)
    reflectance_g                  (float)
    reflectance_b                  (float)
    reflectance_nir                (float)

View dependency caveat
----------------------
The JaiField inherits NerfactoField's directional encoding (``SHEncoding`` on
ray direction).  Different camera angles for the same 3-D point will produce
slightly different reflectance.  This exporter does **not** canonicalize the
view: for each sampled point, the rendered value uses the actual training-camera
ray direction at that pixel.  This is the same behavior as
``jai_export_reflectance_pointcloud.py`` — chosen here because the user wants
"reflectance as observed during training", not a single canonical reflectance.

Usage
-----
::

    python scripts/jai_export_4ch_reflectance_pointcloud.py \
        --load-config outputs/.../config.yml \
        --output      exports/4.21-soybean1-1400/refl_4ch.ply \
        --num-points  1000000 \
        --remove-outliers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.eval_utils import eval_setup


# ---------------------------------------------------------------------------
# Sampling loop — same as the uint8 RGB exporter; we just keep RGB as float.
# ---------------------------------------------------------------------------

def _sample_points(
    pipeline,
    num_points: int,
    num_rays_per_batch: int,
    bbox_min: Optional[List[float]],
    bbox_max: Optional[List[float]],
    want_normals: bool,
    refl_clip_max: float,
    crop_obb=None,
):
    assert isinstance(pipeline.datamanager, VanillaDataManager), (
        "This exporter assumes a VanillaDataManager (the default for jai-nerfacto)."
    )
    assert pipeline.datamanager.train_pixel_sampler is not None
    pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = num_rays_per_batch

    pts_list, rgb_list, nir_list, nrm_list = [], [], [], []
    collected = 0
    bbox_lo = torch.tensor(bbox_min) if bbox_min is not None else None
    bbox_hi = torch.tensor(bbox_max) if bbox_max is not None else None

    pipeline.eval()
    with torch.no_grad():
        while collected < num_points:
            ray_bundle, _ = pipeline.datamanager.next_train(0)
            assert isinstance(ray_bundle, RayBundle)
            outputs = pipeline.model(ray_bundle)

            for k in ("rgb", "nir", "depth", "accumulation"):
                if k not in outputs:
                    raise RuntimeError(
                        f"JAI model outputs missing expected key '{k}'. "
                        f"Found: {sorted(outputs.keys())}"
                    )

            rgb = outputs["rgb"]                    # (N, 3) reflectance R,G,B
            nir = outputs["nir"].squeeze(-1)        # (N,)   reflectance NIR
            depth = outputs["depth"]
            acc = outputs["accumulation"]
            normals = None
            if want_normals:
                if "normals" not in outputs:
                    raise RuntimeError(
                        "Model was asked for normals but outputs['normals'] is missing. "
                        "Retrain with --pipeline.model.predict-normals True or pass --no-normals."
                    )
                normals = outputs["normals"] * 2.0 - 1.0

            points = ray_bundle.origins + ray_bundle.directions * depth

            mask = (acc[..., 0] > 0.5)
            points = points[mask]
            rgb = rgb[mask]
            nir = nir[mask]
            if normals is not None:
                normals = normals[mask]

            if crop_obb is not None:
                m2 = crop_obb.within(points)
                points = points[m2]; rgb = rgb[m2]; nir = nir[m2]
                if normals is not None:
                    normals = normals[m2]
            elif bbox_lo is not None and bbox_hi is not None:
                lo = bbox_lo.to(points.device)
                hi = bbox_hi.to(points.device)
                m2 = ((points > lo) & (points < hi)).all(dim=-1)
                points = points[m2]; rgb = rgb[m2]; nir = nir[m2]
                if normals is not None:
                    normals = normals[m2]

            if points.shape[0] == 0:
                continue
            pts_list.append(points.float().cpu())
            # NOTE: do NOT clamp RGB / NIR to [0, 1] here — reflectance can legitimately
            # exceed 1 on bright specular highlights.  Clamp to a generous upper bound only.
            rgb_list.append(rgb.float().clamp(0.0, refl_clip_max).cpu())
            nir_list.append(nir.float().clamp(0.0, refl_clip_max).cpu())
            if normals is not None:
                nrm_list.append(normals.float().cpu())
            collected += int(points.shape[0])
            print(f"  sampled {collected}/{num_points}", end="\r", flush=True)

    print()
    points_t = torch.cat(pts_list, dim=0)[:num_points]
    rgb_t    = torch.cat(rgb_list, dim=0)[:num_points]
    nir_t    = torch.cat(nir_list, dim=0)[:num_points]
    normals_t = torch.cat(nrm_list, dim=0)[:num_points] if nrm_list else None

    return (
        points_t.numpy().astype(np.float32),
        rgb_t.numpy().astype(np.float32),
        nir_t.numpy().astype(np.float32),
        None if normals_t is None else normals_t.numpy().astype(np.float32),
    )


def _remove_outliers(points, rgb, nir, normals, nb_neighbors=20, std_ratio=10.0):
    try:
        import open3d as o3d
    except ImportError:
        print("[warn] open3d not available; skipping statistical outlier removal.")
        return points, rgb, nir, normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    idx = np.asarray(ind, dtype=np.int64)
    print(f"[cleanup] outlier removal kept {len(idx)} / {len(points)} points")
    return (
        points[idx], rgb[idx], nir[idx],
        None if normals is None else normals[idx],
    )


# ---------------------------------------------------------------------------
# PLY writer — float reflectance + uint8 viewport color
# ---------------------------------------------------------------------------

def _write_ply(
    out_path: Path,
    points: np.ndarray,
    rgb: np.ndarray,
    nir: np.ndarray,
    normals: Optional[np.ndarray],
) -> None:
    n = points.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [
        ("reflectance_r",   "<f4"),
        ("reflectance_g",   "<f4"),
        ("reflectance_b",   "<f4"),
        ("reflectance_nir", "<f4"),
    ]

    arr = np.empty(n, dtype=fields)
    arr["x"] = points[:, 0]; arr["y"] = points[:, 1]; arr["z"] = points[:, 2]

    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    arr["red"]   = rgb_u8[:, 0]
    arr["green"] = rgb_u8[:, 1]
    arr["blue"]  = rgb_u8[:, 2]

    if normals is not None:
        arr["nx"] = normals[:, 0]; arr["ny"] = normals[:, 1]; arr["nz"] = normals[:, 2]

    arr["reflectance_r"]   = rgb[:, 0]
    arr["reflectance_g"]   = rgb[:, 1]
    arr["reflectance_b"]   = rgb[:, 2]
    arr["reflectance_nir"] = nir

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if normals is not None:
        header_lines += [
            "property float nx",
            "property float ny",
            "property float nz",
        ]
    header_lines += [
        "property float reflectance_r",
        "property float reflectance_g",
        "property float reflectance_b",
        "property float reflectance_nir",
        "end_header",
        "",
    ]
    header = "\n".join(header_lines).encode("ascii")

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())

    print(f"[ok] wrote {n} points → {out_path}")
    print(f"     fields: x y z red green blue "
          f"{'nx ny nz ' if normals is not None else ''}"
          f"reflectance_r reflectance_g reflectance_b reflectance_nir")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export JAI point cloud with per-vertex 4-channel float reflectance (R/G/B/NIR).",
    )
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-points", type=int, default=1_000_000)
    ap.add_argument("--num-rays-per-batch", type=int, default=32768)
    ap.add_argument("--remove-outliers", action="store_true")
    ap.add_argument("--std-ratio", type=float, default=10.0)
    ap.add_argument("--no-normals", action="store_true")
    ap.add_argument("--refl-clip-max", type=float, default=1.5,
                    help="Upper clamp for reflectance values (default 1.5 — bright "
                         "specular highlights can legitimately exceed 1.0).")
    ap.add_argument("--bbox-min", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--bbox-max", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--crop-json", type=Path, default=None)
    ap.add_argument("--obb-center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--obb-rotation", type=float, nargs=3, default=None, metavar=("R", "P", "Y"))
    ap.add_argument("--obb-scale", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"))
    args = ap.parse_args()

    if not args.load_config.exists():
        print(f"error: config not found: {args.load_config}", file=sys.stderr)
        return 2

    print(f"[jai-export-4ch] loading pipeline from {args.load_config}")
    _, pipeline, _, _ = eval_setup(args.load_config)

    mcfg = getattr(pipeline.model, "config", None)
    intensity_mode = getattr(mcfg, "intensity_mode", None)
    # Best-effort: look at the dataparser config for the intensity_mode setting
    try:
        dp_cfg = pipeline.datamanager.dataparser.config
        intensity_mode = getattr(dp_cfg, "intensity_mode", intensity_mode)
    except Exception:
        pass
    if intensity_mode is not None and intensity_mode != "reflectance":
        print(f"[jai-export-4ch] WARNING: model was trained with intensity_mode="
              f"{intensity_mode!r}, NOT 'reflectance'.  The 'reflectance_*' float "
              f"channels will contain normalized DN, not physical reflectance.  "
              f"For DN training use jai_export_reflectance_pointcloud.py instead.")

    want_normals = False
    if not args.no_normals:
        pred = bool(getattr(mcfg, "predict_normals", False))
        if pred:
            want_normals = True
            print("[jai-export-4ch] predict_normals=True — capturing per-vertex normals")
        else:
            print("[jai-export-4ch] predict_normals=False — skipping normals")

    crop_obb = None
    if args.crop_json is not None:
        from jai_crop_spec import load_crop_spec, crop_spec_to_oriented_box
        data = load_crop_spec(args.crop_json)
        crop_obb = crop_spec_to_oriented_box(data)
        print(f"[jai-export-4ch] loaded crop OBB from {args.crop_json}")
    elif args.obb_center is not None and args.obb_rotation is not None and args.obb_scale is not None:
        from nerfstudio.data.scene_box import OrientedBox
        crop_obb = OrientedBox.from_params(
            tuple(args.obb_center), tuple(args.obb_rotation), tuple(args.obb_scale),
        )
        print("[jai-export-4ch] using inline OBB args")

    print(f"[jai-export-4ch] sampling {args.num_points} points")
    points, rgb, nir, normals = _sample_points(
        pipeline,
        num_points=args.num_points,
        num_rays_per_batch=args.num_rays_per_batch,
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        want_normals=want_normals,
        refl_clip_max=args.refl_clip_max,
        crop_obb=crop_obb,
    )

    if args.remove_outliers:
        points, rgb, nir, normals = _remove_outliers(
            points, rgb, nir, normals, std_ratio=args.std_ratio,
        )

    _write_ply(args.output, points, rgb, nir, normals)
    print("[jai-export-4ch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
