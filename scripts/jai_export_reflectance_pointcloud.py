"""Export a JAI point cloud with per-vertex RGB + NIR reflectance + normals.

Why a dedicated exporter
------------------------
``ns-export pointcloud`` uses ``generate_point_cloud`` which captures exactly
one RGB output (``rgb_output_name``) and optional normals.  It has no hook for
custom per-vertex scalar attributes.  The JAI model produces **two rendered
outputs in a single forward pass**:

- ``outputs["rgb"]`` — (N, 3) visible branch.  In ``intensity_mode=reflectance``
  training, these three values are the per-pixel (R, G, B) reflectance.  In
  ``intensity_mode=dn`` they are normalized DN and look like a regular image.
- ``outputs["nir"]`` — (N, 1) NIR branch.  In reflectance mode this is NIR
  reflectance; in DN mode it is normalized NIR DN.

This script replicates the sampling loop of ``generate_point_cloud`` but
captures both tensors at each forward pass, so the RGB and NIR values land on
the **same 3-D point** (not two independently-sampled point clouds).  The
result is saved as a PLY with:

::

    x y z
    red green blue          (uint8, from outputs["rgb"])
    nx ny nz                (float, from outputs["normals"], iff predict_normals=True)
    reflectance_nir         (float scalar, from outputs["nir"])

Custom PLY scalar properties are read correctly by Blender, MeshLab, and
open3d's tensor-PointCloud loader.

Honest scope
------------
- Per-vertex (R, G, B) reflectance is EXACT in reflectance-mode training:
  ``outputs["rgb"]`` IS the per-pixel reflectance vector.  In DN-mode
  training, the red/green/blue columns are normalized DN (visually reasonable
  but not physically reflectance).
- Per-vertex ``reflectance_nir`` is likewise reflectance in reflectance-mode
  training and normalized DN otherwise.
- Per-vertex normals are written when the model was trained with
  ``--pipeline.model.predict-normals True``.  Otherwise the column is skipped
  and you can still estimate normals in Blender / open3d from the point
  positions.
- This exporter is a STANDALONE script.  It does NOT modify
  ``nerfstudio/scripts/exporter.py``; the stock ``ns-export pointcloud``
  command still works unchanged.

Usage
-----

::

    python scripts/jai_export_reflectance_pointcloud.py \
        --load-config outputs/.../config.yml \
        --output      exports/12.cucum1/pointcloud_reflectance.ply \
        --num-points  1000000 \
        --remove-outliers

    # Smaller cloud for quick inspection
    python scripts/jai_export_reflectance_pointcloud.py \
        --load-config outputs/.../config.yml \
        --output      exports/12.cucum1/preview.ply \
        --num-points  200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Import order matters — nerfstudio's eval_setup pulls in the full trainer/config stack.
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.eval_utils import eval_setup


# ---------------------------------------------------------------------------
# Sampling loop — single forward pass captures rgb + nir (+ normals)
# ---------------------------------------------------------------------------

def _sample_points(
    pipeline,
    num_points: int,
    num_rays_per_batch: int,
    bbox_min: Optional[List[float]],
    bbox_max: Optional[List[float]],
    want_normals: bool,
    crop_obb=None,
):
    """Run the generate_point_cloud sampling pattern and return stacked tensors.

    Returns
    -------
    points : (P, 3) float32
    rgb    : (P, 3) float32 in [0, 1]
    nir    : (P,)   float32 in [0, 1]
    normals: (P, 3) float32 in [-1, 1] OR None
    """
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

            rgb   = outputs["rgb"]           # (N, 3)
            nir   = outputs["nir"].squeeze(-1)  # (N,)
            depth = outputs["depth"]         # (N, 1)
            acc   = outputs["accumulation"]  # (N, 1)
            normals = None
            if want_normals:
                if "normals" not in outputs:
                    raise RuntimeError(
                        "Model was asked for normals but outputs['normals'] is missing. "
                        "Retrain with --pipeline.model.predict-normals True or pass --no-normals."
                    )
                normals_raw = outputs["normals"]
                # generate_point_cloud asserts [0,1]; rescale to [-1,1]
                normals = normals_raw * 2.0 - 1.0

            points = ray_bundle.origins + ray_bundle.directions * depth  # (N, 3)

            # Opacity mask: drop rays that didn't hit anything
            mask = (acc[..., 0] > 0.5)
            points = points[mask]
            rgb = rgb[mask]
            nir = nir[mask]
            if normals is not None:
                normals = normals[mask]

            # Spatial crop filter — OBB (viewer crop) or AABB (legacy)
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
            rgb_list.append(rgb.float().clamp(0, 1).cpu())
            nir_list.append(nir.float().clamp(0, 1).cpu())
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


# ---------------------------------------------------------------------------
# Outlier removal (optional — uses open3d when available)
# ---------------------------------------------------------------------------

def _remove_outliers(
    points, rgb, nir, normals, nb_neighbors=20, std_ratio=10.0,
):
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
# PLY writer with custom per-vertex scalar
# ---------------------------------------------------------------------------

def _write_ply(
    out_path: Path,
    points: np.ndarray,
    rgb: np.ndarray,
    nir: np.ndarray,
    normals: Optional[np.ndarray],
) -> None:
    """Write a binary little-endian PLY with the custom reflectance_nir scalar."""
    n = points.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a structured numpy array the same way other PLY writers do
    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [("reflectance_nir", "<f4")]

    arr = np.empty(n, dtype=fields)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    arr["red"]   = rgb_u8[:, 0]
    arr["green"] = rgb_u8[:, 1]
    arr["blue"]  = rgb_u8[:, 2]
    if normals is not None:
        arr["nx"] = normals[:, 0]
        arr["ny"] = normals[:, 1]
        arr["nz"] = normals[:, 2]
    arr["reflectance_nir"] = nir

    # Header
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
          f"{'nx ny nz ' if normals is not None else ''}reflectance_nir")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export JAI point cloud with per-vertex RGB + NIR reflectance + normals.",
    )
    ap.add_argument("--load-config", type=Path, required=True,
                    help="Trained JAI model config.yml (from an outputs/ run).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output PLY path.")
    ap.add_argument("--num-points", type=int, default=1_000_000)
    ap.add_argument("--num-rays-per-batch", type=int, default=32768)
    ap.add_argument("--remove-outliers", action="store_true",
                    help="Run open3d statistical outlier removal on the cloud.")
    ap.add_argument("--std-ratio", type=float, default=10.0)
    ap.add_argument("--no-normals", action="store_true",
                    help="Skip per-vertex normals even if the model predicts them.")
    ap.add_argument("--bbox-min", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--bbox-max", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--crop-json", type=Path, default=None,
                    help="Crop-spec JSON from jai_crop_spec.py.  Uses the viewer's "
                         "OBB (oriented bounding box) representation.  Overrides "
                         "--bbox-min/--bbox-max when both are given.")
    ap.add_argument("--obb-center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="OBB center (same as viewer/ns-export). Overridden by --crop-json.")
    ap.add_argument("--obb-rotation", type=float, nargs=3, default=None,
                    metavar=("R", "P", "Y"),
                    help="OBB rotation as RPY radians.")
    ap.add_argument("--obb-scale", type=float, nargs=3, default=None,
                    metavar=("SX", "SY", "SZ"),
                    help="OBB scale / extents.")
    args = ap.parse_args()

    if not args.load_config.exists():
        print(f"error: config not found: {args.load_config}", file=sys.stderr)
        return 2

    print(f"[jai-export] loading pipeline from {args.load_config}")
    _, pipeline, _, _ = eval_setup(args.load_config)

    # Detect whether the model actually predicts normals
    want_normals = False
    if not args.no_normals:
        mcfg = getattr(pipeline.model, "config", None)
        pred = bool(getattr(mcfg, "predict_normals", False))
        if pred:
            want_normals = True
            print("[jai-export] predict_normals=True — will capture per-vertex normals")
        else:
            print("[jai-export] predict_normals=False — skipping normals "
                  "(retrain with --pipeline.model.predict-normals True for normals)")

    # --- Resolve crop region ---
    crop_obb = None
    if args.crop_json is not None:
        from jai_crop_spec import load_crop_spec, crop_spec_to_oriented_box
        data = load_crop_spec(args.crop_json)
        crop_obb = crop_spec_to_oriented_box(data)
        print(f"[jai-export] loaded crop OBB from {args.crop_json}")
    elif args.obb_center is not None and args.obb_rotation is not None and args.obb_scale is not None:
        from nerfstudio.data.scene_box import OrientedBox
        crop_obb = OrientedBox.from_params(
            tuple(args.obb_center), tuple(args.obb_rotation), tuple(args.obb_scale),
        )
        print("[jai-export] using inline OBB args")

    print(f"[jai-export] sampling {args.num_points} points")
    points, rgb, nir, normals = _sample_points(
        pipeline,
        num_points=args.num_points,
        num_rays_per_batch=args.num_rays_per_batch,
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        want_normals=want_normals,
        crop_obb=crop_obb,
    )

    if args.remove_outliers:
        points, rgb, nir, normals = _remove_outliers(
            points, rgb, nir, normals, std_ratio=args.std_ratio,
        )

    _write_ply(args.output, points, rgb, nir, normals)
    print("[jai-export] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
