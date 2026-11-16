"""Shared helpers for the appearance-embedding export fixes (A / A') and their validation.

A   : per point, re-render the ray that generated it with EVERY training-frame
      appearance embedding and average the outputs.
A'  : per point, find the training frames that see it (rendered-depth test),
      render the frame-camera -> point ray with THAT frame's embedding, and
      average (median and view count kept too).

The old exporter (``jai_export_4ch_reflectance_pointcloud.py``) samples random
training pixels, renders them in eval mode (mean embedding) and keeps
``o + d * median_depth``.  Its RNG is unseeded, so the generating ray of a
stored point is recovered here by projecting the point into every training
camera: the generating camera puts it exactly on a pixel centre.

Projection inverts nerfstudio's own ray generation (``Cameras._generate_rays_from_coords``):
    x = (u - cx) / fx,  y = (v - cy) / fy  ->  OPENCV undistort (y still pointing down)
    -> y *= -1  ->  dir_cam = [x, y, -1]  ->  dir_world = R @ dir_cam (normalised).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.fields.base_field import get_normalized_directions

PLY_FLOAT_FIELDS = ("reflectance_r", "reflectance_g", "reflectance_b", "reflectance_nir")


# --------------------------------------------------------------------------- PLY
def read_ply(path: Path) -> Tuple[bytes, np.ndarray]:
    """Return (raw header bytes, structured vertex array) of a binary little-endian PLY."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end]
    fields = []
    n = None
    tmap = {"float": "<f4", "uchar": "u1", "int": "<i4", "double": "<f8", "uint": "<u4"}
    for line in header.decode("ascii").splitlines():
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            n = int(tok[2])
        elif tok and tok[0] == "property":
            fields.append((tok[2], tmap[tok[1]]))
    arr = np.frombuffer(raw[end:], dtype=fields, count=n).copy()
    return header, arr


def write_ply_like(path: Path, header: bytes, old: np.ndarray, rgb: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Copy ``old`` verbatim, replace only the reflectance floats and the derived uint8 viewport colour."""
    new = old.copy()
    new["reflectance_r"], new["reflectance_g"], new["reflectance_b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    new["reflectance_nir"] = nir
    # same viewport-colour rule as jai_export_4ch_reflectance_pointcloud._write_ply
    u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    new["red"], new["green"], new["blue"] = u8[:, 0], u8[:, 1], u8[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(new.tobytes())
    return new


# ------------------------------------------------------------------- projection
def _distort(x, y, dp):
    """OPENCV forward model in nerfstudio's parameter order [k1, k2, k3, k4, p1, p2]."""
    k1, k2, k3, k4, p1, p2 = (dp[..., i] for i in range(6))
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))
    xd = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x)
    yd = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y)
    return xd, yd


def project(cams: Cameras, k: int, pts: torch.Tensor):
    """Project world points into camera k. Returns (u, v, dist, valid) in float64; u,v = pixel coords (col,row)."""
    c2w = cams.camera_to_worlds[k].double()
    R, t = c2w[:, :3], c2w[:, 3]
    pc = (pts.double() - t) @ R  # = R^T (p - t)
    z = -pc[:, 2]
    x = pc[:, 0] / z
    y = -pc[:, 1] / z  # y pointing down, as in nerfstudio before the flip
    if cams.distortion_params is not None:
        x, y = _distort(x, y, cams.distortion_params[k].double())
    u = x * cams.fx[k, 0].double() + cams.cx[k, 0].double()
    v = y * cams.fy[k, 0].double() + cams.cy[k, 0].double()
    w, h = int(cams.width[k, 0]), int(cams.height[k, 0])
    valid = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    dist = (pts.double() - t).norm(dim=-1)
    return u, v, dist, valid


def rays_at(cams: Cameras, cam_idx: torch.Tensor, uv: torch.Tensor) -> RayBundle:
    """Rays of cameras ``cam_idx`` through pixel coords ``uv`` = (u=col, v=row), float."""
    coords = torch.stack([uv[:, 1], uv[:, 0]], -1).float()  # nerfstudio wants (row, col)
    return cams.generate_rays(camera_indices=cam_idx.view(-1, 1).long(), coords=coords)


@torch.no_grad()
def recover_generating_rays(cams: Cameras, pts: torch.Tensor, chunk: int = 1_000_000):
    """For each point find (camera, pixel) whose pixel centre it sits on.

    Returns cam (P,), px (P,) col, py (P,) row, res (P,) best max-abs offset from the pixel centre
    [px], res2 (P,) second-best residual (ambiguity check).
    """
    P = pts.shape[0]
    dev = pts.device
    best = torch.full((P,), np.inf, dtype=torch.float64, device=dev)
    second = torch.full_like(best, np.inf)
    cam = torch.full((P,), -1, dtype=torch.long, device=dev)
    px = torch.zeros((P,), dtype=torch.long, device=dev)
    py = torch.zeros_like(px)
    for k in range(len(cams)):
        for s in range(0, P, chunk):
            sl = slice(s, s + chunk)
            u, v, _, ok = project(cams, k, pts[sl])
            res = torch.maximum((u - u.floor() - 0.5).abs(), (v - v.floor() - 0.5).abs())
            res = torch.where(ok, res, torch.full_like(res, np.inf))
            b = best[sl]
            better = res < b
            second[sl] = torch.where(better, b, torch.minimum(second[sl], res))
            best[sl] = torch.where(better, res, b)
            cam[sl] = torch.where(better, torch.full_like(cam[sl], k), cam[sl])
            px[sl] = torch.where(better, u.floor().long(), px[sl])
            py[sl] = torch.where(better, v.floor().long(), py[sl])
    return cam, px, py, best, second


# -------------------------------------------------------------------- rendering
BRANCHES = {
    "visible": ("field_visible", "proposal_sampler_visible", "density_fns_visible", "renderer_rgb"),
    "nir": ("field_nir", "proposal_sampler_nir", "density_fns_nir", "renderer_nir"),
}


def branch_geometry(model, rb: RayBundle, branch: str):
    """Collider + proposal sampling + density for one branch; nothing embedding-dependent."""
    fname, sname, dname, _ = BRANCHES[branch]
    field = getattr(model, fname)
    rb = model.collider(rb)
    ray_samples, _, _ = getattr(model, sname)(rb, density_fns=getattr(model, dname))
    density, geo = field.get_density(ray_samples)
    weights = ray_samples.get_weights(density)
    d = field.direction_encoding(get_normalized_directions(ray_samples.frustums.directions).view(-1, 3))
    base = torch.cat([d, geo.view(-1, field.geo_feat_dim)], dim=-1)
    return {"field": field, "samples": ray_samples, "weights": weights, "base": base,
            "shape": ray_samples.frustums.directions.shape[:-1]}


def branch_shade(model, g: Dict, branch: str, emb: torch.Tensor) -> torch.Tensor:
    """Rendered colour for embedding ``emb``: (E,) shared by all rays, or (R, E) one per ray."""
    field = g["field"]
    n_rays, n_s = g["shape"]
    if emb.dim() == 1:
        e = emb.view(1, -1).expand(n_rays * n_s, -1)
    else:
        e = emb[:, None, :].expand(n_rays, n_s, -1).reshape(n_rays * n_s, -1)
    h = torch.cat([g["base"], e.to(g["base"].dtype)], dim=-1)
    rgb = field.mlp_head(h).view(n_rays, n_s, -1).to(g["weights"])
    return getattr(model, BRANCHES[branch][3])(rgb=rgb, weights=g["weights"]).float()


def median_depth(model, g: Dict) -> torch.Tensor:
    return model.renderer_depth(weights=g["weights"], ray_samples=g["samples"]).float()


def emb_table(model, branch: str) -> torch.Tensor:
    return getattr(model, BRANCHES[branch][0]).embedding_appearance.embedding.weight


# ------------------------------------------------------------------------ misc
def bayer_site_channel(h: int, w: int, device) -> torch.Tensor:
    """(H, W) long: 0=R, 1=G, 2=B for BayerRG (RGGB)."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    ch = torch.ones((h, w), dtype=torch.long, device=device)
    ch[(yy % 2 == 0) & (xx % 2 == 0)] = 0
    ch[(yy % 2 == 1) & (xx % 2 == 1)] = 2
    return ch


# ------------------------------------------------------- training-frame cameras
def training_frame_cameras(pipe, config_path: Path, tol: float = 1e-7):
    """Return (train_cams, eval_cams, info) in the world frame the checkpoint was TRAINED in.

    jai_dataparser.py changed on 2026-06-11 from normalising poses per split (train frames
    only) to normalising over all frames.  Runs trained before that live in the per-split
    train frame, so the cameras eval_setup() rebuilds today are off by a small similarity
    transform (~0.6 % scale plus re-centring).  Detection: the scale saved at training time
    (<run>/dataparser_transforms.json) vs today's dataparser scale.  If they differ, poses of
    ALL frames are re-normalised with the train split's orient/centre/scale (reproducing the
    training frame for the train split, and putting eval frames in that same frame).
    """
    import json

    from nerfstudio.cameras import camera_utils
    from nerfstudio.data.dataparsers import jai_dataparser as jd

    dm = pipe.datamanager
    saved = json.loads((Path(config_path).parent / "dataparser_transforms.json").read_text())["scale"]
    current = float(dm.train_dataparser_outputs.dataparser_scale)
    info = {"saved_training_scale": saved, "current_scale": current}
    if abs(saved - current) <= tol * max(abs(saved), 1.0):
        info["frame"] = "current"
        return dm.train_dataset.cameras, dm.eval_dataset.cameras, info

    dp = dm.dataparser
    all_pairs, _ = dp._scan()
    train_pairs = dp._train_eval_split(all_pairs, "train")
    eval_pairs = dp._train_eval_split(all_pairs, "val")
    ref = all_pairs[0]
    c2w, *_ = jd._load_nsproc_poses(dp.config.poses_source, all_pairs, ref.width, ref.height)
    P = torch.cat([c2w, torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(len(all_pairs), 1, 4)], 1)
    is_train = torch.tensor([p in train_pairs for p in all_pairs])
    _, T = camera_utils.auto_orient_and_center_poses(P[is_train].clone(), method="up", center_method="poses")
    T4 = torch.cat([T, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], 0)
    P = T4 @ P
    s = 1.0 / float(P[is_train, :3, 3].abs().max())
    P[:, :3, 3] *= s
    if abs(s - saved) > 1e-6 * saved:
        raise RuntimeError(f"legacy per-split normalisation gives scale {s}, training saved {saved}")
    idx = {p.frame_idx: i for i, p in enumerate(all_pairs)}

    def rebuild(cams, pairs):
        c = copy_cameras(cams)
        c.camera_to_worlds = P[[idx[p.frame_idx] for p in pairs], :3, :].to(cams.camera_to_worlds)
        return c
    info.update(frame="legacy_per_split_train", legacy_scale=s)
    return rebuild(dm.train_dataset.cameras, train_pairs), rebuild(dm.eval_dataset.cameras, eval_pairs), info


def copy_cameras(cams: Cameras) -> Cameras:
    return Cameras(camera_to_worlds=cams.camera_to_worlds.clone(), fx=cams.fx, fy=cams.fy, cx=cams.cx, cy=cams.cy,
                   width=cams.width, height=cams.height, distortion_params=cams.distortion_params,
                   camera_type=cams.camera_type)
