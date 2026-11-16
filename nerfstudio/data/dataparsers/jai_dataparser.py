"""JAI FS-3200D dual-source dataparser for nerfstudio.

Scans a scene directory for paired Source0 (BayerRG12) and Source1 (Mono12 NIR)
raw files, validates consistency, and produces structured metadata suitable for
the JAI training pipeline.

Scene directory layout
----------------------
::

    scene_root/
        Source0_000000_2048x1536_16b.raw
        Source0_000000_2048x1536_16b.npy   (optional)
        Source1_000000_2048x1536_16b.raw
        Source1_000000_2048x1536_16b.npy   (optional)
        Source0_000001_2048x1536_16b.raw
        ...
        transforms.json                    (optional, for future use)

Pose integration
----------------
When ``poses_source`` is set to a path containing a ``transforms.json``
produced by ``ns-process-data images`` (from Source0 JPGs/PNGs processed
through COLMAP), the dataparser uses real camera poses and intrinsics
instead of identity placeholders.

The mapping from transforms.json ``frame_NNNNN.jpg`` entries to raw frame
pairs uses the fact that ``ns-process-data`` alphabetically sorts input
files and 1-indexes them:  raw ``frame_idx=0`` → ``frame_00001.jpg``,
``frame_idx=1`` → ``frame_00002.jpg``, etc.

Intrinsics from transforms.json are at the resolution of the processed
images.  They are scaled to match the raw image resolution (typically 2×).

Bit depth handling
------------------
Filenames encode the *container* bit depth (e.g. ``16b`` for uint16 storage).
The *effective* sensor bit depth (e.g. 12 for the JAI FS-3200D) is configured
separately and used for normalization.  See ``jai_utils.py`` for details.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.process_data.jai_utils import (
    DEFAULT_EFFECTIVE_BITS,
    JaiFramePair,
    SourceIntrinsics,
    load_transforms_json,
    parse_source_intrinsics,
    scan_scene_directory,
)


# ---------------------------------------------------------------------------
# Nerfstudio DataParser config + implementation
# ---------------------------------------------------------------------------

@dataclass
class JaiDataParserConfig(DataParserConfig):
    """Configuration for the JAI dual-source dataparser."""

    _target: Type = field(default_factory=lambda: JaiDataParser)
    """Target class to instantiate."""

    data: Path = Path()
    """Scene directory containing Source0/Source1 raw files."""

    poses_source: Optional[Path] = None
    """Path to an ns-process-data output directory (containing transforms.json)
    or directly to a transforms.json file.  When set, real camera poses and
    intrinsics are used instead of identity placeholders."""

    effective_bits: int = DEFAULT_EFFECTIVE_BITS
    """Effective sensor bit depth for normalization (default: 12)."""

    file_ext: str = "raw"
    """File extension to load: ``"raw"`` or ``"npy"``."""

    eval_fraction: float = 0.1
    """Fraction of frames to hold out for evaluation."""

    scene_scale: float = 1.0
    """Scale factor applied to the scene box."""

    intensity_mode: Literal["dn", "reflectance"] = "dn"
    """Input intensity mode.

    - ``"dn"`` (default): Source0/Source1 files store raw DN values as uint16;
      the dataset divides by ``2**effective_bits - 1`` to get [0,1] inputs.
      Backward-compatible with the original JAI pipeline.
    - ``"reflectance"``: Source0/Source1 files store pre-computed reflectance as
      float32 .npy in [0,1].  No DN normalization is applied.  Requires
      ``file_ext = "npy"``.  See ``scripts/jai_compute_reflectance.py``.
    """

    bayer_black_level: Optional[Tuple[float, float, float, float]] = None
    """Optional per-Bayer-channel black level (R, G1, G2, B) in DN units.

    Only used in ``intensity_mode = "dn"`` at the moment — in reflectance mode
    the black level has already been subtracted during preprocessing.  Kept on
    the dataparser so the same knob serves both current DN training and the
    future per-channel normalization path."""

    nir_black_level: Optional[float] = None
    """Optional NIR (Source1) black level in DN units (single scalar).
    Only applied in ``intensity_mode = "dn"``."""

    reflectance_clip_max: float = 1.0
    """Upper clip applied by JaiDataset to reflectance-mode inputs (lower clip stays 0).  1.0 = original behaviour.  Set to the
    NIR branch output bound (e.g. 3.0) to train on unclipped panel-relative values; NaN pixels pass through unchanged (2026-10-27)."""

    plant_mask_dir: Optional[Path] = None
    """Optional directory containing per-frame plant masks as .mask.npy files,
    produced by ``scripts/jai_segment_plant_foundation.py``.

    Filename convention: ``<Source0-stem>.mask.npy`` where ``<Source0-stem>``
    matches the Source0 raw/npy file stem for the same frame
    (e.g. ``Source0_000044_2048x1536_16b.mask.npy``).  Masks are bool (H, W)
    at raw Bayer resolution — the same resolution used by Source0 / Source1 —
    so no resampling is needed and the mask maps 1:1 to both sensors.

    When set, the dataset emits a ``plant_mask`` tensor per frame in addition
    to the existing keys.  The model only USES the mask when
    ``JaiNerfactoModelConfig.use_plant_mask_loss = True`` — otherwise the
    mask is sampled along with the rays but ignored, so setting this path
    alone is safe for debugging / QC without changing the optimization."""


# ---------------------------------------------------------------------------
# Pose loading helpers
# ---------------------------------------------------------------------------

# ns-process-data names: "images/frame_00001.jpg"
_NSPROC_FRAME_RE = re.compile(r"frame_(\d+)\.\w+$")


def _load_nsproc_poses(
    poses_path: Path,
    raw_pairs: List[JaiFramePair],
    raw_width: int,
    raw_height: int,
) -> Tuple[
    torch.Tensor,   # camera_to_worlds (N, 3, 4)
    float, float, float, float,  # fx, fy, cx, cy at raw resolution
    Optional[torch.Tensor],  # distortion_params (N, 6) or None
    List[str],       # warnings
]:
    """Load poses from an ns-process-data transforms.json and map to raw pairs.

    ns-process-data alphabetically sorts input images and 1-indexes them:
      Source0_000000_... → frame_00001.jpg
      Source0_000001_... → frame_00002.jpg
      ...
    So transforms frame number ``K`` corresponds to raw ``frame_idx = K - 1``.

    Intrinsics in transforms.json are at the processed image resolution.
    We scale them to match the raw resolution.
    """
    warnings: List[str] = []

    # Load transforms.json
    if poses_path.is_dir():
        tj_path = poses_path / "transforms.json"
    else:
        tj_path = poses_path
    if not tj_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {tj_path}")

    with open(tj_path, "r") as f:
        tj = json.load(f)

    # --- Parse intrinsics and compute scale ---
    tj_w = int(tj["w"])
    tj_h = int(tj["h"])
    scale_x = raw_width / tj_w
    scale_y = raw_height / tj_h

    fx = float(tj["fl_x"]) * scale_x
    fy = float(tj["fl_y"]) * scale_y
    cx = float(tj["cx"]) * scale_x
    cy = float(tj["cy"]) * scale_y

    # Distortion (OpenCV: k1, k2, p1, p2, k3, k4 — nerfstudio uses 6 params)
    has_distortion = "k1" in tj
    if has_distortion:
        k1 = float(tj.get("k1", 0))
        k2 = float(tj.get("k2", 0))
        p1 = float(tj.get("p1", 0))
        p2 = float(tj.get("p2", 0))
        k3 = float(tj.get("k3", 0))
        k4 = float(tj.get("k4", 0))
        dist_vec = [k1, k2, p1, p2, k3, k4]

    # --- Build frame-index to pose mapping ---
    # Key: raw frame_idx, Value: 4×4 transform matrix
    pose_by_raw_idx: Dict[int, List[List[float]]] = {}
    for frame_entry in tj["frames"]:
        fp = frame_entry["file_path"]
        m = _NSPROC_FRAME_RE.search(fp)
        if m is None:
            warnings.append(f"Could not parse frame number from '{fp}', skipping.")
            continue
        nsproc_num = int(m.group(1))       # 1-indexed
        raw_idx = nsproc_num - 1            # 0-indexed raw frame_idx
        pose_by_raw_idx[raw_idx] = frame_entry["transform_matrix"]

    # --- Map to raw pairs ---
    n = len(raw_pairs)
    c2w_list = []
    unmatched: List[int] = []

    for pair in raw_pairs:
        if pair.frame_idx in pose_by_raw_idx:
            mat = torch.tensor(pose_by_raw_idx[pair.frame_idx], dtype=torch.float32)
            c2w_list.append(mat[:3, :])   # (3, 4)
        else:
            unmatched.append(pair.frame_idx)
            # Identity fallback — will be reported
            c2w_list.append(torch.eye(4, dtype=torch.float32)[:3, :])

    if unmatched:
        warnings.append(
            f"{len(unmatched)} raw frames have no pose in transforms.json: "
            f"{unmatched[:10]}{'...' if len(unmatched) > 10 else ''}"
        )

    matched = n - len(unmatched)
    warnings.insert(0, f"Pose mapping: {matched}/{n} raw frames matched to transforms.json entries.")

    camera_to_worlds = torch.stack(c2w_list, dim=0)  # (N, 3, 4)

    distortion_params = None
    if has_distortion:
        distortion_params = torch.tensor([dist_vec] * n, dtype=torch.float32)

    return camera_to_worlds, fx, fy, cx, cy, distortion_params, warnings


# ---------------------------------------------------------------------------
# DataParser
# ---------------------------------------------------------------------------

@dataclass
class JaiDataParser(DataParser):
    """JAI dataparser that scans a scene directory and pairs frames.

    When ``config.poses_source`` is set, uses real camera poses from
    ns-process-data / COLMAP output.  Otherwise falls back to identity
    placeholders.
    """

    config: JaiDataParserConfig

    def __init__(self, config: JaiDataParserConfig):
        super().__init__(config)
        self._pairs: Optional[List[JaiFramePair]] = None
        self._warnings: Optional[List[str]] = None
        # Cached pose normalization computed over ALL frames (train + eval) so
        # every split lands in the SAME world coordinate frame.
        self._global_norm: Optional[Dict[str, Any]] = None

    def _global_pose_normalization(
        self, all_pairs: List[JaiFramePair], ref: JaiFramePair,
    ) -> Dict[str, Any]:
        """Orient/center/scale poses ONCE over all frames, then cache.

        The transform and scale must be derived from the full set of frames so
        that train and eval (and ns-render camera paths) share one coordinate
        frame.  Computing this per-split puts train and eval cameras in
        different frames, which makes eval / novel-view renders garbage.
        """
        if self._global_norm is not None:
            return self._global_norm

        c2w, fx, fy, cx, cy, dist_params, pose_warnings = _load_nsproc_poses(
            self.config.poses_source, all_pairs, ref.width, ref.height,
        )
        n_all = len(all_pairs)
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n_all, 1, 4)
        poses_4x4 = torch.cat([c2w, bottom], dim=1)  # (N, 4, 4)

        poses_4x4, transform = camera_utils.auto_orient_and_center_poses(
            poses_4x4, method="up", center_method="poses",
        )

        max_trans = float(torch.max(torch.abs(poses_4x4[:, :3, 3])))
        scale = 1.0 / max_trans if max_trans > 0 else 1.0
        poses_4x4[:, :3, 3] *= scale

        pose_warnings.append(
            f"Pose normalization (global, {n_all} frames): auto_orient + "
            f"auto_scale (dataparser_scale={scale:.6f})."
        )

        self._global_norm = {
            # frame_idx -> transformed (3, 4) camera_to_world
            "pose_map": {p.frame_idx: poses_4x4[i, :3, :] for i, p in enumerate(all_pairs)},
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "dist_row": dist_params[0] if dist_params is not None else None,
            "scale": scale,
            "transform": transform,
            "warnings": pose_warnings,
        }
        return self._global_norm

    def _scan(self) -> Tuple[List[JaiFramePair], List[str]]:
        """Lazy scan: only scan the directory once."""
        if self._pairs is None:
            self._pairs, self._warnings = scan_scene_directory(
                self.config.data,
                effective_bits=self.config.effective_bits,
                file_ext=self.config.file_ext,
            )
        return self._pairs, self._warnings

    def _train_eval_split(
        self, pairs: List[JaiFramePair], split: str,
    ) -> List[JaiFramePair]:
        """Split pairs into train / eval sets by evenly spacing eval frames."""
        n = len(pairs)
        if n <= 1:
            return pairs

        n_eval = max(1, int(n * self.config.eval_fraction))
        eval_indices = set(
            int(round(i * (n - 1) / (n_eval - 1))) if n_eval > 1 else {0}
            for i in range(n_eval)
        ) if n_eval > 1 else {0}

        if split == "train":
            return [p for i, p in enumerate(pairs) if i not in eval_indices]
        else:
            return [p for i, p in enumerate(pairs) if i in eval_indices]

    def _generate_dataparser_outputs(
        self, split: str = "train", **kwargs,
    ) -> DataparserOutputs:
        all_pairs, scan_warnings = self._scan()
        pairs = self._train_eval_split(all_pairs, split)

        n = len(pairs)
        if n == 0:
            raise ValueError(f"No frames in split '{split}'.")

        ref = pairs[0]
        image_filenames = [p.source0.path for p in pairs]
        warnings = list(scan_warnings)

        # --- Camera poses and intrinsics ---
        poses_source = self.config.poses_source
        has_real_poses = False
        distortion_params = None

        dataparser_scale = 1.0
        dataparser_transform = torch.eye(4, dtype=torch.float32)[:3, :]

        if poses_source is not None:
            # Normalize over ALL frames once (shared frame for every split),
            # then select this split's subset from the transformed poses.
            norm = self._global_pose_normalization(all_pairs, ref)
            warnings.extend(norm["warnings"])
            has_real_poses = True

            c2w = torch.stack([norm["pose_map"][p.frame_idx] for p in pairs], dim=0)  # (n, 3, 4)
            fx, fy, cx, cy = norm["fx"], norm["fy"], norm["cx"], norm["cy"]
            dataparser_scale = norm["scale"]
            dataparser_transform = norm["transform"]
            if norm["dist_row"] is not None:
                distortion_params = norm["dist_row"].unsqueeze(0).repeat(n, 1)
        else:
            # Placeholder identity poses
            fx = float(ref.width)
            fy = float(ref.width)
            cx = float(ref.width) / 2.0
            cy = float(ref.height) / 2.0
            c2w = torch.eye(4, dtype=torch.float32)[None, :3, :].repeat(n, 1, 1)

        cameras = Cameras(
            camera_to_worlds=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=ref.width,
            height=ref.height,
            camera_type=CameraType.PERSPECTIVE,
            distortion_params=distortion_params,
        )

        # --- Try loading transforms.json from scene root (for per-source intrinsics) ---
        transforms_path = self.config.data / "transforms.json"
        transforms_data = load_transforms_json(transforms_path)
        s0_intrinsics, s1_intrinsics = parse_source_intrinsics(transforms_data)

        # --- Metadata ---
        metadata: Dict[str, Any] = {
            "source0_paths": [p.source0.path for p in pairs],
            "source1_paths": [p.source1.path for p in pairs],
            "frame_indices": [p.frame_idx for p in pairs],
            "viewpoint_ids": [p.viewpoint_id for p in pairs],
            "source_id": [0] * n,

            "width": ref.width,
            "height": ref.height,
            "container_bits": ref.container_bits,
            "effective_bits": ref.effective_bits,

            "shared_viewpoint": True,
            "has_real_poses": has_real_poses,

            "source0_intrinsics": s0_intrinsics,
            "source1_intrinsics": s1_intrinsics,

            "jai_pairs": pairs,
            "jai_warnings": warnings,

            # Reflectance / hybrid-mode plumbing
            "intensity_mode": self.config.intensity_mode,
            "bayer_black_level": self.config.bayer_black_level,
            "nir_black_level": self.config.nir_black_level,
            "reflectance_clip_max": float(getattr(self.config, "reflectance_clip_max", 1.0)),

            # Foreground / plant mask plumbing
            "plant_mask_dir": self.config.plant_mask_dir,
        }

        if transforms_data is not None:
            metadata["transforms_json"] = transforms_data

        scene_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-scene_scale, -scene_scale, -scene_scale],
                 [scene_scale, scene_scale, scene_scale]],
                dtype=torch.float32,
            )
        )

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=dataparser_scale,
            dataparser_transform=dataparser_transform,
        )
